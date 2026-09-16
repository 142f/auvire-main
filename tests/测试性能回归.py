import ast
import datetime
import tempfile
import unittest
import json
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from src.eval import Evaluation, adjust_data
from src.loaders import collate_fn
from src.perf_monitor import PerfMonitor, batches, stage
from src.perf_regression import (
    IndexedDataset, ObservedLoader, RegressionAudit, cpu_copy,
    exact_difference, indexed_collate, source_manifest,
)
from scripts.性能回归 import timings, check_provenance, run


class Samples(Dataset):
    name, max_length = "lavdf", 4

    def __init__(self):
        self.videos = [(f"sample-{i}",) for i in range(21)]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        feature = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10 + index / 100
        labels = torch.tensor([[0., 0., 0.], [1., 0., 1.], [1., 1., 0.], [0., 0., 0.]])
        return [feature, feature + 0.5, labels, [[0.04, 0.08]]]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.dropout = torch.nn.Dropout(0.1)
        self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        result = self.linear(self.dropout(inputs[0] + inputs[1]))
        return [torch.cat((result[..., :1], result[..., 1:].abs() + 0.1), dim=-1)], None


def criterion(predictions, labels, z):
    return (predictions[0] - labels).square().mean()


class SmallEvaluation(Evaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_proposals_list["lavdf"] = [2, 1]


def training_methods():
    # Exercise the real loop without importing the unavailable optional model backend.
    tree = ast.parse(Path("src/training.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Experiment")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"train_one_epoch", "compute_metric", "adjust_metrics"}]
    namespace = dict(torch=torch, datetime=datetime, tqdm=tqdm, average_precision_score=average_precision_score,
                     adjust_data=adjust_data, Evaluation=SmallEvaluation, stage=stage, batches=batches)
    exec(compile(ast.Module(body=methods, type_ignores=[]), "src/training.py", "exec"), namespace)
    return type("LoopFixture", (), {node.name: namespace[node.name] for node in methods})


class PerformanceTests(unittest.TestCase):
    def test_failed_run_is_reported_and_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            args = SimpleNamespace(output=str(output), config=str(Path(folder) / "missing.json"))
            with self.assertRaises(FileNotFoundError):
                run(args)
            report = next(output.glob("*.json"))
            original = report.read_bytes()
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(report.read_bytes(), original)

    def test_disabled_monitor_never_touches_cuda(self):
        monitor = PerfMonitor(False, device="cuda")
        with patch("torch.cuda.Event", side_effect=AssertionError("unexpected event")):
            for _ in monitor.iterate([1, 2, 3], "test"):
                with monitor.stage("forward", gpu=True):
                    pass
        self.assertEqual(monitor.records, [])

    def test_sampling_and_tail(self):
        monitor = PerfMonitor(True, interval=2)
        for _ in monitor.iterate([1, 2, 3, 4, 5], "train"):
            with monitor.stage("forward", gpu=True):
                pass
        self.assertEqual([r["batch"] for r in monitor.records if r["stage"] == "forward"], [1, 3])
        self.assertFalse(monitor.active)
        for record in monitor.records:
            self.assertGreaterEqual(record["cpu_seconds"], 0)

    def test_cuda_events_flush_only_at_boundary(self):
        events = []
        class Event:
            def __init__(self, **kwargs):
                self.waits = 0
                events.append(self)
            def record(self, stream):
                pass
            def synchronize(self):
                self.waits += 1
            def elapsed_time(self, other):
                return 2.0
        monitor = PerfMonitor(True, interval=1, device="cuda")
        with patch("torch.cuda.Event", Event), patch("torch.cuda.current_stream", return_value=None):
            for _ in monitor.iterate([0], "test"):
                for name in ("forward", "loss", "backward"):
                    with monitor.stage(name, gpu=True):
                        pass
                self.assertEqual(sum(e.waits for e in events), 0)
        self.assertEqual(sum(e.waits for e in events), 1)
        self.assertEqual(events[-1].waits, 1)

    def test_exact_comparison_rejects_structural_and_numeric_changes(self):
        value = {"grad": None, "x": torch.ones(3)}
        self.assertIsNone(exact_difference(value, cpu_copy(value)))
        self.assertIsNotNone(exact_difference(None, torch.zeros(1)))
        self.assertIsNotNone(exact_difference(torch.ones(1), torch.ones(1, dtype=torch.float64)))
        self.assertIsNotNone(exact_difference([1], (1,)))
        self.assertIsNotNone(exact_difference({"a": 1, "b": 2}, {"b": 2, "a": 1}))
        self.assertIsNotNone(exact_difference(torch.zeros(1), torch.full((1,), 1e-9)))

    def test_observed_loader_matches_original_order_and_rng(self):
        dataset = Samples()
        originals, observed = [], []
        g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=g1, collate_fn=collate_fn)
        tracked = ObservedLoader(DataLoader(IndexedDataset(dataset), batch_size=2, shuffle=True,
                                generator=g2, collate_fn=indexed_collate), "train")
        for _ in range(2):
            originals.append(list(loader))
            observed.append(list(tracked))
        self.assertIsNone(exact_difference(originals, observed))
        self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))
        self.assertEqual(len(tracked.orders), 2)
        self.assertEqual(len(tracked.orders[0][-1]), 1)

    def test_spawn_workers_preserve_two_epoch_order(self):
        dataset = Samples()
        g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
        options = dict(batch_size=2, shuffle=True, num_workers=2, persistent_workers=True,
                       prefetch_factor=4, multiprocessing_context="spawn")
        original = DataLoader(dataset, generator=g1, collate_fn=collate_fn, **options)
        tracked = ObservedLoader(DataLoader(IndexedDataset(dataset), generator=g2,
                                 collate_fn=indexed_collate, **options), "train")
        for _ in range(2):
            self.assertIsNone(exact_difference(list(original), list(tracked)))
        self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))

    def make_timing_pair(self, folder, index, a, b):
        from src.perf_regression import fingerprint
        paths = [Path(folder) / f"{side}{index}" for side in ("a", "b")]
        metadata = fingerprint(Path(folder), {"batch_size": 64})
        metadata.update(seed=1, mode="benchmark", requested_epochs=4, reference_split=None, checkpoint=None)
        for path, duration, profile in zip(paths, (a, b), (False, True)):
            path.mkdir()
            for name, content in {
                "运行元数据.json": metadata, "数据清单.json": {"samples": [1]},
                "批次顺序.json": [[1]], "验收报告.json": {"mode": "benchmark", "profile": profile,
                    "measured_seconds": [duration] * 3, "epoch_metrics": [{"loss": 1.0}]},
            }.items():
                (path / name).write_text(json.dumps(content), encoding="utf-8")
            (path / "依赖版本.txt").write_text("torch==test", encoding="utf-8")
        return paths

    def test_profiler_overhead_gate_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            pairs = [self.make_timing_pair(folder, i, 100, 101) for i in range(3)]
            args = SimpleNamespace(baselines=[str(p[0]) for p in pairs],
                                   candidates=[str(p[1]) for p in pairs], profiler_overhead=True)
            timings(args)
            path = pairs[0][1] / "验收报告.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["measured_seconds"] = [103] * 3
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(SystemExit):
                timings(args)
            args.candidates[1] = args.candidates[0]
            with self.assertRaises(ValueError):
                timings(args)

    def test_provenance_rejects_changed_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = self.make_timing_pair(folder, 0, 100, 101)
            path = b / "运行元数据.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["matmul_tf32"] = not value["matmul_tf32"]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matmul_tf32"):
                check_provenance(a, b)

    def test_npy_precedence_and_content_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.npz"
            np.savez(path, video_features=np.ones((2, 2)), audio_features=np.ones((2, 2)))
            dataset = Samples()
            dataset.backbone = "other"
            dataset.videos = [(str(path), 1, 0, [])]
            before = source_manifest(dataset)
            self.assertEqual(len(before[0]["sources"]), 1)
            np.save(path.with_suffix(".video.npy"), np.ones((2, 2)))
            self.assertEqual(source_manifest(dataset), before)
            np.save(path.with_suffix(".audio.npy"), np.ones((2, 2)))
            after = source_manifest(dataset)
            self.assertEqual(len(after[0]["sources"]), 2)
            np.save(path.with_suffix(".audio.npy"), np.zeros((2, 2)))
            self.assertNotEqual(source_manifest(dataset), after)

    def run_loop(self, enabled, device, folder):
        torch.manual_seed(123)
        fixture = training_methods()()
        fixture.model = TinyModel().to(device)
        fixture.optimizer = torch.optim.Adam(fixture.model.parameters(), lr=0.001)
        fixture.scheduler_dict = {"name": "step", "obj": torch.optim.lr_scheduler.StepLR(fixture.optimizer, 1)}
        fixture.criterion = criterion
        fixture.device, fixture.dataset, fixture.factor = device, "lavdf", [1]
        fixture.non_blocking_transfer, fixture.show_progress = False, False
        fixture.training_metric_interval_batches = fixture.progress_interval_batches = 50
        fixture.perf_monitor = PerfMonitor(enabled, interval=2, device=device)
        fixture.regression_audit = RegressionAudit(folder)
        fixture.loaders = {split: DataLoader(Samples(), batch_size=2, collate_fn=collate_fn) for split in ("train", "val")}
        metrics, _ = fixture.train_one_epoch(0)
        metrics.pop("duration")
        return metrics, cpu_copy(fixture.model.state_dict()), cpu_copy(fixture.optimizer.state_dict()), fixture.perf_monitor

    def check_loop(self, device):
        with tempfile.TemporaryDirectory() as folder:
            a = self.run_loop(False, device, Path(folder) / "关闭")
            b = self.run_loop(True, device, Path(folder) / "开启")
            self.assertIsNone(exact_difference(a[:3], b[:3]))
            files = sorted((Path(folder) / "关闭").glob("*.pth"))
            self.assertEqual(len([p for p in files if p.name.startswith("训练步")]), 10)
            for path in files:
                self.assertIsNone(exact_difference(
                    torch.load(path, weights_only=False),
                    torch.load(Path(folder) / "开启" / path.name, weights_only=False)), path.name)
            self.assertIn("softnms", {r["stage"] for r in b[3].records})
            self.assertIn("d2h", {r["stage"] for r in b[3].records})

    def test_cpu_training_and_eval_profiler_equality(self):
        self.check_loop("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_and_eval_profiler_equality(self):
        self.check_loop("cuda")

    def test_exception_resets_window(self):
        monitor = PerfMonitor(True, interval=1)
        iterator = monitor.iterate([1], "test")
        next(iterator)
        iterator.close()
        self.assertFalse(monitor.active)


if __name__ == "__main__":
    unittest.main()
