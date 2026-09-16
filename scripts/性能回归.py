"""Run with python -m scripts.性能回归; only load trusted local .pth files."""

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.perf_monitor import PerfMonitor
from src.perf_regression import (
    IndexedDataset, ObservedLoader, RegressionAudit, exact_difference,
    file_hash, fingerprint, indexed_collate, rng_state, source_manifest,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def setup(cfg, seed, audit):
    # Importing training sets the same CUBLAS environment as the normal entrypoint.
    from src.training import Experiment
    from src.seed import seed_everything
    from src.loaders import get_loaders

    experiment = Experiment(cfg, print_config=False, job_info=False)
    seed_everything(seed)
    loaders = get_loaders(
        experiment.dataset, experiment.backbone, experiment.partition, experiment.max_length,
        experiment.batch_size, experiment.workers, performance=experiment.performance,
    )
    manifests = {}
    observed = {}
    for split, loader in loaders.items():
        # Hashing is intentionally outside all timed epochs.
        manifests[split] = source_manifest(loader.dataset)
        options = dict(dataset=IndexedDataset(loader.dataset), batch_size=loader.batch_size,
                       shuffle=split == "train", num_workers=loader.num_workers,
                       worker_init_fn=loader.worker_init_fn, generator=loader.generator,
                       collate_fn=indexed_collate, pin_memory=loader.pin_memory,
                       drop_last=loader.drop_last, persistent_workers=loader.persistent_workers)
        if loader.num_workers:
            options["prefetch_factor"] = loader.prefetch_factor
        observed[split] = ObservedLoader(DataLoader(**options), split, audit)
    experiment.loaders = observed
    experiment.model = experiment.get_model().to(experiment.device)
    experiment.optimizer = experiment.get_optimizer(experiment.optimizer_name)
    experiment.scheduler_dict = {
        "name": experiment.scheduler_name,
        "obj": experiment.get_scheduler(experiment.scheduler_name, experiment.optimizer),
    }
    experiment.criterion = experiment.get_criterion()
    experiment.regression_audit = audit
    return experiment, manifests


def evaluate(experiment, split, audit, phase):
    from src.eval import Evaluation
    evaluation = Evaluation(
        experiment.model, experiment.loaders[split], experiment.criterion,
        experiment.device, experiment.factor, phase=phase,
        non_blocking_transfer=experiment.non_blocking_transfer,
        perf_monitor=experiment.perf_monitor, regression_audit=audit,
    )
    evaluation.compute_metrics()
    return evaluation.metrics


def run(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        _run(args)
    except Exception as error:
        write_json(folder / "失败报告.json", {"status": "失败", "error_type": type(error).__name__,
                                               "message": str(error)})
        raise


def _run(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg.get("config", cfg))
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    if cfg["dataloader"]["batch_size"] != 64:
        raise ValueError("Strict baseline requires batch_size=64")
    folder = Path(args.output)
    audit = RegressionAudit(folder / "数值证据") if args.mode != "benchmark" else None
    experiment, manifests = setup(cfg, seed, audit)
    experiment.perf_monitor = PerfMonitor(args.profile, args.interval, experiment.device)
    metadata = fingerprint(Path.cwd(), cfg)
    metadata.update(seed=seed, mode=args.mode, profile=args.profile, interval=args.interval,
                    requested_epochs=args.epochs, reference_split=args.reference_split)
    metadata["checkpoint"] = {"path": args.checkpoint, "sha256": file_hash(args.checkpoint)} if args.checkpoint else None
    write_json(folder / "运行元数据.json", metadata)
    write_json(folder / "数据清单.json", manifests)
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (folder / "依赖版本.txt").write_text(packages.stdout + packages.stderr, encoding="utf-8")
    if packages.returncode:
        raise RuntimeError("Could not collect dependency versions")
    if audit is not None:
        audit.save("初始状态", {"model": experiment.model.state_dict(),
            "optimizer": experiment.optimizer.state_dict(),
            "scheduler": experiment.scheduler_dict["obj"].state_dict()
            if experiment.scheduler_dict["obj"] is not None else None,
            "rng": rng_state(),
            "loaders": {key: value.loader.generator.get_state() for key, value in experiment.loaders.items()}})

    durations, epoch_metrics = [], []
    best_score, best_epoch = 0, 0
    best_written = False
    test_metrics = None
    if args.mode == "eval":
        checkpoint = torch.load(args.checkpoint, map_location=experiment.device, weights_only=False)
        experiment.model.load_state_dict(checkpoint["model"])
        test_metrics = evaluate(experiment, args.reference_split, audit, f"Fixed {args.reference_split}")
    else:
        epochs = args.epochs if args.epochs is not None else (4 if args.mode == "benchmark" else cfg["epochs"])
        for epoch in range(epochs):
            synchronize(experiment.device)
            start = perf_counter()
            metrics, _ = experiment.train_one_epoch(epoch)
            synchronize(experiment.device)
            durations.append(perf_counter() - start)
            numeric = {key: value for key, value in metrics.items() if key != "duration"}
            epoch_metrics.append(numeric)
            score = sum(value for key, value in metrics.items() if "vap" in key or "var" in key)
            if score > best_score:
                best_score, best_epoch = score, epoch
                if audit is not None:
                    # This isolated harness artifact does not alter production checkpoint paths.
                    torch.save({"model": experiment.model.state_dict()}, folder / "基准最佳.pth")
                    best_written = True
            if audit is not None:
                audit.save(f"轮次_{epoch + 1:03d}", {"metrics": numeric,
                    "scheduler": experiment.scheduler_dict["obj"].state_dict()
                    if experiment.scheduler_dict["obj"] is not None else None})
            print(f"epoch={epoch + 1} wall_seconds={durations[-1]:.6f}", flush=True)
            if args.mode == "capture" and epoch - best_epoch >= experiment.patience:
                break
        if args.mode == "capture":
            if not best_written:
                raise RuntimeError("No best checkpoint: baseline training never improved above zero")
            checkpoint = torch.load(folder / "基准最佳.pth", map_location=experiment.device, weights_only=False)
            experiment.model.load_state_dict(checkpoint["model"])
            test_metrics = evaluate(experiment, "test", audit, "Test")
            audit.save("最终训练状态", {"best_model": experiment.model.state_dict(),
                "optimizer": experiment.optimizer.state_dict(),
                "rng": rng_state(), "best_epoch": best_epoch + 1})

    experiment.perf_monitor.flush()
    write_json(folder / "批次顺序.json", {key: value.orders for key, value in experiment.loaders.items()})
    write_json(folder / "阶段耗时.json", experiment.perf_monitor.records)
    measured = durations[1:]  # The first complete epoch is warmup.
    report = {
        "status": "待验收", "mode": args.mode, "profile": args.profile,
        "epoch_wall_seconds": durations, "measured_seconds": measured,
        "mean_seconds": statistics.mean(measured) if measured else None,
        "median_seconds": statistics.median(measured) if measured else None,
        "epoch_metrics": epoch_metrics, "test_metrics": test_metrics,
        "best_epoch": best_epoch + 1 if epoch_metrics else None,
        "captured_samples": audit.counts if audit else None,
        "captured_steps": min(audit.step, 10) if audit else None,
        "timing_scope": "train_one_epoch: training + validation + scheduler; excludes checkpoint/report writes",
        "full_training_completed": bool(args.mode == "capture" and
            (len(durations) >= cfg["epochs"] or len(durations) - 1 - best_epoch >= experiment.patience)),
        "resources": {"gpu_util": None, "cpu_util": None, "ram": None, "io_wait": None,
                      "note": "Use an external sampler; unavailable here, never interpreted as zero."},
        "note": "Stage timings are sampled and non-additive. Capture timings are not performance evidence.",
    }
    write_json(folder / "验收报告.json", report)
    if audit:
        audit.save("运行结果", {"epoch_metrics": epoch_metrics, "test_metrics": test_metrics,
                                "best_epoch": report["best_epoch"]})


def compare(args):
    baseline, candidate = Path(args.baseline), Path(args.candidate)
    check_provenance(baseline, candidate)
    for name in ("数据清单.json", "批次顺序.json"):
        difference = exact_difference(json.loads((baseline / name).read_text(encoding="utf-8")),
                                      json.loads((candidate / name).read_text(encoding="utf-8")), name)
        if difference:
            raise ValueError(difference)
    left = sorted((baseline / "数值证据").glob("*.pth"))
    right = sorted((candidate / "数值证据").glob("*.pth"))
    if not left or [p.name for p in left] != [p.name for p in right]:
        raise ValueError("Missing evidence or evidence file sets differ")
    for a, b in zip(left, right):
        difference = exact_difference(torch.load(a, map_location="cpu", weights_only=False),
                                      torch.load(b, map_location="cpu", weights_only=False), a.name)
        if difference:
            raise ValueError(difference)
    report = json.loads((candidate / "验收报告.json").read_text(encoding="utf-8"))
    coverage = (report["mode"] == "capture" and report["captured_steps"] == 10
                and report["captured_samples"] == {"train": 1000, "val": 500, "test": 500}
                and report["full_training_completed"])
    print(json.dumps({"numeric_equality": True, "full_training_coverage": coverage,
                      "status": "数值证据一致；E13 归属、性能及原始分支等价性仍需验收"}, ensure_ascii=False, indent=2))


def check_provenance(baseline, candidate):
    left = json.loads((baseline / "运行元数据.json").read_text(encoding="utf-8"))
    right = json.loads((candidate / "运行元数据.json").read_text(encoding="utf-8"))
    for key in ("config", "seed", "mode", "requested_epochs", "reference_split", "python", "platform",
                "torch", "numpy", "cuda", "cudnn", "devices", "threads", "torch_threads",
                "torch_interop_threads", "deterministic", "cudnn_deterministic", "cudnn_benchmark",
                "matmul_tf32", "cudnn_tf32", "mha_fastpath"):
        difference = exact_difference(left[key], right[key], key)
        if difference:
            raise ValueError(difference)
    a = left["checkpoint"]["sha256"] if left["checkpoint"] else None
    b = right["checkpoint"]["sha256"] if right["checkpoint"] else None
    if a != b:
        raise ValueError("Reference checkpoint content differs")
    if (baseline / "依赖版本.txt").read_bytes() != (candidate / "依赖版本.txt").read_bytes():
        raise ValueError("Installed dependency versions differ")


def timings(args):
    if len(args.baselines) != len(args.candidates) or len(args.baselines) < 3:
        raise ValueError("At least three paired independent runs are required")
    paths = [Path(folder).resolve() for folder in args.baselines + args.candidates]
    if len(set(paths)) != len(paths):
        raise ValueError("Each timing run must have a distinct evidence directory")
    for a, b in zip(args.baselines, args.candidates):
        check_provenance(Path(a), Path(b))
        for name in ("数据清单.json", "批次顺序.json"):
            if exact_difference(json.loads((Path(a) / name).read_text(encoding="utf-8")),
                                json.loads((Path(b) / name).read_text(encoding="utf-8"))):
                raise ValueError(f"{name} differs")
        reports = [json.loads((Path(p) / "验收报告.json").read_text(encoding="utf-8")) for p in (a, b)]
        if exact_difference(reports[0]["epoch_metrics"], reports[1]["epoch_metrics"]):
            raise ValueError("Epoch numerical metrics differ")
    baseline, candidate = [], []
    for collection, destination in ((args.baselines, baseline), (args.candidates, candidate)):
        for folder in collection:
            report = json.loads((Path(folder) / "验收报告.json").read_text(encoding="utf-8"))
            if report["mode"] != "benchmark" or len(report["measured_seconds"]) < 3:
                raise ValueError("Performance evidence requires benchmark mode and >=3 measured epochs")
            if report["profile"] != (args.profiler_overhead and destination is candidate):
                raise ValueError("Unexpected profiler mode for timing comparison")
            destination.append(statistics.mean(report["measured_seconds"]))
    noise = (max(baseline) - min(baseline)) / statistics.mean(baseline)
    gains = [(a - b) / a for a, b in zip(baseline, candidate)]
    if args.profiler_overhead:
        passed = all(-gain < 0.02 for gain in gains)
    else:
        passed = all(gain > noise for gain in gains)
    print(json.dumps({"timing_gate_passed": passed, "paired_relative_gain": gains,
                      "baseline_relative_range": noise,
                      "status": "仅计时门槛；不代表优化通过全部验收"}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--config", required=True, help="Raw config or existing results JSON with config key")
    execute.add_argument("--output", required=True, help="New directory; existing paths are rejected")
    execute.add_argument("--mode", choices=("benchmark", "capture", "eval"), required=True)
    execute.add_argument("--seed", type=int)
    execute.add_argument("--epochs", type=int)
    execute.add_argument("--profile", action="store_true")
    execute.add_argument("--interval", type=int, default=100)
    execute.add_argument("--checkpoint")
    execute.add_argument("--reference-split", choices=("val", "test"))
    execute.set_defaults(func=run)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.set_defaults(func=compare)
    timing = commands.add_parser("timings")
    timing.add_argument("--baselines", nargs="+", required=True)
    timing.add_argument("--candidates", nargs="+", required=True)
    timing.add_argument("--profiler-overhead", action="store_true")
    timing.set_defaults(func=timings)
    args = parser.parse_args()
    if args.command == "run":
        if args.interval < 1 or (args.epochs is not None and args.epochs < 1):
            parser.error("interval and epochs must be positive")
        if args.mode == "eval" and (not args.checkpoint or not args.reference_split):
            parser.error("eval requires --checkpoint and --reference-split")
        if args.mode != "eval" and args.checkpoint:
            parser.error("Training comparisons start from the configured seed; --checkpoint is eval-only")
    args.func(args)


if __name__ == "__main__":
    main()
