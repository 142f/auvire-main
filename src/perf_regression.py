"""Explicit, expensive regression capture; never enabled by normal training."""

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.loaders import collate_fn


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def exact_difference(left, right, path="root"):
    """Return the first mismatch, including dtype, shape, container and None."""
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            return f"{path}: tensor mismatch"
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            return f"{path}: numpy mismatch"
    elif isinstance(left, dict):
        if list(left) != list(right):
            return f"{path}: keys/order mismatch"
        for key in left:
            difference = exact_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: length mismatch"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = exact_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
    elif left != right:
        return f"{path}: value mismatch"
    return None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(root, cfg):
    root = Path(root)
    files = sorted({path for folder in ("src", "scripts") for path in (root / folder).rglob("*.py")})
    files += [root / "requirements.txt"]
    sources = {p.relative_to(root).as_posix(): file_hash(p) for p in files if p.is_file()}
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {
        "config": cfg,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "source_files": sources,
        "code_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "commit": git("rev-parse", "HEAD"),
        "worktree": git("status", "--porcelain"),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "threads": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")},
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
    }


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def source_manifest(dataset):
    """Describe the current loader's actual selection without modifying it."""
    result = []
    for index, entry in enumerate(dataset.videos):
        path = entry[0]
        if dataset.backbone == "avhubert":
            path = path.replace("mediapipe/", "").replace("LAV-DF_emb", "LAV-DF_emb_avhubert")
            path = path.replace("AV-Deepfake1M_emb", "AV-Deepfake1M_emb_avhubert")
        stem = os.path.splitext(path)[0]
        npys = [stem + ".video.npy", stem + ".audio.npy"]
        selected = npys if all(os.path.exists(p) for p in npys) else [path]
        result.append({"index": index, "id": entry[0], "metadata": entry[1:],
                       "sources": [{"path": p, "sha256": file_hash(p)} for p in selected]})
    return result


class IndexedDataset(Dataset):
    """Expose IDs in the harness only, without a second sampler iteration."""
    def __init__(self, dataset):
        self.inner = dataset
        self.name = dataset.name
        self.max_length = dataset.max_length

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        sample = self.inner[index]
        if sample is None:
            raise RuntimeError(f"Regression capture cannot accept a skipped sample: {self.inner.videos[index][0]}")
        return index, self.inner.videos[index][0], sample


def indexed_collate(items):
    return collate_fn([item[2] for item in items]), [(item[0], item[1]) for item in items]


class ObservedLoader:
    def __init__(self, loader, split, audit=None):
        self.loader, self.split, self.audit = loader, split, audit
        self.dataset = loader.dataset
        self.orders = []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        order = []
        self.orders.append(order)
        for batch, ids in self.loader:
            order.append(ids)
            if self.audit is not None:
                self.audit.input_batch(self.split, batch, ids)
            yield batch


class RegressionAudit:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.step = 0
        self.pending = None
        self.counts = {"train": 0, "val": 0, "test": 0}
        self.limits = {"train": 1000, "val": 500, "test": 500}
        self.input_chunks = 0
        self.evaluations = 0

    def save(self, name, value):
        torch.save(cpu_copy(value), self.folder / f"{name}.pth")

    def input_batch(self, split, batch, ids):
        count = min(len(ids), self.limits[split] - self.counts[split])
        if count <= 0:
            return
        features, periods = batch
        self.save(f"输入_{self.input_chunks:06d}", {
            "split": split, "ids": ids[:count],
            "tensors": [tensor[:count] for tensor in features], "periods": periods[:count]})
        self.counts[split] += count
        self.input_chunks += 1

    def before_step(self, experiment, predictions, loss):
        if self.step < 10:
            self.pending = cpu_copy({"predictions": predictions, "loss": loss,
                "gradients": {key: param.grad for key, param in experiment.model.named_parameters()}})

    def after_step(self, experiment):
        if self.step < 10:
            self.save(f"训练步_{self.step:02d}", {
                **self.pending, "model": experiment.model.state_dict(),
                "optimizer": experiment.optimizer.state_dict(),
                "scheduler": experiment.scheduler_dict["obj"].state_dict()
                if experiment.scheduler_dict["obj"] is not None else None,
                "rng": rng_state()})
            self.pending = None
        self.step += 1

    def evaluation_predictions(self, evaluation):
        self.save(f"评估_{self.evaluations:04d}_原始预测", {
            "phase": evaluation.phase, "predictions": evaluation.predictions,
            "ground_truth": evaluation.ground_truth, "loss": evaluation.loss})

    def evaluation_metrics(self, evaluation, proposals):
        self.save(f"评估_{self.evaluations:04d}_指标", {
            "phase": evaluation.phase, "proposals": proposals, "metrics": evaluation.metrics})
        self.evaluations += 1
