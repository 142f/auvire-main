import torch
from torch.utils.data import DataLoader

import numpy as np
import random

from src.datasets import LAVDF, AVDeepFake1M


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise RuntimeError("The complete batch is invalid; inspect the dataset error for the source path.")
    videos = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    periods = [item[3] for item in batch]
    rectangular = all(tensor.shape == videos[0].shape for tensor in videos) and all(
        tensor.shape == audios[0].shape for tensor in audios
    )
    if rectangular:
        stacked = torch.utils.data.dataloader.default_collate([item[:3] for item in batch])
        return stacked, periods
    targets = torch.utils.data.dataloader.default_collate([item[2] for item in batch])
    return (videos, audios, targets), periods


def get_dataset(dataset, backbone, partition, split, max_length, without, showsize):
    if dataset == "lavdf":
        return LAVDF(backbone, split, max_length, showsize)
    elif dataset == "avdeepfake1m":
        return AVDeepFake1M(backbone, split, max_length, partition, showsize)
    else:
        raise Exception(f"Dataset {dataset} is not supported.")


def get_loaders(
    dataset,
    backbone,
    partition,
    max_length,
    batch_size,
    workers,
    without="none",
    splits=None,
    showsize=True,
    performance=None,
):
    if splits is None:
        splits = ["train", "val", "test"]

    performance = performance or {}
    persistent_workers = bool(performance.get("persistent_workers", True)) and workers > 0
    prefetch_factor = performance.get("prefetch_factor", 2)
    loaders = {}
    split_seeds = {"train": 0, "val": 1, "test": 2}
    for split in splits:
        generator = torch.Generator()
        generator.manual_seed(split_seeds.get(split, 0))
        loader_options = {
            "dataset": get_dataset(dataset, backbone, partition, split, max_length, without, showsize),
            "batch_size": batch_size,
            "shuffle": split == "train",
            "num_workers": workers,
            "worker_init_fn": seed_worker,
            "generator": generator,
            "collate_fn": collate_fn,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": persistent_workers,
        }
        if workers > 0:
            loader_options["prefetch_factor"] = prefetch_factor
        loaders[split] = DataLoader(**loader_options)
    return loaders
