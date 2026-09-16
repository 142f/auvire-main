"""Opt-in sampled timings. Stage timings are not additive wall-clock timings."""

from contextlib import contextmanager, nullcontext
from time import perf_counter

import torch


class PerfMonitor:
    def __init__(self, enabled=False, interval=100, device="cpu"):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be a positive integer")
        self.enabled = enabled
        self.interval = interval
        self.device = torch.device(device)
        self.active = False
        self.records = []
        self.pending = []
        self.context = {}

    def stage(self, name, gpu=False, always=False):
        if not self.enabled or not (self.active or always):
            return nullcontext()
        return self._stage(name, gpu)

    @contextmanager
    def _stage(self, name, gpu):
        use_cuda = gpu and self.device.type == "cuda"
        start = end = None
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        before = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - before
            record = {**self.context, "stage": name, "cpu_seconds": elapsed}
            if use_cuda:
                end.record(torch.cuda.current_stream(self.device))
                self.pending.append((record, start, end))
            else:
                self.records.append(record)

    def flush(self):
        if self.pending:
            # One boundary synchronization, never one synchronization per stage.
            self.pending[-1][2].synchronize()
            for record, start, end in self.pending:
                record["gpu_seconds"] = start.elapsed_time(end) / 1000
                self.records.append(record)
            self.pending.clear()

    def iterate(self, loader, phase):
        if not self.enabled:
            yield from loader
            return
        iterator = iter(loader)
        try:
            for index in range(len(loader)):
                self.context = {"phase": phase, "batch": index}
                self.active = (index + 1) % self.interval == 0
                with self.stage("data_wait"):
                    batch = next(iterator)
                yield batch
                self.flush()
        finally:
            self.flush()
            self.active = False


def stage(monitor, name, gpu=False, always=False):
    return monitor.stage(name, gpu, always) if monitor is not None else nullcontext()


def batches(monitor, loader, phase):
    return monitor.iterate(loader, phase) if monitor is not None else loader
