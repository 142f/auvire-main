import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import Experiment
from src.config import load_default
from src.perf_monitor import PerfMonitor

def main():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "-d",
        "--dataset_name",
        help="The name of the dataset to train on",
        choices=["lavdf", "avdeepfake1m"],
    )
    args = parser.parse_args()
    dataset = args.dataset_name
    cfg = load_default(dataset)
    # Measurement-only overrides. Unset variables leave the published defaults.
    if os.environ.get("AUVIRE_EPOCHS"):
        cfg["epochs"] = int(os.environ["AUVIRE_EPOCHS"])
    if os.environ.get("AUVIRE_WORKERS"):
        cfg["dataloader"]["workers"] = int(os.environ["AUVIRE_WORKERS"])
    if os.environ.get("AUVIRE_KEEP_CKPT"):
        # Measurement-only override: keep the .pth after training so the arm can be
        # re-inspected offline. Does not change any training numerics.
        cfg["delete_ckpt"] = False
    folder = os.environ.get("AUVIRE_FOLDER", "ckpt")
    e = Experiment(cfg=cfg, folder=folder, print_config=False, job_info=False)
    if os.environ.get("AUVIRE_PERF") == "1":
        e.perf_monitor = PerfMonitor(
            enabled=True,
            interval=max(int(os.environ.get("AUVIRE_PERF_INTERVAL", "1")), 1),
            device=cfg["device"],
        )
    e.run()
    perf_out = os.environ.get("AUVIRE_PERF_OUT")
    monitor = getattr(e, "perf_monitor", None)
    if perf_out and monitor is not None:
        monitor.flush()
        os.makedirs(os.path.dirname(perf_out) or ".", exist_ok=True)
        with open(perf_out, "w", encoding="utf-8") as handle:
            json.dump(monitor.records, handle)


if __name__ == "__main__":
    main()
