"""Follow runs/epoch_times.jsonl and print each epoch's stage times.

Start it in another terminal before or during training:

    python scripts/watch_epoch_times.py

The training process appends the file. Stopping this script does not stop training.
"""

import argparse
import json
import os
import time

DEFAULT_PATH = os.path.join("runs", "epoch_times.jsonl")
STAGE_ORDER = ("train", "val_forward", "transform", "softnms", "ap", "ar", "metrics", "epoch")


def log_path():
    return os.environ.get("AUVIRE_EPOCH_LOG", DEFAULT_PATH)


def format_epoch(records):
    """Render one epoch. The epoch line is the wall-clock total used for percentages."""
    if not records:
        return ""
    epoch = records[0]["epoch"]
    by_stage = {record["stage"]: record for record in records}
    total = by_stage.get("epoch", records[-1])["seconds"]
    ordered = [stage for stage in STAGE_ORDER if stage in by_stage]
    ordered.extend(stage for stage in by_stage if stage not in ordered)
    lines = [f"Epoch {epoch}  {total:.1f}s"]
    for stage in ordered:
        record = by_stage[stage]
        seconds = record["seconds"]
        share = 100.0 * seconds / total if total else 0.0
        batches = record.get("batches")
        batch_text = f"  {batches} batches" if batches is not None else ""
        lines.append(f"  {stage:<12} {seconds:8.1f}s  {share:5.1f}%{batch_text}")
    return "\n".join(lines)


def iter_lines(path, once):
    announced = False
    while not os.path.exists(path):
        if once:
            raise SystemExit(f"No epoch log at {path}")
        if not announced:
            print(f"Waiting for {path}", flush=True)
            announced = True
        time.sleep(1)
    with open(path, "r", encoding="utf-8") as handle:
        pending = ""
        while True:
            chunk = handle.readline()
            if chunk:
                pending += chunk
                if pending.endswith("\n"):
                    yield pending
                    pending = ""
                continue
            if once:
                if pending.strip():
                    yield pending if pending.endswith("\n") else pending + "\n"
                return
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Print per-epoch stage times while training runs")
    parser.add_argument("--path", default=None, help="JSONL path. Defaults to AUVIRE_EPOCH_LOG or runs/epoch_times.jsonl")
    parser.add_argument("--once", action="store_true", help="Print the current file and exit")
    args = parser.parse_args()
    path = args.path or log_path()
    grouped = {}
    for line in iter_lines(path, args.once):
        text = line.strip()
        if not text:
            continue
        record = json.loads(text)
        epoch = record["epoch"]
        grouped.setdefault(epoch, []).append(record)
        print(f"epoch {epoch}  {record['stage']:<12} {record['seconds']:8.1f}s", flush=True)
        if record["stage"] != "epoch":
            seen = {item["stage"] for item in grouped[epoch]}
            upcoming = [stage for stage in STAGE_ORDER if stage not in seen]
            if upcoming:
                print(f"epoch {epoch}  waiting for {upcoming[0]}", flush=True)
            continue
        print(format_epoch(grouped[epoch]), flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
