"""Write the reproducibility record (meta.json) for one experiment arm.

Every number in the paper's tables must be traceable to one of these files: the
exact command, the environment, the code hashes, the hardware, the wall clock,
and the metrics that the run itself printed at the end of the log.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys


def source_hashes(root, files):
    out = {}
    for name in files:
        path = os.path.join(root, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                out[name] = hashlib.md5(handle.read()).hexdigest()
    return out


def gpu_peak(path):
    peak = 0
    util = []
    if not os.path.exists(path):
        return {"peak_memory_mib": None, "mean_utilization_pct": None}
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) != 2:
                continue
            try:
                peak = max(peak, int(parts[0]))
                util.append(int(parts[1]))
            except ValueError:
                continue
    return {
        "peak_memory_mib": peak or None,
        "mean_utilization_pct": round(sum(util) / len(util), 2) if util else None,
    }


def final_metrics(log_path):
    """The last JSON object printed by the run (the test block)."""
    if not os.path.exists(log_path):
        return None
    text = open(log_path, "r", errors="ignore").read()
    blocks = re.findall(r"\{\s*\"tloss\".*?\}", text, flags=re.S)
    if not blocks:
        return None
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None


def versions():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:  # pragma: no cover - diagnostic only
        info["torch_error"] = str(error)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--rc", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {key: value for key, value in os.environ.items() if key.startswith("AUVIRE_")}
    env["GIT_COMMIT"] = os.environ.get("GIT_COMMIT", "unavailable-on-remote")
    payload = {
        "arm": args.arm,
        "command": "python3 scripts/train.py -d lavdf",
        "env": env,
        "exit_code": args.rc,
        "wall_clock_seconds": args.end - args.start,
        "log": os.path.abspath(args.log),
        "git_commit": env["GIT_COMMIT"],
        "source_md5": source_hashes(
            root,
            [
                "src/models.py",
                "src/losses.py",
                "src/eval.py",
                "src/training.py",
                "src/datasets.py",
                "src/metrics.py",
                "src/post_process.py",
                "src/config.py",
                "scripts/train.py",
            ],
        ),
        "versions": versions(),
        **gpu_peak(args.gpu),
        "test_metrics": final_metrics(args.log),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"meta": args.out, "test_metrics": payload["test_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
