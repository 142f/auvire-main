"""Assemble the D5 results table from diag JSON files.

Every row must be traceable to a file under results/diag/ produced by
scripts/diag_boundary.py (which now folds the metrics into the same JSON).
"""
import argparse
import glob
import json
import os


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="+", help="glob(s) of diag json files")
    parser.add_argument("--out", default=None, help="optional markdown output")
    args = parser.parse_args()

    rows = []
    for pattern in args.pattern:
        for path in sorted(glob.glob(pattern)):
            try:
                payload = load(path)
            except (OSError, ValueError):
                continue
            metrics = payload.get("metrics")
            if not metrics:
                continue
            ap, ar = metrics["ap"], metrics["ar"]
            rows.append(
                {
                    "file": os.path.basename(path),
                    "limit": payload.get("limit"),
                    "ap50": 100 * ap["0.5"],
                    "ap75": 100 * ap["0.75"],
                    "ap95": 100 * ap["0.95"],
                    "ar100": 100 * ar["100"],
                    "unreachable": payload.get("unreachable_095"),
                    "segments": payload.get("segments"),
                }
            )

    base = next((row for row in rows if row["file"].endswith("_a10.json") or "baseline" in row["file"]), None)
    header = "| run | limit | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | dAP@0.95 |\n|---|---|---|---|---|---|---|"
    lines = [header]
    for row in rows:
        delta = "" if base is None else f"{row['ap95'] - base['ap95']:+.2f}"
        lines.append(
            f"| {row['file']} | {row['limit']} | {row['ap50']:.2f} | {row['ap75']:.2f} | "
            f"{row['ap95']:.2f} | {row['ar100']:.2f} | {delta} |"
        )
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
