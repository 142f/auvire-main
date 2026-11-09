import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.eval import Evaluation
from src.loaders import get_loaders
from src.losses import CombinedLoss
from src.models import Model


def main():
    parser = argparse.ArgumentParser(description="Boundary diagnostic (read-only)")
    parser.add_argument("--json", required=True, help="results json providing config")
    parser.add_argument("--ckpt", required=True, help=".pth to evaluate")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 条视频（0=全部）")
    args = parser.parse_args()

    with open(args.json, "r") as handle:
        cfg = json.load(handle)["config"]
    model_cfg, encoder = cfg["model"], cfg["model"]["encoder"]
    factor = [1] * encoder["nlayers"]["retain"] + [
        2 ** (i + 1) for i in range(encoder["nlayers"]["downsample"])
    ]
    model = Model(
        max_length=cfg["dataset"]["params"]["max_length"],
        d_model=model_cfg["d_model"],
        win_size=model_cfg["win_size"],
        num_heads=model_cfg["num_heads"],
        operation=model_cfg["operation"],
        reconstruction=model_cfg["reconstruction"],
        encoder=encoder,
        dropout=model_cfg["dropout"],
        use_ln=model_cfg["conv"]["use_ln"],
        use_rl=model_cfg["conv"]["use_rl"],
        use_do=model_cfg["conv"]["use_do"],
        model_type=model_cfg["type"],
        factor=factor,
        device=args.device,
        quality=model_cfg.get("quality"),
    )
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    model.to(args.device).eval()

    criterion = CombinedLoss(
        alpha=cfg["criterion"]["params"]["alpha"],
        gamma=cfg["criterion"]["params"]["gamma"],
        composition=cfg["criterion"]["composition"],
        factor=factor,
        quality_beta=cfg["criterion"]["params"].get("quality_beta", 2.0),
        quality_weight=cfg["criterion"]["params"].get("quality_weight", 0.5),
    )
    loaders = get_loaders(
        dataset=cfg["dataset"]["name"],
        backbone=cfg["dataset"]["backbone"],
        partition=cfg["dataset"]["params"].get("partition", "whole"),
        splits=[args.split],
        showsize=False,
        max_length=cfg["dataset"]["params"]["max_length"],
        batch_size=cfg["dataloader"]["batch_size"],
        workers=cfg["dataloader"]["workers"],
        performance=cfg.get("performance"),
    )
    if args.limit:
        loaders[args.split].dataset.videos = loaders[args.split].dataset.videos[: args.limit]
        print(f"[diag] 限制为前 {args.limit} 条视频")
    evaluation = Evaluation(
        model=model,
        loader=loaders[args.split],
        criterion=criterion,
        device=args.device,
        factor=factor,
        phase=f"Diag {args.split}",
        show_progress=True,
    )
    os.environ["AUVIRE_BOUNDARY_DIAG"] = args.out
    evaluation.compute_metrics()
    print(json.dumps({k: v for k, v in evaluation.metrics.items()}, ensure_ascii=False))
    # Fold the metrics into the same JSON so a sweep can be parsed without the log.
    try:
        with open(args.out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        payload = {}
    payload["metrics"] = evaluation.metrics
    payload["limit"] = args.limit
    payload["ckpt"] = os.path.abspath(args.ckpt)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
