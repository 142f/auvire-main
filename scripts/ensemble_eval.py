"""Evaluate a checkpoint whose quality branch is an ensemble of read-out heads.

All heads must share one architecture (they are trained from the same frozen
backbone, only the random initialisation differs). Logits are averaged, which is
the standard output-level ensemble; averaging the *weights* would be invalid
because the heads start from different random permutations.
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.eval import Evaluation
from src.loaders import get_loaders
from src.losses import CombinedLoss
from src.models import Model


class QualityEnsemble(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return [torch.stack([out[i] for out in outputs], dim=0).mean(dim=0) for i in range(len(outputs[0]))]


def main():
    parser = argparse.ArgumentParser(description="Ensembled quality read-out evaluation")
    parser.add_argument("--json", required=True, help="results json providing the config")
    parser.add_argument("--ckpt", required=True, help="backbone checkpoint (no quality head needed)")
    parser.add_argument("--heads", required=True, help="comma separated checkpoints carrying quality.* weights")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.json, "r") as handle:
        cfg = json.load(handle)["config"]
    model_cfg, encoder = cfg["model"], cfg["model"]["encoder"]
    factor = [1] * encoder["nlayers"]["retain"] + [2 ** (i + 1) for i in range(encoder["nlayers"]["downsample"])]
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
        # The backbone config (baseline json) carries no quality section; the heads
        # define the branch architecture, so build it explicitly.
        quality=(model_cfg.get("quality") or {"enabled": True, "readout": True, "use_offsets": True}),
    )
    base = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(base, strict=False)
    missing = [key for key in missing if not key.startswith("quality.")]
    print(f"[ensemble] backbone loaded; missing(non-quality)={missing} unexpected={unexpected}")

    heads = []
    for path in [item for item in args.heads.split(",") if item]:
        head = copy.deepcopy(model.quality)
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        quality_state = {key[len("quality."):]: value for key, value in state.items() if key.startswith("quality.")}
        if not quality_state:
            raise RuntimeError(f"{path} carries no quality.* weights")
        head.load_state_dict(quality_state)
        heads.append(head)
        print(f"[ensemble] + {os.path.basename(path)} ({len(quality_state)} tensors)")
    model.quality = QualityEnsemble(heads)
    model.to(args.device).eval()

    criterion = CombinedLoss(
        alpha=cfg["criterion"]["params"]["alpha"],
        gamma=cfg["criterion"]["params"]["gamma"],
        composition=["focal", "diou", "rec"],
        factor=factor,
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
        print(f"[ensemble] 限制为前 {args.limit} 条视频")
    evaluation = Evaluation(
        model=model,
        loader=loaders[args.split],
        criterion=criterion,
        device=args.device,
        factor=factor,
        phase=f"Ens {args.split}",
        show_progress=True,
    )
    os.environ["AUVIRE_BOUNDARY_DIAG"] = args.out
    evaluation.compute_metrics()
    print(json.dumps(evaluation.metrics, ensure_ascii=False))
    try:
        with open(args.out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        payload = {}
    payload["metrics"] = evaluation.metrics
    payload["limit"] = args.limit
    payload["heads"] = args.heads
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
