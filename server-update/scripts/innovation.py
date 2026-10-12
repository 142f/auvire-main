import argparse

from src.config import load_default
from src.training import Experiment

REFINE = {"enabled": True, "hidden": 32, "kernel_size": 5, "levels": None, "gate": None}
REFINE_GATE = {
    "enabled": True,
    "hidden": 32,
    "kernel_size": 5,
    "levels": None,
    "gate": {"type": "probability", "gamma": 1.0, "detach": True},
}

VARIANTS = {
    # Innovation 2 (LHB-CR): coarse-to-fine boundary refinement, plain and
    # gated by the frame-level forgery probability.
    "refine": {"refinement": REFINE, "uncertainty": False},
    "refine_gate": {"refinement": REFINE_GATE, "uncertainty": False},
    # Kendall-style homoscedastic uncertainty weighting of the loss terms.
    "uncertainty": {"refinement": None, "uncertainty": True},
    "gate_uncertainty": {"refinement": REFINE_GATE, "uncertainty": True},
}


def main():
    parser = argparse.ArgumentParser(description="LHB-CR innovation variants")
    parser.add_argument("-v", "--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("-d", "--dataset_name", choices=["lavdf", "avdeepfake1m"], default="lavdf")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    cfg = load_default(args.dataset_name)
    if variant["refinement"] is not None:
        cfg["model"]["refinement"] = variant["refinement"]
    cfg["criterion"]["uncertainty"] = {"enabled": variant["uncertainty"]}
    Experiment(cfg, f"results/innovation/{args.variant}", print_config=False, job_info=False).run()


if __name__ == "__main__":
    main()