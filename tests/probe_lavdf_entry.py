"""Probe scripts/train.py for the lavdf default without starting training.

Runs ``python scripts/train.py -h`` and ``load_default(\"lavdf\")``.
Does not construct ``Experiment`` and does not call ``Experiment.run``.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ROOT if not previous else ROOT + os.pathsep + previous
    help_run = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "-h"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if help_run.returncode != 0:
        sys.stderr.write(help_run.stdout)
        sys.stderr.write(help_run.stderr)
        return help_run.returncode
    help_text = help_run.stdout + help_run.stderr
    if "lavdf" not in help_text or "-d" not in help_text:
        sys.stderr.write(help_text)
        sys.stderr.write("help text does not offer -d lavdf\n")
        return 2

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import src.config
    import src.losses
    import src.models
    import src.training

    cfg = src.config.load_default("lavdf")
    enabled = cfg["model"]["refinement"]["enabled"]
    print("dataset=%s" % cfg["dataset"]["name"])
    print("epochs=%s" % cfg["epochs"])
    print("refinement.enabled=%s" % str(bool(enabled)).lower())
    print("imports=src.config,src.training,src.models,src.losses")
    print("entry=scripts/train.py")
    print("choice=lavdf")
    print("help_exit=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
