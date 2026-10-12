"""The server update set is the six screenshot files, byte-for-byte, plus the lavdf train command."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD = os.path.join(ROOT, "server-update")
COMMAND_FILE = os.path.join(ROOT, "server-update-command.txt")
RELATIVE_PATHS = (
    "scripts/innovation.py",
    "scripts/predict.py",
    "src/config.py",
    "src/losses.py",
    "src/models.py",
    "src/training.py",
)
COMMAND = (
    "cd /3240608030/auvire-main\n"
    'export PYTHONPATH="/3240608030/auvire-main"\n'
    "python scripts/train.py -d lavdf\n"
)
FORBIDDEN = (
    "scripts/train.py",
    "auvire-main.zip",
    "results",
    "runs",
    "utils",
    "ckpt",
)


def _payload_files():
    found = []
    for dirpath, dirnames, filenames in os.walk(PAYLOAD):
        for name in filenames:
            full = os.path.join(dirpath, name)
            found.append(os.path.relpath(full, PAYLOAD).replace("\\", "/"))
    return tuple(sorted(found))


def test_payload_is_exactly_the_six_sources():
    assert os.path.isdir(PAYLOAD)
    assert _payload_files() == tuple(sorted(RELATIVE_PATHS))
    for rel in RELATIVE_PATHS:
        source = os.path.join(ROOT, *rel.split("/"))
        staged = os.path.join(PAYLOAD, *rel.split("/"))
        with open(source, "rb") as handle:
            source_bytes = handle.read()
        with open(staged, "rb") as handle:
            staged_bytes = handle.read()
        assert staged_bytes == source_bytes
        assert len(staged_bytes) > 0
    joined = " ".join(_payload_files())
    for forbidden in FORBIDDEN:
        assert forbidden not in _payload_files()
        assert f"/{forbidden}/" not in f"/{joined}/"


def test_server_command_is_the_lavdf_train_entry():
    with open(COMMAND_FILE, "rb") as handle:
        raw = handle.read()
    assert raw.decode("utf-8") == COMMAND
    lines = COMMAND.splitlines()
    assert lines == [
        "cd /3240608030/auvire-main",
        'export PYTHONPATH="/3240608030/auvire-main"',
        "python scripts/train.py -d lavdf",
    ]
    assert "scripts/innovation.py" not in COMMAND


def test_train_entry_accepts_lavdf_without_training():
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
    assert help_run.returncode == 0
    help_text = help_run.stdout + help_run.stderr
    assert "lavdf" in help_text
    assert "-d" in help_text

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import src.config
    import src.losses
    import src.models
    import src.training

    calls = {"run": 0}
    original = src.training.Experiment.run

    def _refuse_run(self, *args, **kwargs):
        calls["run"] += 1
        raise AssertionError("Experiment.run must not be called")

    src.training.Experiment.run = _refuse_run
    try:
        cfg = src.config.load_default("lavdf")
        assert calls["run"] == 0
    finally:
        src.training.Experiment.run = original

    assert cfg["dataset"]["name"] == "lavdf"
    assert cfg["epochs"] == 100
    assert cfg["model"]["refinement"]["enabled"] is False


def test_probe_script_exits_zero():
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ROOT if not previous else ROOT + os.pathsep + previous
    first = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tests", "probe_lavdf_entry.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tests", "probe_lavdf_entry.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    assert "dataset=lavdf" in first.stdout
    assert "epochs=100" in first.stdout
    assert "refinement.enabled=false" in first.stdout
    assert "Experiment.run" not in first.stdout
