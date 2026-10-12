"""The pasted LAV-DF log is compared by reading the stored result JSON."""

import json
import subprocess
import sys
from pathlib import Path

import tests.lavdf_log_compare as compare

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "tests" / "fixtures" / "lavdf_server_epoch20_23.log"


def _loaded():
    text = LOG.read_text(encoding="utf-8")
    pasted, facts, blocks, refinement = compare.build(ROOT, text)
    return text, pasted, facts, blocks, refinement


def test_pasted_test_json_is_the_log_object():
    text, pasted, _, _, _ = _loaded()
    obj, _ = json.JSONDecoder().raw_decode(text, text.find("{"))
    assert pasted["values"] == obj
    assert pasted["raw"] == text[text.find("{") : text.find("{") + len(pasted["raw"])]


def test_stored_blocks_match_the_json_files():
    _, _, _, blocks, _ = _loaded()
    ablation = json.loads((ROOT / compare.ABLATION_REL).read_text(encoding="utf-8"))
    grid = json.loads((ROOT / compare.GRID_REL).read_text(encoding="utf-8"))
    task = json.loads((ROOT / compare.TASK_REL).read_text(encoding="utf-8"))
    assert blocks[0]["values"] == ablation["results"][0]["test"]
    grid_blocks = [block for block in blocks if block["source"] == "grid"]
    assert [block["values"] for block in grid_blocks] == [item["test"] for item in grid["results"]]
    assert compare.task_block(blocks)["values"] == task["lavdf"]["lavdf"]


def test_deltas_subtract_stored_from_pasted():
    _, pasted, _, blocks, _ = _loaded()
    for block in blocks:
        delta = compare.deltas_for(pasted, block)
        for key in compare.COMPARE_KEYS:
            assert delta[key] == pasted["values"][key] - block["values"][key]
        assert pasted["values"]["tap@0.95"] > block["values"]["tap@0.95"]


def test_close_thresholds_use_task_and_highest_grid_block():
    _, pasted, _, blocks, _ = _loaded()
    chosen = (compare.task_block(blocks), compare.grid_max_block(blocks))
    grid_values = [block["values"]["tap@0.95"] for block in blocks if block["source"] == "grid"]
    assert chosen[1]["values"]["tap@0.95"] == max(grid_values)
    for block in chosen:
        delta = compare.deltas_for(pasted, block)
        assert abs(delta["tap@0.5"]) < 0.05
        assert abs(delta["tap@0.75"]) < 0.05
        assert delta["tap@0.95"] > 3


def test_log_facts_and_refinement_gate():
    text, _, facts, _, refinement = _loaded()
    assert facts["early_stop_epoch"] == 23
    assert facts["best_epoch"] == 13
    assert "loading best checkpoint from epoch 13" in text
    assert facts["epoch_13_validation_present"] is False
    assert facts["epochs"][20]["lr"] == "1.00e-03"
    assert facts["epochs"][21]["lr"] == "1.00e-04"
    assert facts["epochs"][23]["wait"] == "10/10"
    for epoch in range(20, 24):
        assert facts["epochs"][epoch]["checkpoint"] == "checkpoint unchanged"
    assert refinement["enabled"] is False
    assert refinement["train_mentions_refinement"] is False


def test_report_carries_source_numbers_and_does_not_assign_the_gain():
    text, pasted, facts, blocks, refinement = _loaded()
    others = compare.other_higher_lavdf_tests(ROOT, pasted["values"]["tap@0.95"])
    report = compare.render(pasted, facts, blocks, refinement, others)
    assert pasted["raw"] in report
    for block in blocks:
        assert block["raw"].strip() in report
    assert "early stop at epoch 23" in report
    assert "best checkpoint epoch 13" in report
    assert "refinement was not shown to be enabled" in report
    assert "does not attribute the tap@0.95 difference to LHB-CR" in report
    assert "召回概括成提高" in report
    assert "15 条 dev" in report
    task_delta = compare.deltas_for(pasted, compare.task_block(blocks))
    grid_delta = compare.deltas_for(pasted, compare.grid_max_block(blocks))
    assert repr(task_delta["tap@0.95"]) in report
    assert repr(grid_delta["tap@0.95"]) in report
    assert "sign=%s" % ("positive" if task_delta["tap@0.5"] > 0 else "negative") in report
    assert "更高和最接近只限于这三份同一配置记录" in report
    assert "并不是仓库里每一份已保存 LAV-DF 全量测试的最高分" in report
    assert others
    for item in others:
        assert ("%r" % item["tap@0.95"]) in report
        assert item["path"] in report
        loaded = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
        tests = []
        compare._walk_tests(loaded, tests)
        assert tests[item["index"]]["tap@0.95"] == item["tap@0.95"]
        assert item["tap@0.95"] > pasted["values"]["tap@0.95"]


def test_cli_reads_the_json_files(tmp_path):
    comparison = tmp_path / "comparison.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests" / "lavdf_log_compare.py"),
            "--log",
            str(LOG),
            "--write-comparison",
            str(comparison),
            "--print-deltas",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "all_tap95_above=True" in completed.stdout
    assert "below_0.05=False" not in completed.stdout
    assert comparison.is_file()
    body = comparison.read_text(encoding="utf-8")
    assert "early stop at epoch 23" in body
    assert "refinement was not shown to be enabled" in body
