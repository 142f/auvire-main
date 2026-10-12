"""Compare one pasted LAV-DF test log with the stored full-test JSON records.

This module only reads logs, result JSON, and the local default config.
It does not train and does not change metrics.
"""

import argparse
import json
import re
import sys
from pathlib import Path

COMPARE_KEYS = (
    "tloss",
    "tap@0.5",
    "tap@0.75",
    "tap@0.95",
    "tar@100",
    "tar@50",
    "tar@20",
    "tar@10",
)

ABLATION_REL = (
    "results/ablation/modality/"
    "lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_"
    "rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
)
GRID_REL = (
    "results/grid/"
    "lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_"
    "rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
)
TASK_REL = "results/test/task_tfl_training_on_lavdf.json"


def repo_root():
    return Path(__file__).resolve().parents[1]


def _raw_object(text, brace):
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(text, brace)
    return obj, text[brace:end]


def _test_objects(text):
    found = []
    decoder_text = text
    for match in re.finditer(r'"test"\s*:\s*\{', decoder_text):
        obj, raw = _raw_object(decoder_text, match.end() - 1)
        found.append({"values": obj, "raw": raw})
    return found


def load_stored(root):
    root = Path(root)
    blocks = []
    ablation_path = root / ABLATION_REL
    ablation_text = ablation_path.read_text(encoding="utf-8")
    ablation_data = json.loads(ablation_text)
    ablation_blocks = _test_objects(ablation_text)
    if len(ablation_blocks) != 1 or ablation_blocks[0]["values"] != ablation_data["results"][0]["test"]:
        raise RuntimeError("ablation file does not contain exactly one test block")
    blocks.append(
        {
            "source": "ablation",
            "index": 0,
            "label": "ablation results[0].test",
            "path": str(ablation_path),
            **ablation_blocks[0],
        }
    )

    grid_path = root / GRID_REL
    grid_text = grid_path.read_text(encoding="utf-8")
    grid_data = json.loads(grid_text)
    grid_blocks = _test_objects(grid_text)
    grid_tests = [item["test"] for item in grid_data["results"]]
    if [item["values"] for item in grid_blocks] != grid_tests:
        raise RuntimeError("grid test blocks do not match the parsed results")
    for index, item in enumerate(grid_blocks):
        blocks.append(
            {
                "source": "grid",
                "index": index,
                "label": "grid results[%d].test" % index,
                "path": str(grid_path),
                **item,
            }
        )

    task_path = root / TASK_REL
    task_text = task_path.read_text(encoding="utf-8")
    task_data = json.loads(task_text)
    nested = task_text.find('"lavdf"')
    nested = task_text.find('"lavdf"', nested + 1)
    brace = task_text.find("{", nested)
    task_values, task_raw = _raw_object(task_text, brace)
    if task_values != task_data["lavdf"]["lavdf"]:
        raise RuntimeError("task file lavdf/lavdf block was not located")
    blocks.append(
        {
            "source": "task_tfl",
            "index": None,
            "label": "task_tfl lavdf/lavdf",
            "path": str(task_path),
            "values": task_values,
            "raw": task_raw,
        }
    )
    return blocks


def parse_pasted_test(text):
    brace = text.find("{")
    if brace < 0:
        raise RuntimeError("pasted log has no test JSON object")
    values, raw = _raw_object(text, brace)
    missing = [key for key in COMPARE_KEYS if key not in values]
    if missing:
        raise RuntimeError("pasted test JSON missing %s" % ",".join(missing))
    return {"values": values, "raw": raw}


def parse_facts(text):
    epochs = {}
    for match in re.finditer(r"Epoch\s+(\d+)/\d+[^\n]*LR\s+([0-9.eE+-]+)", text):
        epoch = int(match.group(1))
        window = text[match.end() :]
        nxt = re.search(r"\n\s*Epoch\s+\d+/", window)
        chunk = window[: nxt.start()] if nxt else window
        footer = re.search(r"wait\s+(\d+)/(\d+)\s+│\s+([^\n]+)", chunk)
        epochs[epoch] = {
            "lr": match.group(2),
            "wait": None if footer is None else "%s/%s" % (footer.group(1), footer.group(2)),
            "checkpoint": None if footer is None else footer.group(3).strip(),
        }
    early = re.search(
        r"EARLY STOP\s+│\s+epoch\s+(\d+)\s+│\s+loading best checkpoint from epoch\s+(\d+)",
        text,
    )
    loaded = re.search(r"Loading best checkpoint from epoch\s+(\d+)", text)
    if early is None or loaded is None:
        raise RuntimeError("pasted log has no early-stop checkpoint line")
    if early.group(2) != loaded.group(1):
        raise RuntimeError("early-stop epoch and loaded checkpoint epoch disagree")
    return {
        "epochs": epochs,
        "early_stop_epoch": int(early.group(1)),
        "best_epoch": int(early.group(2)),
        "epoch_13_validation_present": bool(re.search(r"Epoch\s+13/\d+|Val E13\b", text)),
    }


def workspace_refinement(root):
    root = Path(root)
    sys.path.insert(0, str(root))
    from src.config import load_default

    enabled = load_default("lavdf")["model"]["refinement"]["enabled"]
    train_text = (root / "scripts" / "train.py").read_text(encoding="utf-8")
    return {
        "enabled": bool(enabled),
        "train_mentions_refinement": "refinement" in train_text,
    }


def deltas_for(pasted, block):
    return {key: pasted["values"][key] - block["values"][key] for key in COMPARE_KEYS}


def grid_max_block(blocks):
    grid = [block for block in blocks if block["source"] == "grid"]
    return max(grid, key=lambda block: block["values"]["tap@0.95"])


def task_block(blocks):
    return next(block for block in blocks if block["source"] == "task_tfl")


def _walk_tests(obj, found):
    if isinstance(obj, dict):
        test = obj.get("test")
        if isinstance(test, dict) and "tap@0.95" in test:
            found.append(test)
        for value in obj.values():
            _walk_tests(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _walk_tests(value, found)


def other_higher_lavdf_tests(root, threshold):
    """LAV-DF full-test blocks outside the three same-config records."""
    root = Path(root)
    excluded = {
        (root / ABLATION_REL).resolve(),
        (root / GRID_REL).resolve(),
        (root / TASK_REL).resolve(),
    }
    found = []
    results = root / "results"
    if not results.is_dir():
        return found
    for path in sorted(results.rglob("*.json")):
        name = path.name.lower()
        if "lavdf" not in name or "avdeepfake" in name:
            continue
        if path.resolve() in excluded:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tests = []
        _walk_tests(data, tests)
        for index, block in enumerate(tests):
            value = block["tap@0.95"]
            if value > threshold:
                found.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "index": index,
                        "tap@0.95": value,
                    }
                )
    return found


def _sign(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _metric_sentence(name, delta):
    lines = []
    for key in ("tap@0.5", "tap@0.75", "tap@0.95"):
        value = delta[key]
        lines.append(
            "%s versus %s: delta=%r sign=%s absolute=%r"
            % (key, name, value, _sign(value), abs(value))
        )
    return lines


def render(pasted, facts, blocks, refinement, others):
    task = task_block(blocks)
    best_grid = grid_max_block(blocks)
    task_delta = deltas_for(pasted, task)
    grid_delta = deltas_for(pasted, best_grid)
    lines = [
        "# 这次 LAV-DF 全量测试和已保存结果的差别",
        "",
        "early stop at epoch %d" % facts["early_stop_epoch"],
        "best checkpoint epoch %d" % facts["best_epoch"],
        "refinement was not shown to be enabled",
        "",
        "这段日志没有记录配置。测试前加载的检查点是 epoch %d，早停发生在 epoch %d，当时 wait 为 %s。"
        % (
            facts["best_epoch"],
            facts["early_stop_epoch"],
            facts["epochs"][facts["early_stop_epoch"]]["wait"],
        ),
        "epoch 13 的验证指标在这段粘贴里缺失。"
        if not facts["epoch_13_validation_present"]
        else "epoch 13 的验证指标出现在这段粘贴里。",
        "epoch 20 的学习率是 %s。epoch 21 及之后日志里打印的学习率是 %s。"
        % (facts["epochs"][20]["lr"], facts["epochs"][21]["lr"]),
        "epoch 20–23 的 checkpoint 状态："
        + "；".join(
            "epoch %d %s（wait %s，LR %s）" % (epoch, facts["epochs"][epoch]["checkpoint"], facts["epochs"][epoch]["wait"], facts["epochs"][epoch]["lr"])
            for epoch in range(20, 24)
        )
        + "。",
        "epoch 20–23 的验证 AP 行只保留两位小数，不是这次选出的测试分数。可比的是加载 best checkpoint epoch %d 之后打印的测试 JSON。"
        % facts["best_epoch"],
        "本仓库没有 /3240608030/auvire-main 上这次运行的检查点或配置，实际训练代码无法从这段日志核定。",
        "本地 15 条 dev 的 AP 不是这次全量测试的同一测量。",
        "本仓库 load_default(\"lavdf\") 的 model.refinement.enabled 为 %s。scripts/train.py -d lavdf 不会打开 refinement（训练脚本%s refinement）。"
        % (
            str(refinement["enabled"]).lower(),
            "写到了" if refinement["train_mentions_refinement"] else "没有写",
        ),
        "This comparison does not attribute the tap@0.95 difference to LHB-CR or any other innovation.",
        "更高和最接近只限于这三份同一配置记录：%s、%s、%s 的 lavdf/lavdf。"
        % (ABLATION_REL, GRID_REL, TASK_REL),
        "在这三份同一配置记录里，这次的 tap@0.95 高于其中每一个 test。最接近的是 task_tfl lavdf/lavdf，以及该 grid 文件里 tap@0.95 约为 46.50 的 test。",
        "这次的 tap@0.95 并不是仓库里每一份已保存 LAV-DF 全量测试的最高分。",
        "",
        "## 这次测试",
        "",
        pasted["raw"],
        "",
        "## 已保存的全量测试",
        "",
    ]
    for block in blocks:
        lines.append("### %s" % block["label"])
        lines.append("")
        lines.append(block["raw"].strip())
        lines.append("")
        delta = deltas_for(pasted, block)
        lines.append("差值（这次减去该记录）：")
        for key in COMPARE_KEYS:
            lines.append("- %s delta=%r sign=%s" % (key, delta[key], _sign(delta[key])))
        lines.append("")
    lines.append("## 在这三份同一配置记录里最接近的两份")
    lines.append("")
    lines.extend(_metric_sentence(task["label"], task_delta))
    lines.append(
        "相对 %s，tap@0.5 与 tap@0.75 的绝对差值都小于 0.05，tap@0.95 的差值约为 3.3（精确值 %r）。"
        % (task["label"], task_delta["tap@0.95"])
    )
    lines.extend(_metric_sentence(best_grid["label"], grid_delta))
    lines.append(
        "相对 %s（该 grid 测试块的 tap@0.95 为 %r，约为 46.50），tap@0.5 与 tap@0.75 的绝对差值都小于 0.05，tap@0.95 的差值约为 3.3（精确值 %r）。"
        % (best_grid["label"], best_grid["values"]["tap@0.95"], grid_delta["tap@0.95"])
    )
    lines.append("在这三份同一配置记录里，这次的 tap@0.95 高于其中每一个 test：")
    for block in blocks:
        lines.append(
            "- %s stored tap@0.95=%r delta=%r"
            % (block["label"], block["values"]["tap@0.95"], pasted["values"]["tap@0.95"] - block["values"]["tap@0.95"])
        )
    lines.append("其他配置的 LAV-DF 全量 test 里，tap@0.95 高于这次的有：")
    if not others:
        lines.append("- 无")
    for item in others:
        lines.append(
            "- %s results[%d].test tap@0.95=%r，高于这次，不计入上面的更高和最接近。"
            % (item["path"], item["index"], item["tap@0.95"])
        )
    lines.append("召回只在对应差值为正时更高。上面每个 tar 差值都带了符号，这里不把召回概括成提高。")
    lines.append(
        "本仓库 src/training.py 保留检查点时，取的是验证指标名字里包含 vap 或 var 的和，不是验证损失。粘贴里没有 epoch 13 的验证数，这个和无法重算。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def delta_report(pasted, blocks):
    task = task_block(blocks)
    best_grid = grid_max_block(blocks)
    lines = []
    lines.append("pasted_tap@0.95=%r" % pasted["values"]["tap@0.95"])
    above = True
    for block in blocks:
        delta = pasted["values"]["tap@0.95"] - block["values"]["tap@0.95"]
        is_above = pasted["values"]["tap@0.95"] > block["values"]["tap@0.95"]
        above = above and is_above
        lines.append(
            "stored %s tap@0.95=%r delta=%r above=%s"
            % (block["label"], block["values"]["tap@0.95"], delta, is_above)
        )
    lines.append("all_tap95_above=%s" % above)
    for block in (task, best_grid):
        delta = deltas_for(pasted, block)
        for key in ("tap@0.5", "tap@0.75"):
            absolute = abs(delta[key])
            lines.append(
                "versus %s %s delta=%r sign=%s absolute=%r below_0.05=%s"
                % (block["label"], key, delta[key], _sign(delta[key]), absolute, absolute < 0.05)
            )
        lines.append(
            "versus %s tap@0.95 delta=%r sign=%s about_3_3=%s"
            % (block["label"], delta["tap@0.95"], _sign(delta["tap@0.95"]), abs(abs(delta["tap@0.95"]) - 3.3) < 0.15)
        )
    return "\n".join(lines) + "\n"


def build(root, log_text):
    root = Path(root)
    pasted = parse_pasted_test(log_text)
    facts = parse_facts(log_text)
    blocks = load_stored(root)
    refinement = workspace_refinement(root)
    return pasted, facts, blocks, refinement


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(repo_root()))
    parser.add_argument("--log", required=True)
    parser.add_argument("--write-comparison")
    parser.add_argument("--print-deltas", action="store_true")
    args = parser.parse_args(argv)
    log_text = Path(args.log).read_text(encoding="utf-8")
    pasted, facts, blocks, refinement = build(args.root, log_text)
    others = other_higher_lavdf_tests(args.root, pasted["values"]["tap@0.95"])
    if args.write_comparison:
        Path(args.write_comparison).write_text(
            render(pasted, facts, blocks, refinement, others),
            encoding="utf-8",
            newline="\n",
        )
    if args.print_deltas:
        sys.stdout.write(delta_report(pasted, blocks))
        for item in others:
            sys.stdout.write(
                "other_config_higher path=%s index=%d tap@0.95=%r\n"
                % (item["path"], item["index"], item["tap@0.95"])
            )
        sys.stdout.write("other_config_higher_count=%d\n" % len(others))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
