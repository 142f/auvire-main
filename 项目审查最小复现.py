"""仅使用获准源码和合成张量复现审查发现；不读取数据、权重或 fairseq。

运行：python -B 项目审查最小复现.py
输出：项目审查验证结果.json
环境需要 torch、torchvision、numpy、scikit-learn。
为避免缺失的 transformers 及入口脚本副作用，通过 AST 装载目标定义。
此方式仅隔离导入和入口副作用，不修改被测函数/类的函数体；不验证完整环境。
"""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.ops import FeaturePyramidNetwork as FPN
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RESULTS = {}


def read_allowed(relative):
    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative], cwd=ROOT,
        capture_output=True,
    )
    if checked.returncode != 1:
        raise RuntimeError(f"文件被忽略或无法核实：{relative}")
    return (ROOT / relative).read_text(encoding="utf-8")


def load_definitions(relative, namespace, names=None, assignments=False):
    tree = ast.parse(read_allowed(relative), filename=relative)
    tree.body = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and
            (names is None or node.name in names))
        or (assignments and isinstance(node, ast.Assign))
    ]
    exec(compile(tree, relative, "exec"), namespace)
    return namespace


def check(name, function):
    try:
        value = function()
        value = json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else list(x)))
        RESULTS[name] = {"status": "completed", "result": value}
    except Exception as exc:
        RESULTS[name] = {"status": "raised", "exception": type(exc).__name__, "message": str(exc)}
    print(name, json.dumps(RESULTS[name], ensure_ascii=False), flush=True)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    ns = dict(torch=torch, nn=nn, torchvision=torchvision, FPN=FPN, OrderedDict=OrderedDict)
    load_definitions("src/models.py", ns, assignments=True)
    load_definitions("src/losses.py", ns)
    from typing import List, Union
    ns.update(np=np, List=List, Union=Union, Tensor=torch.Tensor)
    load_definitions("src/metrics.py", ns)
    load_definitions("src/post_process.py", ns)
    ns.update(average_precision_score=average_precision_score, accuracy_score=accuracy_score,
              roc_auc_score=roc_auc_score)
    load_definitions("src/eval.py", ns)
    ns.update(copy=copy, os=os, datetime=datetime, json=json)
    load_definitions("src/config.py", ns, assignments=True)
    load_definitions("src/training.py", ns)
    ns["Dataset"] = torch.utils.data.Dataset
    load_definitions("src/datasets.py", ns)
    load_definitions("src/loaders.py", ns, names={"collate_fn"})
    RESULTS["environment"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "cuda_available": torch.cuda.is_available(), "device": "cpu",
                              "limitations": "AST隔离导入；无数据/权重；不运行DistilBERT或AV-HuBERT"}

    check("AP_perfect_single_GT_expected_1", lambda: ns["AP"]([0.5])(
        torch.tensor([[[0.9, 0., 25.], [0.8, 50., 75.]]]), [[[0., 1.]]]))
    check("DFD_second_sigmoid", lambda: {
        "probabilities": [0.01, 0.2, 0.8],
        "current_scores": torch.sigmoid(torch.tensor([0.01, 0.2, 0.8])).tolist(),
        "current_labels_at_05": (torch.sigmoid(torch.tensor([0.01, 0.2, 0.8])) > 0.5).tolist(),
    })
    ds = ns["LAVDF"].__new__(ns["LAVDF"])
    ds.max_length = 512
    check("label_past_512", lambda: ds.period2target([[20.0, 21.0]]).shape)
    check("empty_collate", lambda: ns["collate_fn"]([None, None]))
    check("single_proposal_soft_nms", lambda: ns["soft_nms_torch_parallel"](
        torch.tensor([[[0.9, 0., 25.]]]), 0.7234, 0.1968, 0.4123, 25).shape)
    cfg = copy.deepcopy(ns["DEFAULT_LAVDF"])
    cfg["device"] = "cpu"
    experiment = ns["Experiment"](cfg, print_config=False, job_info=False)
    check("job_info_false_has_job", lambda: hasattr(experiment, "job"))
    ns_logger = dict(os=os, json=json)
    load_definitions("src/logger.py", ns_logger, names={"Logger"})
    check("logging_false_has_path", lambda: hasattr(ns_logger["Logger"](None, None, False), "path"))
    cfg2 = copy.deepcopy(cfg)
    cfg2["optimization"]["lr"] *= 0.1
    check("different_lr_same_filename", lambda: ns["get_filename"](cfg) == ns["get_filename"](cfg2))

    def model_smoke():
        model = experiment.get_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = [torch.randn(2, 512, 768), torch.randn(2, 512, 768)]
        targets = torch.stack([ds.period2target([]), ds.period2target([[1., 2.]])])
        calls = []
        hook = model.reconstruction_model.register_forward_hook(lambda *args: calls.append(1))
        outputs, errors = model(inputs)
        hook.remove()
        loss = experiment.get_criterion()(outputs, targets, errors)
        optimizer.zero_grad()
        loss.backward()
        no_grad = [name for name, p in model.named_parameters() if p.grad is None]
        zero_grad = [name for name, p in model.named_parameters() if p.grad is not None and not p.grad.any()]
        all_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        first = next(model.parameters())
        before = first.detach().clone()
        optimizer.step()
        result = {"output_shapes": [list(x.shape) for x in outputs], "error_shape": list(errors.shape),
                  "loss": float(loss.detach()), "all_gradients_finite": bool(all_finite),
                  "parameter_tensors_without_grad": no_grad, "parameter_tensors_all_zero_grad": zero_grad,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "first_parameter_changed": bool((first.detach() != before).any()),
                  "reconstruction_calls": len(calls)}
        model.eval()
        with torch.no_grad():
            pairs, _ = model.get_reconstruction_pairs(inputs[1].transpose(1, 2), inputs[0].transpose(1, 2))
            result["av_aa_predictions_identical"] = torch.equal(pairs[0]["prediction"], pairs[1]["prediction"])
            result["cnn_batch1_shapes"] = [list(p.shape) for p in model([x[:1] for x in inputs])[0]]
        return result
    check("default_CNN_forward_backward_optimizer", model_smoke)

    def transformer_batch1():
        m = ns["TransformerEncoderModel"](16, 8, {"retain": 1, "downsample": 1}, 2, True, 0., 16, 3, "cpu")
        return {"B1": [list(p.shape) for p in m(torch.randn(1, 16, 16))],
                "B2": [list(p.shape) for p in m(torch.randn(2, 16, 16))],
                "expected_B1": [[1, 8, 16], [1, 8, 8]]}
    check("transformer_FPN_batch1", transformer_batch1)
    rec = ns["CombinedLoss"](0.98, 2, ["focal", "diou", "rec"], [1])
    check("partial_fake_reconstruction_is_zero", lambda: rec.reconstruction_loss(
        torch.ones(1, 512), ds.period2target([[1., 2.]]).unsqueeze(0)).tolist())
    itw_ns = dict(torch=torch, MAX_FRAMES_PER_SEGMENT=512, MAX_FPS=25, np=np,
                  APPROXIMATE_SEGMENT_LENGTH_SEC=20)
    load_definitions("src/itw.py", itw_ns, names={"transform_features", "get_offsets", "valid_metadata", "get_interval_union", "get_total_length"})
    check("ITW_nonempty_metadata_errors", lambda: itw_ns["valid_metadata"]({
        "video": {"exists": True}, "audio": {"exists": True}, "errors": ["probe warning"]}))
    check("ITW_interval_union_fills_gap", lambda: {"expected_duration": 0.2,
        "current_duration": itw_ns["get_total_length"]([[0., 0.1], [1., 1.1]])})
    check("ITW_29s_segment_exceeds_model_length", lambda: itw_ns["get_offsets"]({
        "video": {"duration": 29., "fps": 25}, "audio": {"duration": 29., "framerate": 16000}}))
    if torch.cuda.is_available():
        check("ITW_GPU_feature_CPU_padding", lambda: itw_ns["transform_features"](torch.ones(3, 768, device="cuda"), 3).shape)
    source_files = ["src/models.py", "src/training.py", "src/datasets.py", "src/loaders.py", "src/losses.py",
                    "src/metrics.py", "src/eval.py", "src/post_process.py", "src/config.py", "src/logger.py", "src/itw.py"]
    RESULTS["source_sha256_LF"] = {f: hashlib.sha256(read_allowed(f).encode()).hexdigest() for f in source_files}
    output = ROOT / "项目审查验证结果.json"
    output.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
