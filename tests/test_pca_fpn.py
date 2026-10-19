"""PCA-FPN（UMMAFormer）迁移的回归测试。

覆盖：
1. 默认关闭时走原标准 FPN 分支，state_dict 不含新参数（基线不受影响）；
2. 开启时输出级数、长度、顺序与 factor 一致，可直接替换 FPN；
3. 评估模式下确定性（dropout 关闭）；
4. 前向/反向可跑通且损失有限；
5. 配置守卫：pca_fpn 需要 fpn=True；
6. 环境变量 AUVIRE_PCA_FPN 覆盖。
7. 批 A 契约：末端无 LayerNorm（P1）、跨尺度残差带门控标量（P2）、
   context 只允许来自"更粗"（更短）的级、不做降采样对齐（P5）。
8. P4a 契约：pos=none 不加参数且逐位等价旧行为；scale 零初始化且注入尺度身份；
   time 为固定正弦 PE 且确定性；pos=both 在确定性算法下可反向；环境变量经编码器生效。
"""

import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import Model, ParallelCrossAttentionFPN, CrossScaleAttentionBlock  # noqa: E402


def _factor(retain, downsample):
    return [1] * retain + [2 ** (i + 1) for i in range(downsample)]


def _build(
    pca_fpn,
    fpn=True,
    retain=1,
    downsample=1,
    encoder_type="cnn",
    max_length=64,
    dropout_main=0.0,
    use_do=False,
):
    return Model(
        max_length=max_length,
        d_model=16,
        win_size=3,
        num_heads=4,
        operation="subtraction",
        reconstruction={"nlayers": {"pre": 1, "downsample": 1, "upsample": 1, "post": 1}, "modality": ["av", "aa", "vv"]},
        encoder={"nlayers": {"retain": retain, "downsample": downsample}, "fpn": fpn, "pca_fpn": pca_fpn},
        dropout={"main": dropout_main, "head": 0.0},
        use_ln=True,
        use_rl=True,
        use_do=use_do,
        model_type={"reconstruction": "cnn", "encoder": encoder_type},
        factor=_factor(retain, downsample),
        device="cpu",
    )


def _inputs(batch=2, length=64):
    torch.manual_seed(0)
    return [torch.randn(batch, length, 768), torch.randn(batch, length, 768)]


def test_disabled_uses_standard_fpn_and_adds_no_params():
    model = _build(pca_fpn=False)
    encoder = model.encoder_model.model
    assert hasattr(encoder, "pyramid")
    assert not hasattr(encoder, "pca_pyramid")
    assert not any("pca_pyramid" in key for key in model.state_dict())


def test_enabled_uses_pca_pyramid():
    model = _build(pca_fpn=True)
    encoder = model.encoder_model.model
    assert hasattr(encoder, "pca_pyramid")
    assert not hasattr(encoder, "pyramid")
    assert any("pca_pyramid" in key for key in model.state_dict())


def test_enabled_output_levels_match_factor():
    retain, downsample, max_length = 2, 2, 64
    model = _build(pca_fpn=True, retain=retain, downsample=downsample, max_length=max_length)
    model.eval()
    with torch.no_grad():
        outputs, dissimilarity, features = model.get_features(_inputs(length=max_length))
    expected = [max_length // f for f in _factor(retain, downsample)]
    assert len(features) == retain + downsample
    assert [f.shape[-1] for f in features] == expected
    assert [o.shape[-2] for o in outputs] == expected
    assert all(o.shape[-1] == 3 for o in outputs)
    assert all(f.shape[1] == 16 for f in features)
    assert all(torch.isfinite(f).all() for f in features)


def test_eval_deterministic():
    model = _build(pca_fpn=True)
    model.eval()
    x = _inputs()
    with torch.no_grad():
        out_a = model(x)[0]
        out_b = model(x)[0]
    assert all(torch.equal(a, b) for a, b in zip(out_a, out_b))


def test_backward_runs_and_loss_finite():
    model = _build(pca_fpn=True)
    model.train()
    outputs, dissimilarity = model(_inputs())
    loss = sum(o.sum() for o in outputs) + dissimilarity.sum()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.encoder_model.model.pca_pyramid.parameters()]
    assert any(g is not None for g in grads)


def test_guard_requires_fpn():
    try:
        _build(pca_fpn=True, fpn=False)
    except ValueError:
        return
    raise AssertionError("pca_fpn without fpn should raise ValueError")


def test_env_override():
    os.environ["AUVIRE_PCA_FPN"] = "1"
    try:
        model = _build(pca_fpn=False)
        assert hasattr(model.encoder_model.model, "pca_pyramid")
        assert not hasattr(model.encoder_model.model, "pyramid")
    finally:
        os.environ.pop("AUVIRE_PCA_FPN", None)


def test_module_shapes_and_alignment():
    module = ParallelCrossAttentionFPN(in_channels_list=[16, 16, 32], out_channels=16, num_heads=4)
    features = [torch.randn(2, 16, 64), torch.randn(2, 16, 64), torch.randn(2, 32, 32)]
    out = module(features)
    assert [tuple(o.shape) for o in out] == [
        (2, 16, 64),
        (2, 16, 64),
        (2, 16, 32),
    ]
    assert all(torch.isfinite(o).all() for o in out)

    block = CrossScaleAttentionBlock(d_model=16, num_heads=4)
    attended = block(torch.randn(2, 64, 16), torch.randn(2, 32, 16))
    assert tuple(attended.shape) == (2, 64, 16)


def test_no_terminal_layer_norm():
    """P1：PCA-FPN 末端不得再对分类头输入做跨通道归一化（标准 FPN 亦无）。"""
    module = ParallelCrossAttentionFPN(in_channels_list=[16, 16, 32], out_channels=16, num_heads=4)
    assert not hasattr(module, "norms")
    assert not any("norms" in key for key in module.state_dict())
    model = _build(pca_fpn=True)
    assert not any("pca_pyramid.norms" in key for key in model.state_dict())


def test_gated_residual_scale_parameter():
    """P2：跨尺度残差必须带可学习门控标量，且初始值较小（0.1）。"""
    module = ParallelCrossAttentionFPN(in_channels_list=[16, 16, 32], out_channels=16, num_heads=4)
    scales = [b.scale for b in module.blocks]
    assert len(scales) == 3
    assert all(isinstance(s, nn.Parameter) for s in scales)
    assert all(torch.allclose(s.detach(), torch.tensor(0.1)) for s in scales)
    assert any("blocks.0.scale" in key for key in module.state_dict())


def test_attention_dropout_follows_host_use_do():
    """R2：卷积式编码器未启用 dropout（use_do=False）时，PCA-FPN 也不得引入额外随机路径
    （torchvision FPN 无 dropout），否则变体臂会比基线臂多一层正则化。"""
    off = _build(pca_fpn=True, dropout_main=0.1, use_do=False)
    assert all(b.dropout == 0.0 for b in off.encoder_model.model.pca_pyramid.blocks)
    on = _build(pca_fpn=True, dropout_main=0.1, use_do=True)
    assert all(b.dropout == 0.1 for b in on.encoder_model.model.pca_pyramid.blocks)


def test_finer_levels_excluded_and_same_length_kept():
    """P5：更细（更长）的级必须被排除（否则 nearest 对齐 = 隔点抽取混叠）；
    等长级无需插值，必须保留（标准 FPN 的 top-down 链就是靠相邻同级/更粗级传递的）。"""
    module = ParallelCrossAttentionFPN(in_channels_list=[16, 16, 16], out_channels=16, num_heads=4)
    module.eval()
    torch.manual_seed(1)
    a = torch.randn(2, 16, 64)
    b = torch.randn(2, 16, 64)  # 与 a 等长
    c = torch.randn(2, 16, 32)  # 更粗
    with torch.no_grad():
        base = module([a, b, c])
        finer_changed = module([a + 1.0, b, c])
        same_length_changed = module([a, b + 1.0, c])
        coarser_changed = module([a, b, c + 1.0])
    # 最粗级（长度 32）唯一的邻级更长，必须被排除：改动更细级不得影响它的输出
    assert torch.equal(base[2], finer_changed[2])
    # 该级自身输入的改动必须生效
    assert not torch.equal(base[2], coarser_changed[2])
    # 等长邻级必须进入 context：改动 b 必须让 a 所在级的输出变化
    assert not torch.equal(base[0], same_length_changed[0])
    # 更粗邻级必须通过上采样进入 context
    assert not torch.equal(base[1], coarser_changed[1])


def test_finest_level_receives_non_adjacent_coarser_levels():
    """P5b：四级金字塔（512,512,256,128）下，第 0 级必须能收到非相邻的更粗级（256/128）。
    直接使用编码器构造出的 pca_pyramid，以确保覆盖真实接线（mode="all"），而不是单独构造的模块。"""
    model = _build(pca_fpn=True, retain=2, downsample=2)
    pyramid = model.encoder_model.model.pca_pyramid
    assert pyramid.mode == "all"
    pyramid.eval()
    torch.manual_seed(0)
    feats = [
        torch.randn(1, 16, 512),
        torch.randn(1, 16, 512),
        torch.randn(1, 16, 256),
        torch.randn(1, 16, 128),
    ]
    with torch.no_grad():
        base = pyramid(feats)
        changed_128 = pyramid(feats[:3] + [feats[3] + 1.0])
        changed_256 = pyramid(feats[:2] + [feats[2] + 1.0] + feats[3:])
    assert not torch.equal(base[0], changed_128[0])
    assert not torch.equal(base[0], changed_256[0])


def test_pool_mode_env_override_restores_adjacent_semantics():
    """AUVIRE_PCA_POOL=adjacent 退回批 A 语义：最细级只看得见等长相邻级，256/128 到不了它。
    这条同时锁住了两条臂的差异，避免它们被混为一谈。"""
    os.environ["AUVIRE_PCA_POOL"] = "adjacent"
    try:
        model = _build(pca_fpn=True, retain=2, downsample=2)
        pyramid = model.encoder_model.model.pca_pyramid
        assert pyramid.mode == "adjacent"
        pyramid.eval()
        torch.manual_seed(0)
        feats = [
            torch.randn(1, 16, 512),
            torch.randn(1, 16, 512),
            torch.randn(1, 16, 256),
            torch.randn(1, 16, 128),
        ]
        with torch.no_grad():
            base = pyramid(feats)
            changed_128 = pyramid(feats[:3] + [feats[3] + 1.0])
            changed_256 = pyramid(feats[:2] + [feats[2] + 1.0] + feats[3:])
            changed_equal = pyramid([feats[0], feats[1] + 1.0] + feats[2:])
        assert torch.equal(base[0], changed_128[0])
        assert torch.equal(base[0], changed_256[0])
        assert not torch.equal(base[0], changed_equal[0])
    finally:
        os.environ.pop("AUVIRE_PCA_POOL", None)


def test_alignment_runs_under_deterministic_algorithms():
    """R3：项目启用 torch.use_deterministic_algorithms(True)，PCA-FPN 的对齐与反向必须
    在该模式下可跑通（linear/bilinear 等无非确定性反向实现的插值已被排除）。"""
    module = ParallelCrossAttentionFPN(in_channels_list=[16, 16, 16], out_channels=16, num_heads=4)
    module.train()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        out = module([torch.randn(2, 16, 64), torch.randn(2, 16, 64), torch.randn(2, 16, 32)])
        loss = sum(o.sum() for o in out)
        loss.backward()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in module.parameters())


def test_pos_none_adds_no_parameters_and_matches_plain():
    """P4a：pos=none（默认）不创建尺度嵌入、不加时间 PE——state_dict 与 P5b 完全一致；
    默认构造与显式 none 在同一份权重下输出逐位相同，保证基线数值不受影响。"""
    default = ParallelCrossAttentionFPN(in_channels_list=[16, 16, 32], out_channels=16, num_heads=4)
    explicit = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 32], out_channels=16, num_heads=4, pos="none"
    )
    assert default.pos == "none" and explicit.pos == "none"
    assert not hasattr(default, "level_embed")
    assert not hasattr(explicit, "level_embed")
    assert not any("level_embed" in key for key in default.state_dict())
    assert set(default.state_dict().keys()) == set(explicit.state_dict().keys())
    explicit.load_state_dict(default.state_dict())
    default.eval()
    explicit.eval()
    torch.manual_seed(0)
    feats = [torch.randn(2, 16, 64), torch.randn(2, 16, 64), torch.randn(2, 32, 32)]
    with torch.no_grad():
        a = default([f.clone() for f in feats])
        b = explicit([f.clone() for f in feats])
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_pos_scale_creates_zero_initialised_level_embed():
    """P4a：pos=scale 创建形状 (levels, out_channels) 的零初始化尺度嵌入。"""
    module = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 32], out_channels=16, num_heads=4, pos="scale"
    )
    assert hasattr(module, "level_embed")
    assert tuple(module.level_embed.shape) == (3, 16)
    assert torch.equal(module.level_embed.detach(), torch.zeros(3, 16))
    assert any("level_embed" in key for key in module.state_dict())


def test_pos_scale_zero_init_matches_none():
    """P4a：零初始化的尺度嵌入不改变输出——启用 scale 的初始状态与 none 逐位一致。"""
    plain = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 32], out_channels=16, num_heads=4, pos="none"
    )
    scaled = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 32], out_channels=16, num_heads=4, pos="scale"
    )
    scaled.load_state_dict(plain.state_dict(), strict=False)
    plain.eval()
    scaled.eval()
    torch.manual_seed(1)
    feats = [torch.randn(2, 16, 64), torch.randn(2, 16, 64), torch.randn(2, 32, 32)]
    with torch.no_grad():
        a = plain([f.clone() for f in feats])
        b = scaled([f.clone() for f in feats])
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_pos_scale_injects_scale_identity_into_attention():
    """P4a：等长同源两级（如四级金字塔里的 512/512）在尺度嵌入设为逐级不同常量后，
    输出必须相对零初始化版本变化——证明尺度身份确实进入了注意力（query/context 两个方向），
    而不是把同级与别级退化为均匀平均。"""
    module = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16], out_channels=16, num_heads=4, pos="scale"
    )
    module.eval()
    torch.manual_seed(0)
    shared = torch.randn(2, 16, 64)
    with torch.no_grad():
        zero = module([shared.clone(), shared.clone()])
        module.level_embed.data[0].fill_(1.0)
        module.level_embed.data[1].fill_(-1.0)
        embedded = module([shared.clone(), shared.clone()])
    assert not torch.equal(zero[0], embedded[0])
    assert not torch.equal(zero[1], embedded[1])


def test_pos_time_changes_output_and_is_deterministic():
    """P4a：时间 PE 改变输出（注入生效），且两次调用逐位相同（固定正弦、无随机数）。"""
    plain = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 32], out_channels=16, num_heads=4, pos="none"
    )
    timed = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 32], out_channels=16, num_heads=4, pos="time"
    )
    # time 不新增参数：state_dict 与 none 完全一致
    timed.load_state_dict(plain.state_dict())
    plain.eval()
    timed.eval()
    torch.manual_seed(2)
    feats = [torch.randn(2, 16, 64), torch.randn(2, 16, 64), torch.randn(2, 32, 32)]
    with torch.no_grad():
        a = plain([f.clone() for f in feats])
        b = timed([f.clone() for f in feats])
        c = timed([f.clone() for f in feats])
    assert not torch.equal(a[0], b[0])
    assert all(torch.equal(x, y) for x, y in zip(b, c))


def test_pos_both_backward_under_deterministic_algorithms():
    """P4a：pos=both 在 torch.use_deterministic_algorithms(True) 下前向/反向可跑通
    （正弦 PE 与 nearest 对齐都是确定性算子）。"""
    module = ParallelCrossAttentionFPN(
        in_channels_list=[16, 16, 16], out_channels=16, num_heads=4, pos="both"
    )
    module.train()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        out = module([torch.randn(2, 16, 64), torch.randn(2, 16, 64), torch.randn(2, 16, 32)])
        loss = sum(o.sum() for o in out)
        loss.backward()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in module.parameters())


def test_env_pca_pos_scale_reaches_encoder():
    """P4a：AUVIRE_PCA_POS 经编码器生效（真实接线，而非单独构造的模块）。"""
    os.environ["AUVIRE_PCA_POS"] = "scale"
    try:
        model = _build(pca_fpn=True, retain=2, downsample=2)
        pyramid = model.encoder_model.model.pca_pyramid
        assert pyramid.pos == "scale"
        assert tuple(pyramid.level_embed.shape) == (4, 16)
        assert any("level_embed" in key for key in model.state_dict())
    finally:
        os.environ.pop("AUVIRE_PCA_POS", None)


def test_pca_pos_arm_filenames_are_distinct():
    """P4a：不同 AUVIRE_PCA_POS 必须产出不同的实验名后缀——否则变体臂会与 none/P5b 臂
    共用同一份产物，被 check_complete 判定为"已完成"而静默跳过训练。未识别的取值也必须
    给出独立后缀（否则"名义变体臂、实际 none 臂"会被当成同一份结果复用）。"""
    from src.config import load_default
    from src.training import get_filename, pca_pos_tag

    cfg = load_default("lavdf")
    cfg["model"]["encoder"]["pca_fpn"] = True

    names = {}
    for value in ["", "none", "scale", "time", "both"]:
        if value:
            os.environ["AUVIRE_PCA_POS"] = value
        else:
            os.environ.pop("AUVIRE_PCA_POS", None)
        try:
            names[value] = get_filename(cfg)
        finally:
            os.environ.pop("AUVIRE_PCA_POS", None)

    assert names[""] == names["none"]
    assert names["scale"].endswith("_pe_s")
    assert names["time"].endswith("_pe_t")
    assert names["both"].endswith("_pe_st")
    assert len({names[""], names["scale"], names["time"], names["both"]}) == 4

    # 大小写/空白会被归一化到同一后缀
    os.environ["AUVIRE_PCA_POS"] = "ScaLe "
    try:
        assert get_filename(cfg).endswith("_pe_s")
    finally:
        os.environ.pop("AUVIRE_PCA_POS", None)

    # 未识别取值：仍给独立后缀，不与 none 臂同名
    os.environ["AUVIRE_PCA_POS"] = "bogus"
    try:
        assert pca_pos_tag() == ["pe_bogus"]
        assert get_filename(cfg).endswith("_pe_bogus")
    finally:
        os.environ.pop("AUVIRE_PCA_POS", None)
