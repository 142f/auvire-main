import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork as FPN

from collections import OrderedDict

RECONSTRUCTION = {
    "av": {"input": "audio", "output": "video"},
    "va": {"input": "video", "output": "audio"},
    "vv": {"input": "video", "output": "video"},
    "aa": {"input": "audio", "output": "audio"},
}


class Mask(nn.Module):

    def __init__(self, num_tokens, win_size, device):
        super().__init__()
        self.num_tokens = num_tokens
        self.win_size = win_size
        self.device = device
        self.mask = self.construct_mask(num_tokens, win_size).to(device)

    def construct_mask(self, num_tokens, win_size):
        return torch.stack(
            [
                torch.cat(
                    (
                        torch.ones((i,), dtype=torch.bool),
                        torch.zeros((win_size,), dtype=torch.bool),
                        torch.ones(num_tokens - i, dtype=torch.bool),
                    )
                )
                for i in range(num_tokens)
            ],
            dim=0,
        )[:, win_size // 2 : -win_size // 2]

    def forward(self, factor):
        if self.win_size > 0:
            return self.mask[::factor, ::factor]
        else:
            return None


class DownsamplingTransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attn_mask=None,
        kernel_size=2,
        stride=2,
        dropout=0.0,
        attn_dropout=0.0,
        ff_expansion=4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_mask = attn_mask
        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ff_expansion, in_channels),
            nn.Dropout(dropout),
        )

        # Downsampling layer (using Conv1d)
        self.downsample = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )  # 1D convolution for sequence
        self.norm3 = nn.LayerNorm(out_channels)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, in_channels)

        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=self.attn_mask)
        x = x + attn_out

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        # Downsampling (with Conv1d)
        x = x.permute(0, 2, 1)  # Reshape for Conv1d: (B, C, L)
        x = self.downsample(x)  # Downsample along the sequence dimension
        x = x.permute(0, 2, 1)  # Reshape back: (B, L_new, C_new)
        x = self.norm3(x)

        return x


class UpsamplingTransformerBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attn_mask=None,
        dropout=0.0,
        attn_dropout=0.0,
        ff_expansion=4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_mask = attn_mask
        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ff_expansion, in_channels),
            nn.Dropout(dropout),
        )

        # Downsampling layer (using Conv1d)
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm3 = nn.LayerNorm(out_channels)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, in_channels)

        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=self.attn_mask)
        x = x + attn_out

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        # Downsampling (with Conv1d)
        x = x.permute(0, 2, 1)  # Reshape for Conv1d: (B, C, L)
        x = self.upsample(x)  # Downsample along the sequence dimension
        x = x.permute(0, 2, 1)  # Reshape back: (B, L_new, C_new)
        x = self.norm3(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, timesteps, dropout, use_ln, use_rl, use_do
    ):
        super(ConvBlock, self).__init__()
        self.use_ln = use_ln
        self.use_rl = use_rl
        self.use_do = use_do
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        if use_ln:
            self.norm = nn.LayerNorm([out_channels, timesteps])
        if use_rl:
            self.relu = nn.ReLU()
        if use_do:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.use_ln:
            x = self.norm(x)
        if self.use_rl:
            x = self.relu(x)
        if self.use_do:
            x = self.dropout(x)
        return x


class ConvTransposeBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, timesteps, dropout, use_ln, use_rl, use_do
    ):
        super(ConvTransposeBlock, self).__init__()
        self.use_ln = use_ln
        self.use_rl = use_rl
        self.use_do = use_do
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
        if self.use_ln:
            self.norm = nn.LayerNorm([out_channels, timesteps])
        if self.use_rl:
            self.relu = nn.ReLU()
        if self.use_do:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.deconv(x)
        if self.use_ln:
            x = self.norm(x)
        if self.use_rl:
            x = self.relu(x)
        if self.use_do:
            x = self.dropout(x)
        return x


class Projection(nn.Module):

    def __init__(self, nlayers, in_channels, out_channels, timesteps, use_ln, use_rl, use_do, dropout, pre):
        super().__init__()
        self.projection = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=in_channels if not pre or (pre and i == 0) else out_channels,
                        out_channels=out_channels if pre or (not pre and i == nlayers - 1) else in_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        timesteps=timesteps,
                        dropout=dropout,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                    )
                    for i in range(nlayers)
                ]
            )
        )

    def forward(self, x):
        return self.projection(x)


class TransformerDownUpCrossModalReconstructionModel(nn.Module):

    def __init__(
        self,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        max_length,
        win_size,
        use_ln,
        use_rl,
        use_do,
        dropout,
        device,
    ):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.attn_mask = Mask(num_tokens=max_length, win_size=win_size, device=device)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.positions = torch.arange(0, max_length).unsqueeze(0).to(device)
        self.encoder = nn.Sequential(
            *nn.ModuleList(
                [
                    DownsamplingTransformerBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        num_heads=num_heads,
                        attn_mask=self.attn_mask(2**i),
                        dropout=dropout,
                    )
                    for i in range(nlayers["downsample"])
                ]
            )
        )
        self.decoder = nn.Sequential(
            *nn.ModuleList(
                [
                    UpsamplingTransformerBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        num_heads=num_heads,
                        attn_mask=self.attn_mask(2 ** (nlayers["upsample"] - i)),
                        dropout=dropout,
                    )
                    for i in range(nlayers["upsample"])
                ]
            )
        )

    def forward(self, x):
        x = self.projection_pre(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed(self.positions)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = self.projection_post(x)
        return x


class PreTrainedTransformerCrossModalReconstructionModel(nn.Module):

    def __init__(
        self,
        input_dimension,
        d_model,
        nlayers,
        max_length,
        win_size,
        use_ln,
        use_rl,
        use_do,
        dropout,
        device,
    ):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.attn_mask = Mask(num_tokens=max_length, win_size=win_size, device=device)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.positions = torch.arange(0, max_length).unsqueeze(0).to(device)
        from transformers import DistilBertModel

        self.transformer = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", torch_dtype=torch.float32, attn_implementation="sdpa"
        ).transformer

    def forward(self, x):
        x = self.projection_pre(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed(self.positions)
        x = self.transformer(x, attn_mask=self.attn_mask(1), head_mask=[None] * 6)[0]
        x = x.permute(0, 2, 1)
        x = self.projection_post(x)
        return x


class AutoRegressiveTransformerEncoderCrossModalReconstructionModel(nn.Module):

    def __init__(
        self,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        max_length,
        win_size,
        use_ln,
        use_rl,
        use_do,
        dropout,
        device,
    ):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.src_mask = self.get_src_mask(max_length, win_size, device)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.positions = torch.arange(0, max_length).unsqueeze(0).to(device)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=2 * d_model, batch_first=True),
            num_layers=nlayers["downsample"] + nlayers["upsample"],
        )

    def get_src_mask(self, num_tokens, win_size, device):
        mask = torch.stack(
            [
                torch.cat(
                    (
                        torch.ones((i,), dtype=torch.bool),
                        torch.zeros((win_size,), dtype=torch.bool),
                        torch.ones(num_tokens - i, dtype=torch.bool),
                    )
                )
                for i in range(num_tokens)
            ],
            dim=0,
        )[:, win_size:]
        mask[0, 0] = False
        return mask.to(device)

    def forward(self, x):
        x = self.projection_pre(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed(self.positions)
        x = self.model(x, mask=self.src_mask)
        x = x.permute(0, 2, 1)
        x = self.projection_post(x)
        return x


class CNNDownUpCrossModalReconstructionModel(nn.Module):

    def __init__(self, input_dimension, d_model, nlayers, win_size, max_length, use_ln, use_rl, use_do, dropout):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.encoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=2,
                        padding=win_size // 2,
                        timesteps=max_length // 2 ** (i + 1),
                        dropout=dropout,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                    )
                    for i in range(nlayers["downsample"])
                ]
            )
        )
        self.decoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvTransposeBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=2,
                        padding=win_size // 2,
                        timesteps=max_length // 2 ** (nlayers["upsample"] - i - 1),
                        dropout=dropout,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                    )
                    for i in range(nlayers["upsample"])
                ]
            )
        )

    def forward(self, x):
        x = self.projection_pre(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.projection_post(x)
        return x


class ReconstructionModel(nn.Module):

    def __init__(
        self,
        model_type,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        use_ln,
        use_rl,
        use_do,
        dropout,
        max_length,
        win_size,
        device,
    ):
        super().__init__()
        if model_type in ["cnn"]:
            self.model = CNNDownUpCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                win_size=win_size,
                max_length=max_length,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
            )
        elif model_type in ["transformer"]:
            self.model = TransformerDownUpCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                num_heads=num_heads,
                max_length=max_length,
                win_size=win_size,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
                device=device,
            )
        elif model_type in ["distilbert"]:
            self.model = PreTrainedTransformerCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=768,
                nlayers=nlayers,
                max_length=max_length,
                win_size=win_size,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
                device=device,
            )
        elif model_type in ["autoregressive"]:
            self.model = AutoRegressiveTransformerEncoderCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                num_heads=num_heads,
                max_length=max_length,
                win_size=win_size,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
                device=device,
            )
        else:
            raise Exception(f"model type {model_type} not suported")

    def forward(self, x):
        return self.model(x)


class CrossScaleAttentionBlock(nn.Module):
    """PCA-FPN 的单层：让某一金字塔级去交叉关注其（已对齐到该级长度）的相邻级。

    Query 取自本级特征，Key/Value 取自邻级特征。相比标准 FPN 的逐级单向相加，
    这里实现了尺度间的双向信息交换，用于保留/放大细微的跨模态重建残差。

    ``pos="time"``/``"both"`` 时另给 q/k 注入固定正弦时间 PE（``_temporal_pe``），
    补齐 CNN 路径缺失的时间位置先验；v 不加，避免位置偏置污染值本身承载的内容。
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be a positive multiple of num_heads")
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # 门控残差：初始以很小的步长注入跨尺度信息。原实现为无门控的全量相加，
        # 在训练初期注意力输出尚未标定时会直接覆盖本级特征，破坏分类头所需的
        # 特征幅度（表现为分数标定偏移）。这里把残差缩放到可学习标量 scale 上。
        self.scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _temporal_pe(length, channels, device, dtype):
        """固定正弦时间位置编码，形状 (1, length, channels)；纯算术、确定性。"""
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )
        pe = torch.zeros(length, channels, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0).to(dtype)

    def forward(self, query, context, pos="none"):
        # query: (B, Lq, C)；context: (B, Lk, C)
        b, lq, c = query.shape
        q = self.q_proj(self.norm_q(query))
        k, v = self.kv_proj(self.norm_kv(context)).chunk(2, dim=-1)
        if pos in {"time", "both"}:
            # 固定正弦时间 PE：只加在 q/k 上，v 不加（值本身承载内容，位置偏置会污染值）。
            # CNN 路径本身没有 pos_embed、SDPA 也没有 mask，跨尺度注意力因此完全没有时间
            # 位置先验。context 由多个"长度等于目标级长度"的邻级沿时间拼接而成，故按目标
            # 长度逐段生成同一段 PE 再平铺，保证每个邻级段内的相对位置一致。
            q = q + self._temporal_pe(lq, c, q.device, q.dtype)
            repeats = k.shape[1] // lq if lq > 0 else 1
            if repeats > 0 and repeats * lq == k.shape[1]:
                pe = self._temporal_pe(lq, c, k.device, k.dtype).repeat(1, repeats, 1)
            else:
                pe = self._temporal_pe(k.shape[1], c, k.device, k.dtype)
            k = k + pe
        head_dim = c // self.num_heads
        q = q.view(b, lq, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, head_dim).transpose(1, 2)
        # 使用内存高效的 SDPA，避免物化完整注意力矩阵
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(b, lq, c)
        return query + self.scale * self.out_proj(attended)


def pca_pool_mode():
    """PCA-FPN 的候选集范围，可用环境变量 ``AUVIRE_PCA_POOL`` 覆盖。

    默认 ``"all"``：金字塔的全部级都可作为 context，最细级因此能直接收到非相邻的更粗级；
    ``"adjacent"``/``"adj"``：只取相邻一级（批 A 的旧行为），仅用于消融对照。
    """
    value = os.environ.get("AUVIRE_PCA_POOL", "").strip().lower()
    return "adjacent" if value in {"adjacent", "adj"} else "all"


def pca_pos_mode():
    """PCA-FPN 跨尺度注意力的位置/尺度先验，可用环境变量 ``AUVIRE_PCA_POS`` 覆盖。

    默认 ``"none"``：与 P5b 逐位一致——不创建尺度嵌入、不加时间 PE。
    ``"scale"``：给每级特征加一个零初始化的可学习尺度嵌入（query 与 context 两个方向都带
    尺度身份），让注意力能区分"自己这一级"与"别级"，避免四级金字塔里 512/512/256/128 的
    等长同源级被当成同一尺度，从而退化为近似均匀平均。
    ``"time"``：给 q/k 各加固定正弦时间 PE（v 不加），注入时间位置先验（CNN 路径没有
    pos_embed，SDPA 也无 mask）。
    ``"both"``：两者同时启用。

    未被识别的取值一律归为 ``"none"``（与 ``pca_pool_mode`` 一致）；训练脚本会为未识别取值
    打印告警并给出独立产物后缀，避免静默地与 none 臂混用同一份产物。
    """
    value = os.environ.get("AUVIRE_PCA_POS", "").strip().lower()
    return value if value in {"none", "scale", "time", "both"} else "none"


class ParallelCrossAttentionFPN(nn.Module):
    """Parallel Cross-Attention FPN（迁移自 UMMAFormer, ACM MM 2023）。

    保留金字塔的多尺度构造，但把"top-down 单向相加"替换为"每级并行 + 交叉注意力"的
    融合；输出与输入在数量、长度、顺序上完全一致，可直接替换既有 FPN。

    每级只把"实际长度不超过本级"的级作为 context：更粗级用 nearest 上采样对齐，等长级直接
    拼接。更细（更长）的级被排除，因此不存在 nearest 降采样（等价于隔点抽取）带来的混叠；
    输出端也不做任何幅度归一化，以保持与标准 FPN 的输出标定一致。

    候选集覆盖金字塔的全部级（编码器默认 ``mode="all"``），因此最细级能直接收到非相邻的
    更粗级（例如四级金字塔里的 256/128），而不是只能看见与自己等长的相邻级。

    位置/尺度先验（``pos``，取值见 ``pca_pos_mode``）：``scale`` 给每级 lateral 加一个零初始化
    的可学习尺度嵌入（query 与 context 两个方向都带，使注意力能区分同级与别级）；``time`` 给
    q/k 加固定正弦时间 PE；``both`` 为两者同时启用。默认 ``none`` 不创建任何新参数、不加 PE，
    与 P5b 逐位一致。
    """

    def __init__(self, in_channels_list, out_channels, num_heads, mode="adjacent", pos="none", dropout=0.0):
        super().__init__()
        if mode not in {"adjacent", "all"}:
            raise ValueError("mode must be 'adjacent' or 'all'")
        if pos not in {"none", "scale", "time", "both"}:
            raise ValueError("pos must be one of 'none', 'scale', 'time', 'both'")
        self.mode = mode
        self.pos = pos
        self.lateral = nn.ModuleList([nn.Conv1d(c, out_channels, 1) for c in in_channels_list])
        self.blocks = nn.ModuleList(
            [CrossScaleAttentionBlock(out_channels, num_heads, dropout) for _ in in_channels_list]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv1d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list]
        )
        # 尺度嵌入：仅 scale/both 时创建，零初始化 → 启用初期输出与 pos="none" 逐位一致。
        # 刻意在 pos="none" 时不创建该属性，保证 state_dict 与 P5b 完全一致（基线与旧
        # checkpoint 都不受影响）。
        if pos in {"scale", "both"}:
            self.level_embed = nn.Parameter(torch.zeros(len(in_channels_list), out_channels))
        # 注意：这里刻意不做末端 LayerNorm。标准 FPN 的输出是 lateral + 上采样相加 + smooth
        # 卷积，不含任何归一化；而逐时间步的跨通道 LayerNorm 会把每个时间步的特征重新
        # 标准化，压缩分类头输入的动态范围，使不同视频之间的分数失去可比性（AP 是跨视频
        # 全局排序，因此会被显著破坏）。

    @staticmethod
    def _align(feature, length):
        """把长度不超过目标长度的邻级特征对齐到目标级长度（保持通道数不变）。

        更粗级（更短）做 nearest 上采样；等长级原样返回。调用方已排除更细（更长）的级，
        因此这里永远不会走"变短"方向——nearest 降采样等价于隔点抽取，会引入混叠。

        必须使用确定性算子：项目启用了 torch.use_deterministic_algorithms(True)，
        而 linear/bilinear 插值的 CUDA 反向没有确定性实现。这里沿用标准 FPN 的
        nearest 插值（其反向是确定性的）。
        """
        if feature.shape[-1] == length:
            return feature
        return F.interpolate(feature, size=length, mode="nearest")

    def forward(self, features):
        # features: list of (B, C_in_i, L_i) -> list of (B, out_channels, L_i)
        laterals = [conv(feature) for conv, feature in zip(self.lateral, features)]
        level_embed = getattr(self, "level_embed", None)
        if level_embed is not None:
            # 尺度身份：query 取自本级、context 取自邻级，两者都来自 laterals，因此同一个
            # 尺度嵌入会随特征一起进入两个方向，注意力才能区分"自己这一级"与"别级"。
            laterals = [lat + level_embed[j].view(1, -1, 1) for j, lat in enumerate(laterals)]
        levels = len(laterals)
        outputs = []
        for index, current in enumerate(laterals):
            length = current.shape[-1]
            if self.mode == "all":
                candidates = [j for j in range(levels) if j != index]
            else:
                candidates = [j for j in (index - 1, index + 1) if 0 <= j < levels]
            # 只排除"更细"（实际长度更长）的级：对更细级做 nearest 对齐等价于隔点抽取，会引入混叠。
            # 更粗级走上采样对齐，等长级直接拼接（无需插值，不产生混叠）。
            neighbours = [j for j in candidates if laterals[j].shape[-1] <= length]
            context = (
                torch.cat([self._align(laterals[j], length) for j in neighbours], dim=-1)
                if neighbours
                else current
            )
            attended = self.blocks[index](
                current.transpose(1, 2), context.transpose(1, 2), pos=self.pos
            )
            attended = self.smooth[index](attended.transpose(1, 2))
            outputs.append(attended)
        return outputs


class TransformerEncoderModel(nn.Module):

    def __init__(
        self, input_dimension, d_model, nlayers, num_heads, fpn, pca_fpn, dropout, max_length, win_size, device
    ):
        super().__init__()
        self.attn_mask = Mask(num_tokens=max_length, win_size=win_size, device=device)
        self.model = nn.ModuleList(
            [
                DownsamplingTransformerBlock(
                    in_channels=input_dimension if i == 0 else d_model,
                    out_channels=d_model,
                    num_heads=num_heads,
                    attn_mask=self.attn_mask(1),
                    kernel_size=1,
                    stride=1,
                    dropout=dropout,
                )
                for i in range(nlayers["retain"])
            ]
            + [
                DownsamplingTransformerBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    num_heads=num_heads,
                    attn_mask=self.attn_mask(2**i if fpn else 1),
                    kernel_size=2 if fpn else 1,
                    stride=2 if fpn else 1,
                    dropout=dropout,
                )
                for i in range(nlayers["downsample"])
            ]
        )
        self.fpn = fpn
        self.pca_fpn = pca_fpn
        if self.pca_fpn and not self.fpn:
            raise ValueError("pca_fpn requires fpn=True so that the pyramid levels are constructed")
        if self.pca_fpn:
            self.pca_pyramid = ParallelCrossAttentionFPN(
                in_channels_list=[d_model] * (nlayers["retain"] + nlayers["downsample"]),
                out_channels=d_model,
                num_heads=num_heads,
                # P5b：候选集默认用全部级（而非只看相邻级），配合"长度 ≤ 本级"的过滤，
                # 保证最细级也能直接收到非相邻的更粗级（如 256/128），而不是只看到
                # 与自己等长的相邻级。AUVIRE_PCA_POOL=adjacent 可退回批 A 的旧行为，
                # 仅用于消融对照。
                mode=pca_pool_mode(),
                pos=pca_pos_mode(),
                dropout=dropout,
            )
        elif self.fpn:
            self.pyramid = FPN(
                in_channels_list=[d_model] * (nlayers["retain"] + nlayers["downsample"]), out_channels=d_model
            )

    def forward(self, x):
        intermediate = []
        out = x
        for i, m in enumerate(self.model):
            out = m(out)
            intermediate.append((f"feat{i}", out.permute(0, 2, 1).unsqueeze(-1)))
        if self.pca_fpn:
            return self.pca_pyramid([feature.squeeze(-1) for _, feature in intermediate])
        if self.fpn:
            out = self.pyramid(OrderedDict(intermediate))
            return [out[x].squeeze() for x in out]
        else:
            return [out.permute(0, 2, 1)]


class CNNEncoderModel(nn.Module):

    def __init__(
        self, input_dimension, d_model, nlayers, win_size, max_length, fpn, pca_fpn, num_heads, use_ln, use_rl, use_do, dropout
    ):
        super().__init__()
        self.model = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=input_dimension if i == 0 else d_model,
                    out_channels=d_model,
                    kernel_size=win_size,
                    stride=1,
                    padding="same",
                    timesteps=max_length,
                    dropout=dropout,
                    use_ln=use_ln,
                    use_rl=use_rl,
                    use_do=use_do,
                )
                for i in range(nlayers["retain"])
            ]
            + [
                ConvBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=win_size,
                    stride=2 if fpn else 1,
                    padding=win_size // 2 if fpn else "same",
                    timesteps=max_length // 2 ** (i + 1) if fpn else max_length,
                    dropout=dropout,
                    use_ln=use_ln,
                    use_rl=use_rl,
                    use_do=use_do,
                )
                for i in range(nlayers["downsample"])
            ]
        )
        self.fpn = fpn
        self.pca_fpn = pca_fpn
        if self.pca_fpn and not self.fpn:
            raise ValueError("pca_fpn requires fpn=True so that the pyramid levels are constructed")
        if self.pca_fpn:
            self.pca_pyramid = ParallelCrossAttentionFPN(
                in_channels_list=[d_model] * (nlayers["retain"] + nlayers["downsample"]),
                out_channels=d_model,
                num_heads=num_heads,
                mode=pca_pool_mode(),
                pos=pca_pos_mode(),
                # 与宿主编码器保持同一 dropout 制度：本编码器是卷积式的（ConvBlock），
                # 其 dropout 由 use_do 控制。torchvision FPN 本身不含任何 dropout，因此
                # 这里不能引入额外随机路径，否则变体臂会比基线臂多出一层正则化。
                dropout=dropout if use_do else 0.0,
            )
        elif self.fpn:
            self.pyramid = FPN(
                in_channels_list=[d_model] * (nlayers["retain"] + nlayers["downsample"]), out_channels=d_model
            )

    def forward(self, x):
        intermediate = []
        out = x
        for i, m in enumerate(self.model):
            out = m(out)
            intermediate.append((f"feat{i}", out.unsqueeze(-1)))
        if self.pca_fpn:
            return self.pca_pyramid([feature.squeeze(-1) for _, feature in intermediate])
        if self.fpn:
            out = self.pyramid(OrderedDict(intermediate))
            return [out[x].squeeze(dim=-1) for x in out]
        else:
            return [out]


class EncoderModel(nn.Module):

    def __init__(
        self,
        model_type,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        fpn,
        pca_fpn,
        use_ln,
        use_rl,
        use_do,
        dropout,
        max_length,
        win_size,
        device,
    ):
        super().__init__()
        if model_type in ["cnn"]:
            self.model = CNNEncoderModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                win_size=win_size,
                max_length=max_length,
                fpn=fpn,
                pca_fpn=pca_fpn,
                num_heads=num_heads,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
            )
        elif model_type in ["transformer"]:
            self.model = TransformerEncoderModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                num_heads=num_heads,
                fpn=fpn,
                pca_fpn=pca_fpn,
                dropout=dropout,
                max_length=max_length,
                win_size=win_size,
                device=device,
            )
        else:
            raise Exception(f"model type {model_type} not suported")

    def forward(self, x):
        return self.model(x)


class HeadModule(nn.Module):

    def __init__(self, d_model, output_size, timesteps, use_ln, use_rl, use_do, dropout, final_relu):
        super().__init__()
        submodules = [
            ConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                timesteps=timesteps,
                dropout=dropout,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
            ),
            ConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                timesteps=timesteps,
                dropout=dropout,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
            ),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=output_size,
                kernel_size=3,
                padding="same",
            ),
        ]
        if final_relu:
            submodules.append(nn.ReLU())
        self.module = nn.Sequential(*submodules)

    def forward(self, x):
        return self.module(x)


class HeadModel(nn.Module):

    def __init__(self, d_model, factor, output_size, max_length, fpn, use_ln, use_rl, use_do, dropout, final_relu):
        super().__init__()
        self.fpn = fpn
        if self.fpn:
            self.model = nn.ModuleList(
                [
                    HeadModule(
                        d_model=d_model,
                        output_size=output_size,
                        timesteps=max_length // f,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                        dropout=dropout,
                        final_relu=final_relu,
                    )
                    for f in factor
                ]
            )
        else:
            self.model = nn.ModuleList(
                [
                    HeadModule(
                        d_model=d_model,
                        output_size=output_size,
                        timesteps=max_length,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                        dropout=dropout,
                        final_relu=final_relu,
                    )
                ]
            )

    def forward(self, x):
        return [self.model[i](f) for i, f in enumerate(x)]


class Model(nn.Module):

    def __init__(
        self,
        max_length,
        d_model,
        win_size,
        num_heads,
        operation,
        reconstruction,
        encoder,
        dropout,
        use_ln,
        use_rl,
        use_do,
        model_type,
        factor,
        device,
    ):
        super().__init__()
        self.device = device
        self.input_dimension = 768
        self.cls_output_size = 1
        self.reg_output_size = 2
        self.max_length = max_length
        self.model_type = model_type
        self.d_model = d_model
        self.win_size = win_size
        self.num_heads = num_heads
        self.operation = operation
        self.reconstruction_nlayers = reconstruction["nlayers"]
        self.reconstruction_modality = [RECONSTRUCTION[m] for m in reconstruction["modality"]]
        self.encoder_nlayers = encoder["nlayers"]
        self.fpn = encoder["fpn"]
        # PCA-FPN 默认关闭；可由配置键开启，或用环境变量 AUVIRE_PCA_FPN 覆盖。
        self.encoder_pca_fpn = bool(encoder.get("pca_fpn", False)) or os.environ.get(
            "AUVIRE_PCA_FPN", ""
        ).strip().lower() in {"1", "true", "yes"}
        self.use_ln = use_ln
        self.use_rl = use_rl
        self.use_do = use_do
        self.dropout_main = dropout["main"]
        self.dropout_head = dropout["head"]
        self.factor = factor
        self.reconstruction_model = ReconstructionModel(
            model_type=self.model_type["reconstruction"],
            input_dimension=self.input_dimension,
            d_model=self.d_model,
            nlayers=self.reconstruction_nlayers,
            num_heads=self.num_heads,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_main,
            max_length=self.max_length,
            win_size=self.win_size,
            device=self.device,
        )
        self.encoder_model = EncoderModel(
            model_type=self.model_type["encoder"],
            input_dimension=self.input_dimension * len(self.reconstruction_modality),
            d_model=self.d_model,
            nlayers=self.encoder_nlayers,
            num_heads=self.num_heads,
            fpn=self.fpn,
            pca_fpn=self.encoder_pca_fpn,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_main,
            max_length=self.max_length,
            win_size=self.win_size,
            device=self.device,
        )
        self.classification = HeadModel(
            d_model=self.d_model,
            factor=self.factor,
            output_size=self.cls_output_size,
            max_length=self.max_length,
            fpn=self.fpn,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_head,
            final_relu=False,
        )
        self.regression = HeadModel(
            d_model=self.d_model,
            factor=self.factor,
            output_size=self.reg_output_size,
            max_length=self.max_length,
            fpn=self.fpn,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_head,
            final_relu=True,
        )

    def get_model_inputs(self, x):
        return x[0].permute((0, 2, 1)), x[1].permute((0, 2, 1))

    def get_reconstruction_pairs(self, audio, video):
        # Each modality keeps its own forward during training: sharing the
        # repeated audio reconstruction changes dropout RNG consumption and the
        # backward accumulation order, moving validation AP outside the
        # baseline tolerance. Evaluation is deterministic, so there the
        # repeated forward is shared and the result stays bit for bit equal.
        if self.training:
            reconstructed = [
                self.reconstruction_model(audio) if modality["input"] == "audio" else self.reconstruction_model(video)
                for modality in self.reconstruction_modality
            ]
        else:
            cache = {}
            reconstructed = []
            for modality in self.reconstruction_modality:
                key = modality["input"]
                if key not in cache:
                    cache[key] = self.reconstruction_model(audio if key == "audio" else video)
                reconstructed.append(cache[key])
        pairs = [
            (
                {"target": audio, "prediction": reconstructed_}
                if self.reconstruction_modality[i]["output"] == "audio"
                else {"target": video, "prediction": reconstructed_}
            )
            for i, reconstructed_ in enumerate(reconstructed)
        ]
        if self.operation == "subtraction":
            dissimilarity = (
                torch.stack([torch.abs(pair["target"] - pair["prediction"]) for pair in pairs], dim=0)
                .sum(dim=0)
                .mean(dim=-2)
            )
        elif self.operation == "multiplication":
            dissimilarity = torch.stack(
                [
                    1 - torch.nn.functional.cosine_similarity(pair["target"], pair["prediction"], dim=-2)
                    for pair in pairs
                ],
                dim=0,
            ).sum(dim=0)
        else:
            raise Exception(f"operation {self.operation} is not supported")
        return pairs, dissimilarity

    def get_encoder_inputs(self, pairs):
        if self.operation == "subtraction":
            return torch.cat([pair["prediction"] - pair["target"] for pair in pairs], dim=-2)
        elif self.operation == "multiplication":
            return torch.cat([pair["prediction"] * pair["target"] for pair in pairs], dim=-2)
        else:
            raise Exception(f"operation {self.operation} is not supported")

    def _forward_with_features(self, src):
        # Inputs
        video, audio = self.get_model_inputs(src)

        # Reconstruction target/prediction pairs & corresponding dissimilarity
        pairs, dissimilarity = self.get_reconstruction_pairs(audio, video)

        # Encoder inputs
        encoder_inputs = self.get_encoder_inputs(pairs)

        # Processing
        if self.model_type["encoder"] in ["transformer"]:
            encoder_inputs = encoder_inputs.permute((0, 2, 1))
        features = self.encoder_model(encoder_inputs)

        # Outputs
        logits = self.classification(features)
        boundaries = self.regression(features)
        outputs = [torch.cat((l, b), dim=1).permute((0, 2, 1)) for l, b in zip(logits, boundaries)]
        return outputs, dissimilarity, features

    def forward(self, src):
        outputs, dissimilarity, _ = self._forward_with_features(src)
        return outputs, dissimilarity

    def get_features(self, src):
        return self._forward_with_features(src)
