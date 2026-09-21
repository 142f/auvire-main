import math
from numbers import Integral, Real

import torchvision
import torch.nn as nn
import torch


def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return torchvision.ops.sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_obj = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        return self.loss_obj(inputs, targets)


class CombinedLoss(nn.Module):

    def __init__(
        self, alpha, gamma, composition, factor,
        enable_ib_ecl=False, ib_ecl_weight=0.1, ib_ecl_beta=0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l_focal = 1.0 if "focal" in composition else 0.0
        self.l_diou = 1.0 if "diou" in composition else 0.0
        self.l_sl1 = 1.0 if "sl1" in composition else 0.0
        self.l_rec = 1.0 if "rec" in composition else 0.0
        self.l_det = 1.0 if "det" in composition else 0.0
        self.factor = factor
        self.bce = BCELoss()
        # Independent switch: the legacy composition and all its weights stay intact.
        if not isinstance(enable_ib_ecl, bool):
            raise TypeError("enable_ib_ecl must be a bool")
        self.enable_ib_ecl = enable_ib_ecl
        self.ib_ecl_weight = ib_ecl_weight
        self.ib_ecl_beta = ib_ecl_beta
        if self.enable_ib_ecl:
            for name, value in (("ib_ecl_weight", ib_ecl_weight), ("ib_ecl_beta", ib_ecl_beta)):
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite real number")
            if ib_ecl_weight < 0 or ib_ecl_beta <= 0:
                raise ValueError("ib_ecl_weight must be >= 0 and ib_ecl_beta must be > 0")
            if not self.l_diou:
                raise ValueError("IB-ECL requires the original DIoU loss to anchor common boundary bias")

    def ib_ecl(self, inputs, targets):
        """Instance-balanced consistency of decoded absolute endpoints.

        inputs: nonempty list of [B, T_level, 3] (logit, left, right).
        targets: [B, T, 3] (binary label, GT left, GT right).
        Both predicted and GT offsets use ORIGINAL FRAME units, at every level.
        Padding/background with label 0 is excluded; no feature-value-based mask
        is inferred. The existing dataset must align its labels with its features.
        This method returns a scalar and never detaches the prediction consensus.
        """
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValueError("IB-ECL expects a nonempty list/tuple of prediction levels")
        if targets.ndim != 3 or targets.shape[-1] != 3:
            raise ValueError("IB-ECL targets must have shape [B, T, 3]")
        if len(inputs) > len(self.factor):
            raise ValueError("There must be a scale factor for every prediction level")
        if not torch.all((targets[..., 0] == 0) | (targets[..., 0] == 1)):
            raise ValueError("IB-ECL target labels must be finite binary values (0 or 1)")

        # Keep double precision for gradcheck; avoid half/bfloat16 accumulation.
        dtype = torch.float64 if targets.dtype == torch.float64 or any(
            prediction.dtype == torch.float64 for prediction in inputs
        ) else torch.float32
        predicted, ground_truth, times, batch_ids, level_ids = [], [], [], [], []
        zero = targets.new_zeros((), dtype=dtype)
        for level, prediction in enumerate(inputs):
            stride = self.factor[level]
            if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
                raise ValueError("IB-ECL scale factors must be positive integers")
            target_level = targets[:, ::stride, :]
            if prediction.shape != target_level.shape or prediction.device != targets.device:
                raise ValueError(f"IB-ECL level {level}: prediction shape/device must match targets[:, ::{stride}, :]")
            if not prediction.is_floating_point():
                raise TypeError("IB-ECL predictions must be floating-point tensors")
            # An empty sum retains the graph without reading ignored values or
            # overflowing on a large half-precision prediction sum.
            zero = zero + prediction[:, :0, 1:3].sum(dtype=dtype)
            batch, index = (target_level[..., 0] == 1).nonzero(as_tuple=True)
            predicted.append(prediction[batch, index, 1:3].to(dtype=dtype))
            ground_truth.append(target_level[batch, index, 1:3].detach().to(dtype=dtype))
            times.append((index * stride).to(dtype=dtype))
            batch_ids.append(batch)
            level_ids.append(torch.full_like(batch, level))

        offsets = torch.cat(predicted)
        if offsets.shape[0] == 0:
            return zero
        gt_offsets = torch.cat(ground_truth)
        if not torch.all(torch.isfinite(offsets) & (offsets >= 0)):
            raise ValueError("IB-ECL positive predicted offsets must be finite and non-negative")
        if not torch.all(torch.isfinite(gt_offsets) & (gt_offsets >= 0)):
            raise ValueError("IB-ECL positive GT offsets must be finite and non-negative")
        position = torch.cat(times)
        endpoints = torch.stack((position - offsets[:, 0], position + offsets[:, 1]), dim=-1)
        gt_endpoints = torch.stack((position - gt_offsets[:, 0], position + gt_offsets[:, 1]), dim=-1)

        # Exact GT endpoint keys, not connected components of the positive mask:
        # adjacent/overlapping instances keep the assignment in the original GT.
        keys = torch.cat((torch.cat(batch_ids).to(dtype=dtype)[:, None], gt_endpoints), dim=1)
        instances, instance_id = torch.unique(keys, dim=0, return_inverse=True)
        pairs, pair_id, nodes_per_pair = torch.unique(
            torch.stack((instance_id, torch.cat(level_ids)), dim=1),
            dim=0, return_inverse=True, return_counts=True,
        )
        levels_per_instance = torch.bincount(pairs[:, 0], minlength=instances.shape[0])
        weights = 1.0 / (
            levels_per_instance[instance_id].to(dtype=dtype)
            * nodes_per_pair[pair_id].to(dtype=dtype)
        )
        # w_i = 1 / (number of active levels * nodes in this instance/level).
        # Shift the coordinate origin by the SAME GT endpoints for every node
        # of an instance before summation. Algebraically this is still z_i-mu_g:
        # (z_i-z_gt) - sum_j w_j*(z_j-z_gt) = z_i - mu_g.
        # It is NOT another GT regression term; common boundary bias still cancels.
        # Centering avoids roundoff from summing large absolute frame positions.
        centered = endpoints - instances[instance_id, 1:]
        consensus = centered.new_zeros((instances.shape[0], 2)).index_add(
            0, instance_id, weights[:, None] * centered
        )  # Differentiable mu_g-z_gt; never detach the consensus.
        duration = (instances[:, 2] - instances[:, 1]).clamp_min(1)
        deviations = (centered - consensus[instance_id]) / duration[instance_id, None]
        node_losses = torch.nn.functional.smooth_l1_loss(
            deviations, torch.zeros_like(deviations), beta=self.ib_ecl_beta, reduction="none"
        ).mean(dim=-1)
        loss = (weights * node_losses).sum() / instances.shape[0]
        if not torch.isfinite(loss):
            raise FloatingPointError("IB-ECL produced a non-finite loss; inspect input/GT magnitudes")
        return loss + zero

    def smooth_l1(self, inputs, targets):
        loss = torch.nn.functional.smooth_l1_loss(inputs[:, :, 1:] / 25, targets[:, :, 1:] / 25, reduction="none")
        loss = loss.mean(dim=-1)
        loss = targets[:, :, 0] * loss
        return loss

    def focal(self, inputs, targets):
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs[:, :, 0],
            targets[:, :, 0],
            self.alpha,
            self.gamma,
            "none",
        )
        return loss

    def diou(self, inputs, targets):
        loss = ctr_diou_loss_1d(
            input_offsets=inputs[:, :, 1:].reshape((-1, 2)),
            target_offsets=targets[:, :, 1:].view(-1, 2),
            reduction="none",
        ).view(inputs.shape[0], inputs.shape[1])
        loss = targets[:, :, 0] * loss
        return loss

    def num_positives(self, targets):
        den = targets[:, :, 0].sum(dim=-1)
        den[den == 0] = 1
        return den

    def localization_loss(self, inputs, targets):
        return torch.stack(
            [
                (
                    self.l_focal * self.focal(input_, targets[:, :: self.factor[i], :])
                    + self.l_diou * self.diou(input_, targets[:, :: self.factor[i], :])
                    + self.l_sl1 * self.smooth_l1(input_, targets[:, :: self.factor[i], :])
                ).sum(dim=-1)
                / self.num_positives(targets[:, :: self.factor[i], :])
                for i, input_ in enumerate(inputs)
            ],
            dim=-1,
        ).mean(dim=-1)

    def detection_loss(self, inputs, targets):
        alpha = 4
        video_level_inputs = [
            torch.sum(input_[:, :, 0] * torch.softmax(alpha * input_[:, :, 0], dim=-1), dim=-1) for input_ in inputs
        ]
        video_level_targets = [torch.any(targets[:, :, 0], dim=-1).float() for _ in inputs]
        return torch.stack(
            [self.bce(input_, target_) for input_, target_ in zip(video_level_inputs, video_level_targets)], dim=-1
        ).mean(dim=-1)

    def reconstruction_loss(self, dissimilarity, targets):
        fakes = torch.any(targets[:, :, 0], dim=-1).view(-1, 1)
        masked_dissimilarity = dissimilarity.masked_fill(fakes, 0.0)
        return masked_dissimilarity.mean(dim=-1)

    def forward(self, inputs, targets, errors):
        loss_loc = self.localization_loss(inputs, targets)
        loss_det = self.detection_loss(inputs, targets)
        loss_rec = self.reconstruction_loss(errors, targets)
        loss = (loss_loc + self.l_det * loss_det + self.l_rec * loss_rec) / (1 + self.l_det + self.l_rec)
        loss = loss.mean()
        # IB-ECL is a training-only regularizer. Evaluation.get_predictions()
        # executes under torch.no_grad(), so validation/test loss and inference
        # retain the baseline objective and incur no ECL grouping/aggregation.
        # The disabled path remains the exact legacy path as well.
        if self.enable_ib_ecl and self.ib_ecl_weight > 0 and torch.is_grad_enabled():
            loss = loss + self.ib_ecl_weight * self.ib_ecl(inputs, targets)
        return loss
