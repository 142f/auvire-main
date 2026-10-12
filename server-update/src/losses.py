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

    """Weighted sum of the localization / detection / reconstruction terms.

    ``uncertainty`` optionally enables Kendall-style homoscedastic
    uncertainty weighting (Kendall et al., CVPR 2018; analytically
    reparameterised, cf. the softmax-normalised update of 2024) on the three
    task terms: each term ``c`` gets a learnable log-variance ``s_c``, its
    contribution becomes ``e^{-s_c} L_c`` and the objective adds Kendall's
    ``s_c`` barrier, which keeps the optimum at ``e^{-s_c} = 1 / L_c`` and
    prevents the weights from collapsing onto a single term. Terms whose
    composition flag is off stay at zero and are excluded from the barrier,
    so their log-variance never moves. The log-variances start at zero, so
    an enabled-but-untrained criterion is numerically identical to the
    baseline objective (differences below float reassociation, verified by
    the identity check); the parameters live in this module, so the trainer
    only has to add ``criterion.parameters()`` to the optimizer.
    """

    def __init__(self, alpha, gamma, composition, factor, consistency=None, uncertainty=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l_focal = 1.0 if "focal" in composition else 0.0
        self.l_diou = 1.0 if "diou" in composition else 0.0
        self.l_sl1 = 1.0 if "sl1" in composition else 0.0
        self.l_rec = 1.0 if "rec" in composition else 0.0
        self.l_det = 1.0 if "det" in composition else 0.0
        consistency = consistency or {}
        self.l_smooth = float(consistency.get("smoothness", 0.0))
        self.l_hier = float(consistency.get("hierarchy", 0.0))
        self.factor = factor
        self.bce = BCELoss()
        uncertainty = uncertainty or {}
        self.uncertainty_enabled = bool(uncertainty.get("enabled", False))
        if self.uncertainty_enabled:
            # loc / det / rec log-variances, zero-initialised to unit weight.
            self.uncertainty_log_variance = nn.Parameter(torch.zeros(3))
            self.uncertainty_active = (
                1.0,
                1.0 if self.l_det > 0 else 0.0,
                1.0 if self.l_rec > 0 else 0.0,
            )
        else:
            self.uncertainty_log_variance = None

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

    def temporal_smoothness(self, inputs, targets):
        """Penalize boundary jitter between adjacent manipulated frames.

        ``/25`` converts frame offsets to seconds, matching the convention of
        :meth:`smooth_l1`, so the regularizer stays scale-comparable to the
        other loss terms. Returns a per-frame term: the jump between frame
        t-1 and t is attributed to frame t, frame 0 carries a zero.
        """
        boundaries = inputs[:, :, 1:] / 25
        mask = targets[:, :, 0]
        pair_mask = mask[:, 1:] * mask[:, :-1]
        jump = (boundaries[:, 1:] - boundaries[:, :-1]).abs().mean(dim=-1)
        return torch.nn.functional.pad(pair_mask * jump, (1, 0))

    def hierarchical_agreement(self, inputs, targets_list):
        """Penalize disagreement between boundary predictions of adjacent FPN levels.

        The coarser level boundaries are nearest-upsampled to the finer level
        resolution; both predict the same frame-space offsets (converted to
        seconds, as in :meth:`smooth_l1`), so they should agree inside
        manipulated frames.
        """
        terms = []
        for i in range(len(inputs) - 1):
            fine = inputs[i][:, :, 1:] / 25
            coarse = inputs[i + 1][:, :, 1:] / 25
            if coarse.shape[1] != fine.shape[1]:
                coarse = torch.nn.functional.interpolate(
                    coarse.permute((0, 2, 1)), size=fine.shape[1], mode="nearest"
                ).permute((0, 2, 1))
            mask = targets_list[i][:, :, 0]
            disagreement = (fine - coarse).abs().mean(dim=-1)
            # Same reduction as focal / DIoU: sum over time, divide by the
            # positive-frame count. A raw sum grows with duration and drowns
            # the localization loss on long clips.
            terms.append((mask * disagreement).sum(dim=-1) / self.num_positives(targets_list[i]))
        if not terms:
            return inputs[0][:, :, 0].sum(dim=-1) * 0
        return torch.stack(terms, dim=-1).mean(dim=-1)

    def localization_loss(self, inputs, targets):
        targets_list = [targets[:, :: self.factor[i], :] for i in range(len(inputs))]
        losses = []
        for i, input_ in enumerate(inputs):
            loss = (
                self.l_focal * self.focal(input_, targets_list[i])
                + self.l_diou * self.diou(input_, targets_list[i])
                + self.l_sl1 * self.smooth_l1(input_, targets_list[i])
            )
            if self.l_smooth:
                loss = loss + self.l_smooth * self.temporal_smoothness(input_, targets_list[i])
            losses.append(loss.sum(dim=-1) / self.num_positives(targets_list[i]))
        loss = torch.stack(losses, dim=-1).mean(dim=-1)
        if self.l_hier:
            loss = loss + self.l_hier * self.hierarchical_agreement(inputs, targets_list)
        return loss

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
        if self.uncertainty_log_variance is None:
            # Smoothness and hierarchy already scale their own terms. They are not
            # 0/1 composition flags, so they do not belong in this normalizer.
            # With both weights at 0 the divisor matches the baseline exactly.
            loss = (loss_loc + self.l_det * loss_det + self.l_rec * loss_rec) / (
                1 + self.l_det + self.l_rec
            )
        else:
            log_variance = self.uncertainty_log_variance
            weights = torch.exp(-log_variance)
            loss = (
                weights[0] * loss_loc
                + self.l_det * weights[1] * loss_det
                + self.l_rec * weights[2] * loss_rec
            ) / (1 + self.l_det + self.l_rec)
            barrier = (
                self.uncertainty_active[0] * log_variance[0]
                + self.uncertainty_active[1] * log_variance[1]
                + self.uncertainty_active[2] * log_variance[2]
            ) / (1 + self.l_det + self.l_rec)
            loss = loss + barrier
        return loss.mean()
