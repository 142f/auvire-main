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

    def __init__(self, alpha, gamma, composition, factor, quality_beta=2.0, quality_weight=0.5,
                 rank_weight=0.5, rank_margin=0.5, quality_floor=0.0, quality_alpha=0.5,
                 quality_target="iou", quality_threshold=0.95):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l_focal = 1.0 if "focal" in composition else 0.0
        self.l_diou = 1.0 if "diou" in composition else 0.0
        self.l_sl1 = 1.0 if "sl1" in composition else 0.0
        self.l_rec = 1.0 if "rec" in composition else 0.0
        self.l_det = 1.0 if "det" in composition else 0.0
        self.l_quality = float(quality_weight) if "quality" in composition else 0.0
        self.quality_beta = float(quality_beta)
        # MAQS-R (backup arm): hard-pair ranking term on the quality logits.
        self.l_rank = float(rank_weight) if "rank" in composition else 0.0
        self.rank_margin = float(rank_margin)
        # Optional target remap: concentrate the quality regression on the high-IoU
        # band that AP@0.95 actually measures (0.0 keeps the plain IoU target).
        self.quality_floor = float(quality_floor)
        # The quality branch must be *calibrated* over the whole timeline, so its
        # focal imbalance weight is symmetric by default (the 0.98 used for the
        # positive-starved classification task would stop it from learning to
        # suppress the spurious boxes that dominate the high-recall ranks).
        self.quality_alpha = float(quality_alpha)
        # Target family: "iou" regresses the interval IoU, "binary" predicts the
        # event AP@0.95 actually scores (IoU above the threshold). The binary
        # variant turns the branch into P(this box is a true positive).
        self.quality_target = str(quality_target).lower()
        self.quality_threshold = float(quality_threshold)
        self.factor = factor
        self.bce = BCELoss()

    def smooth_l1(self, inputs, targets):
        loss = torch.nn.functional.smooth_l1_loss(inputs[:, :, 1:3] / 25, targets[:, :, 1:] / 25, reduction="none")
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
            input_offsets=inputs[:, :, 1:3].reshape((-1, 2)),
            target_offsets=targets[:, :, 1:].view(-1, 2),
            reduction="none",
        ).view(inputs.shape[0], inputs.shape[1])
        loss = targets[:, :, 0] * loss
        return loss

    def quality(self, inputs, targets):
        """Quality Focal Loss on the third scoring pathway (MAQS).

        The target is the IoU between the interval this centre would emit with
        the *current* offsets and the ground-truth interval. It is computed on
        the prediction detached, so the branch learns to predict the localization
        quality of the regressor rather than to move the regressor. Positives and
        negatives are handled uniformly: a frame outside every fake period has
        zero target offsets, and its IoU collapses to 0.
        """
        logits = inputs[:, :, 3]
        with torch.no_grad():
            pred_left, pred_right = inputs[:, :, 1].detach(), inputs[:, :, 2].detach()
            gt_left, gt_right = targets[:, :, 1], targets[:, :, 2]
            intersection = torch.minimum(pred_left, gt_left) + torch.minimum(pred_right, gt_right)
            union = torch.maximum(pred_left, gt_left) + torch.maximum(pred_right, gt_right)
            iou = intersection / union.clamp(min=1e-8)
            if self.quality_floor > 0.0:
                floor = self.quality_floor
                iou = ((iou - floor) / (1.0 - floor)).clamp(min=0.0, max=1.0)
            if self.quality_target == "binary":
                iou = (iou >= self.quality_threshold).to(iou.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(logits, iou, self.quality_alpha, self.gamma, "none")
        weight = (iou - torch.sigmoid(logits)).abs().pow(self.quality_beta)
        return weight * loss

    def quality_loss(self, inputs, targets):
        return torch.stack(
            [
                self.quality(input_, targets[:, :: self.factor[i], :]).sum(dim=-1)
                / self.num_positives(targets[:, :: self.factor[i], :])
                for i, input_ in enumerate(inputs)
            ],
            dim=-1,
        ).mean(dim=-1)

    def ranking(self, inputs, targets):
        """Hard-pair ranking on the quality logits (MAQS-R backup arm).

        For every video the best-positive proposal (IoU >= 0.95) must outrank the
        hardest near-miss (0.4 <= IoU <= 0.9), which is exactly the comparison
        AP@0.95 makes when a near-miss sits above a true positive.
        """
        logits = inputs[:, :, 3]
        with torch.no_grad():
            pred_left, pred_right = inputs[:, :, 1].detach(), inputs[:, :, 2].detach()
            gt_left, gt_right = targets[:, :, 1], targets[:, :, 2]
            intersection = torch.minimum(pred_left, gt_left) + torch.minimum(pred_right, gt_right)
            union = torch.maximum(pred_left, gt_left) + torch.maximum(pred_right, gt_right)
            iou = intersection / union.clamp(min=1e-8)
        positive = iou >= 0.95
        negative = (iou >= 0.4) & (iou <= 0.9)
        valid = positive.any(dim=1) & negative.any(dim=1)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        best_positive = logits.masked_fill(~positive, float("-inf")).max(dim=1).values
        worst_negative = logits.masked_fill(~negative, float("-inf")).max(dim=1).values
        margin = self.rank_margin + worst_negative - best_positive
        return torch.nn.functional.softplus(margin)[valid].mean()

    def ranking_loss(self, inputs, targets):
        return torch.stack(
            [self.ranking(input_, targets[:, :: self.factor[i], :]) for i, input_ in enumerate(inputs)],
            dim=-1,
        ).mean(dim=-1)

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
        use_quality = self.l_quality and inputs[0].shape[-1] > 3
        loss_quality = self.quality_loss(inputs, targets) if use_quality else None
        loss_det = self.detection_loss(inputs, targets)
        loss_rec = self.reconstruction_loss(errors, targets)
        # The historical normalisation is kept untouched so that the quality term
        # does not silently re-weight the localization / reconstruction terms;
        # it enters as an additive term with its own weight instead.
        loss = (loss_loc + self.l_det * loss_det + self.l_rec * loss_rec) / (1 + self.l_det + self.l_rec)
        if loss_quality is not None:
            loss = loss + self.l_quality * loss_quality
        if self.l_rank and inputs[0].shape[-1] > 3:
            loss = loss + self.l_rank * self.ranking_loss(inputs, targets)
        return loss.mean()
