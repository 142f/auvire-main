from typing import List, Union

import torch
from torch import Tensor

import numpy as np


def iou_1d(proposal, target) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.0)
    union = outer_end - outer_begin
    return inter / union


def _label_batch(labels, videos, device):
    """Pad ragged segments. An empty label contributes 0, matching torch.tensor(label).shape[0]."""
    rows = [torch.tensor(labels[index]).to(device) for index in range(videos)]
    counts = [0 if row.numel() == 0 else row.shape[0] for row in rows]
    width = max(counts, default=0)
    dtype = next((row.dtype for row, count in zip(rows, counts) if count), torch.float32)
    padded = torch.zeros((videos, width, 2), dtype=dtype, device=device)
    valid = torch.zeros((videos, width), dtype=torch.bool, device=device)
    for index, (row, count) in enumerate(zip(rows, counts)):
        if count:
            padded[index, :count] = row
            valid[index, :count] = True
    return padded, valid, sum(counts)


def _interval_iou_batch(proposals, targets):
    """iou_1d over a batch. proposals is (V, P, 2) and targets is (V, L, 2)."""
    proposal_begin = proposals[:, :, 0].unsqueeze(-1)
    proposal_end = proposals[:, :, 1].unsqueeze(-1)
    target_begin = targets[:, :, 0].unsqueeze(1)
    target_end = targets[:, :, 1].unsqueeze(1)
    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)
    inter = torch.clamp(inner_end - inner_begin, min=0.0)
    union = outer_end - outer_begin
    return inter / union


def _first_unused_hits(hits):
    """Match labels in order. Each label takes the first proposal that is still unused."""
    videos, proposals, labels = hits.shape
    taken = torch.zeros((videos, proposals), dtype=torch.bool, device=hits.device)
    chosen = torch.zeros((videos, proposals), dtype=torch.bool, device=hits.device)
    rows = torch.arange(videos, device=hits.device)
    for label_index in range(labels):
        candidates = hits[:, :, label_index] & ~taken
        pick = candidates.any(dim=1)
        index = candidates.to(torch.uint8).argmax(dim=1)
        chosen[rows[pick], index[pick]] = True
        taken[rows[pick], index[pick]] = True
    return chosen


class AP:
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5, device: str = "cpu"):
        super().__init__()
        self.iou_thresholds: List[float] = iou_thresholds if type(iou_thresholds) is list else [iou_thresholds]
        self.device = device
        self.n_labels = 0
        self.ap: dict = {}

    def __call__(self, proposals, labels) -> dict:
        label_rows, label_valid, self.n_labels = _label_batch(labels, proposals.shape[0], self.device)
        if label_rows.shape[1] == 0:
            ious = proposals.new_zeros((proposals.shape[0], proposals.shape[1], 0))
        else:
            ious = _interval_iou_batch(proposals[:, :, 1:] / 25.0, label_rows)
        confidence = proposals[:, :, 0].reshape(-1)
        # Confidence does not depend on the IoU threshold, so one stable order is enough.
        order = confidence.sort(stable=True, descending=True).indices
        confidence = confidence[order]
        for iou_threshold in self.iou_thresholds:
            hits = (ious > iou_threshold) & label_valid[:, None, :]
            is_tp = _first_unused_hits(hits).reshape(-1)[order]
            values = torch.column_stack([confidence, is_tp])
            curve = self.calculate_curve(values)
            self.ap[iou_threshold] = self.calculate_ap(curve).item()
        return self.ap

    def calculate_curve(self, values):
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)).to(self.device) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1.0, 0.0]]).to(self.device), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    def get_values(
        self,
        iou_threshold: float,
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ) -> Tensor:
        n_labels = len(labels)
        n_proposals = len(proposals)
        if n_labels > 0:
            ious = iou_1d(proposals[:, 1:] / fps, labels)
        else:
            ious = torch.zeros((n_proposals, 0)).to(self.device)

        # values: (confidence, is_TP) rows
        n_labels = ious.shape[1]
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break

        is_TP = torch.zeros(n_proposals, dtype=torch.bool).to(self.device)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes).to(self.device)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values


class AR:
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(
        self,
        n_proposals_list: Union[List[int], int] = 100,
        iou_thresholds: List[float] = None,
        device: str = "cpu",
    ):
        super().__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_proposals_list = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.n_proposals_list = torch.tensor(self.n_proposals_list)
        self.iou_thresholds = iou_thresholds
        self.device = device
        self.ar: dict = {}

    def __call__(self, proposals, labels) -> dict:
        label_rows, label_valid, _ = _label_batch(labels, proposals.shape[0], self.device)
        limit = int(self.n_proposals_list.max())
        ranked = proposals[:, :limit]
        if label_rows.shape[1] == 0:
            ious = ranked.new_zeros((ranked.shape[0], ranked.shape[1], 0))
        else:
            ious = _interval_iou_batch(ranked[:, :, 1:] / 25.0, label_rows)
        selected = (self.n_proposals_list - 1).to(ious.device)
        best = ious.cummax(1).values[:, selected, :]
        thresholds = torch.tensor(self.iou_thresholds, device=best.device)[:, None, None]
        hits = best.unsqueeze(1) > thresholds
        hits = hits & label_valid[:, None, None, :].to(hits.device)
        true_positives = hits.sum(dim=-1)
        per_video = label_valid.sum(dim=1).to(device=true_positives.device, dtype=true_positives.dtype)
        false_negatives = per_video[:, None, None] - true_positives
        # Totals stay on CPU, including when the score tensor is on CUDA.
        values = torch.zeros((len(proposals), len(self.iou_thresholds), len(self.n_proposals_list), 2))
        values[..., 0] = true_positives
        values[..., 1] = false_negatives
        totals = values.sum(dim=0)
        recall = totals[:, :, 0] / (totals[:, :, 0] + totals[:, :, 1])
        for index, count in enumerate(self.n_proposals_list):
            self.ar[count.item()] = recall[:, index].mean().item()
        return self.ar

    def get_values(
        self,
        iou_thresholds: List[float],
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ):
        n_proposals_list = self.n_proposals_list
        max_proposals = max(n_proposals_list)

        proposals = proposals[:max_proposals]
        n_labels = len(labels)

        if n_labels > 0:
            ious = iou_1d(proposals[:, 1:] / fps, labels)
        else:
            ious = torch.zeros((max_proposals, 0)).to(self.device)

        # values: matrix of (TP, FN), shapes (n_iou_thresholds, n_proposal_thresholds, 2)
        iou_max = ious.cummax(0).values[n_proposals_list - 1]  # shape (n_iou_thresholds, n_labels)
        iou_max = iou_max[None]

        iou_thresholds = torch.tensor(iou_thresholds)[:, None, None]
        TP = (iou_max > iou_thresholds.to(self.device)).sum(-1)
        FN = n_labels - TP
        values = torch.stack([TP, FN], dim=-1)

        return values
