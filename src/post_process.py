import torch


def soft_nms_torch_parallel(proposal, sigma, t1, t2, fps, device="cpu"):
    """Batched soft-NMS. The result is bit for bit the historical implementation:
    the same selection order, the same suppression arithmetic, only the
    intermediates are reused instead of reallocated, and the caller's tensor is
    no longer written through.
    """
    proposal = proposal.to(device)
    # The score column is a view into ``proposal`` when the device already matches;
    # the in-place suppression below must not reach the caller's predictions.
    t_score = proposal[:, :, 0].clone()
    t_start = proposal[:, :, 1] / fps
    t_end = proposal[:, :, 2] / fps
    # Frame lengths never change; rows selected below only gain NaN, which
    # propagates through the intersection exactly like recomputing x - y does.
    len_anchors = t_end - t_start

    videos = t_score.shape[0]
    r_start = torch.empty((videos, 101), dtype=t_start.dtype, device=device)
    r_end = torch.empty((videos, 101), dtype=t_end.dtype, device=device)
    r_score = torch.empty((videos, 101), dtype=t_score.dtype, device=device)
    scratch_min = torch.empty_like(t_start)
    scratch_max = torch.empty_like(t_start)
    scratch_union = torch.empty_like(t_start)
    zero = torch.zeros((), dtype=t_start.dtype, device=device)

    step = 0
    while torch.sum(t_score[0, :] > -torch.inf) > 1 and step < 101:
        _, max_index = torch.max(t_score, -1)
        max_index_2d = (torch.arange(videos, device=device), max_index)
        selected_start = t_start[max_index_2d].unsqueeze(1)
        selected_end = t_end[max_index_2d].unsqueeze(1)
        width = t_end[max_index_2d] - t_start[max_index_2d]

        # iou_with_anchors_torch, with its temporaries reused across iterations.
        torch.maximum(t_start, selected_start, out=scratch_min)
        torch.minimum(t_end, selected_end, out=scratch_max)
        inter_len = scratch_max.sub_(scratch_min)
        torch.maximum(inter_len, zero, out=inter_len)
        union_len = scratch_union.copy_(len_anchors).sub_(inter_len).add_(selected_end).sub_(selected_start)
        ious = inter_len.div_(union_len)

        idx = ious > t1 + (t2 - t1) * width.unsqueeze(1)
        idx[max_index_2d] = False
        t_score[idx] *= torch.exp(ious[idx].square_().div_(-sigma))

        r_score[:, step] = t_score[max_index_2d]
        r_start[:, step] = t_start[max_index_2d]
        r_end[:, step] = t_end[max_index_2d]

        t_start[max_index_2d] = torch.nan
        t_end[max_index_2d] = torch.nan
        t_score[max_index_2d] = -torch.inf

        step += 1

    if step == 0:
        # The original loop could not start; keep its empty (0, 3) result.
        empty = torch.empty(0, device=device)
        return torch.stack((empty, empty.clone(), empty.clone()), dim=-1)
    return torch.stack((r_score[:, :step], r_start[:, :step] * fps, r_end[:, :step] * fps), dim=-1)