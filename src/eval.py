from src.metrics import AP, AR, iou_1d
from src.post_process import soft_nms_torch_parallel
from src.perf_monitor import stage, batches
from src.datasets import pad_feature_batch

import os

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm


def adjust_data(data, task, dataset, device, non_blocking=False):
    if task not in {"tfl", "dfd"}:
        raise ValueError(f"Unknown task: {task}")

    if dataset in {"lavdf", "avdeepfake1m"}:
        data, fake_periods = data
        video_features, audio_features, labels = data
        max_length = labels.shape[1]
        video_features = pad_feature_batch(video_features, max_length, device, non_blocking)
        audio_features = pad_feature_batch(audio_features, max_length, device, non_blocking)
        adjusted = {
            "video_features": video_features,
            "audio_features": audio_features,
            "fake_periods": fake_periods,
        }
        if task == "tfl":
            adjusted["labels"] = labels.float().to(device, non_blocking=non_blocking)
        else:
            target = labels[:, :, 0].amax(dim=-1).tolist()
            adjusted["video_target"] = target
            adjusted["audio_target"] = target
        return adjusted
    if task == "dfd":
        data, audio_target = data
        video_features, audio_features, video_target = data
        video_features = video_features.to(device, non_blocking=non_blocking)
        audio_features = audio_features.to(device, non_blocking=non_blocking)
        return {
            "video_features": video_features,
            "audio_features": audio_features,
            "video_target": video_target,
            "audio_target": audio_target,
        }
    raise ValueError(f"Dataset {dataset} does not provide temporal-localization labels")


class Evaluation:

    def __init__(
        self,
        model,
        loader,
        criterion,
        device,
        factor,
        generalization=False,
        task="tfl",
        show_progress=False,
        phase="Validation",
        progress_interval_batches=50,
        non_blocking_transfer=False,
        perf_monitor=None,
        regression_audit=None,
    ):
        self.model = model
        self.loader = loader
        self.max_length = loader.dataset.max_length
        self.dataset = loader.dataset.name
        self.criterion = criterion
        self.device = device
        self.metrics_device = "cpu"
        self.loss = 0
        self.ground_truth = []
        self.predictions = None
        self.sigma, self.t1, self.t2, self.fps = 0.7234, 0.1968, 0.4123, 25
        self.confidence_thres = 0.5
        self.n_proposals_list = {
            "lavdf": [100, 50, 20, 10],
            "avdeepfake1m": [50, 30, 20, 10, 5],
            "generalization": [100, 50, 30, 20, 10, 5],
        }
        self.ar_iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.ap_iou_thresholds = {
            "lavdf": [0.5, 0.75, 0.95],
            "avdeepfake1m": [0.5, 0.75, 0.9, 0.95],
            "generalization": [0.5, 0.75, 0.9, 0.95],
        }
        self.metrics = {}
        self.factor = factor
        self.generalization = generalization
        self.task = task
        self.show_progress = show_progress
        self.phase = phase
        self.progress_interval_batches = max(int(progress_interval_batches), 1)
        self.non_blocking_transfer = non_blocking_transfer
        self._prediction_batches = None
        self.perf_monitor = perf_monitor
        self.regression_audit = regression_audit

    def get_predictions(self):
        self.model.eval()
        loss_total = None
        loss_weight = 0
        with torch.no_grad():
            progress = tqdm(
                batches(self.perf_monitor, self.loader, self.phase),
                total=len(self.loader),
                desc=self.phase,
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                position=1,
                disable=not self.show_progress,
                mininterval=1.0,
            )
            for iteration, data in enumerate(progress):
                with stage(self.perf_monitor, "h2d", gpu=True):
                    data_adjusted = adjust_data(
                        data,
                        self.task,
                        self.dataset,
                        self.device,
                        non_blocking=self.non_blocking_transfer,
                    )
                with stage(self.perf_monitor, "forward", gpu=True):
                    p, z = self.model([data_adjusted["video_features"], data_adjusted["audio_features"]])
                if self.task == "tfl":
                    batch_size = data_adjusted["labels"].shape[0]
                    with stage(self.perf_monitor, "loss", gpu=True):
                        batch_loss = self.criterion(p, data_adjusted["labels"], z).detach()
                    weighted_loss = batch_loss * batch_size
                    loss_total = weighted_loss if loss_total is None else loss_total + weighted_loss
                    loss_weight += batch_size
                    with stage(self.perf_monitor, "d2h", gpu=True):
                        self.update_predictions(p, data_adjusted["fake_periods"])
                    if self.show_progress and (iteration + 1) % self.progress_interval_batches == 0:
                        progress.set_postfix(
                            loss=f"{(loss_total / loss_weight).item():.4f}", refresh=False
                        )
                elif self.task == "dfd":
                    overall_target = [
                        max(x, y) for x, y in zip(data_adjusted["video_target"], data_adjusted["audio_target"])
                    ]
                    with stage(self.perf_monitor, "d2h", gpu=True):
                        self.update_predictions(p, overall_target)

            progress.close()

        self._finalize_predictions()
        self.loss = (loss_total / loss_weight).item() if loss_weight else 0.0
        if self.regression_audit is not None:
            self.regression_audit.evaluation_predictions(self)

    def compute_dfd_metrics(self, proposals):
        y_score = torch.sigmoid(proposals[:, :, 0]).max(dim=-1)[0].cpu().numpy()
        y_true = np.array(self.ground_truth)
        self.metrics = {
            "acc": accuracy_score(y_true, y_score > 0.5),
            "ap": average_precision_score(y_true, y_score),
            "auc": roc_auc_score(y_true, y_score),
        }

    def compute_tfl_metrics(self, proposals):
        ap = AP(
            iou_thresholds=self.ap_iou_thresholds["generalization" if self.generalization else self.dataset],
            device=self.metrics_device,
        )
        ap_score = ap(proposals, self.ground_truth)
        ar = AR(
            n_proposals_list=self.n_proposals_list["generalization" if self.generalization else self.dataset],
            iou_thresholds=self.ar_iou_thresholds,
            device=self.metrics_device,
        )
        ar_score = ar(proposals, self.ground_truth)
        self.metrics = {"loss": self.loss, "ap": ap_score, "ar": ar_score}

    def _boundary_diag(self, proposals):
        """只读诊断：把 @0.95 的失败拆成"边界差几帧"与"段长"两个维度。

        ``AUVIRE_BOUNDARY_DIAG=<out.json>`` 时写盘；不改变任何评测数值。
        每段取 IoU 最高的提案作为配对，统计起/止边界的绝对误差（帧）。
        """
        import json

        path = os.environ["AUVIRE_BOUNDARY_DIAG"].strip()
        errors, lengths, ious = [], [], []
        for index, periods in enumerate(self.ground_truth):
            boxes = [b for b in (periods or []) if float(b[1]) > float(b[0])]
            if not boxes:
                continue
            pred = self.predictions_proposals[index][:, 1:] / self.fps
            gt = torch.tensor([[float(a), float(b)] for a, b in boxes], dtype=pred.dtype)
            matrix = iou_1d(pred, gt)
            best, which = matrix.max(dim=0)
            for g, p in enumerate(which.tolist()):
                ious.append(float(best[g]))
                errors.append([abs(float(pred[p, 0]) - float(gt[g, 0])) * self.fps,
                               abs(float(pred[p, 1]) - float(gt[g, 1])) * self.fps])
                lengths.append((float(gt[g, 1]) - float(gt[g, 0])) * self.fps)

        def bucket(values, edges):
            return {f"{lo}-{hi}": int(sum(1 for v in values if lo <= v < hi))
                    for lo, hi in zip(edges, edges[1:])}

        edges = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 1e9]
        payload = {
            "videos": len(self.ground_truth),
            "segments": len(ious),
            "iou_ge": {str(t): int(sum(1 for v in ious if v >= t)) for t in (0.5, 0.75, 0.9, 0.95, 1.0)},
            "start_err_frames": bucket([e[0] for e in errors], edges),
            "end_err_frames": bucket([e[1] for e in errors], edges),
            "seg_len_frames": bucket(lengths, [0, 10, 20, 40, 80, 1e9]),
            "best_iou_hist": bucket(ious, [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0000001]),
            "unreachable_095": int(sum(1 for v in ious if v < 0.95)),
            "near_miss_08_095": int(sum(1 for v in ious if 0.8 <= v < 0.95)),
            "err_by_len": {},
            "fps": self.fps,
        }
        for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)):
            sel = [e for e, l in zip(errors, lengths) if lo <= l < hi]
            if sel:
                payload["err_by_len"][f"{lo}-{hi}"] = {
                    "n": len(sel),
                    "start_med_frames": round(float(torch.tensor([e[0] for e in sel]).median()), 3),
                    "end_med_frames": round(float(torch.tensor([e[1] for e in sel]).median()), 3),
                }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"[boundary_diag] 已写入 {path}，配对段数 {len(ious)}")

    def _rank_diag(self, proposals):
        """只读诊断：AP@0.95 的排序损失被谁吃掉了。

        把全部提案按分数全局排序（与 AP 的口径一致），在每个召回水平上统计前 k 个提案里
        非 TP 的三类来源：
          real_video —— 该视频没有任何 GT（纯真视频的高分误报）
          near_miss  —— 与本视频 GT 的 IoU 落在 (0.5, 0.95]（近失：框差一点）
          other      —— 本视频有 GT，但该提案 IoU<=0.5（段外误报）
        AUVIRE_RANK_DIAG=<out.json> 时写盘；不改变任何评测数值。
        """
        import json

        path = os.environ["AUVIRE_RANK_DIAG"].strip()
        scores = proposals[:, :, 0]
        boxes = proposals[:, :, 1:3] / self.fps
        videos, count = scores.shape
        best = torch.zeros((videos, count), dtype=scores.dtype)
        has_gt = torch.zeros(videos, dtype=torch.bool)
        per_video_gt = []
        for video, periods in enumerate(self.ground_truth):
            valid = [b for b in (periods or []) if float(b[1]) > float(b[0])]
            per_video_gt.append(len(valid))
            if not valid:
                continue
            has_gt[video] = True
            target = torch.tensor([[float(a), float(b)] for a, b in valid], dtype=boxes.dtype)
            best[video] = iou_1d(boxes[video], target).max(dim=1).values

        flat_best = best.reshape(-1)
        flat_video = torch.arange(videos).repeat_interleave(count)
        order = torch.argsort(scores.reshape(-1), descending=True, stable=True)

        hits = (best > 0.95) & has_gt[:, None]
        taken = torch.zeros_like(hits)
        matched = torch.zeros_like(hits)
        rows = torch.arange(videos)
        for _ in range(max(per_video_gt, default=0)):
            candidates = hits & ~taken
            pick = candidates.any(dim=1)
            index = candidates.to(torch.uint8).argmax(dim=1)
            matched[rows[pick], index[pick]] = True
            taken[rows[pick], index[pick]] = True

        is_tp = matched.reshape(-1)[order]
        ordered_best = flat_best[order]
        ordered_has_gt = has_gt[flat_video[order]]
        total_gt = int(sum(per_video_gt))
        cum_tp = torch.cumsum(is_tp.float(), dim=0)
        levels = {}
        for recall_target in (0.5, 0.8, 0.9, 0.95):
            need = torch.tensor(recall_target * total_gt)
            k = int(torch.searchsorted(cum_tp, need).item()) + 1
            k = min(k, len(ordered_best))
            prefix_has_gt = ordered_has_gt[:k]
            prefix_is_tp = is_tp[:k]
            prefix_best = ordered_best[:k]
            levels[str(recall_target)] = {
                "k": k,
                "tp": int(prefix_is_tp.sum()),
                "real_video_fp": int((~prefix_has_gt).sum()),
                "near_miss_fp": int((prefix_has_gt & ~prefix_is_tp & (prefix_best > 0.5)).sum()),
                "other_fp": int((prefix_has_gt & ~prefix_is_tp & (prefix_best <= 0.5)).sum()),
                "precision": round(float(prefix_is_tp.float().mean()), 4),
            }
        payload = {
            "videos": videos,
            "proposals_per_video": count,
            "total_proposals": int(videos * count),
            "gt_segments": total_gt,
            "videos_without_gt": int((~has_gt).sum()),
            "at_recall": levels,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"[rank_diag] 已写入 {path}")
    def compute_metrics(self):
        self.get_predictions()
        if self.show_progress:
            tqdm.write(f"{self.phase}: post-processing and computing metrics...")
        with stage(self.perf_monitor, "transform", always=True):
            self.transform_predictions()
        with stage(self.perf_monitor, "softnms", always=True):
            proposals = soft_nms_torch_parallel(
                self.predictions, self.sigma, self.t1, self.t2, self.fps, self.metrics_device
            )
            self.predictions_proposals = proposals.detach().cpu()
            if os.environ.get("AUVIRE_BOUNDARY_DIAG"):
                self._boundary_diag(proposals)
            if os.environ.get("AUVIRE_RANK_DIAG"):
                self._rank_diag(proposals)
        with stage(self.perf_monitor, "metrics", always=True):
            if self.task == "tfl":
                self.compute_tfl_metrics(proposals)
            elif self.task == "dfd":
                self.compute_dfd_metrics(proposals)
            else:
                raise ValueError(f"Unknown task: {self.task}")
        if self.regression_audit is not None:
            self.regression_audit.evaluation_metrics(self, proposals)

    def update_predictions(self, predictions, labels):
        self.ground_truth.extend(labels)
        if isinstance(predictions, list):
            if self._prediction_batches is None:
                self._prediction_batches = [[] for _ in predictions]
            for batches, prediction in zip(self._prediction_batches, predictions):
                batches.append(prediction.detach().cpu())
        else:
            if self._prediction_batches is None:
                self._prediction_batches = []
            self._prediction_batches.append(predictions.detach().cpu())

    def _finalize_predictions(self):
        if self._prediction_batches is None:
            raise RuntimeError("Evaluation loader produced no predictions")
        if self._prediction_batches and isinstance(self._prediction_batches[0], list):
            self.predictions = [torch.cat(batches, dim=0) for batches in self._prediction_batches]
        else:
            self.predictions = torch.cat(self._prediction_batches, dim=0)

    def transform_predictions(self):
        idx = torch.arange(0, self.max_length).to(self.metrics_device)
        # AUVIRE_FINE_ONLY=1：只保留 stride=1 两级的提案，用于检验粗层（2/4 帧步长）
        # 是否稀释 AP@0.95。默认关闭，不改变任何既有行为。
        fine_only = os.environ.get("AUVIRE_FINE_ONLY", "").strip().lower() in {"1", "true", "yes"}
        # AUVIRE_LEVEL_SCALE="w1,w2,w3,w4"：按级乘到置信度上，用于检验粗层提案的排序是否被
        # 系统性高估（0 训练成本的 level-aware 重打分）。默认关闭。
        score_scale = None
        if isinstance(self.predictions, list):
            strides = [self.factor[i] for i, _ in enumerate(self.predictions)]
            widths = [tensor.shape[1] for tensor in self.predictions]
            idx = torch.cat([idx[:: stride] for stride in strides])
            self.predictions = torch.cat(self.predictions, dim=1)
            if fine_only:
                keep = torch.cat([torch.full((width,), stride == 1, dtype=torch.bool)
                                  for stride, width in zip(strides, widths)]).to(self.metrics_device)
                idx, self.predictions = idx[keep], self.predictions[:, keep]
            weights = [w.strip() for w in os.environ.get("AUVIRE_LEVEL_SCALE", "").split(",") if w.strip()]
            if weights and len(weights) == len(widths):
                columns = [torch.full((width,), float(w)) for w, width in zip(weights, widths)]
                score_scale = torch.cat(columns).to(self.metrics_device)
                if fine_only:
                    score_scale = score_scale[keep]
        probability = torch.sigmoid(self.predictions[:, :, 0])
        if self.predictions.shape[2] > 3:
            # MAQS decode: re-weight the authenticity score with the predicted
            # localization quality, then drop the extra columns so that soft-NMS,
            # AP and AR observe exactly the historical 3-column layout.
            # AUVIRE_Q_ALPHA=1.0 keeps the plain authenticity score (fusion off).
            quality_alpha = float(os.environ.get("AUVIRE_Q_ALPHA", "0.8"))
            quality = torch.sigmoid(self.predictions[:, :, 3])
            if os.environ.get("AUVIRE_Q_SHUFFLE"):
                # Read-only control: destroy the *content* of the quality signal
                # while keeping its marginal distribution, so any remaining gain
                # cannot be attributed to localization quality.
                flat = quality.reshape(-1)[torch.randperm(quality.numel())]
                quality = flat.reshape(quality.shape)
            if quality_alpha >= 1.0:
                probability = probability
            else:
                probability = probability.clamp_min(1e-12).pow(quality_alpha) * quality.clamp_min(1e-12).pow(
                    1.0 - quality_alpha
                )
            self.predictions = self.predictions[:, :, :3].contiguous()
        self.predictions[:, :, 0] = probability
        if score_scale is not None:
            self.predictions[:, :, 0] = self.predictions[:, :, 0] * score_scale
        self.predictions[:, :, 1] = torch.clamp(idx - self.predictions[:, :, 1], min=0.0)
        self.predictions[:, :, 2] = torch.clamp(idx + self.predictions[:, :, 2], max=self.max_length)
        if os.environ.get("AUVIRE_ORACLE_RANK"):
            # Read-only diagnostic (never a reported setting): replace the score by
            # the best IoU this proposal reaches against any ground-truth segment.
            # The resulting AP@0.95 is the value a *perfect* quality predictor would
            # obtain, i.e. the ceiling of any score-side mechanism.
            boxes = self.predictions[:, :, 1:3] / self.fps
            scores = torch.zeros(boxes.shape[:2], dtype=boxes.dtype, device=boxes.device)
            for video, periods in enumerate(self.ground_truth):
                valid = [b for b in (periods or []) if float(b[1]) > float(b[0])]
                if not valid:
                    continue
                target = torch.tensor([[float(a), float(b)] for a, b in valid], dtype=boxes.dtype)
                scores[video] = iou_1d(boxes[video], target).max(dim=1).values
            self.predictions[:, :, 0] = scores
        if os.environ.get("AUVIRE_WIDTH_RANGE"):
            # Read-only diagnostic: sink proposals whose interval length (in frames)
            # falls outside the requested range, to measure how much of the
            # high-recall error is caused by absurd box widths.
            low, high = (float(part) for part in os.environ["AUVIRE_WIDTH_RANGE"].split(","))
            widths = self.predictions[:, :, 2] - self.predictions[:, :, 1]
            keep = (widths >= low) & (widths <= high)
            self.predictions[:, :, 0] = self.predictions[:, :, 0] * keep.to(self.predictions.dtype)
        _, indexes = torch.sort(self.predictions[:, :, 0], dim=1, descending=True)
        first_indices = torch.arange(self.predictions.shape[0])[:, None]
        self.predictions = self.predictions[first_indices, indexes]
