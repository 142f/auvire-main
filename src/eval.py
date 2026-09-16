from src.metrics import AP, AR
from src.post_process import soft_nms_torch_parallel

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
        video_features = video_features.to(device, non_blocking=non_blocking)
        audio_features = audio_features.to(device, non_blocking=non_blocking)
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

    def get_predictions(self):
        self.model.eval()
        loss_total = None
        loss_weight = 0
        with torch.no_grad():
            progress = tqdm(
                self.loader,
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
                data_adjusted = adjust_data(
                    data,
                    self.task,
                    self.dataset,
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )
                p, z = self.model([data_adjusted["video_features"], data_adjusted["audio_features"]])
                if self.task == "tfl":
                    batch_size = data_adjusted["labels"].shape[0]
                    batch_loss = self.criterion(p, data_adjusted["labels"], z).detach()
                    weighted_loss = batch_loss * batch_size
                    loss_total = weighted_loss if loss_total is None else loss_total + weighted_loss
                    loss_weight += batch_size
                    self.update_predictions(p, data_adjusted["fake_periods"])
                    if self.show_progress and (iteration + 1) % self.progress_interval_batches == 0:
                        progress.set_postfix(
                            loss=f"{(loss_total / loss_weight).item():.4f}", refresh=False
                        )
                elif self.task == "dfd":
                    overall_target = [
                        max(x, y) for x, y in zip(data_adjusted["video_target"], data_adjusted["audio_target"])
                    ]
                    self.update_predictions(p, overall_target)

            progress.close()

        self._finalize_predictions()
        self.loss = (loss_total / loss_weight).item() if loss_weight else 0.0

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

    def compute_metrics(self):
        self.get_predictions()
        if self.show_progress:
            tqdm.write(f"{self.phase}: post-processing and computing metrics...")
        self.transform_predictions()
        proposals = soft_nms_torch_parallel(
            self.predictions, self.sigma, self.t1, self.t2, self.fps, self.metrics_device
        )
        if self.task == "tfl":
            self.compute_tfl_metrics(proposals)
        elif self.task == "dfd":
            self.compute_dfd_metrics(proposals)
        else:
            raise ValueError(f"Unknown task: {self.task}")

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
        if isinstance(self.predictions, list):
            idx = torch.cat([idx[:: self.factor[i]] for i, _ in enumerate(self.predictions)])
            self.predictions = torch.cat(self.predictions, dim=1)
        self.predictions[:, :, 0] = torch.sigmoid(self.predictions[:, :, 0])
        self.predictions[:, :, 1] = torch.clamp(idx - self.predictions[:, :, 1], min=0.0)
        self.predictions[:, :, 2] = torch.clamp(idx + self.predictions[:, :, 2], max=self.max_length)
        _, indexes = torch.sort(self.predictions[:, :, 0], dim=1, descending=True)
        first_indices = torch.arange(self.predictions.shape[0])[:, None]
        self.predictions = self.predictions[first_indices, indexes]
