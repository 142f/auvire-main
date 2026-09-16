import torch

from sklearn.metrics import average_precision_score
from tqdm import tqdm

import os
import datetime
import json
import re
import shutil
import unicodedata

from src.loaders import get_loaders
from src.models import Model
from src.eval import Evaluation, adjust_data
from src.logger import Logger
from src.losses import CombinedLoss
from src.seed import seed_everything
from src.perf_monitor import stage, batches

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_filename(cfg):
    return "_".join(
        map(
            str,
            [
                cfg["dataset"]["name"],
                "b",
                cfg["dataset"]["backbone"],
                "t",
                cfg["model"]["type"]["reconstruction"],
                cfg["model"]["type"]["encoder"],
                "h",
                cfg["model"]["num_heads"],
                "d",
                cfg["model"]["d_model"],
                "l",
                f'r{cfg["model"]["encoder"]["nlayers"]["retain"]}' + f'd{cfg["model"]["encoder"]["nlayers"]["downsample"]}',
                "w",
                cfg["model"]["win_size"],
                "o",
                cfg["model"]["operation"],
                "rl",
                f'r{cfg["model"]["reconstruction"]["nlayers"]["pre"]}'
                + f'd{cfg["model"]["reconstruction"]["nlayers"]["downsample"]}'
                + f'u{cfg["model"]["reconstruction"]["nlayers"]["upsample"]}'
                + f's{cfg["model"]["reconstruction"]["nlayers"]["post"]}',
                "rm",
                "_".join(cfg["model"]["reconstruction"]["modality"]),
                "f",
                cfg["model"]["encoder"]["fpn"],
                "conv",
                "".join(
                    [
                        "l" if cfg["model"]["conv"]["use_ln"] else "-",
                        "r" if cfg["model"]["conv"]["use_rl"] else "-",
                        "d" if cfg["model"]["conv"]["use_do"] else "-",
                    ]
                ),
                "c",
                "_".join(cfg["criterion"]["composition"]),
            ],
        )
    )


def check_complete(path, seeds):
    if os.path.exists(path):
        try:
            with open(path, "r") as hundle:
                json_file = json.load(hundle)
        except:
            return {
                "complete": False,
                "exists": True,
                "corrupted": True,
                "seeds_ended": {},
            }
        if "results" in json_file:
            all_seeds_started = set(seeds) == {j["seed"] for j in json_file["results"]}
            all_seeds_ended = set(seeds) == {j["seed"] for j in json_file["results"] if "test" in j}
            if all_seeds_started and all_seeds_ended:
                return {
                    "complete": True,
                    "exists": True,
                    "corrupted": False,
                    "seeds_ended": {j["seed"] for j in json_file["results"] if "test" in j},
                }
            else:
                return {
                    "complete": False,
                    "exists": True,
                    "corrupted": False,
                    "seeds_ended": {j["seed"] for j in json_file["results"] if "test" in j},
                }
        else:
            return {
                "complete": False,
                "exists": True,
                "corrupted": True,
                "seeds_ended": {},
            }
    else:
        return {
            "complete": False,
            "exists": False,
            "corrupted": False,
            "seeds_ended": {},
        }


def get_job_info():
    import subprocess

    job_id = os.environ.get("SLURM_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task_id is not None:
        job = subprocess.check_output(f"scontrol show job {job_id}_{array_task_id}", shell=True)
    else:
        job = subprocess.check_output(f"scontrol show job {job_id}", shell=True)
    info = dict([z.split("=", 1) for y in job.decode("utf-8").split("\n") for z in y.split(" ") if z])
    try:
        gpu = subprocess.check_output(f"nvidia-smi --query-gpu=gpu_name --format=csv,noheader", shell=True)
        info["gpu"] = gpu.decode("utf-8").split("\n")[0]
    except:
        info["gpu"] = "NA"
    return info


class Experiment:

    def __init__(self, cfg, folder=None, print_config=True, job_info=True):
        self.job = {}
        if job_info:
            self.job = get_job_info()
            if print_config:
                print(json.dumps(self.job, indent=2))
        if print_config:
            print(json.dumps(cfg, indent=2))
        self.cfg = cfg
        self.device = cfg["device"]
        self.logging = cfg["logging"]
        self.disable_tqdm = cfg["disable_tqdm"]
        stderr = __import__("sys").stderr
        force_progress = os.environ.get("AUVIRE_FORCE_TQDM", "").lower() in {"1", "true", "yes"}
        self.show_progress = not self.disable_tqdm and (
            force_progress or getattr(stderr, "isatty", lambda: False)()
        )
        self.delete_ckpt = cfg["delete_ckpt"]
        self.seeds = cfg["seeds"]
        self.epochs = cfg["epochs"]
        self.patience = cfg["patience"]
        self.dataset = cfg["dataset"]["name"]
        self.backbone = cfg["dataset"]["backbone"]
        self.max_length = cfg["dataset"]["params"]["max_length"]
        self.partition = cfg["dataset"]["params"]["partition"]
        self.batch_size = cfg["dataloader"]["batch_size"]
        self.workers = cfg["dataloader"]["workers"]
        self.model_type = cfg["model"]["type"]
        self.d_model = cfg["model"]["d_model"]
        self.win_size = cfg["model"]["win_size"]
        self.num_heads = cfg["model"]["num_heads"]
        self.operation = cfg["model"]["operation"]
        self.reconstruction = cfg["model"]["reconstruction"]
        self.encoder = cfg["model"]["encoder"]
        self.use_ln = cfg["model"]["conv"]["use_ln"]
        self.use_rl = cfg["model"]["conv"]["use_rl"]
        self.use_do = cfg["model"]["conv"]["use_do"]
        self.dropout = cfg["model"]["dropout"]
        self.criterion_composition = cfg["criterion"]["composition"]
        self.alpha = cfg["criterion"]["params"]["alpha"]
        self.gamma = cfg["criterion"]["params"]["gamma"]
        self.lr = cfg["optimization"]["lr"]
        self.scheduler_name = cfg["optimization"]["scheduler"]["name"]
        self.scheduler_params = dict(cfg["optimization"]["scheduler"].get("params") or {})
        self.optimizer_name = cfg["optimization"]["optimizer"]["name"]
        performance_defaults = {
            "progress_interval_batches": 50,
            "training_metric_interval_batches": 50,
            "result_flush_interval_epochs": 5,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "non_blocking_transfer": True,
        }
        self.performance = {**performance_defaults, **(cfg.get("performance") or {})}
        self.progress_interval_batches = max(int(self.performance["progress_interval_batches"]), 1)
        self.training_metric_interval_batches = max(
            int(self.performance["training_metric_interval_batches"]), 1
        )
        self.result_flush_interval_epochs = max(
            int(self.performance["result_flush_interval_epochs"]), 1
        )
        self.non_blocking_transfer = bool(self.performance["non_blocking_transfer"])
        self.factor = [1] * self.encoder["nlayers"]["retain"] + [2 ** (i + 1) for i in range(self.encoder["nlayers"]["downsample"])]
        if folder is not None:
            self.folder = folder
            self.filename = get_filename(cfg)
            self.ckpt_folder = "/".join(["ckpt"] + folder.split(os.sep)[1:])
            if not os.path.exists(self.ckpt_folder):
                os.makedirs(self.ckpt_folder)
            self.ckpt_path = "/".join([self.ckpt_folder] + [f"{self.filename}.pth"])

    def get_model(self):
        return Model(
            max_length=self.max_length,
            d_model=self.d_model,
            win_size=self.win_size,
            num_heads=self.num_heads,
            operation=self.operation,
            reconstruction=self.reconstruction,
            encoder=self.encoder,
            dropout=self.dropout,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            model_type=self.model_type,
            factor=self.factor,
            device=self.device,
        )

    def get_criterion(self):
        return CombinedLoss(
            alpha=self.alpha,
            gamma=self.gamma,
            composition=self.criterion_composition,
            factor=self.factor,
        )

    def get_scheduler(self, name, optimizer):
        params = dict(self.scheduler_params)
        if name == "none":
            if params:
                raise ValueError("scheduler params must be empty when scheduler name is 'none'")
            return None
        elif name == "reduceonplateau":
            patience = params.get("patience", 7)
            factor = params.get("factor", 0.1)
            min_lr = params.get("min_lr", 0)
            if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
                raise ValueError("scheduler.params.patience must be a non-negative integer")
            if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not 0 < factor < 1:
                raise ValueError("scheduler.params.factor must be a number between 0 and 1")
            min_lrs = min_lr if isinstance(min_lr, (list, tuple)) else [min_lr]
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 for value in min_lrs):
                raise ValueError("scheduler.params.min_lr must contain only non-negative numbers")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", **params)
        elif name == "step":
            params.setdefault("step_size", max(self.epochs // 5, 1))
            return torch.optim.lr_scheduler.StepLR(optimizer, **params)
        elif name == "cosineanealing":
            params.setdefault("T_max", self.epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        raise ValueError(f"scheduler {name} not supported")

    def get_optimizer(self, name):
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise Exception(f"optimizer {name} not supported")

    def compute_metric(self, labels, predictions):
        return sum(
            [
                average_precision_score(
                    labels[:, :: self.factor[i], 0].cpu().numpy(),
                    torch.sigmoid(prediction_[:, :, 0]).detach().cpu().numpy(),
                    average="micro",
                )
                for i, prediction_ in enumerate(predictions)
            ]
        ) / len(predictions)

    def adjust_metrics(self, loss, metric_name, metric, validation):
        l = {"loss": validation["loss"]}
        m = {f"{x}@{y}": validation[x][y] for x in ["ap", "ar"] for y in validation[x]}
        validation = {**l, **m}
        pf_keys = ["vloss", "vap@0.5"]
        t = {"loss": loss, metric_name: 100 * metric}
        v = {f"v{x}": (99 * int(x != "loss") + 1) * validation[x] for x in validation}
        metrics = {**t, **v}
        postfix = {x: v[x] for x in pf_keys}
        postfix = {**t, **postfix}
        return metrics, postfix

    @staticmethod
    def _clock(value):
        """Format a timedelta without microseconds for stable console logs."""
        return str(value).split(".", 1)[0]

    def _write(self, message):
        try:
            if self.show_progress:
                tqdm.write(message)
            else:
                print(message, flush=True)
        except UnicodeEncodeError:
            fallback = re.sub(r"\033\[[0-9;]*m", "", message).translate(
                str.maketrans({
                    "═": "=", "─": "-", "│": "|", "┃": "|", "█": "#", "░": ".",
                    "↑": "^", "↓": "v", "→": "-", "—": "-", "🏆": "BEST", "✔": "OK", "⛔": "STOP",
                })
            )
            if self.show_progress:
                tqdm.write(fallback)
            else:
                print(fallback, flush=True)

    def _color(self, text, code):
        if not self.show_progress:
            return text
        return f"\033[{code}m{text}\033[0m"

    @staticmethod
    def _display_width(text):
        text = re.sub(r"\033\[[0-9;]*m", "", text)
        return sum(
            0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} or ord(char) >= 0x1F300 else 1
            for char in text
        )

    def _summary_row(self, plain, colored, width):
        padding = max(width - self._display_width(plain) - 2, 0)
        return f"  {colored}{' ' * padding}"

    def _trend(self, value, previous, precision, lower_is_better=False):
        if previous is None:
            return "—", self._color("—", "90")
        current_rounded = round(value, precision)
        previous_rounded = round(previous, precision)
        if current_rounded == previous_rounded:
            return "→", self._color("→", "33")
        rising = current_rounded > previous_rounded
        arrow = "↑" if rising else "↓"
        improved = not rising if lower_is_better else rising
        return arrow, self._color(arrow, "32" if improved else "31")

    def _format_metric_rows(self, label, metrics, previous_metrics, width):
        entries = []
        prefix = f"{label:<4} │ "
        for key, value in metrics.items():
            if not key.startswith(f"v{label.lower()}@"):
                continue
            previous = previous_metrics.get(key) if previous_metrics else None
            arrow, colored_arrow = self._trend(value, previous, 2)
            threshold = key.split("@", 1)[1]
            entries.append((f"@{threshold:<4} {value:6.2f} {arrow}", f"@{threshold:<4} {value:6.2f} {colored_arrow}"))

        rows = []
        plain_row, colored_row = prefix, prefix
        for plain_entry, colored_entry in entries:
            separator = "  │  " if plain_row != prefix else ""
            if self._display_width(plain_row + separator + plain_entry) > width - 2 and plain_row != prefix:
                rows.append(self._summary_row(plain_row, colored_row, width))
                plain_row = " " * len(prefix) + plain_entry
                colored_row = " " * len(prefix) + colored_entry
            else:
                plain_row += separator + plain_entry
                colored_row += separator + colored_entry
        if entries:
            rows.append(self._summary_row(plain_row, colored_row, width))
        return rows

    def _format_epoch_summary(
        self,
        metrics,
        duration,
        total_duration,
        best_epoch,
        wait,
        previous_metrics=None,
        improved=False,
        stopped=False,
    ):
        lr = self.optimizer.param_groups[0]["lr"]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch = metrics["epoch"]
        if self.disable_tqdm:
            fields = [
                f"epoch={epoch}/{self.epochs}",
                f"duration={self._clock(duration)}",
                f"elapsed={self._clock(total_duration)}",
                f"loss={metrics['loss']:.4f}",
                f"ap={metrics['ap']:.2f}",
                f"vloss={metrics['vloss']:.4f}",
            ]
            fields.extend(
                f"{key.replace('@', '').replace('.', '')}={value:.2f}"
                for key, value in metrics.items()
                if key.startswith(("vap@", "var@"))
            )
            fields.extend(
                [f"best={best_epoch + 1}", f"wait={wait}/{self.patience}", f"lr={lr:.2e}"]
            )
            return [f"[{timestamp}] " + " ".join(fields)]

        width = max(82, min(shutil.get_terminal_size(fallback=(104, 24)).columns, 110))
        ratio = min(max(epoch / max(self.epochs, 1), 0.0), 1.0)
        bar_width = max(8, min(24, width - 72))
        completed = round(ratio * bar_width)
        progress_bar = "█" * completed + "░" * (bar_width - completed)
        eta = total_duration / epoch * max(self.epochs - epoch, 0) if epoch else datetime.timedelta(0)

        train_arrow, train_arrow_color = self._trend(
            metrics["loss"], previous_metrics.get("loss") if previous_metrics else None, 4, lower_is_better=True
        )
        val_arrow, val_arrow_color = self._trend(
            metrics["vloss"], previous_metrics.get("vloss") if previous_metrics else None, 4, lower_is_better=True
        )
        gap = abs(metrics["loss"] - metrics["vloss"])

        border = "═" * width
        divider = "─" * width
        header_plain = (
            f"Epoch {epoch:>{len(str(self.epochs))}}/{self.epochs} ┃ {progress_bar} {ratio:>6.1%} "
            f"│ ETA {self._clock(eta):>8} │ LR {lr:.2e}"
        )

        # --- Loss + Best：同一行，使用 ┊ 明确区分“训练状态”和“最佳 checkpoint” ---
        best_text = f"🏆 E{best_epoch + 1}"
        loss_best_plain = (
            f"Loss │ Train {metrics['loss']:.4f}{train_arrow} │ Val {metrics['vloss']:.4f}{val_arrow} "
            f"│ Gap {gap:.4f}  ┊  Best │ {best_text}"
        )
        loss_best_colored = (
            f"Loss │ Train {metrics['loss']:.4f}{train_arrow_color} │ Val {metrics['vloss']:.4f}{val_arrow_color} "
            f"│ Gap {gap:.4f}  ┊  Best │ {self._color(best_text, '32' if improved else '36')}"
        )

        # --- AP + AR：优先压缩成真正的一行；终端过窄时才安全回退成两行 ---
        def _compact_metric_entries(label):
            plain_entries = []
            colored_entries = []
            prefix = f"v{label.lower()}@"
            for key, value in metrics.items():
                if not key.startswith(prefix):
                    continue
                previous = previous_metrics.get(key) if previous_metrics else None
                arrow, colored_arrow = self._trend(value, previous, 2)
                threshold = key.split("@", 1)[1]
                # AP 阈值保持 0.50 / 0.75 / 0.95；AR 保持 100 / 50 / 20 / 10。
                if label == "AP":
                    threshold_text = f"{float(threshold):.2f}"
                else:
                    threshold_text = threshold
                plain_entries.append(f"{threshold_text} {value:.2f}{arrow}")
                colored_entries.append(f"{threshold_text} {value:.2f}{colored_arrow}")
            return plain_entries, colored_entries

        ap_plain_entries, ap_colored_entries = _compact_metric_entries("AP")
        ar_plain_entries, ar_colored_entries = _compact_metric_entries("AR")

        ap_plain = " · ".join(ap_plain_entries)
        ap_colored = " · ".join(ap_colored_entries)
        ar_plain = " · ".join(ar_plain_entries)
        ar_colored = " · ".join(ar_colored_entries)

        metric_plain = f"AP │ {ap_plain}  ┊  AR │ {ar_plain}"
        metric_colored = f"AP │ {ap_colored}  ┊  AR │ {ar_colored}"

        if self._display_width(metric_plain) <= width - 2:
            metrics_rows = [self._summary_row(metric_plain, metric_colored, width)]
        else:
            # 极窄终端下避免物理换行破坏边框；正常 tmux/SSH 宽度下不会进入这里。
            metrics_rows = [
                self._summary_row(f"AP │ {ap_plain}", f"AP │ {ap_colored}", width),
                self._summary_row(f"AR │ {ar_plain}", f"AR │ {ar_colored}", width),
            ]

        rows = []
        # 仅第一个 epoch 输出开头双线，后续 epoch 沿用上一个的结尾双线作为开头
        if epoch == 1:
            rows.append(self._color(border, "36"))
        rows.append(self._summary_row(header_plain, self._color(header_plain, "36"), width))
        rows.append(divider)
        rows.append(self._summary_row(loss_best_plain, loss_best_colored, width))
        rows.append(divider)
        rows.extend(metrics_rows)
        rows.append(divider)

        checkpoint = "Saved ✔ (best)" if improved else "checkpoint unchanged"
        footer_plain = (
            f"Time │ epoch {self._clock(duration)} │ elapsed {self._clock(total_duration)} "
            f"│ wait {wait}/{self.patience} │ {checkpoint}"
        )
        footer_colored = footer_plain.replace(
            checkpoint, self._color(checkpoint, "32" if improved else "90")
        )
        rows.append(self._summary_row(footer_plain, footer_colored, width))
        if stopped:
            stop_plain = f"⛔ EARLY STOP │ epoch {epoch} │ loading best checkpoint from epoch {best_epoch + 1}"
            rows.append(self._summary_row(stop_plain, self._color(stop_plain, "1;31"), width))
        rows.append(self._color(border, "36"))
        return rows

    def train_one_epoch(self, epoch):
        start_epoch = datetime.datetime.now()
        monitor = getattr(self, "perf_monitor", None)
        audit = getattr(self, "regression_audit", None)
        self.model.train()
        tot_loss, tot_metric = 0, 0
        metric_samples = 0
        metric_name = "ap"
        with tqdm(
            unit="batch",
            total=len(self.loaders["train"]),
            desc=f"Train E{epoch + 1}",
            dynamic_ncols=True,
            leave=False,
            position=1,
            disable=not self.show_progress,
            mininterval=1.0,
        ) as tepoch:
            for iteration, data in enumerate(batches(monitor, self.loaders["train"], f"Train E{epoch + 1}")):
                tepoch.update(1)
                with stage(monitor, "h2d", gpu=True):
                    data_adjusted = adjust_data(
                        data,
                        "tfl",
                        self.dataset,
                        self.device,
                        non_blocking=self.non_blocking_transfer,
                    )
                with stage(monitor, "forward", gpu=True):
                    p, z = self.model([data_adjusted["video_features"], data_adjusted["audio_features"]])
                self.optimizer.zero_grad()
                with stage(monitor, "loss", gpu=True):
                    loss_ = self.criterion(p, data_adjusted["labels"], z)
                detached_loss = loss_.detach()
                tot_loss = detached_loss if iteration == 0 else tot_loss + detached_loss
                with stage(monitor, "backward", gpu=True):
                    loss_.backward()
                if audit is not None:
                    audit.before_step(self, p, loss_)
                with stage(monitor, "optimizer", gpu=True):
                    self.optimizer.step()
                if audit is not None:
                    audit.after_step(self)
                should_sample_metric = (
                    (iteration + 1) % self.training_metric_interval_batches == 0
                    or iteration + 1 == len(self.loaders["train"])
                )
                if should_sample_metric:
                    tot_metric += self.compute_metric(data_adjusted["labels"], p)
                    metric_samples += 1
                if self.show_progress and (iteration + 1) % self.progress_interval_batches == 0:
                    tepoch.set_postfix(
                        {
                            "loss": f"{(tot_loss / (iteration + 1)).item():.4f}",
                            metric_name: f"{100.0 * tot_metric / max(metric_samples, 1):.2f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        },
                        refresh=False,
                    )

        e = Evaluation(
            model=self.model,
            loader=self.loaders["val"],
            criterion=self.criterion,
            device=self.device,
            factor=self.factor,
            show_progress=self.show_progress,
            phase=f"Val E{epoch + 1}",
            perf_monitor=monitor,
            regression_audit=audit,
            progress_interval_batches=self.progress_interval_batches,
            non_blocking_transfer=self.non_blocking_transfer,
        )
        e.compute_metrics()
        epoch_loss = (tot_loss / len(self.loaders["train"])).item()
        metrics, _ = self.adjust_metrics(
            loss=epoch_loss,
            metric_name=metric_name,
            metric=tot_metric / max(metric_samples, 1),
            validation=e.metrics,
        )
        metrics["training_ap_sampled_batches"] = metric_samples
        duration = datetime.datetime.now() - start_epoch
        metrics["epoch"] = epoch + 1
        metrics["duration"] = str(duration)
        if self.scheduler_dict["obj"] is not None:
            if self.scheduler_dict["name"] == "reduceonplateau":
                current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
                self.scheduler_dict["obj"].step(current_score)
            else:
                self.scheduler_dict["obj"].step()
        return metrics, duration

    def training_process(self):
        best_score = 0
        best_epoch = 0
        metrics_train_val = []
        previous_metrics = None
        process_start = datetime.datetime.now()
        epoch_progress = tqdm(
            total=self.epochs,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            position=0,
            disable=not self.show_progress,
            mininterval=1.0,
        )
        try:
            for epoch in range(self.epochs):
                metrics, duration = self.train_one_epoch(epoch)
                metrics_train_val.append(metrics)
                current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
                improved = current_score > best_score
                if improved:
                    best_score = current_score
                    best_epoch = epoch
                    torch.save({"optimizer": self.optimizer.state_dict(), "model": self.model.state_dict()}, self.ckpt_path)

                flush_results = improved or (epoch + 1) % self.result_flush_interval_epochs == 0
                self.logger.update(
                    "results",
                    self.results + [{"job": self.job, "seed": self.seed, "training": metrics_train_val}],
                    flush=flush_results,
                )

                wait = epoch - best_epoch
                stopped = wait >= self.patience
                total_duration = datetime.datetime.now() - process_start
                for line in self._format_epoch_summary(
                    metrics,
                    duration,
                    total_duration,
                    best_epoch,
                    wait,
                    previous_metrics=previous_metrics,
                    improved=improved,
                    stopped=stopped,
                ):
                    self._write(line)
                if improved and self.disable_tqdm:
                    self._write(f"Saved best checkpoint: epoch={epoch + 1} path={self.ckpt_path}")

                epoch_progress.update(1)
                epoch_progress.set_postfix(
                    best=best_epoch + 1,
                    wait=f"{wait}/{self.patience}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )
                previous_metrics = metrics
                if stopped:
                    if self.disable_tqdm:
                        self._write(
                            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] early_stop "
                            f"epoch={epoch + 1} best_epoch={best_epoch + 1} patience={self.patience}"
                        )
                    break
        finally:
            epoch_progress.close()
            self.logger.flush()

        self._write(f"Loading best checkpoint from epoch {best_epoch + 1}; starting test...")
        checkpoint = torch.load(self.ckpt_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        e = Evaluation(
            model=self.model,
            loader=self.loaders["test"],
            criterion=self.criterion,
            device=self.device,
            factor=self.factor,
            show_progress=self.show_progress,
            phase="Test",
            progress_interval_batches=self.progress_interval_batches,
            non_blocking_transfer=self.non_blocking_transfer,
        )
        e.compute_metrics()
        l = {"tloss": e.metrics["loss"]}
        m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
        metrics_test = {**l, **m}
        print(json.dumps(metrics_test, indent=2))
        if self.delete_ckpt:
            os.remove(self.ckpt_path)
        return metrics_train_val, metrics_test

    def run(self):
        print(f"[{datetime.datetime.now()}] Experiment starts")
        self.logger = Logger(folder=self.folder, filename=self.filename, enable=self.logging)
        print(self.logger.path)
        check = check_complete(self.logger.path, self.seeds)
        if not check["exists"] or check["corrupted"]:
            self.logger.create()
            self.logger.update("config", self.cfg)
            self.results = []
            self.logger.update("results", self.results)
        elif not check["complete"]:
            self.logger.update("config", self.cfg)
            self.results = [r for r in self.logger.get_values("results") if "test" in r]
            self.logger.update("results", self.results)
        else:
            self.logger.update("config", self.cfg)

        if not check["complete"]:
            for self.seed in self.seeds:
                print(f"[{datetime.datetime.now()}] Seed {self.seed}")
                if self.seed not in check["seeds_ended"]:
                    start_time = datetime.datetime.now()
                    seed_everything(self.seed)
                    self.loaders = get_loaders(
                        dataset=self.dataset,
                        backbone=self.backbone,
                        partition=self.partition,
                        max_length=self.max_length,
                        batch_size=self.batch_size,
                        workers=self.workers,
                        performance=self.performance,
                    )
                    self.model = self.get_model()
                    self.model.to(self.device)
                    self.optimizer = self.get_optimizer(self.optimizer_name)
                    self.scheduler_dict = {
                        "name": self.scheduler_name,
                        "obj": self.get_scheduler(self.scheduler_name, self.optimizer),
                    }
                    self.criterion = self.get_criterion()
                    training, test = self.training_process()
                    end_time = datetime.datetime.now()
                    duration = str(end_time - start_time)
                    self.results.append({"job": self.job, "seed": self.seed, "training": training, "test": test, "duration": duration})
                    self.logger.update("results", self.results)
                    print(f"duration: {duration}")
        self.logger.flush(pretty=True)
        print(f"[{datetime.datetime.now()}] Experiment ends")
