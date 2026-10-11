from src.training import Experiment
from src.eval import Evaluation
from src.loaders import get_loaders
from src.logger import Logger

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.backends.mha.set_fastpath_enabled(False)


def get_json_file(dataset):
    if dataset == "lavdf":
        return "ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    elif dataset == "avdeepfake1m":
        return "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def skip_check(path, train_ds, test_ds):
    if os.path.exists(path):
        with open(path, "r") as hundle:
            json_file = json.load(hundle)
        if train_ds in json_file:
            evaluated_datasets = set(json_file[train_ds].keys())
            if test_ds in evaluated_datasets:
                return True
    return False


def main():
    device = "cuda:0"
    workers = 4
    batch_size = 64
    tasks = [
        "dfd",
        "tfl",
    ]
    train_datasets = [
        "lavdf",
        "avdeepfake1m",
    ]
    test_datasets = [
        "lavdf",
        "avdeepfake1m",  # This is the validation set evaluation. For test set get predictions with `scripts/predict.py` and evaluate through Codabench (https://deepfakes1m.github.io/2024/evaluation). Cf. README.md for more details.
    ]
    loader_cache = {}
    for task in tasks:
        for train_dataset in train_datasets:
            print(f"\nTask:{task} TrainDataset:{train_dataset}")

            logger = Logger(folder=f"results/test", filename=f"task_{task}_training_on_{train_dataset}", enable=True)
            if not os.path.exists(logger.path):
                logger.create()

            json_file = get_json_file(train_dataset)
            with open(json_file, "r") as hundle:
                data = json.load(hundle)

            filename = Path(json_file).with_suffix("")
            configuration = data["config"]

            test_loaders = []
            for test_dataset in test_datasets:
                if skip_check(path=logger.path, train_ds=train_dataset, test_ds=test_dataset):
                    print(f"Skipping TestDataset: {test_dataset}...")
                    continue
                cache_key = (
                    test_dataset,
                    configuration["dataset"]["backbone"],
                    configuration["dataset"]["params"]["max_length"],
                    batch_size,
                    workers,
                    json.dumps(configuration.get("performance"), sort_keys=True),
                )
                test_loader = loader_cache.get(cache_key)
                if test_loader is None:
                    try:
                        test_loader = get_loaders(
                            dataset=test_dataset,
                            backbone=configuration["dataset"]["backbone"],
                            partition="whole",
                            max_length=configuration["dataset"]["params"]["max_length"],
                            batch_size=batch_size,
                            workers=workers,
                            splits=["test"],
                            showsize=False,
                            performance=configuration.get("performance"),
                        )["test"]
                    except (FileNotFoundError, OSError, KeyError) as exc:
                        print(f"Skipping TestDataset: {test_dataset} ({exc})")
                        continue
                    loader_cache[cache_key] = test_loader
                test_loaders.append((test_dataset, test_loader))

            if test_loaders:
                experiment = Experiment(cfg=configuration, print_config=False, job_info=False)
                model = experiment.get_model()
                model.to(device)

                for test_dataset, test_loader in test_loaders:
                    print(f"Evaluating on TestDataset: {test_dataset}...")
                    ckpt_path = filename.with_suffix(".pth")
                    checkpoint = torch.load(ckpt_path, weights_only=False)
                    model.load_state_dict(checkpoint["model"])
                    e = Evaluation(
                        model=model,
                        loader=test_loader,
                        criterion=experiment.get_criterion(),
                        device=device,
                        factor=experiment.factor,
                        generalization=True,
                        task=task,
                        progress_interval_batches=experiment.progress_interval_batches,
                        non_blocking_transfer=experiment.non_blocking_transfer,
                    )
                    e.compute_metrics()
                    if task == "dfd":
                        performance = {f"t{x}": (99 * int(x != "loss") + 1) * e.metrics[x] for x in e.metrics}
                    else:
                        l = {"tloss": e.metrics["loss"]}
                        m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
                        performance = {**l, **m}
                    logger.update([train_dataset, test_dataset], performance)
                    print(json.dumps({"task": task, "train": train_dataset, "test": test_dataset, "metrics": performance}))


if __name__ == "__main__":
    main()
