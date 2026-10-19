from torch.utils.data import Dataset
import torch
import os
import numpy as np
import json


def _path_sort_key(path):
    """Return a platform-independent key for persisted relative data paths."""
    return str(path).replace("\\", "/").casefold()


def _iter_files_stably(directory):
    """Yield directory entries in a deterministic order on every filesystem."""
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs.sort(key=str.casefold)
        for name in sorted(files, key=str.casefold):
            yield root, name


def _ensure_stable_order(videos):
    """Sort only manifests that are not already in canonical path order."""
    keys = [_path_sort_key(video[0]) for video in videos]
    if any(current > following for current, following in zip(keys, keys[1:])):
        videos.sort(key=lambda video: _path_sort_key(video[0]))


def _lavdf_feature_dir(split):
    """Prefer the published embedding tree, then the flat local split directory."""
    nested = os.path.join("data", "LAV-DF_emb", split)
    if os.path.isdir(nested):
        return nested
    return os.path.join("data", split)


def _lavdf_metadata_path():
    nested = os.path.join("data", "LAV-DF_emb", "metadata.min.json")
    if os.path.exists(nested):
        return nested
    return os.path.join("data", "metadata.min.json")


def _lavdf_metadata_key(root, metadata):
    """Match either ``<split>/<id>/<nest>`` embeddings or flat ``<split>/<id>`` clips."""
    parts = [part for part in root.replace("\\", "/").split("/") if part]
    candidates = []
    if len(parts) >= 3:
        candidates.append("/".join(parts[-3:-1]) + ".mp4")
    if len(parts) >= 2:
        candidates.append("/".join(parts[-2:]) + ".mp4")
    for key in candidates:
        if key in metadata:
            return key
    raise KeyError(candidates[0] if candidates else root)


def _frame_index(seconds):
    # int(9.2 * 25) is 229: the product is 229.99999999999997. 1e-6 frame is
    # below one millisecond (0.025 frame), so a timestamp inside a frame stays
    # on the same lower frame as the original truncation.
    return int(float(seconds) * 25.0 + 1e-6)


def _label_mode():
    """训练标签的边界口径，由 ``AUVIRE_LABEL_MODE`` 选择（默认 ``baseline``）。

    ``baseline``：既有实现，两条边界都向下截断到整数帧；评测把预测边界按 ``frame/25``
    还原成秒，于是标签告诉模型的边界恒比真实秒边界早 ``frac(seconds*25)/25``（最多 1 帧）。
    ``continuous``：边界用连续帧坐标（``seconds * 25``，与评测 ``frame/25`` 同一时间轴），
    偏移可以精确等于真实边界。默认关闭时与既有基线逐位一致。
    """
    value = os.environ.get("AUVIRE_LABEL_MODE", "").strip().lower()
    return "continuous" if value in {"continuous", "cont", "c"} else "baseline"


def _period2target_continuous(fake_periods, max_length):
    """连续边界版（``AUVIRE_LABEL_MODE=continuous``）。

    帧号 f 对应时刻 f/25，故段 ``[s, e]`` 在帧坐标上是 ``[s*25, e*25]``。掩码取完整落在
    段内的帧（``ceil(s*25)`` … ``floor(e*25)``），使左右偏移都非负——``ctr_diou_loss_1d``
    有 ``(target_offsets >= 0.0).all()`` 断言。
    """
    tfl_target = torch.zeros((max_length, 3))
    for fake_period in fake_periods:
        start_f = float(fake_period[0]) * 25.0
        end_f = float(fake_period[1]) * 25.0
        start = int(np.ceil(start_f - 1e-6))
        end = int(np.floor(end_f + 1e-6))
        if end < start:
            continue
        if start < 0 or end >= max_length:
            # 本地 LAV-DF 片段最长 495 帧，不会走到这里；保留与 baseline 相同的填充语义，
            # 但把写入范围夹回 [0, max_length-1]，不做负索引写入。
            start = max(start, 0)
            end = min(end, max_length - 1)
            if end < start:
                continue
        tfl_target[start : end + 1, 0] = 1
        offsets = torch.arange(end - start + 1, dtype=tfl_target.dtype)
        # clamp 只为吸收 start_f/end_f 的 float64 取自整数时的 1e-13 级误差
        # （例如 4.68*25 = 117.00000000000001），量级低于 1e-9 帧。
        tfl_target[start : end + 1, 1] = torch.clamp(offsets - (start_f - start), min=0.0)
        tfl_target[start : end + 1, 2] = torch.clamp((end_f - start) - offsets, min=0.0)
    return tfl_target


def period2target(fake_periods, max_length):
    """Convert temporal fake periods to the shared localization target format."""
    if _label_mode() == "continuous":
        return _period2target_continuous(fake_periods, max_length)
    tfl_target = torch.zeros((max_length, 3))
    for fake_period in fake_periods:
        start = _frame_index(fake_period[0])
        end = _frame_index(fake_period[1])
        # Out-of-range periods keep the original index fill, including its error.
        if start < 0 or end >= max_length:
            tfl_target[start : end + 1, 0] = 1
            for i in range(start, end + 1):
                tfl_target[i, 1] = i - start
                tfl_target[i, 2] = end - i
            continue
        if end < start:
            continue
        tfl_target[start : end + 1, 0] = 1
        offsets = torch.arange(end - start + 1, dtype=tfl_target.dtype)
        tfl_target[start : end + 1, 1] = offsets
        tfl_target[start : end + 1, 2] = offsets.flip(0)
    return tfl_target


def _prefix_view(array, length):
    """Prefix view without an extra copy.

    The returned tensor keeps the source buffer alive (npz arrays own their
    bytes, the read-only memmap is held through ``base``), so the values stay
    valid after the file object is released. Cloning here would duplicate every
    feature once per sample without changing any value.
    """
    return torch.from_numpy(np.ascontiguousarray(array[:length]))


def _load_compact_features(data_path, max_length):
    """Load the unpadded prefix shared by the video and audio streams."""
    base_path = os.path.splitext(data_path)[0]
    video_npy = f"{base_path}.video.npy"
    audio_npy = f"{base_path}.audio.npy"
    if not (os.path.exists(video_npy) and os.path.exists(audio_npy)):
        with np.load(data_path, allow_pickle=False) as data:
            video_np = data["video_features"]
            audio_np = data["audio_features"]
            length = min(int(video_np.shape[0]), int(audio_np.shape[0]), max_length)
            return _prefix_view(video_np, length), _prefix_view(audio_np, length)

    # The read-only mappings stay alive through the returned tensors' ``base``
    # and are released together with them; no explicit close and no copy.
    video_array = np.load(video_npy, mmap_mode="c", allow_pickle=False)
    audio_array = np.load(audio_npy, mmap_mode="c", allow_pickle=False)
    length = min(int(video_array.shape[0]), int(audio_array.shape[0]), max_length)
    return _prefix_view(video_array, length), _prefix_view(audio_array, length)


def _load_padded_features(data_path, max_length):
    """Load numeric features and copy them once into fixed-size tensors."""
    video, audio = _load_compact_features(data_path, max_length)
    video_features = torch.zeros((max_length, video.shape[1]), dtype=video.dtype)
    audio_features = torch.zeros((max_length, audio.shape[1]), dtype=audio.dtype)
    if video.shape[0]:
        video_features[: video.shape[0]].copy_(video)
        audio_features[: audio.shape[0]].copy_(audio)
    return video_features, audio_features


def pad_feature_batch(features, max_length, device="cpu", non_blocking=False):
    """Right-pad a batch of feature tensors with zeros up to ``max_length``."""
    if torch.is_tensor(features):
        if features.shape[1] == max_length:
            return features.to(device, non_blocking=non_blocking)
        out = features.new_zeros((features.shape[0], max_length, features.shape[2]), device=torch.device(device))
        width = min(features.shape[1], max_length)
        if width:
            out[:, :width].copy_(features[:, :width].to(device, non_blocking=non_blocking))
        return out

    width = features[0].shape[1]
    dtype = features[0].dtype
    out = torch.zeros((len(features), max_length, width), dtype=dtype, device=torch.device(device))
    for index, tensor in enumerate(features):
        count = min(tensor.shape[0], max_length)
        if count:
            out[index, :count].copy_(tensor[:count].to(device, non_blocking=non_blocking))
    return out


class LAVDF(Dataset):

    def __init__(self, backbone, split, max_length, showsize=True):
        self.backbone = backbone
        self.name = "lavdf"
        if split == "val":
            split = "dev"
        os.makedirs("utils", exist_ok=True)
        if os.path.exists(f"utils/lavdf_{split}.json"):
            with open(f"utils/lavdf_{split}.json", "r") as hundle:
                self.videos = json.load(hundle)
            _ensure_stable_order(self.videos)
        else:
            directory = _lavdf_feature_dir(split)
            with open(_lavdf_metadata_path(), "r") as f:
                metadata = {x["file"]: (int(x["modify_video"]), int(x["modify_audio"]), x["fake_periods"]) for x in json.load(f)}
            self.videos = []
            for root, name in _iter_files_stably(directory):
                if name == "features.npz":
                    fn = _lavdf_metadata_key(root, metadata)
                    video_target, audio_target, fake_periods = metadata[fn]
                    self.videos.append((os.path.join(root, name), video_target, audio_target, fake_periods))
            self.videos.sort(key=lambda video: _path_sort_key(video[0]))
            with open(f"utils/lavdf_{split}.json", "w") as hundle:
                json.dump(self.videos, hundle, indent=2)

        self.max_length = max_length
        if showsize:
            _, counts = np.unique([f"{x[1]}{x[2]}" for x in self.videos], return_counts=True)
            print(split)
            print(f"Real Video Real Audio: {counts[0]}")
            print(f"Real Video Fake Audio: {counts[1]}")
            print(f"Fake Video Real Audio: {counts[2]}")
            print(f"Fake Video Fake Audio: {counts[3]}\n")

    def __len__(self):
        return len(self.videos)

    def period2target(self, fake_periods):
        return period2target(fake_periods, self.max_length)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = f"index={idx}"
        try:
            data_path, video_target, audio_target, fake_periods = self.videos[idx]
            if self.backbone == "avhubert":
                data_path = data_path.replace("mediapipe/", "").replace("LAV-DF_emb", "LAV-DF_emb_avhubert")
            video_features, audio_features = _load_compact_features(data_path, self.max_length)
            tfl_target = self.period2target(fake_periods)
            return [video_features, audio_features, tfl_target, fake_periods]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return None


class AVDeepFake1M(Dataset):

    def __init__(self, backbone, split, max_length, partition="partial", showsize=True):
        assert partition in ["partial", "whole"]
        self.backbone = backbone
        self.name = "avdeepfake1m"
        self.num_partial_samples = 200000
        os.makedirs("utils", exist_ok=True)
        if split == "test":
            split = "val"
        if partition == "partial" or split != "train":
            self.create_video_dict(filename=f"utils/avdeepfake1m_{split}.json", split=split, partition=partition)
        else:
            self.create_video_dict(filename=f"utils/avdeepfake1m_{split}_whole.json", split=split, partition=partition)

        self.max_length = max_length
        if showsize:
            _, counts = np.unique([f"{x[1]}{x[2]}" for x in self.videos], return_counts=True)
            print(split)
            print(f"Real Video Real Audio: {counts[0]}")
            print(f"Real Video Fake Audio: {counts[1]}")
            print(f"Fake Video Real Audio: {counts[2]}")
            print(f"Fake Video Fake Audio: {counts[3]}\n")

    def __len__(self):
        return len(self.videos)

    def create_video_dict(self, filename, split, partition):
        if os.path.exists(filename):
            self.read_json_file(filename)
        else:
            self.videos = self.get_video_paths(split, partition)
            self.write_json_file(filename, self.videos)

    def read_json_file(self, filename):
        with open(filename, "r") as hundle:
            self.videos = json.load(hundle)
        _ensure_stable_order(self.videos)

    def write_json_file(self, filename, variable):
        with open(filename, "w") as hundle:
            json.dump(variable, hundle, indent=2)

    def get_video_paths(self, split, partition):
        directory = f"data/AV-Deepfake1M_emb/{split}"
        metadata = self.get_metadata(split)
        self.videos = []
        for root, name in _iter_files_stably(directory):
            if name == "features.npz":
                fn = "/".join(os.path.dirname(root).replace("\\", "/").split("/")[-4:]) + ".mp4"
                targets = metadata[fn]
                self.videos.append((os.path.join(root, name), targets[0], targets[1], targets[2], targets[3], targets[4]))
                if (split == "train") and (partition == "partial") and (len(self.videos) >= self.num_partial_samples):
                    break
        self.videos.sort(key=lambda video: _path_sort_key(video[0]))
        return self.videos

    def get_metadata(self, split):
        with open(f"data/AV-Deepfake1M_emb/{split}_metadata.json", "r") as f:
            metadata = {
                x["file"]: (
                    (0, 0, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                    if x["modify_type"] == "real"
                    else (
                        (1, 1, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                        if x["modify_type"] == "both_modified"
                        else (
                            (1, 0, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                            if x["modify_type"] == "visual_modified"
                            else (0, 1, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                        )
                    )
                )
                for x in json.load(f)
            }
        return metadata

    def period2target(self, fake_segments):
        return period2target(fake_segments, self.max_length)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = f"index={idx}"
        try:
            data_path, video_target, audio_target, visual_fake_segments, audio_fake_segments, fake_periods = self.videos[idx]
            if self.backbone == "avhubert":
                data_path = data_path.replace("mediapipe/", "").replace("AV-Deepfake1M_emb", "AV-Deepfake1M_emb_avhubert")
            video_features, audio_features = _load_compact_features(data_path, self.max_length)
            tfl_target = self.period2target(fake_periods)
            return [video_features, audio_features, tfl_target, fake_periods]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return None
