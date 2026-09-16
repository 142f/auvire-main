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


def period2target(fake_periods, max_length):
    """Convert temporal fake periods to the shared localization target format."""
    tfl_target = torch.zeros((max_length, 3))
    for fake_period in fake_periods:
        fake_period_indx = [int(x * 25) for x in fake_period]
        tfl_target[fake_period_indx[0] : fake_period_indx[1] + 1, 0] = 1
        for i in range(fake_period_indx[0], fake_period_indx[1] + 1):
            tfl_target[i, 1] = i - fake_period_indx[0]
            tfl_target[i, 2] = fake_period_indx[1] - i
    return tfl_target


def _load_padded_features(data_path, max_length):
    """Load numeric NPZ features and copy them once into fixed-size tensors."""
    base_path = os.path.splitext(data_path)[0]
    video_npy = f"{base_path}.video.npy"
    audio_npy = f"{base_path}.audio.npy"
    if not (os.path.exists(video_npy) and os.path.exists(audio_npy)):
        with np.load(data_path, allow_pickle=False) as data:
            video_source = torch.tensor(data["video_features"])
            audio_source = torch.tensor(data["audio_features"])
        length = min(video_source.shape[0], audio_source.shape[0], max_length)
        video_features = torch.cat(
            [video_source[:length], torch.zeros((max_length - length, video_source.shape[1]))]
        )
        audio_features = torch.cat(
            [audio_source[:length], torch.zeros((max_length - length, audio_source.shape[1]))]
        )
        return video_features, audio_features

    video_array = np.load(video_npy, mmap_mode="c", allow_pickle=False)
    audio_array = np.load(audio_npy, mmap_mode="c", allow_pickle=False)
    try:
        video_source = torch.from_numpy(video_array)
        audio_source = torch.from_numpy(audio_array)

        length = min(video_source.shape[0], audio_source.shape[0], max_length)
        video_features = torch.zeros(
            (max_length, video_source.shape[1]), dtype=video_source.dtype
        )
        audio_features = torch.zeros(
            (max_length, audio_source.shape[1]), dtype=audio_source.dtype
        )
        video_features[:length].copy_(video_source[:length])
        audio_features[:length].copy_(audio_source[:length])
    finally:
        mmap_video = getattr(video_array, "_mmap", None)
        mmap_audio = getattr(audio_array, "_mmap", None)
        if mmap_video is not None:
            mmap_video.close()
        if mmap_audio is not None:
            mmap_audio.close()
    return video_features, audio_features


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
            directory = f"data/LAV-DF_emb/{split}"
            with open("data/LAV-DF_emb/metadata.min.json", "r") as f:
                metadata = {x["file"]: (int(x["modify_video"]), int(x["modify_audio"]), x["fake_periods"]) for x in json.load(f)}
            self.videos = []
            for root, name in _iter_files_stably(directory):
                if name == "features.npz":
                    fn = "/".join(os.path.dirname(root).replace("\\", "/").split("/")[-2:]) + ".mp4"
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
            video_features, audio_features = _load_padded_features(data_path, self.max_length)
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
            video_features, audio_features = _load_padded_features(data_path, self.max_length)
            tfl_target = self.period2target(fake_periods)
            return [video_features, audio_features, tfl_target, fake_periods]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return None
