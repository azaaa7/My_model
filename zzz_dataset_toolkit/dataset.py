from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .transforms import (
    add_gaussian_noise_snr,
    build_replay_augmenter,
    is_image_file,
    simulate_jpeg_compression_cv2,
    threshold_mask,
)


SampleList = Union[str, os.PathLike[str], Sequence[str], Sequence[Sequence[str]], np.ndarray]


def _is_path_like(value: object) -> bool:
    return isinstance(value, (str, os.PathLike))


def _rows_to_samples(data: np.ndarray | Sequence[Sequence[str]]) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for item in data:
        if len(item) < 2:
            raise ValueError(f"Each sample row must contain video_dir and mask_dir, got: {item}")
        video_dir, mask_dir = item[:2]
        result.append((str(video_dir), str(mask_dir)))
    return result


def _load_npy_samples(path: str | os.PathLike[str]) -> List[Tuple[str, str]]:
    data = np.load(path, allow_pickle=True)
    return _rows_to_samples(data)


def _load_samples(samples: SampleList) -> List[Tuple[str, str]]:
    if _is_path_like(samples):
        return _load_npy_samples(samples)

    if isinstance(samples, np.ndarray):
        if samples.ndim == 1 and all(_is_path_like(item) for item in samples.tolist()):
            result: List[Tuple[str, str]] = []
            for path in samples.tolist():
                result.extend(_load_npy_samples(path))
            return result
        return _rows_to_samples(samples)

    if all(_is_path_like(item) for item in samples):
        result: List[Tuple[str, str]] = []
        for path in samples:
            result.extend(_load_npy_samples(path))
        return result

    return _rows_to_samples(samples)


def _sample_indices(
    video_length: int,
    mode: str,
    num_frames: int = 5,
) -> List[int]:
    half = num_frames // 2

    if mode == "train":
        center = random.randint(0, video_length - 1)
        return [min(max(center + offset, 0), video_length - 1) for offset in range(-half, half + 1)]

    if mode == "val":
        center = video_length // 2
        return [min(max(center + offset, 0), video_length - 1) for offset in range(-half, half + 1)]

    if mode == "test":
        return list(range(video_length))

    raise ValueError(f"Unknown mode: {mode}")


def _validate_num_frames(num_frames: int) -> None:
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError(f"num_frames must be an int, got {type(num_frames).__name__}")
    if num_frames <= 0 or num_frames % 2 == 0:
        raise ValueError(f"num_frames must be a positive odd integer, got {num_frames}")


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _align_mask_to_frame(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    if (mask_h, mask_w) == (frame_h, frame_w):
        return mask
    return cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)


def _derive_sample_name(video_dir: str) -> str:
    p = Path(video_dir)
    if len(p.parents) >= 2:
        prefix = p.parents[1].name
    else:
        prefix = p.parent.name
    return f"{prefix}_{p.name}.jpg"


class VideoInpaintingDataset(Dataset):
    """
    可配置奇数帧输入的视频 inpainting 数据集。

    训练 / 验证:
        return frames, center_mask, H, W, name
    测试 / 推理:
        return frames, all_masks, H, W, name
    """

    def __init__(
        self,
        samples: SampleList,
        mode: str = "train",
        input_size: int = 512,
        gt_ratio: int = 1,
        num_frames: int = 5,
        augment_prob: float = 0.75,
        robust_noise_snr: int = 0,
        robust_jpeg_quality: int = 0,
    ):
        _validate_num_frames(num_frames)

        self.samples = _load_samples(samples)
        self.mode = mode
        self.input_size = input_size
        self.gt_ratio = gt_ratio
        self.num_frames = num_frames
        self.augment_prob = augment_prob
        self.robust_noise_snr = robust_noise_snr
        self.robust_jpeg_quality = robust_jpeg_quality

        self.to_tensor = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])
        self.replay_aug = build_replay_augmenter()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_dir, mask_dir = self.samples[idx]
        frame_list = sorted([p for p in os.listdir(video_dir) if is_image_file(p)])
        mask_list = sorted([p for p in os.listdir(mask_dir) if is_image_file(p)])

        if len(frame_list) != len(mask_list):
            raise ValueError(
                f"Frame count and mask count mismatch in {video_dir}: "
                f"{len(frame_list)} vs {len(mask_list)}"
            )

        video_length = len(frame_list)
        indices = _sample_indices(video_length, self.mode, self.num_frames)
        name = _derive_sample_name(video_dir)

        frames: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        original_h, original_w = None, None

        for frame_idx in indices:
            frame_path = str(Path(video_dir) / frame_list[frame_idx])
            mask_path = str(Path(mask_dir) / mask_list[frame_idx])

            frame = _read_image(frame_path)
            mask = _read_image(mask_path)
            mask = _align_mask_to_frame(mask, frame)
            mask = threshold_mask(mask)

            if original_h is None:
                original_h, original_w = frame.shape[:2]

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks.append(mask)

        if self.mode == "train" and random.random() < self.augment_prob:
            aug = self.replay_aug(image=frames[0], mask=masks[0])
            replay = aug["replay"]
            frames[0], masks[0] = aug["image"], aug["mask"]
            for i, (img, mask) in enumerate(zip(frames[1:], masks[1:])):
                aug = A.ReplayCompose.replay(replay, image=img, mask=mask)
                frames[i + 1], masks[i + 1] = aug["image"], aug["mask"]

        if self.mode == "test" and self.robust_noise_snr > 0:
            frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]

        if self.mode == "test" and 1 <= self.robust_jpeg_quality <= 100:
            frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

        frame_tensors = []
        mask_tensors = []
        for img, mask in zip(frames, masks):
            img = cv2.resize(img, (self.input_size, self.input_size))
            mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio))
            mask = threshold_mask(mask)

            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0

            frame_tensors.append(self.to_tensor(img).unsqueeze(0))
            mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

        frames_out = torch.cat(frame_tensors, dim=0)
        masks_out = torch.cat(mask_tensors, dim=0)

        if self.mode in ("train", "val"):
            return frames_out, masks_out[self.num_frames // 2], original_h, original_w, name
        return frames_out, masks_out, original_h, original_w, name


def build_dataloader(
    samples: SampleList,
    mode: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    drop_last: bool | None = None,
    **dataset_kwargs,
):
    dataset = VideoInpaintingDataset(samples=samples, mode=mode, **dataset_kwargs)
    if shuffle is None:
        shuffle = mode == "train"
    if drop_last is None:
        drop_last = mode == "train"
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
