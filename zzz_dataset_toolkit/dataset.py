from __future__ import annotations

import os
import random
import re
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


def _numeric_key(path_or_name: str | os.PathLike[str]) -> int | str:
    """Extract trailing integer from filename for correct numeric sort.

    Ensures "10.png" follows "2.png" rather than string ordering.
    """
    name = Path(str(path_or_name)).stem
    nums = re.findall(r"\d+", name)
    if nums:
        return int(nums[-1])
    return name


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
    val_num_frames: int = 0,
) -> List[int]:
    half = num_frames // 2

    if mode == "train":
        center = random.randint(0, video_length - 1)
        return [min(max(center + offset, 0), video_length - 1) for offset in range(-half, half + 1)]

    if mode == "val":
        if val_num_frames > 0:
            # 等间隔采样 val_num_frames 帧，保证每个视频帧数一致
            if video_length <= val_num_frames:
                return list(range(video_length))
            step = (video_length - 1) / (val_num_frames - 1) if val_num_frames > 1 else 0.0
            return [min(round(i * step), video_length - 1) for i in range(val_num_frames)]
        # fallback: 中心连续帧 (与 train 相同逻辑)
        center = video_length // 2
        return [min(max(center + offset, 0), video_length - 1) for offset in range(-half, half + 1)]

    if mode == "test":
        return list(range(video_length))

    raise ValueError(f"Unknown mode: {mode}")


def _sample_multi_clip_indices(
    video_length: int,
    mode: str,
    num_clips: int = 4,
    num_frames: int = 4,
    stride: int = 1,
) -> list[list[int]]:
    """Sample ``num_clips`` clips from a video, each of ``num_frames`` frames.

    Train: randomly sample non-overlapping starting positions, sorted chronologically.
           If the video is too short, pad by repeating the last frame.
    Val/Test: sequential sliding windows.

    Returns:
        List of clip index lists, e.g. [[0,1,2,3], [5,6,7,8], ...].
    """
    clip_len = num_frames * stride
    max_start = video_length - clip_len

    if mode == "train":
        if max_start < 0:
            # Video too short — pad with last frame repeats
            clips: list[list[int]] = []
            for _ in range(num_clips):
                clip: list[int] = []
                for i in range(num_frames):
                    clip.append(min(i * stride, video_length - 1))
                clips.append(clip)
            return clips

        starts = random.sample(
            range(0, max_start + 1),
            k=min(num_clips, max_start + 1),
        )
        starts = sorted(starts)

        # If fewer than num_clips possible starts, pad by repeating
        while len(starts) < num_clips:
            starts.append(random.randint(0, max_start))
        starts = sorted(starts)

        clips = []
        for s in starts:
            clip = [min(s + i * stride, video_length - 1) for i in range(num_frames)]
            clips.append(clip)
        return clips

    # Val / Test: sequential windows
    step = num_frames  # non-overlapping by default
    clips = []
    for start in range(0, video_length, step):
        clip = [min(start + i * stride, video_length - 1) for i in range(num_frames)]
        clips.append(clip)

    # Ensure exactly num_clips clips (pad with last clip if needed)
    if len(clips) > num_clips:
        # Evenly subsample
        indices = [round(i * (len(clips) - 1) / (num_clips - 1)) for i in range(num_clips)] if num_clips > 1 else [0]
        clips = [clips[i] for i in indices]
    elif len(clips) < num_clips:
        while len(clips) < num_clips:
            clips.append(clips[-1])  # repeat last clip
    return clips


def _sample_eval_windows(
    video_length: int,
    num_clips: int,
    num_frames: int,
    stride: int = 1,
) -> list[dict[str, list[list[int]] | list[list[bool]]]]:
    """Build sequential fixed-size windows for eval-time inference."""
    assert video_length > 0
    assert num_clips > 0
    assert num_frames > 0
    assert stride > 0

    all_clips: list[list[int]] = []
    all_valid: list[list[bool]] = []
    step = num_frames * stride

    for start in range(0, video_length, step):
        clip: list[int] = []
        valid: list[bool] = []

        for i in range(num_frames):
            idx = start + i * stride
            if idx < video_length:
                clip.append(idx)
                valid.append(True)
            else:
                clip.append(video_length - 1)
                valid.append(False)

        all_clips.append(clip)
        all_valid.append(valid)

    windows: list[dict[str, list[list[int]] | list[list[bool]]]] = []
    for i in range(0, len(all_clips), num_clips):
        window = all_clips[i:i + num_clips]
        valid_window = all_valid[i:i + num_clips]

        while len(window) < num_clips:
            window.append(window[-1])
            valid_window.append([False] * num_frames)

        windows.append({
            "frame_indices": window,
            "valid_mask": valid_window,
        })

    return windows


_sample_test_windows = _sample_eval_windows


def _validate_num_frames(num_frames: int, *, allow_even: bool = False) -> None:
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError(f"num_frames must be an int, got {type(num_frames).__name__}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames}")
    if not allow_even and num_frames % 2 == 0:
        raise ValueError(
            f"num_frames must be a positive odd integer in baseline mode, got {num_frames}. "
            f"Set use_tfcu_adapter=true or num_clips>1 to allow even frames."
        )


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
    可配置的视频 inpainting 数据集。

    模式 1 — Baseline (num_clips=1):
        训练: return frames [T,3,H,W], center_mask [1,H,W], H, W, name
        验证: return frames [T,3,H,W], all_masks [T,1,H,W], H, W, name

    模式 2 — TFCU (num_clips>1):
        训练/验证/测试: return frames [N,T,3,H,W], masks [N,T,1,H,W], H, W, name
    """

    def __init__(
        self,
        samples: SampleList,
        mode: str = "train",
        input_size: int = 512,
        gt_ratio: int = 1,
        num_frames: int = 5,
        val_num_frames: int = 0,
        dataset_repeat: int = 1,
        augment_prob: float = 0.75,
        robust_noise_snr: int = 0,
        robust_jpeg_quality: int = 0,
        num_clips: int = 1,
        clip_stride: int = 1,
        use_tfcu_adapter: bool = False,
        test_max_clips: int = 4,
        val_full_video: bool = False,
        test_full_video: bool = True,
    ):
        self.use_tfcu_adapter = bool(use_tfcu_adapter)
        allow_even = self.use_tfcu_adapter or num_clips > 1
        _validate_num_frames(num_frames, allow_even=allow_even)

        self.samples = _load_samples(samples)
        self.mode = mode
        self.input_size = input_size
        self.gt_ratio = gt_ratio
        self.num_frames = num_frames
        self.val_num_frames = val_num_frames
        self.dataset_repeat = dataset_repeat
        self.augment_prob = augment_prob
        self.robust_noise_snr = robust_noise_snr
        self.robust_jpeg_quality = robust_jpeg_quality
        self.num_clips = num_clips
        self.clip_stride = clip_stride
        self.test_max_clips = test_max_clips
        self.val_full_video = bool(val_full_video)
        self.test_full_video = bool(test_full_video)
        self.use_eval_windows = (
            self.num_clips > 1
            and (
                (self.mode == "val" and self.val_full_video)
                or (self.mode == "test" and self.test_full_video)
            )
        )

        self.to_tensor = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])
        self.replay_aug = build_replay_augmenter()
        self.eval_items: list[dict[str, object]] | None = None

        if self.use_eval_windows:
            self.eval_items = []
            for sample_idx, (video_dir, _mask_dir) in enumerate(self.samples):
                frame_list = sorted(
                    [p for p in os.listdir(video_dir) if is_image_file(p)],
                    key=_numeric_key,
                )
                video_length = len(frame_list)
                if video_length <= 0:
                    continue

                windows = _sample_eval_windows(
                    video_length=video_length,
                    num_clips=self.num_clips,
                    num_frames=self.num_frames,
                    stride=self.clip_stride,
                )
                name = _derive_sample_name(video_dir)

                for window_id, item in enumerate(windows):
                    self.eval_items.append({
                        "sample_idx": sample_idx,
                        "video_id": name,
                        "window_id": window_id,
                        "frame_indices": item["frame_indices"],
                        "valid_mask": item["valid_mask"],
                        "is_last_window": window_id == len(windows) - 1,
                    })

    def __len__(self) -> int:
        if self.use_eval_windows and self.eval_items is not None:
            return len(self.eval_items)
        return len(self.samples) * self.dataset_repeat

    def __getitem__(self, idx: int):
        if self.use_eval_windows and self.eval_items is not None:
            item = self.eval_items[idx]
            sample_idx = int(item["sample_idx"])
            video_dir, mask_dir = self.samples[sample_idx]
            frame_list = sorted(
                [p for p in os.listdir(video_dir) if is_image_file(p)], key=_numeric_key,
            )
            mask_list = sorted(
                [p for p in os.listdir(mask_dir) if is_image_file(p)], key=_numeric_key,
            )

            if len(frame_list) != len(mask_list):
                raise ValueError(
                    f"Frame count and mask count mismatch in {video_dir}: "
                    f"{len(frame_list)} vs {len(mask_list)}"
                )

            return self._get_multi_clip_by_indices(
                video_dir=video_dir,
                mask_dir=mask_dir,
                frame_list=frame_list,
                mask_list=mask_list,
                clip_indices=item["frame_indices"],
                video_id=str(item["video_id"]),
                window_id=int(item["window_id"]),
                valid_mask=item["valid_mask"],
                is_last_window=bool(item["is_last_window"]),
            )

        idx = idx % len(self.samples)  # 支持 dataset_repeat：取模映射到原始样本
        video_dir, mask_dir = self.samples[idx]
        frame_list = sorted(
            [p for p in os.listdir(video_dir) if is_image_file(p)], key=_numeric_key,
        )
        mask_list = sorted(
            [p for p in os.listdir(mask_dir) if is_image_file(p)], key=_numeric_key,
        )

        if len(frame_list) != len(mask_list):
            raise ValueError(
                f"Frame count and mask count mismatch in {video_dir}: "
                f"{len(frame_list)} vs {len(mask_list)}"
            )

        video_length = len(frame_list)
        name = _derive_sample_name(video_dir)

        # ── Multi-clip mode ───────────────────────────────────────────
        if self.num_clips > 1:
            return self._get_multi_clip(
                video_dir, mask_dir, frame_list, mask_list, video_length, name,
            )

        # ── Single-clip mode (original behaviour) ─────────────────────
        indices = _sample_indices(video_length, self.mode, self.num_frames, self.val_num_frames)

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

            if self.mode == "train":
                # Train: resize mask to model output resolution for efficient loss computation.
                mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio))
                mask = threshold_mask(mask)
            # val / test: keep mask at original (aligned) resolution.
            # align_logits_and_masks will upsample logits to match.

            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0

            frame_tensors.append(self.to_tensor(img).unsqueeze(0))
            mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

        frames_out = torch.cat(frame_tensors, dim=0)
        masks_out = torch.cat(mask_tensors, dim=0)

        if self.mode == "train":
            return frames_out, masks_out[self.num_frames // 2], original_h, original_w, name
        return frames_out, masks_out, original_h, original_w, name

    # ------------------------------------------------------------------
    # Multi-clip sampling
    # ------------------------------------------------------------------

    def _get_multi_clip_by_indices(
        self,
        video_dir: str,
        mask_dir: str,
        frame_list: list[str],
        mask_list: list[str],
        clip_indices,
        video_id: str,
        window_id: int,
        valid_mask,
        is_last_window: bool,
    ):
        """Load a fixed eval window without re-sampling clip indices."""
        all_frames: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        original_h, original_w = None, None

        for clip in clip_indices:
            frames: list[np.ndarray] = []
            masks: list[np.ndarray] = []

            for frame_idx in clip:
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

            if self.mode == "test" and self.robust_noise_snr > 0:
                frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]
            if self.mode == "test" and 1 <= self.robust_jpeg_quality <= 100:
                frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

            frame_tensors = []
            mask_tensors = []
            for img, mask in zip(frames, masks):
                img = cv2.resize(img, (self.input_size, self.input_size))
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0

                frame_tensors.append(self.to_tensor(img).unsqueeze(0))
                mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

            all_frames.append(torch.cat(frame_tensors, dim=0))
            all_masks.append(torch.cat(mask_tensors, dim=0))

        frames_out = torch.stack(all_frames, dim=0)
        masks_out = torch.stack(all_masks, dim=0)

        return {
            "images": frames_out,
            "masks": masks_out,
            "video_id": video_id,
            "window_id": window_id,
            "frame_indices": torch.tensor(clip_indices, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "is_last_window": is_last_window,
            "original_h": original_h,
            "original_w": original_w,
            "name": video_id,
        }

    def _get_multi_clip(
        self,
        video_dir: str,
        mask_dir: str,
        frame_list: list[str],
        mask_list: list[str],
        video_length: int,
        name: str,
    ):
        """Sample ``num_clips`` clips from the same video, chronologically ordered.

        Test mode: num_clips auto-expands to cover all frames, but capped at
        ``test_max_clips`` to avoid OOM (default 4 = 16 frames at T=4).
        """
        actual_num_clips = self.num_clips
        if self.mode in ("test", "val"):
            max_clips = max(1, (video_length + self.num_frames - 1) // self.num_frames)
            cap = getattr(self, "test_max_clips", self.num_clips)
            actual_num_clips = min(max(self.num_clips, max_clips), cap)

        all_clip_indices = _sample_multi_clip_indices(
            video_length,
            self.mode,
            num_clips=actual_num_clips,
            num_frames=self.num_frames,
            stride=self.clip_stride,
        )

        all_frames: list[torch.Tensor] = []   # each: [T, 3, H, W]
        all_masks: list[torch.Tensor] = []    # each: [T, 1, H, W]
        original_h, original_w = None, None

        for clip_indices in all_clip_indices:
            frames: list[np.ndarray] = []
            masks: list[np.ndarray] = []

            for frame_idx in clip_indices:
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

            # Augmentation (apply same replay to all frames in the clip)
            if self.mode == "train" and random.random() < self.augment_prob:
                aug = self.replay_aug(image=frames[0], mask=masks[0])
                replay = aug["replay"]
                frames[0], masks[0] = aug["image"], aug["mask"]
                for i, (img, mask) in enumerate(zip(frames[1:], masks[1:])):
                    aug = A.ReplayCompose.replay(replay, image=img, mask=mask)
                    frames[i + 1], masks[i + 1] = aug["image"], aug["mask"]

            # Robustness perturbations (test only)
            if self.mode == "test" and self.robust_noise_snr > 0:
                frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]
            if self.mode == "test" and 1 <= self.robust_jpeg_quality <= 100:
                frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

            # Resize & convert to tensor
            frame_tensors = []
            mask_tensors = []
            for img, mask in zip(frames, masks):
                img = cv2.resize(img, (self.input_size, self.input_size))
                if self.mode == "train":
                    mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio))
                    mask = threshold_mask(mask)
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0
                frame_tensors.append(self.to_tensor(img).unsqueeze(0))
                mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

            all_frames.append(torch.cat(frame_tensors, dim=0))     # [T, 3, H, W]
            all_masks.append(torch.cat(mask_tensors, dim=0))       # [T, 1, H, W]

        frames_out = torch.stack(all_frames, dim=0)   # [N, T, 3, H, W]
        masks_out = torch.stack(all_masks, dim=0)     # [N, T, 1, H, W]

        # TFCU mode: always return full [N,T,*,*,*] masks for temporal loss
        # Baseline mode (num_clips=1): return center-frame mask for training
        if self.num_clips > 1:
            return frames_out, masks_out, original_h, original_w, name
        if self.mode == "train":
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
    num_clips: int = 1,
    clip_stride: int = 1,
    use_tfcu_adapter: bool = False,
    test_max_clips: int = 4,
    val_full_video: bool = False,
    test_full_video: bool = True,
    **dataset_kwargs,
):
    dataset = VideoInpaintingDataset(
        samples=samples,
        mode=mode,
        num_clips=num_clips,
        clip_stride=clip_stride,
        use_tfcu_adapter=use_tfcu_adapter,
        test_max_clips=test_max_clips,
        val_full_video=val_full_video,
        test_full_video=test_full_video,
        **dataset_kwargs,
    )
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
