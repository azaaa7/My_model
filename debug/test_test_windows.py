#!/usr/bin/env python3
"""Tests for sequential eval-time window sampling.

Run:
    python -m pytest debug/test_test_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zzz_dataset_toolkit.dataset import VideoInpaintingDataset, _sample_eval_windows


def collect_valid_frames(windows):
    covered = set()

    for window in windows:
        frame_indices = window["frame_indices"]
        valid_mask = window["valid_mask"]

        for clip, valid_clip in zip(frame_indices, valid_mask):
            for idx, valid in zip(clip, valid_clip):
                if valid:
                    covered.add(idx)

    return covered


def make_video_pair(root: Path, length: int = 40) -> tuple[str, str]:
    video_dir = root / "frames"
    mask_dir = root / "masks"
    video_dir.mkdir()
    mask_dir.mkdir()

    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    mask = np.zeros((12, 16, 3), dtype=np.uint8)
    mask[:, :8] = 255

    for idx in range(length):
        cv2.imwrite(str(video_dir / f"{idx:04d}.png"), frame + idx % 255)
        cv2.imwrite(str(mask_dir / f"{idx:04d}.png"), mask)

    return str(video_dir), str(mask_dir)


def test_sample_eval_windows_cover_all_frames():
    windows = _sample_eval_windows(
        video_length=40,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    assert len(windows) == 3
    covered = collect_valid_frames(windows)
    assert covered
    assert covered == set(range(40))


def test_sample_eval_windows_short_video():
    windows = _sample_eval_windows(
        video_length=10,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    assert len(windows) == 1
    assert windows[0]["frame_indices"][2] == [8, 9, 9, 9]
    assert windows[0]["valid_mask"][2] == [True, True, False, False]
    assert collect_valid_frames(windows) == set(range(10))


def test_window_shape_is_fixed():
    windows = _sample_eval_windows(
        video_length=10,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    for window in windows:
        assert len(window["frame_indices"]) == 4
        assert len(window["valid_mask"]) == 4

        for clip in window["frame_indices"]:
            assert len(clip) == 4

        for valid_clip in window["valid_mask"]:
            assert len(valid_clip) == 4


def test_window_order_is_monotonic():
    windows = _sample_eval_windows(
        video_length=40,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    previous_last = -1

    for window in windows:
        for clip, valid_clip in zip(window["frame_indices"], window["valid_mask"]):
            valid_indices = [idx for idx, valid in zip(clip, valid_clip) if valid]
            if not valid_indices:
                continue

            assert valid_indices[0] > previous_last
            previous_last = valid_indices[-1]


def test_val_full_video_uses_eval_windows(tmp_path):
    sample = make_video_pair(tmp_path, length=40)
    ds = VideoInpaintingDataset(
        samples=[sample],
        mode="val",
        input_size=16,
        num_frames=4,
        num_clips=4,
        clip_stride=1,
        use_tfcu_adapter=True,
        val_full_video=True,
    )

    assert len(ds) == 3
    item = ds[2]
    assert item["images"].shape == (4, 4, 3, 16, 16)
    assert item["valid_mask"].sum().item() == 8
    assert item["frame_indices"][0].tolist() == [32, 33, 34, 35]
    assert item["frame_indices"][1].tolist() == [36, 37, 38, 39]


def test_val_fast_path_is_preserved(tmp_path):
    sample = make_video_pair(tmp_path, length=40)
    ds = VideoInpaintingDataset(
        samples=[sample],
        mode="val",
        input_size=16,
        num_frames=4,
        num_clips=4,
        clip_stride=1,
        use_tfcu_adapter=True,
        val_full_video=False,
    )

    assert len(ds) == 1
    item = ds[0]
    assert not isinstance(item, dict)
