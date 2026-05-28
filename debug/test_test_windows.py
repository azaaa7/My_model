#!/usr/bin/env python3
"""Tests for sequential test-time window sampling.

Run:
    python -m pytest debug/test_test_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zzz_dataset_toolkit.dataset import _sample_test_windows


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


def test_sample_test_windows_cover_all_frames():
    windows = _sample_test_windows(
        video_length=40,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    assert len(windows) == 3
    covered = collect_valid_frames(windows)
    assert covered
    assert covered == set(range(40))


def test_sample_test_windows_short_video():
    windows = _sample_test_windows(
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
    windows = _sample_test_windows(
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
    windows = _sample_test_windows(
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
