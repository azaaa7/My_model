from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from zzz_dataset_toolkit import VideoInpaintingDataset, build_dataloader


def _make_image(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 4, (255, 255, 255), -1)
    return img


def build_dummy_dataset(root: Path, num_videos: int = 2, num_frames: int = 6) -> list[tuple[str, str]]:
    fake_root = root / "fake"
    mask_root = root / "mask"
    samples: list[tuple[str, str]] = []

    for vid in range(num_videos):
        video_name = f"video_{vid:02d}"
        fake_dir = fake_root / video_name
        mask_dir = mask_root / video_name
        fake_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(num_frames):
            frame = _make_image(128, 128, (20 * vid, 20 * idx, 80))
            mask = np.zeros((128, 128, 3), dtype=np.uint8)
            if idx % 2 == 0:
                cv2.rectangle(mask, (32, 32), (96, 96), (255, 255, 255), -1)

            cv2.imwrite(str(fake_dir / f"{idx:05d}.jpg"), frame)
            cv2.imwrite(str(mask_dir / f"{idx:05d}.png"), mask)

        samples.append((str(fake_dir), str(mask_dir)))

    return samples


def main() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="zzz_dataset_demo_"))
    try:
        samples = build_dummy_dataset(temp_root)

        train_ds = VideoInpaintingDataset(
            samples=samples,
            mode="train",
            input_size=256,
            gt_ratio=1,
            num_frames=7,
        )
        train_loader = build_dataloader(
            samples=samples,
            mode="train",
            batch_size=2,
            input_size=256,
            gt_ratio=1,
            num_frames=7,
            num_workers=0,
        )

        sample = train_ds[0]
        frames, center_mask, h, w, name = sample
        print("single sample:")
        print("  frames shape:", tuple(frames.shape))
        print("  center mask shape:", tuple(center_mask.shape))
        print("  original hw:", (h, w))
        print("  name:", name)

        batch = next(iter(train_loader))
        batch_frames, batch_mask, batch_h, batch_w, batch_name = batch
        print("batch:")
        print("  frames shape:", tuple(batch_frames.shape))
        print("  mask shape:", tuple(batch_mask.shape))
        print("  original h:", batch_h)
        print("  original w:", batch_w)
        print("  sample names:", batch_name)

        assert batch_frames.shape == (2, 7, 3, 256, 256)
        assert batch_mask.shape == (2, 1, 256, 256)
        assert isinstance(batch_frames, torch.Tensor)
        assert isinstance(batch_mask, torch.Tensor)
        print("demo verify: OK")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
