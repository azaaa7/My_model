from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")


import importlib.util
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


PathLike = Union[str, Path]


def _default_hrnet_path() -> Path:
    exp_root = Path(__file__).resolve().parents[2]
    return exp_root / "ZZZ_model" / "models" / "hrnet.py"


def _load_hrnet_class(hrnet_path: Optional[PathLike] = None):
    path = Path(hrnet_path) if hrnet_path is not None else _default_hrnet_path()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HRNet source file not found: {path}")

    spec = importlib.util.spec_from_file_location("zzz_model_hrnet", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load HRNet module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HRNet


class LightweightDecoder(nn.Module):
    """Small per-frame segmentation head for HRNet features."""

    def __init__(self, in_channels: int = 32, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.net(feat)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class SimpleHRNetInpaintingDetector(nn.Module):
    """
    Simple video inpainting detector.

    Input:
        clip: [B, T, 3, H, W]

    Output:
        logits: [B, T, 1, H, W]
    """

    def __init__(
        self,
        hrnet_path: Optional[PathLike] = None,
        hrnet_extra_name: str = "w32_extra",
        freeze_backbone: bool = False,
        decoder_channels: int = 32,
    ):
        super().__init__()
        HRNet = _load_hrnet_class(hrnet_path)
        self.backbone = HRNet(extra_name=hrnet_extra_name)
        self.decoder = LightweightDecoder(in_channels=32, hidden_channels=decoder_channels)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"clip must have shape [B, T, 3, H, W], got {tuple(clip.shape)}")

        batch_size, num_frames, channels, height, width = clip.shape
        if channels != 3:
            raise ValueError(f"clip channel dimension must be 3, got {channels}")

        frames = clip.reshape(batch_size * num_frames, channels, height, width)
        feats = self.backbone(frames)
        logits = self.decoder(feats, output_size=(height, width))
        return logits.reshape(batch_size, num_frames, 1, height, width)
