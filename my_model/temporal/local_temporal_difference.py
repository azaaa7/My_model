"""Local temporal difference module — captures adjacent-frame inconsistency.

Inpainting regions may look natural in single frames but exhibit anomalies in
texture, boundary, and motion consistency across consecutive frames.  This
module computes per-pixel frame-to-frame differences and fuses them with the
original features via a learned gating mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    """Conv3×3 → GroupNorm → GELU (local helper, avoids circular imports)."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        if out_ch % groups != 0:
            groups = 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalTemporalDifferenceModule(nn.Module):
    """Compute and fuse consecutive-frame differences into feature maps.

    For frame t the diff is ``x_t - x_{t-1}``; for t=0 (first frame) we use
    ``x_0 - x_0 = 0`` so no spurious signal is injected.

    Input:
        x: [B, N, T, C, H, W]

    Output:
        out: [B, N, T, C, H, W]  —  x + residual delta
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, C, H, W]
        B, N, T, C, H, W = x.shape

        # Previous frame (first frame is its own "previous")
        prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)       # [B,N,T,C,H,W]
        diff = x - prev
        abs_diff = diff.abs()

        # Concatenate along channel dim
        feat = torch.cat([x, diff, abs_diff], dim=3)                # [B,N,T,3C,H,W]
        feat = feat.reshape(B * N * T, C * 3, H, W)

        delta = self.fuse(feat)                                     # [BNT, C, H, W]
        gate = self.gate(feat)                                      # [BNT, C, H, W]

        delta = delta * gate
        delta = delta.reshape(B, N, T, C, H, W)

        return x + delta
