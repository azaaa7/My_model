"""Mask / boundary prompt encoder for future TFCU-Inpaint extensions.

Encodes predicted mask and its Sobel boundary into a spatial prompt that can
be fused with temporal features.  First version is not wired into the adapter
by default — enable via ``use_mask_prompt: true`` in config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskPromptEncoder(nn.Module):
    """Encode (mask, boundary) pair into a channel-wise prompt map.

    Args:
        channels: output channel count, should match the feature dimension
            at the injection level (e.g. 256 for P4).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sobel_boundary(mask: torch.Tensor) -> torch.Tensor:
        """Sobel edge magnitude for a single-channel mask.

        Args:
            mask: [*, 1, H, W]  binary or soft mask.

        Returns:
            boundary: [*, 1, H, W]  gradient magnitude.
        """
        device = mask.device
        dtype = mask.dtype

        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(mask, kx, padding=1)
        gy = F.conv2d(mask, ky, padding=1)

        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        mask_logits: torch.Tensor,
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        """Produce mask + boundary prompt.

        Args:
            mask_logits: [B, T, 1, H_img, W_img]
            out_size: target spatial size, e.g. (32, 32) for P4.

        Returns:
            prompt: [B, T, C, out_h, out_w]
        """
        B, T, _, H_img, W_img = mask_logits.shape

        mask = torch.sigmoid(mask_logits)
        mask = mask.reshape(B * T, 1, H_img, W_img)
        mask = F.interpolate(
            mask, size=out_size, mode="bilinear", align_corners=False,
        )

        boundary = self.sobel_boundary(mask)

        prompt = torch.cat([mask, boundary], dim=1)   # [B*T, 2, h, w]
        prompt = self.encoder(prompt)                  # [B*T, C, h, w]

        C = prompt.shape[1]
        prompt = prompt.reshape(B, T, C, out_size[0], out_size[1])
        return prompt
