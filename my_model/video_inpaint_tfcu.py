"""Video Inpainting Detection with TFCU-style Temporal Adapter.

Wraps the existing DINOv3+LoRA+DPT-FPN backbone and inserts a lightweight
temporal adapter at the P4 FPN level.  The adapter captures:

1. Local consecutive-frame differences (within each clip).
2. Forward historical memory (cross-clip, causal — never looks at future clips).

The result is injected back into P4 via a learnable residual coefficient
(alpha, initialised to 0) so the model degrades gracefully to the original
single-frame backbone at the start of training.

Input:
    video: [B, N, T, 3, H, W]   or   [B, T, 3, H, W]  (backward-compatible)

Output:
    logits: [B, N, T, 1, H, W]   or   [B, T, 1, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .temporal import TFCUInpaintAdapter


class VideoInpaintTFCU(nn.Module):
    """DINOv3 + LoRA + DPT-FPN + TFCU-Inpaint temporal adapter.

    Args:
        base_model: existing ``DINOv3ViTL16InpaintingDetector`` with
            ``use_dpt_fpn=True``.
        cfg: configuration dict (supports the same keys as the config YAML).
    """

    def __init__(self, base_model: nn.Module, cfg: dict):
        super().__init__()
        if not getattr(base_model, "use_dpt_fpn", False):
            raise ValueError(
                "VideoInpaintTFCU requires base_model with use_dpt_fpn=True"
            )

        self.base = base_model

        # ── temporal adapter config ──────────────────────────────────
        self.num_clips = int(cfg.get("num_clips", 4))
        self.num_frames = int(cfg.get("num_frames", 4))
        channels = int(cfg.get("neck_channels", 256))

        self.temporal_adapter = TFCUInpaintAdapter(
            channels=channels,
            memory_len=int(cfg.get("memory_len", 4)),
            use_memory=bool(cfg.get("use_memory", True)),
            use_spatial_pool=bool(cfg.get("use_spatial_pool", False)),
            detach_memory=bool(cfg.get("detach_memory", True)),
        )

        self._input_size = int(cfg.get("input_size", 512))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_fpn_features(
        self, frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract FPN pyramid from flat frame tensor (delegates to base)."""
        return self.base.extract_fpn_features(frames)

    def decode(
        self, P2: torch.Tensor, P3: torch.Tensor, P4: torch.Tensor,
        P5: torch.Tensor,
    ) -> torch.Tensor:
        """Decode FPN pyramid to logits (delegates to base)."""
        return self.base.decode_fpn(P2, P3, P4, P5)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal adaptation.

        Args:
            video: [B, N, T, 3, H, W]  or  [B, T, 3, H, W]

        Returns:
            logits: [B, N, T, 1, H, W]  or  [B, T, 1, H, W]
        """
        # ── normalise input shape ────────────────────────────────────
        squeeze_n = False
        if video.dim() == 5:
            # Backward-compatible: [B, T, 3, H, W] → add N=1
            B, T, C, H, W = video.shape
            video = video[:, None]       # [B, 1, T, 3, H, W]
            N = 1
            squeeze_n = True
        elif video.dim() == 6:
            B, N, T, C, H, W = video.shape
        else:
            raise ValueError(
                f"Expected 5D or 6D input, got shape {tuple(video.shape)}"
            )

        # ── flatten & extract FPN pyramid ────────────────────────────
        frames = video.reshape(B * N * T, C, H, W)       # [BNT, 3, H, W]
        P2, P3, P4, P5 = self.extract_fpn_features(frames)
        # P2: [BNT, 256, 128, 128]
        # P3: [BNT, 256,  64,  64]
        # P4: [BNT, 256,  32,  32]
        # P5: [BNT, 256,  16,  16]

        # ── temporal adapter at P4 ───────────────────────────────────
        P4 = self.temporal_adapter(P4, B=B, N=N, T=T)

        # ── decode ───────────────────────────────────────────────────
        logits = self.decode(P2, P3, P4, P5)              # [BNT, 1, 512, 512]
        logits = logits.reshape(B, N, T, 1, H, W)

        if squeeze_n:
            logits = logits[:, 0]                          # [B, T, 1, H, W]

        return logits
