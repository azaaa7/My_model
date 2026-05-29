"""Video Inpainting Detection with TFCU-style Temporal Adapter.

Wraps the existing DINOv3+LoRA+DPT-FPN backbone and inserts a lightweight
temporal adapter at the fused F32 feature level for the fused32 pyramid neck
(or at P4 for the legacy DPT reassemble neck).  The adapter captures:

1. Local consecutive-frame differences (within each clip).
2. Forward historical memory (cross-clip, causal — never looks at future clips).

The result is injected via a learnable residual coefficient (alpha, initialised
to 0) so the model degrades gracefully to the original single-frame backbone at
the start of training.

Input:
    video: [B, N, T, 3, H, W]   or   [B, T, 3, H, W]  (backward-compatible)

Output:
    logits: [B, N, T, 1, H, W]   or   [B, T, 1, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .temporal import TFCUInpaintAdapter


class GatedTemporalInjector(nn.Module):
    """Apply a TFCU adapter as a gated residual branch."""

    def __init__(self, tfcu_module: nn.Module, init: float = -3.0):
        super().__init__()
        self.tfcu = tfcu_module
        self.logit_gate = nn.Parameter(torch.tensor(float(init)))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_gate)

    def forward(self, x: torch.Tensor, B: int, N: int, T: int) -> torch.Tensor:
        enhanced = self.tfcu(x, B=B, N=N, T=T)
        return x + self.gate * (enhanced - x)


class NoOpTemporalInjector(nn.Module):
    """No-parameter temporal adapter used for semantic-anchor no-temporal ablations."""

    def forward(self, x: torch.Tensor, B: int, N: int, T: int) -> torch.Tensor:
        return x


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
        self.neck_variant = str(
            cfg.get("neck_variant", getattr(base_model, "neck_variant", "fused32_pyramid"))
        ).strip().lower()
        self.temporal_insert_level = str(
            cfg.get("temporal_insert_level", "F32" if self.neck_variant == "fused32_pyramid" else "P4")
        ).strip().upper()

        if self.neck_variant == "fused32_pyramid" and self.temporal_insert_level != "F32":
            raise ValueError("neck_variant=fused32_pyramid requires temporal_insert_level=F32")
        if self.neck_variant == "dpt_reassemble" and self.temporal_insert_level != "P4":
            raise ValueError("neck_variant=dpt_reassemble requires temporal_insert_level=P4")
        if self.neck_variant == "semantic_anchor_mfce" and self.temporal_insert_level not in {"P4", "NONE"}:
            raise ValueError("neck_variant=semantic_anchor_mfce requires temporal_insert_level=P4 or NONE")
        if self.neck_variant not in {"fused32_pyramid", "dpt_reassemble", "semantic_anchor_mfce"}:
            raise ValueError(f"Unknown neck_variant: {self.neck_variant}")

        if self.neck_variant == "semantic_anchor_mfce" and self.temporal_insert_level == "NONE":
            self.temporal_adapter = NoOpTemporalInjector()
        else:
            use_memory = bool(cfg.get("use_memory", True)) and self.num_clips > 1
            tfcu = TFCUInpaintAdapter(
                channels=channels,
                memory_len=int(cfg.get("memory_len", 4)),
                use_memory=use_memory,
                use_spatial_pool=bool(cfg.get("use_spatial_pool", False)),
                detach_memory=bool(cfg.get("detach_memory", True)),
            )
            if self.neck_variant == "semantic_anchor_mfce" and self.temporal_insert_level == "P4":
                self.temporal_adapter = GatedTemporalInjector(tfcu, init=float(cfg.get("p4_gate_init", -3.0)))
            else:
                self.temporal_adapter = tfcu

        self._input_size = int(cfg.get("input_size", 512))
        # Encoder chunk size — process at most this many frames at once
        # through DINOv3 to avoid OOM.  0 or negative = no chunking.
        self._encoder_chunk = int(cfg.get("encoder_chunk", 0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_fpn_features(
        self, frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract FPN pyramid from flat frame tensor (delegates to base)."""
        return self.base.extract_fpn_features(frames)

    def extract_pyramid_features(
        self,
        frames: torch.Tensor,
        *,
        return_f32: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract pyramid features from flat frame tensor (delegates to base)."""
        return self.base.extract_pyramid_features(frames, return_f32=return_f32)

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

        # ── flatten & extract features (chunked for VRAM) ───────────
        frames = video.reshape(B * N * T, C, H, W)       # [BNT, 3, H, W]
        total_frames = frames.shape[0]
        chunk = self._encoder_chunk if self._encoder_chunk > 0 else total_frames

        if self.neck_variant == "semantic_anchor_mfce":
            p4_parts = []
            detail_parts: dict[str, list[torch.Tensor]] = {}
            layer_attn_parts = []
            for start in range(0, total_frames, chunk):
                end = min(start + chunk, total_frames)
                features = self.base.extract_semantic_anchor_features(frames[start:end])
                p4_parts.append(features["p4"])
                if "layer_attn" in features:
                    layer_attn_parts.append(features["layer_attn"])
                detail = features.get("detail")
                if isinstance(detail, dict):
                    for key in ("p1", "p2", "p3"):
                        if key in detail:
                            detail_parts.setdefault(key, []).append(detail[key])

            P4 = torch.cat(p4_parts, dim=0)             # [BNT, C, 32, 32]
            if self.temporal_insert_level == "P4":
                P4 = self.temporal_adapter(P4, B=B, N=N, T=T)

            detail_cat = {
                key: torch.cat(parts, dim=0)
                for key, parts in detail_parts.items()
            } or None
            if layer_attn_parts:
                self.base.last_aux = {"layer_attn": torch.cat(layer_attn_parts, dim=0)}
            logits = self.base.decode_semantic_anchor(P4, detail=detail_cat)
            logits = logits.reshape(B, N, T, 1, H, W)
            if squeeze_n:
                logits = logits[:, 0]
            return logits

        if self.temporal_insert_level == "F32":
            f32_parts = []
            for start in range(0, total_frames, chunk):
                end = min(start + chunk, total_frames)
                features = self.extract_pyramid_features(frames[start:end], return_f32=True)
                f32_parts.append(features["f32"])

            f32 = torch.cat(f32_parts, dim=0)             # [BNT, 256, 32, 32]

            # ── temporal adapter at fused F32 ────────────────────────
            f32 = self.temporal_adapter(f32, B=B, N=N, T=T)

            features = self.base.build_pyramid_from_f32(f32=f32, frames=frames)
            P2, P3, P4, P5 = features["p2"], features["p3"], features["p4"], features["p5"]
        else:
            P2_parts, P3_parts, P4_parts, P5_parts = [], [], [], []
            for start in range(0, total_frames, chunk):
                end = min(start + chunk, total_frames)
                p2, p3, p4, p5 = self.extract_fpn_features(frames[start:end])
                P2_parts.append(p2)
                P3_parts.append(p3)
                P4_parts.append(p4)
                P5_parts.append(p5)

            P2 = torch.cat(P2_parts, dim=0)   # [BNT, 256, 128, 128]
            P3 = torch.cat(P3_parts, dim=0)   # [BNT, 256,  64,  64]
            P4 = torch.cat(P4_parts, dim=0)   # [BNT, 256,  32,  32]
            P5 = torch.cat(P5_parts, dim=0)   # [BNT, 256,  16,  16]

            # ── legacy temporal adapter at P4 ───────────────────────
            P4 = self.temporal_adapter(P4, B=B, N=N, T=T)

        # ── decode ───────────────────────────────────────────────────
        logits = self.decode(P2, P3, P4, P5)              # [BNT, 1, 512, 512]
        logits = logits.reshape(B, N, T, 1, H, W)

        if squeeze_n:
            logits = logits[:, 0]                          # [B, T, 1, H, W]

        return logits
