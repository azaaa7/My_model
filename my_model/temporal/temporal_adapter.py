"""TFCU-Inpaint Adapter — the main temporal module entry-point.

Receives P4 features at shape [B*N*T, C, H, W], applies local consecutive-frame
difference and forward historical memory attention, then injects the result
back into P4 via a learnable residual scalar (alpha, initialised to 0).

The adapter processes clips sequentially (n = 0 … N-1) — the current clip can
only attend to past memory, never to future clips.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .local_temporal_difference import LocalTemporalDifferenceModule
from .memory_attention import InpaintMemoryAttention


class TFCUInpaintAdapter(nn.Module):
    """Minimal TFCU-style temporal adapter for video inpainting detection.

    Args:
        channels: feature dimension (e.g. 256 for P4).
        memory_len: how many past clips to keep in the memory buffer.
        use_memory: if False, skip cross-clip memory attention entirely.
        use_spatial_pool: pool spatial dims before memory attention for OOM
            avoidance (P4 32×32 → 16×16 → back).
        detach_memory: if True, detach stored memory so gradients don't flow
            across clips.  Recommended for first version to stabilise training.
    """

    def __init__(
        self,
        channels: int = 256,
        memory_len: int = 4,
        use_memory: bool = True,
        use_spatial_pool: bool = False,
        detach_memory: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.memory_len = memory_len
        self.use_memory = use_memory
        self.detach_memory = detach_memory

        self.local = LocalTemporalDifferenceModule(channels=channels)

        self.memory_attn = InpaintMemoryAttention(
            channels=channels,
            num_heads=8,
            use_spatial_pool=use_spatial_pool,
            pool_size=16,
        )
        if not self.use_memory:
            for param in self.memory_attn.parameters():
                param.requires_grad = False

        # Project temporal-enhanced features back to residual space
        self.temporal_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Learnable residual coefficient — init to 0 so model degrades
        # gracefully to the original single-frame backbone.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P4: torch.Tensor,
        B: int,
        N: int,
        T: int,
    ) -> torch.Tensor:
        """Apply temporal enhancement to P4 features.

        Args:
            P4: [B*N*T, C, H, W]
            B:  batch size.
            N:  number of clips per sample.
            T:  frames per clip.

        Returns:
            P4_out: [B*N*T, C, H, W]
        """
        _, C, H, W = P4.shape
        x = P4.reshape(B, N, T, C, H, W)          # [B, N, T, C, H, W]

        # 1. Local consecutive-frame difference (within each clip independently)
        x = self.local(x)

        # 2. Forward historical memory (cross-clip, causal)
        enhanced: list[torch.Tensor] = []
        state: list[torch.Tensor] = []              # FIFO memory buffer

        for n in range(N):
            cur = x[:, n]                           # [B, T, C, H, W]

            if (not self.use_memory) or len(state) == 0:
                cur_enhanced = cur
            else:
                # Stack at most memory_len past clips  →  [B, K, T, C, H, W]
                mem = torch.stack(state[-self.memory_len:], dim=1)
                cur_enhanced = self.memory_attn(cur, mem)

            enhanced.append(cur_enhanced)
            state.append(self._encode_memory(cur_enhanced))

        # 3. Project back & inject as residual
        temporal = torch.stack(enhanced, dim=1)     # [B, N, T, C, H, W]
        temporal = temporal.reshape(B * N * T, C, H, W)
        temporal = self.temporal_proj(temporal)

        P4_out = P4 + self.alpha * temporal
        return P4_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_memory(self, cur: torch.Tensor) -> torch.Tensor:
        """Encode current clip features into a memory entry.

        First version detaches to avoid gradient explosion across clips.
        """
        if self.detach_memory:
            return cur.detach()
        return cur
