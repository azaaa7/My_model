"""Forward historical memory attention for inpainting detection.

Accumulates history across clips in chronological order.  The current clip
can only attend to past memory — never to future clips.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InpaintMemoryAttention(nn.Module):
    """Cross-attention from current clip features to historical memory features.

    Args:
        channels: feature dimension (default 256 for P4 level).
        num_heads: attention heads.
        dropout: attention dropout rate.
        use_spatial_pool: if True, average-pool spatial dims to reduce memory
            before attention, then upsample back.  Useful for OOM avoidance.
        pool_size: target spatial size when ``use_spatial_pool=True``.
    """

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_spatial_pool: bool = False,
        pool_size: int = 16,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_spatial_pool = use_spatial_pool
        self.pool_size = pool_size

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        cur: torch.Tensor,
        mem: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention from current clip to historical memory.

        Args:
            cur: current clip features  [B, T, C, H, W]
            mem: history  memory  features  [B, K, T, C, H, W]

        Returns:
            enhanced features  [B, T, C, H, W]
        """
        B, T, C, H, W = cur.shape
        K = mem.shape[1]
        spatial_h, spatial_w = H, W

        # --- optional spatial down-sampling for memory efficiency ---------
        if self.use_spatial_pool and (H > self.pool_size or W > self.pool_size):
            cur_small = self._spatial_pool(cur, self.pool_size)           # [B,T,C,ps,ps]
            mem_small = self._spatial_pool_memory(mem, self.pool_size)    # [B,K,T,C,ps,ps]
            attn_out = self._attention_forward(cur_small, mem_small, B, T, K)
            attn_out = F.interpolate(
                attn_out.reshape(B * T, C, self.pool_size, self.pool_size),
                size=(spatial_h, spatial_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T, C, spatial_h, spatial_w)
        else:
            attn_out = self._attention_forward(cur, mem, B, T, K)

        return attn_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attention_forward(
        self,
        cur: torch.Tensor,
        mem: torch.Tensor,
        B: int,
        T: int,
        K: int,
    ) -> torch.Tensor:
        """Core multi-head cross-attention + FFN with residual connections."""
        C = self.channels
        _, _, _, H, W = cur.shape

        # Build token sequences
        # Query: [B, T*H*W, C]
        q = cur.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)

        # Key / Value: [B, K*T*H*W, C]
        kv = mem.permute(0, 1, 2, 4, 5, 3).reshape(B, K * T * H * W, C)

        # Pre-norm
        qn = self.norm1(q)
        kvn = self.norm1(kv)

        # Multi-head cross-attention (batch_first)
        attn_out, _ = self.attn(
            query=self.q_proj(qn),
            key=self.k_proj(kvn),
            value=self.v_proj(kvn),
            need_weights=False,
        )

        # Residual + FFN
        x = q + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))

        # Reshape back to [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)

    # ------------------------------------------------------------------
    # Spatial pooling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_pool(x: torch.Tensor, size: int) -> torch.Tensor:
        """Average-pool a [B,T,C,H,W] tensor to [B,T,C,size,size]."""
        shape_in = x.shape
        x_flat = x.reshape(-1, shape_in[-3], shape_in[-2], shape_in[-1])  # [B*T, C, H, W]
        pooled = F.adaptive_avg_pool2d(x_flat, (size, size))
        return pooled.reshape(*shape_in[:-2], size, size)

    @staticmethod
    def _spatial_pool_memory(mem: torch.Tensor, size: int) -> torch.Tensor:
        """Average-pool a [B,K,T,C,H,W] memory tensor."""
        shape_in = mem.shape
        mem_flat = mem.reshape(-1, shape_in[-3], shape_in[-2], shape_in[-1])
        pooled = F.adaptive_avg_pool2d(mem_flat, (size, size))
        return pooled.reshape(*shape_in[:-2], size, size)
