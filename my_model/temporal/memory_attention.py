"""Forward historical memory attention for inpainting detection.

Accumulates history across clips in chronological order.  The current clip
can only attend to past memory — never to future clips.

Uses manual multi-head attention (not nn.MultiheadAttention) to avoid
PyTorch-version-specific ``scaled_dot_product_attention`` backend issues
(e.g. ``permute(sparse_coo)`` under ``torch.no_grad()`` in PyTorch 2.7).
"""

from __future__ import annotations

import math

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
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_spatial_pool = use_spatial_pool
        self.pool_size = pool_size

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    # ------------------------------------------------------------------
    # Backward-compatible checkpoint loading
    # ------------------------------------------------------------------

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Intercept old nn.MultiheadAttention in_proj_weight and split into
        separate q_proj / k_proj / v_proj weights."""
        in_proj_key = prefix + "attn.in_proj_weight"
        in_proj_bias = prefix + "attn.in_proj_bias"
        out_proj_key = prefix + "attn.out_proj.weight"
        out_proj_bias = prefix + "attn.out_proj.bias"

        if in_proj_key in state_dict:
            w = state_dict.pop(in_proj_key)       # [3*C, C]
            C = self.channels
            state_dict[prefix + "q_proj.weight"] = w[:C]
            state_dict[prefix + "k_proj.weight"] = w[C:2*C]
            state_dict[prefix + "v_proj.weight"] = w[2*C:]
        if in_proj_bias in state_dict:
            b = state_dict.pop(in_proj_bias)       # [3*C]
            C = self.channels
            state_dict[prefix + "q_proj.bias"] = b[:C]
            state_dict[prefix + "k_proj.bias"] = b[C:2*C]
            state_dict[prefix + "v_proj.bias"] = b[2*C:]
        if out_proj_key in state_dict:
            state_dict[prefix + "out_proj.weight"] = state_dict.pop(out_proj_key)
        if out_proj_bias in state_dict:
            state_dict[prefix + "out_proj.bias"] = state_dict.pop(out_proj_bias)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        spatial_h, spatial_w = H, W

        # --- optional spatial down-sampling for memory efficiency ---------
        if self.use_spatial_pool and (H > self.pool_size or W > self.pool_size):
            cur_small = self._spatial_pool(cur, self.pool_size)
            mem_small = self._spatial_pool_memory(mem, self.pool_size)
            attn_out = self._attention_forward(cur_small, mem_small, B, T)
            attn_out = F.interpolate(
                attn_out.reshape(B * T, C, self.pool_size, self.pool_size),
                size=(spatial_h, spatial_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T, C, spatial_h, spatial_w)
        else:
            attn_out = self._attention_forward(cur, mem, B, T)

        return attn_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attention_forward(
        self,
        cur: torch.Tensor,    # [B, T, C, H, W]
        mem: torch.Tensor,    # [B, K, T, C, H, W]
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Manual multi-head cross-attention + FFN with residual."""
        C = self.channels
        H, W = cur.shape[-2:]
        K = mem.shape[1]

        # --- build token sequences ---------------------------------------
        q = cur.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)       # [B, Q, C]
        kv = mem.permute(0, 1, 2, 4, 5, 3).reshape(B, K * T * H * W, C)  # [B, KV, C]

        # Pre-norm
        qn = self.norm1(q)
        kvn = self.norm1(kv)

        # Project
        Q = self.q_proj(qn)   # [B, Q, C]
        K_ = self.k_proj(kvn) # [B, KV, C]
        V = self.v_proj(kvn)  # [B, KV, C]

        # --- manual multi-head attention ---------------------------------
        Q = Q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q, D]
        K_ = K_.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, KV, D]
        V = V.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, KV, D]

        attn_weights = (Q @ K_.transpose(-2, -1)) * self.scale            # [B, H, Q, KV]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = attn_weights @ V                                        # [B, H, Q, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, C)             # [B, Q, C]

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
        x_flat = x.reshape(-1, shape_in[-3], shape_in[-2], shape_in[-1])
        pooled = F.adaptive_avg_pool2d(x_flat, (size, size))
        return pooled.reshape(*shape_in[:-2], size, size)

    @staticmethod
    def _spatial_pool_memory(mem: torch.Tensor, size: int) -> torch.Tensor:
        """Average-pool a [B,K,T,C,H,W] memory tensor."""
        shape_in = mem.shape
        mem_flat = mem.reshape(-1, shape_in[-3], shape_in[-2], shape_in[-1])
        pooled = F.adaptive_avg_pool2d(mem_flat, (size, size))
        return pooled.reshape(*shape_in[:-2], size, size)
