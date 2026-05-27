#!/usr/bin/env python3
"""Shape unit tests for TFCU-Inpaint temporal modules.

Run:
    python debug/test_tfcu_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from my_model.temporal import (
    InpaintMemoryAttention,
    LocalTemporalDifferenceModule,
    MaskPromptEncoder,
    TFCUInpaintAdapter,
)


def test_local_temporal_difference() -> None:
    """Verify LocalTemporalDifferenceModule preserves shape."""
    print("[test] LocalTemporalDifferenceModule ...", end=" ")
    B, N, T, C, H, W = 2, 4, 4, 256, 32, 32
    x = torch.randn(B, N, T, C, H, W)
    m = LocalTemporalDifferenceModule(channels=C)
    y = m(x)
    assert y.shape == x.shape, f"{y.shape} != {x.shape}"
    print("OK")


def test_memory_attention() -> None:
    """Verify InpaintMemoryAttention output shape."""
    print("[test] InpaintMemoryAttention ...", end=" ")
    B, K, T, C, H, W = 1, 4, 4, 256, 32, 32
    cur = torch.randn(B, T, C, H, W)
    mem = torch.randn(B, K, T, C, H, W)
    m = InpaintMemoryAttention(channels=C, num_heads=8)
    y = m(cur, mem)
    assert y.shape == cur.shape, f"{y.shape} != {cur.shape}"
    print("OK")


def test_memory_attention_spatial_pool() -> None:
    """Verify spatial-pooled memory attention."""
    print("[test] InpaintMemoryAttention (spatial pool) ...", end=" ")
    B, K, T, C, H, W = 1, 4, 4, 256, 32, 32
    cur = torch.randn(B, T, C, H, W)
    mem = torch.randn(B, K, T, C, H, W)
    m = InpaintMemoryAttention(channels=C, num_heads=8, use_spatial_pool=True, pool_size=16)
    y = m(cur, mem)
    assert y.shape == cur.shape, f"{y.shape} != {cur.shape}"
    print("OK")


def test_tfcu_adapter() -> None:
    """Verify TFCUInpaintAdapter output shape."""
    print("[test] TFCUInpaintAdapter ...", end=" ")
    B, N, T, C, H, W = 1, 4, 4, 256, 32, 32
    P4 = torch.randn(B * N * T, C, H, W)
    m = TFCUInpaintAdapter(channels=C, memory_len=4)
    y = m(P4, B=B, N=N, T=T)
    assert y.shape == P4.shape, f"{y.shape} != {P4.shape}"
    print("OK")


def test_tfcu_adapter_no_memory() -> None:
    """Verify adapter without memory attention."""
    print("[test] TFCUInpaintAdapter (no memory) ...", end=" ")
    B, N, T, C, H, W = 1, 2, 3, 256, 32, 32
    P4 = torch.randn(B * N * T, C, H, W)
    m = TFCUInpaintAdapter(channels=C, memory_len=4, use_memory=False)
    y = m(P4, B=B, N=N, T=T)
    assert y.shape == P4.shape, f"{y.shape} != {P4.shape}"
    print("OK")


def test_mask_prompt_encoder() -> None:
    """Verify MaskPromptEncoder output shape."""
    print("[test] MaskPromptEncoder ...", end=" ")
    B, T, H, W = 2, 4, 512, 512
    C = 256
    mask_logits = torch.randn(B, T, 1, H, W)
    m = MaskPromptEncoder(channels=C)
    y = m(mask_logits, out_size=(32, 32))
    expected = (B, T, C, 32, 32)
    assert y.shape == expected, f"{y.shape} != {expected}"
    print("OK")


def test_alpha_initial_zero() -> None:
    """Verify TFCUInpaintAdapter alpha starts at 0 for graceful degradation."""
    print("[test] alpha initialised to 0 ...", end=" ")
    m = TFCUInpaintAdapter(channels=256)
    assert torch.allclose(m.alpha, torch.tensor(0.0)), f"alpha={m.alpha.item()}"
    print("OK")


def test_memory_causal() -> None:
    """Verify adapter processes clips sequentially (no future leak).

    Strategy: feed two clips with very different P4 values; the first
    clip's output should NOT depend on the second clip.
    """
    print("[test] causal memory (no future leak) ...", end=" ")
    B, N, T, C, H, W = 1, 2, 2, 32, 8, 8
    # Clip 0: all ones, Clip 1: all zeros
    P4_clip0 = torch.ones(B * T, C, H, W)
    P4_clip1 = torch.zeros(B * T, C, H, W)
    P4 = torch.cat([P4_clip0, P4_clip1], dim=0)

    m = TFCUInpaintAdapter(channels=C, memory_len=4, use_memory=True, detach_memory=False)
    y = m(P4, B=B, N=N, T=T)

    # Clip 0 output should NOT be all zeros (clip 1 hasn't been seen yet)
    y_clip0 = y[:B*T]
    assert not torch.allclose(y_clip0, torch.zeros_like(y_clip0)), \
        "Clip 0 output is zero — memory may be leaking from the future."
    print("OK")


def test_deterministic_without_memory() -> None:
    """Without memory, first clip outputs should be deterministic."""
    print("[test] deterministic without memory ...", end=" ")
    B, N, T, C, H, W = 1, 3, 2, 64, 8, 8
    torch.manual_seed(42)
    m = TFCUInpaintAdapter(channels=C, use_memory=False)
    m.eval()
    P4 = torch.randn(B * N * T, C, H, W)
    with torch.no_grad():
        y1 = m(P4, B=B, N=N, T=T)
        y2 = m(P4, B=B, N=N, T=T)
    assert torch.allclose(y1, y2, atol=1e-6), "Outputs differ across runs"
    print("OK")


def main() -> None:
    tests = [
        test_local_temporal_difference,
        test_memory_attention,
        test_memory_attention_spatial_pool,
        test_tfcu_adapter,
        test_tfcu_adapter_no_memory,
        test_mask_prompt_encoder,
        test_alpha_initial_zero,
        test_memory_causal,
        test_deterministic_without_memory,
    ]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"  {passed}/{len(tests)} tests passed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
