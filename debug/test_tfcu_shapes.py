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
import torch.nn as nn

from my_model.dinov3_dpt_fpn import ViTMultiLayerFusionPyramidNeck
from my_model.temporal import (
    InpaintMemoryAttention,
    LocalTemporalDifferenceModule,
    MaskPromptEncoder,
    TFCUInpaintAdapter,
)
from my_model.video_inpaint_tfcu import VideoInpaintTFCU


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


def test_fused32_pyramid_shape() -> None:
    """Verify fused32 pyramid neck output shapes."""
    print("[test] ViTMultiLayerFusionPyramidNeck shapes ...", end=" ")
    feats = {
        5: torch.randn(2, 1024, 32, 32),
        11: torch.randn(2, 1024, 32, 32),
        17: torch.randn(2, 1024, 32, 32),
        23: torch.randn(2, 1024, 32, 32),
    }
    neck = ViTMultiLayerFusionPyramidNeck(
        in_ch=1024,
        out_ch=256,
        layers=(5, 11, 17, 23),
    )
    out = neck(feats)
    assert out["f32"].shape == (2, 256, 32, 32), out["f32"].shape
    assert out["p2"].shape == (2, 256, 128, 128), out["p2"].shape
    assert out["p3"].shape == (2, 256, 64, 64), out["p3"].shape
    assert out["p4"].shape == (2, 256, 32, 32), out["p4"].shape
    assert out["p5"].shape == (2, 256, 16, 16), out["p5"].shape
    print("OK")


class CaptureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_shape = None

    def forward(self, x, B: int, N: int, T: int):
        self.seen_shape = tuple(x.shape)
        return x


class FakeFused32Base(nn.Module):
    use_dpt_fpn = True
    neck_variant = "fused32_pyramid"

    def __init__(self, channels: int = 32):
        super().__init__()
        self.channels = channels

    def extract_pyramid_features(self, frames, *, return_f32=False):
        batch = frames.shape[0]
        f32 = torch.randn(batch, self.channels, 32, 32, device=frames.device)
        if return_f32:
            return {"f32": f32}
        return self.build_pyramid_from_f32(f32=f32, frames=frames)

    def build_pyramid_from_f32(self, f32, *, original_features=None, frames=None):
        batch, channels = f32.shape[:2]
        return {
            "f32": f32,
            "p2": torch.zeros(batch, channels, 128, 128, device=f32.device),
            "p3": torch.zeros(batch, channels, 64, 64, device=f32.device),
            "p4": f32,
            "p5": torch.zeros(batch, channels, 16, 16, device=f32.device),
        }

    def decode_fpn(self, P2, P3, P4, P5):
        return torch.zeros(P2.shape[0], 1, 512, 512, device=P2.device)


class FakeDPTBase(FakeFused32Base):
    neck_variant = "dpt_reassemble"

    def extract_fpn_features(self, frames):
        batch = frames.shape[0]
        channels = self.channels
        return (
            torch.zeros(batch, channels, 128, 128, device=frames.device),
            torch.zeros(batch, channels, 64, 64, device=frames.device),
            torch.randn(batch, channels, 32, 32, device=frames.device),
            torch.zeros(batch, channels, 16, 16, device=frames.device),
        )


def test_video_inpaint_tfcu_forward_fused32_shape() -> None:
    """Verify VideoInpaintTFCU keeps output shape on fused32 path."""
    print("[test] VideoInpaintTFCU fused32 forward shape ...", end=" ")
    cfg = {
        "neck_channels": 32,
        "neck_variant": "fused32_pyramid",
        "temporal_insert_level": "F32",
        "memory_len": 2,
        "use_memory": False,
    }
    model = VideoInpaintTFCU(FakeFused32Base(channels=32), cfg)
    video = torch.randn(1, 2, 4, 3, 512, 512)
    with torch.no_grad():
        logits = model(video)
    assert logits.shape == (1, 2, 4, 1, 512, 512), logits.shape
    print("OK")


def test_tfcu_insert_receives_f32() -> None:
    """Verify fused32 path sends [B*N*T,C,32,32] to temporal_adapter."""
    print("[test] temporal insert receives F32 ...", end=" ")
    cfg = {
        "neck_channels": 32,
        "neck_variant": "fused32_pyramid",
        "temporal_insert_level": "F32",
        "memory_len": 2,
        "use_memory": False,
    }
    model = VideoInpaintTFCU(FakeFused32Base(channels=32), cfg)
    capture = CaptureAdapter()
    model.temporal_adapter = capture
    video = torch.randn(1, 2, 4, 3, 512, 512)
    with torch.no_grad():
        _ = model(video)
    assert capture.seen_shape == (8, 32, 32, 32), capture.seen_shape
    print("OK")


def test_image_stem_skip_zero_init() -> None:
    """Verify image stem skip is zero-scale initialised and backpropagates."""
    print("[test] image stem skip zero init ...", end=" ")
    neck = ViTMultiLayerFusionPyramidNeck(
        in_ch=16,
        out_ch=32,
        layers=(5, 11),
        use_image_stem_skip=True,
    )
    assert neck.stem2_scale.item() == 0.0
    assert neck.stem3_scale.item() == 0.0

    feats = {
        5: torch.randn(1, 16, 32, 32, requires_grad=True),
        11: torch.randn(1, 16, 32, 32, requires_grad=True),
    }
    frames_a = torch.randn(1, 3, 512, 512, requires_grad=True)
    frames_b = torch.randn(1, 3, 512, 512, requires_grad=True)
    out_a = neck(feats, frames=frames_a)
    out_b = neck(feats, frames=frames_b)
    assert torch.allclose(out_a["p2"], out_b["p2"], atol=1e-6)
    assert torch.allclose(out_a["p3"], out_b["p3"], atol=1e-6)
    loss = sum(value.mean() for value in out_a.values())
    loss.backward()
    print("OK")


def test_image_stem_skip_disabled_has_no_trainable_unused_scales() -> None:
    """Verify disabled stem skip does not leave DDP-visible unused params."""
    print("[test] image stem skip disabled scales frozen ...", end=" ")
    neck = ViTMultiLayerFusionPyramidNeck(
        in_ch=16,
        out_ch=32,
        layers=(5, 11),
        use_image_stem_skip=False,
    )
    assert neck.stem2_scale.item() == 0.0
    assert neck.stem3_scale.item() == 0.0
    assert not neck.stem2_scale.requires_grad
    assert not neck.stem3_scale.requires_grad
    print("OK")


def test_legacy_dpt_reassemble_p4_fallback() -> None:
    """Verify legacy DPT neck path still inserts temporal adapter at P4."""
    print("[test] legacy DPT P4 fallback ...", end=" ")
    cfg = {
        "neck_channels": 32,
        "neck_variant": "dpt_reassemble",
        "temporal_insert_level": "P4",
        "memory_len": 2,
        "use_memory": False,
    }
    model = VideoInpaintTFCU(FakeDPTBase(channels=32), cfg)
    capture = CaptureAdapter()
    model.temporal_adapter = capture
    video = torch.randn(1, 2, 4, 3, 512, 512)
    with torch.no_grad():
        logits = model(video)
    assert logits.shape == (1, 2, 4, 1, 512, 512), logits.shape
    assert capture.seen_shape == (8, 32, 32, 32), capture.seen_shape
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
        test_fused32_pyramid_shape,
        test_video_inpaint_tfcu_forward_fused32_shape,
        test_tfcu_insert_receives_f32,
        test_image_stem_skip_zero_init,
        test_image_stem_skip_disabled_has_no_trainable_unused_scales,
        test_legacy_dpt_reassemble_p4_fallback,
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
