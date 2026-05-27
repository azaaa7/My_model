"""DINOv3 multi-layer feature extraction + DPT Reassemble Neck + FPN Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────

def get_gn_groups(channels: int, preferred: int = 32) -> int:
    """Return a valid GroupNorm group count for the given channels."""
    for g in [preferred, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ── basic building block ─────────────────────────────────────────────────

class ConvGNAct(nn.Module):
    """Conv3×3 → GroupNorm → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(get_gn_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── DPT Reassemble ──────────────────────────────────────────────────────

class ReassembleBlock(nn.Module):
    """Project a 32×32 DINO token-map to a target spatial scale.

    scale mapping (input_size=512, patch=16 → token map 32×32):
        "x4"   32→128   (for shallow block → P2)
        "x2"   32→ 64   (for mid-shallow block → P3)
        "x1"   32→ 32   (for mid-deep block → P4)
        "down2" 32→ 16  (for deepest block → P5)
    """

    def __init__(self, in_ch: int = 1024, out_ch: int = 256, scale: str = "x1"):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(get_gn_groups(out_ch), out_ch),
            nn.GELU(),
        ]

        if scale == "x4":
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(out_ch, out_ch),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(out_ch, out_ch),
            ]
        elif scale == "x2":
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(out_ch, out_ch),
            ]
        elif scale == "x1":
            layers += [ConvGNAct(out_ch, out_ch)]
        elif scale == "down2":
            layers += [
                ConvGNAct(out_ch, out_ch, stride=2),
                ConvGNAct(out_ch, out_ch),
            ]
        else:
            raise ValueError(f"Unknown scale: {scale}")

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DPTReassembleNeck(nn.Module):
    """Convert 4 same-resolution DINO token-maps to FPN multi-scale features.

    Input:  dict {layer_idx: [N, 1024, 32, 32]}  for layers [5, 11, 17, 23]
    Output: dict {"p2": [N,256,128,128], "p3": [N,256,64,64],
                  "p4": [N,256,32,32],   "p5": [N,256,16,16]}
    """

    def __init__(self, in_ch: int = 1024, out_ch: int = 256, layers: tuple[int, ...] = (5, 11, 17, 23)):
        super().__init__()
        self.layer_indices = list(layers)
        scales = ["x4", "x2", "x1", "down2"]
        self.reassemble = nn.ModuleDict()
        for idx, scale in zip(self.layer_indices, scales):
            self.reassemble[str(idx)] = ReassembleBlock(in_ch, out_ch, scale)

    def forward(self, feats: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        l0, l1, l2, l3 = self.layer_indices
        return {
            "p2": self.reassemble[str(l0)](feats[l0]),
            "p3": self.reassemble[str(l1)](feats[l1]),
            "p4": self.reassemble[str(l2)](feats[l2]),
            "p5": self.reassemble[str(l3)](feats[l3]),
        }


# ── FPN Decoder ──────────────────────────────────────────────────────────

class FPNDecoder(nn.Module):
    """Top-down FPN decoder with final upsampling to input resolution.

    Input:  {"p2": [N,256,128,128], "p3": [N,256,64,64],
             "p4": [N,256,32,32],   "p5": [N,256,16,16]}
    Output: [N, 1, 512, 512]  logits (no sigmoid)
    """

    def __init__(self, channels: int = 256, out_channels: int = 1):
        super().__init__()

        # FPN lateral + output convs (one per level)
        self.lat5 = ConvGNAct(channels, channels)
        self.lat4 = ConvGNAct(channels, channels)
        self.lat3 = ConvGNAct(channels, channels)
        self.lat2 = ConvGNAct(channels, channels)

        # Final upsampling stages: 128→256→512
        self.up1 = ConvGNAct(channels, 128)
        self.up0 = ConvGNAct(128, 64)

        # Prediction head
        self.head = nn.Sequential(
            ConvGNAct(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        p2, p3, p4, p5 = feats["p2"], feats["p3"], feats["p4"], feats["p5"]

        # Top-down FPN
        f5 = self.lat5(p5)                                                   # [N,256,16,16]

        f4 = p4 + F.interpolate(f5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        f4 = self.lat4(f4)                                                   # [N,256,32,32]

        f3 = p3 + F.interpolate(f4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f3 = self.lat3(f3)                                                   # [N,256,64,64]

        f2 = p2 + F.interpolate(f3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        f2 = self.lat2(f2)                                                   # [N,256,128,128]

        # Final upsampling: 128 → 256 → 512
        x = F.interpolate(f2, size=(256, 256), mode="bilinear", align_corners=False)
        x = self.up1(x)                                                      # [N,128,256,256]

        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = self.up0(x)                                                      # [N, 64,512,512]

        return self.head(x)                                                  # [N,  1,512,512]
