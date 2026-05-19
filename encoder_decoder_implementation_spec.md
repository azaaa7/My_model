# Encoder and Decoder Implementation Specification
## For a DINOv3-based Video Inpainting Localization Model

This document summarizes the technical details of the **encoder** and **decoder** modules only.  
It is intended to be handed to an AI coding assistant or engineer to implement the modules in PyTorch.

---

# 1. Scope

This specification covers:

1. **DINOv3 Multi-layer Patch Encoder**
2. **Patch Feature Fusion**
3. **Frequency-guided Boundary Decoder**
4. **High-frequency / Boundary Cue Branch**
5. **Recommended module interfaces**
6. **Shape conventions**
7. **Implementation notes and pseudo-code guidance**

This file does **not** cover the memory bank, temporal error accumulation, suspicious memory, or anomaly query transformer in detail, except where their outputs are needed by the decoder.

---

# 2. Overall Position in the Full Network

The full model can be summarized as:

```text
Video Frames
    ↓
DINOv3 Multi-layer Patch Encoder
    ↓
Patch Feature Fusion
    ↓
[Memory / Temporal / Query Modules]
    ↓
Frequency-guided Boundary Decoder
    ↓
Pixel-level Video Inpainting Mask
```

The encoder produces dense patch-level visual features.  
The decoder upsamples coarse anomaly representations into full-resolution masks and refines mask boundaries with high-frequency cues.

---

# 3. Input and Output Conventions

## 3.1 Input Video Tensor

Use the following standard format:

```python
video: Tensor[B, T, 3, H, W]
```

Where:

- `B`: batch size
- `T`: number of frames in one clip
- `H, W`: spatial image size
- Input frames are RGB

Recommended initial settings:

```text
T = 5 or 7
H = W = 384
```

---

## 3.2 Per-frame Encoder Processing

The DINOv3 encoder is applied frame-by-frame.

Before encoding:

```python
frames = video.reshape(B * T, 3, H, W)
```

After encoding, features are reshaped back to:

```python
features: Tensor[B, T, C, h, w]
```

For a patch stride of 16:

```text
h = H / 16
w = W / 16
```

Example:

```text
Input size: 384 × 384
Patch grid: 24 × 24
```

---

# 4. DINOv3 Multi-layer Patch Encoder

## 4.1 Purpose

The encoder extracts dense patch-level features from each video frame.  
Instead of using only the last transformer block output, the encoder should expose **multi-layer patch tokens**, because:

- earlier / mid layers preserve more local spatial detail;
- later layers provide stronger semantic and structural cues;
- localization tasks benefit from multi-level representation fusion.

The output of this stage will be used by downstream anomaly modeling modules.

---

## 4.2 Backbone Behavior

Assume a DINOv3 ViT backbone that produces:

- patch tokens only, without using the CLS token in the downstream localization branch;
- hidden states from several selected transformer layers.

Recommended selected layer strategy:

```text
Choose 4 layers across the backbone depth:
- one shallow-to-middle layer
- two intermediate layers
- one deep layer
```

Example for a deep ViT with L blocks:

```text
selected_layers = [L/4, L/2, 3L/4, L]
```

If exact indices are required, choose the nearest valid transformer block indices.

---

## 4.3 Output Tokens from Each Layer

For each selected layer `l`, obtain:

```python
tokens_l: Tensor[B*T, N, C_l]
```

Where:

- `N = h * w`
- `C_l` is the DINOv3 embedding dimension for that backbone

If the model returns CLS + patch tokens, remove CLS:

```python
patch_tokens_l = tokens_l[:, 1:, :]
```

---

## 4.4 Reshape to Spatial Patch Grids

Each layer's patch tokens should be reshaped to:

```python
patch_map_l: Tensor[B*T, C_l, h, w]
```

using:

```python
patch_map_l = patch_tokens_l.transpose(1, 2).reshape(B*T, C_l, h, w)
```

---

# 5. Patch Feature Fusion

## 5.1 Purpose

The patch feature fusion module converts multi-layer DINOv3 features into a unified dense feature map:

```python
fused_features: Tensor[B, T, C, h, w]
```

Recommended target channel dimension:

```text
C = 256
```

---

## 5.2 Recommended Fusion Design

For each selected layer feature map:

```text
LayerNorm / normalization
→ 1×1 Conv or Linear projection
→ channel dimension unified to C
```

Let:

```python
proj_l(patch_map_l): Tensor[B*T, C, h, w]
```

Then concatenate all projected feature maps:

```python
concat_features: Tensor[B*T, 4C, h, w]
```

A practical fusion block:

```text
Concat(4 projected feature maps)
→ Conv 1×1: 4C → 2C
→ GroupNorm or LayerNorm-like 2D normalization
→ GELU
→ Conv 3×3: 2C → C
→ GroupNorm
→ GELU
```

Output:

```python
fused_features_flat: Tensor[B*T, C, h, w]
```

Then reshape:

```python
fused_features = fused_features_flat.reshape(B, T, C, h, w)
```

---

## 5.3 Why Use a 3×3 Conv in Fusion

Although DINO patch tokens are transformer-derived, adding a shallow local convolution after concatenation can help:

- smooth inconsistencies across neighboring patch positions;
- reintroduce a small amount of locality;
- create a more stable interface for memory and decoder modules.

---

## 5.4 Suggested PyTorch-style Module Interface

```python
class DinoMultiLayerPatchEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        selected_layers,
        out_dim=256,
        patch_size=16,
        freeze_backbone=True
    ):
        ...

    def forward(self, video):
        # video: [B, T, 3, H, W]
        # return:
        # fused_features: [B, T, C, h, w]
        # optional multi_scale_features: list of [B, T, C, h, w]
        ...
```

---

# 6. Encoder Output Requirements for Downstream Modules

The downstream memory / temporal / query modules should receive:

```python
encoder_features: Tensor[B, T, C, h, w]
```

For each frame `t`:

```python
F_t = encoder_features[:, t]  # [B, C, h, w]
```

Optionally, the encoder may also expose flattened tokens:

```python
tokens_t: Tensor[B, N, C]
```

via:

```python
tokens_t = F_t.flatten(2).transpose(1, 2)
```

This token format is useful for:

- memory retrieval;
- patch affinity matching;
- transformer cross-attention.

---

# 7. Decoder Inputs

The final decoder should not depend only on the DINO features.  
It should fuse coarse anomaly outputs from the middle modules with high-frequency image cues.

Recommended decoder inputs:

```python
query_feature: Tensor[B, Cq, h, w]
accum_error_map: Tensor[B, 1, h, w]
coarse_mask_logits: Tensor[B, 1, h, w]
encoder_feature_t: Tensor[B, Ce, h, w]
boundary_features: multi-scale features from high-frequency branch
```

A practical channel choice:

```text
Ce = 256
Cq = 256
```

If `query_feature` is unavailable, the decoder can still be implemented using:

```text
encoder_feature_t + accum_error_map + coarse_mask_logits
```

---

# 8. Frequency-guided Boundary Decoder

## 8.1 Purpose

DINO patch maps are typically low resolution, e.g.:

```text
H/16 × W/16
```

If the model upsamples directly from patch grids to full resolution, the mask boundaries can become:

- coarse;
- blurry;
- inaccurate for thin or small tampered regions.

The decoder should therefore:

1. use coarse anomaly features from the main model;
2. use high-frequency / boundary cues from original frames;
3. progressively upsample from low resolution to full resolution.

---

# 9. High-frequency / Boundary Cue Branch

## 9.1 Purpose

This branch extracts detail-rich features from raw frames to improve mask boundaries.

It should provide multi-scale features at:

```text
1/4 resolution
1/8 resolution
1/16 resolution
```

---

## 9.2 Recommended Inputs

For each frame `I_t`, construct a cue tensor.

Minimal version:

```text
RGB image
+ Laplacian map
+ Sobel edge magnitude
```

Possible channel layout:

```python
cue_t: Tensor[B, 5, H, W]
```

with:

- 3 RGB channels
- 1 Laplacian channel
- 1 Sobel magnitude channel

More advanced version:

```text
RGB image
+ Laplacian
+ Sobel-x
+ Sobel-y
+ optional temporal difference
```

Possible layout:

```python
cue_t: Tensor[B, 7, H, W]
```

---

## 9.3 Optional Temporal Boundary Cue

A simple and implementable temporal cue:

```python
delta_t = abs(I_t - I_{t-1})
```

or:

```python
delta_t = abs(I_t - mean_of_neighbor_frames)
```

This cue is easier to implement than FFT-based temporal frequency features and is a good first version.

For a later advanced version, the branch can be extended with:

- temporal FFT magnitude;
- local temporal variance;
- high-frequency fluctuation statistics.

---

## 9.4 Boundary Cue Network

Recommended lightweight CNN:

```text
Input cue map [B, K, H, W]

Stem:
Conv 3×3, stride 2      -> [B, 64, H/2,  W/2]
GroupNorm + GELU
Conv 3×3, stride 2      -> [B, 96, H/4,  W/4]
GroupNorm + GELU

Stage 1:
Residual block(s)       -> [B, 96, H/4,  W/4]
Output B_1_4

Stage 2:
Conv 3×3, stride 2      -> [B, 128, H/8,  W/8]
Residual block(s)
Output B_1_8

Stage 3:
Conv 3×3, stride 2      -> [B, 192, H/16, W/16]
Residual block(s)
Output B_1_16
```

Outputs:

```python
boundary_features = {
    "1_4":  Tensor[B,  96, H/4,  W/4],
    "1_8":  Tensor[B, 128, H/8,  W/8],
    "1_16": Tensor[B, 192, H/16, W/16],
}
```

---

# 10. Decoder Fusion at 1/16 Resolution

## 10.1 Inputs at 1/16

Assume the main model provides:

```python
query_feature:      [B, 256, h, w]
encoder_feature_t:  [B, 256, h, w]
accum_error_map:    [B,   1, h, w]
coarse_mask_logits: [B,   1, h, w]
boundary_1_16:      [B, 192, h, w]
```

Concatenate:

```python
decoder_input_1_16 = concat(
    query_feature,
    encoder_feature_t,
    accum_error_map,
    coarse_mask_logits,
    boundary_1_16
)
```

Channel count:

```text
256 + 256 + 1 + 1 + 192 = 706
```

Project to a compact decoder channel dimension, e.g. 256:

```text
Conv 1×1: 706 → 256
GroupNorm
GELU
Conv 3×3: 256 → 256
GroupNorm
GELU
```

Result:

```python
D_1_16: Tensor[B, 256, h, w]
```

---

# 11. Progressive Upsampling Decoder

## 11.1 Stage: 1/16 → 1/8

Upsample:

```python
U_1_8 = bilinear_upsample(D_1_16, scale_factor=2)
```

Fuse with boundary feature:

```python
concat_1_8 = concat(U_1_8, boundary_1_8)
```

Recommended block:

```text
Concat [256 + 128]
→ Conv 3×3: 384 → 192
→ GroupNorm
→ GELU
→ Conv 3×3: 192 → 192
→ GroupNorm
→ GELU
```

Output:

```python
D_1_8: Tensor[B, 192, H/8, W/8]
```

---

## 11.2 Stage: 1/8 → 1/4

Upsample:

```python
U_1_4 = bilinear_upsample(D_1_8, scale_factor=2)
```

Fuse with boundary feature:

```python
concat_1_4 = concat(U_1_4, boundary_1_4)
```

Recommended block:

```text
Concat [192 + 96]
→ Conv 3×3: 288 → 128
→ GroupNorm
→ GELU
→ Conv 3×3: 128 → 128
→ GroupNorm
→ GELU
```

Output:

```python
D_1_4: Tensor[B, 128, H/4, W/4]
```

---

## 11.3 Stage: 1/4 → Full Resolution

Upsample twice or use a small two-step head:

### Option A: Two-step refinement

```text
1/4 → 1/2
1/2 → 1
```

Example:

```text
Upsample ×2
Conv 3×3: 128 → 64
GN + GELU

Upsample ×2
Conv 3×3: 64 → 32
GN + GELU

Conv 1×1: 32 → 1
```

Output:

```python
mask_logits: Tensor[B, 1, H, W]
```

---

# 12. Decoder Output

Final decoder output:

```python
mask_logits: Tensor[B, 1, H, W]
```

Sigmoid probabilities:

```python
mask_prob = torch.sigmoid(mask_logits)
```

Binary inference:

```python
mask_binary = (mask_prob > 0.5).float()
```

---

# 13. Optional Edge Prediction Head

To explicitly supervise boundaries, add an auxiliary edge head from `D_1_4`:

```text
D_1_4
→ Conv 3×3: 128 → 64
→ GELU
→ Conv 1×1: 64 → 1
→ Upsample to H × W
```

Output:

```python
edge_logits: Tensor[B, 1, H, W]
```

This can be trained with an edge supervision map derived from the ground-truth mask.

---

# 14. Recommended Decoder Module Interface

```python
class FrequencyGuidedBoundaryDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim=256,
        query_dim=256,
        use_query_feature=True,
        use_edge_head=True,
    ):
        ...

    def forward(
        self,
        encoder_feature_t,      # [B, 256, h, w]
        query_feature,          # [B, 256, h, w] or None
        accum_error_map,         # [B, 1, h, w]
        coarse_mask_logits,      # [B, 1, h, w]
        boundary_features,       # dict with 1_16, 1_8, 1_4
    ):
        # return:
        # mask_logits: [B, 1, H, W]
        # edge_logits: optional [B, 1, H, W]
        ...
```

---

# 15. Recommended High-frequency Branch Interface

```python
class HighFrequencyBoundaryBranch(nn.Module):
    def __init__(
        self,
        in_channels=5,
        channels=(96, 128, 192)
    ):
        ...

    def forward(self, cue_t):
        # cue_t: [B, K, H, W]
        # returns:
        # {
        #   "1_4":  [B,  96, H/4,  W/4],
        #   "1_8":  [B, 128, H/8,  W/8],
        #   "1_16": [B, 192, H/16, W/16],
        # }
        ...
```

---

# 16. Encoder + Decoder Training Losses

Although the full model has additional losses, the encoder and decoder directly interact with the following objectives.

## 16.1 Final Mask Segmentation Loss

```text
BCE loss + Dice loss
```

\[
L_{seg}=L_{BCE}+L_{Dice}
\]

---

## 16.2 Auxiliary Edge Loss

If using `edge_logits`:

\[
L_{edge}=BCE(edge\_logits, edge\_gt)
\]

Where `edge_gt` can be extracted from the binary mask using morphological gradient, Sobel filtering, or a contour operator.

---

## 16.3 Optional Deep Supervision

The decoder may output low-resolution intermediate masks:

```text
1/16 mask
1/8 mask
1/4 mask
full-resolution mask
```

Each intermediate mask can be supervised with a downsampled ground-truth mask.

This can stabilize early training.

---

# 17. Practical Design Choices

## 17.1 Use GroupNorm Instead of BatchNorm

Batch sizes for video models are often small.  
Use:

```text
GroupNorm
```

instead of BatchNorm for better stability.

Recommended:

```text
GroupNorm(num_groups=8 or 16)
```

---

## 17.2 Keep Decoder Lightweight

The decoder should refine the localization output, not dominate the paper's novelty.  
Avoid building an excessively large segmentation head.

Recommended philosophy:

```text
Strong encoder + strong middle anomaly reasoning + lightweight decoder
```

---

## 17.3 Coordinate Alignment

Make sure all low-resolution maps follow the same spatial grid:

```text
encoder_feature_t
query_feature
accum_error_map
coarse_mask_logits
boundary_1_16
```

All should be aligned at:

```text
H/16 × W/16
```

before concatenation.

---

## 17.4 Input Resolution Constraints

If using patch stride 16, choose:

```text
H and W divisible by 16
```

Examples:

```text
384, 512, 640
```

---

# 18. Minimal Implementation Path

If the implementation assistant needs a simpler first version, implement in this order:

## Version 1
- DINOv3 last-layer patch tokens only
- Simple 1×1 projection to 256 channels
- Decoder with:
  - encoder feature
  - coarse mask logits
  - bilinear upsampling
- No boundary branch

## Version 2
- Multi-layer token fusion
- High-frequency boundary branch
- Progressive 1/16 → 1/8 → 1/4 → 1 decoder

## Version 3
- Add optional query feature fusion
- Add edge head
- Add temporal boundary cues

---

# 19. Recommended First Full Version

A strong but still feasible version is:

```text
Encoder:
- DINOv3 multi-layer token extraction
- 4 selected layers
- each projected to 256 channels
- concatenate and fuse with 1×1 + 3×3 conv

Decoder:
- high-frequency branch using RGB + Laplacian + Sobel magnitude
- fuse query feature, encoder feature, accumulation map, coarse mask, boundary_1_16
- progressive upsampling with boundary_1_8 and boundary_1_4
- full-resolution binary mask head
- optional edge prediction head
```

---

# 20. Expected Tensor Summary

| Name | Shape |
|---|---|
| `video` | `[B, T, 3, H, W]` |
| `frames_flat` | `[B*T, 3, H, W]` |
| `patch_tokens_l` | `[B*T, N, C_l]` |
| `patch_map_l` | `[B*T, C_l, h, w]` |
| `encoder_features` | `[B, T, 256, h, w]` |
| `encoder_feature_t` | `[B, 256, h, w]` |
| `query_feature` | `[B, 256, h, w]` |
| `accum_error_map` | `[B, 1, h, w]` |
| `coarse_mask_logits` | `[B, 1, h, w]` |
| `boundary_1_16` | `[B, 192, h, w]` |
| `boundary_1_8` | `[B, 128, H/8, W/8]` |
| `boundary_1_4` | `[B, 96, H/4, W/4]` |
| `mask_logits` | `[B, 1, H, W]` |
| `edge_logits` | `[B, 1, H, W]` |

---

# 21. Suggested File / Class Organization

```text
models/
├── encoder/
│   ├── dino_multilayer_encoder.py
│   └── patch_feature_fusion.py
│
├── decoder/
│   ├── high_frequency_branch.py
│   ├── boundary_decoder.py
│   └── edge_head.py
│
└── utils/
    ├── image_filters.py
    └── shape_utils.py
```

---

# 22. Implementation Checklist

Before training, verify:

- [ ] DINO output token count matches `h * w`
- [ ] CLS token is removed if present
- [ ] all selected layer maps reshape correctly
- [ ] encoder output is `[B, T, 256, h, w]`
- [ ] decoder fusion inputs have aligned spatial sizes
- [ ] mask logits restore to original `H × W`
- [ ] high-frequency branch outputs all three scales
- [ ] edge head output shape matches GT edge shape
- [ ] model works with `H, W` divisible by 16

---

# 23. Recommended Prompt to Send to a Coding AI

You can give the coding AI the following instruction:

> Implement the encoder and decoder described in this Markdown file using PyTorch.  
> The encoder should wrap a DINOv3 backbone, extract patch tokens from 4 configurable transformer layers, project each layer to 256 channels, reshape tokens to patch grids, concatenate them, and fuse them into `[B, T, 256, H/16, W/16]`.  
> The decoder should include a high-frequency boundary branch that takes RGB + Laplacian + Sobel magnitude cues and produces 1/4, 1/8, and 1/16 features.  
> The final frequency-guided boundary decoder should fuse encoder features, query features, accumulation maps, coarse mask logits, and boundary features at 1/16 scale, then progressively upsample to full resolution with skip fusion at 1/8 and 1/4 scale, outputting full-resolution mask logits and an optional edge head.
