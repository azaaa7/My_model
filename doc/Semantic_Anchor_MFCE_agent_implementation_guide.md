# Semantic-Anchor MFCE + P4 Gated TFCU 实现指导文档

面向 coding agent 的结构设计、接口约束、实现步骤与实验验收标准

## 1. 背景与目标

当前要实现的不是普通 FPN，也不是把 DINOv3 的不同层硬映射成 P1/P2/P3/P4/P5。正确方向是：把 DINOv3 ViT 的多层 1/16 token feature 先在原生分辨率上自适应融合，形成 P4 semantic anchor；然后在 P4 上做上下文增强和时序注入，再用 top-down decoder 恢复到 P3/P2/P1 和最终 mask。

目标结构：**Semantic-Anchor MFCE + P4 Gated TFCU**。

## 2. 核心原则

1. DINOv3 ViT-L/16 在 512×512 输入下，L5/L11/L17/L23 都是 32×32，即 1/16。
2. layer 23 是最强语义层，应保留在 1/16 作为 P4 semantic anchor，不应再下采样成 1/32 主 P5。
3. 多层融合采用 MFCE-style layer attention，而不是简单 concat。
4. TFCU 优先插入 P4；P3 TFCU 作为第二阶段可选消融。
5. 如果需要更大感受野，用 ASPP 或 global context 回注 P4，不做主 P5。
6. P3/P2/P1 由 top-down decoder 和 detail stem 恢复。

## 3. 总体结构

```text
Input video clip: [B, T, 3, 512, 512]
    ↓
DINOv3 ViT-L/16 frozen backbone + LoRA
    ↓
L5, L11, L17, L23: [B, T, 1024, 32, 32]
    ↓
Semantic-Anchor MFCE:
  1×1 projection per layer → [B, T, 256, 32, 32]
  spatial layer attention over 4 layers
  weighted sum → P4_sem
    ↓
ASPP at P4
    ↓
P4 = P4 + gate4 * TFCU(P4)
    ↓
P4 → P3 → P2 → P1 → logits
    ↓
mask
```

## 4. 尺度约定

| Level | Stride | Resolution at 512 | Source | TFCU |
|---|---:|---:|---|---|
| P1 | 1/2 | 256×256 | decoder + detail stem | No |
| P2 | 1/4 | 128×128 | decoder + detail stem | No |
| P3 | 1/8 | 64×64 | upsampled from P4 | Optional later |
| P4 | 1/16 | 32×32 | fused DINOv3 layers | Yes |
| P5 | 1/32 | 16×16 | not a main branch | No |

## 5. Agent direct task

```text
Implement a new model variant named semantic_anchor_mfce.

Requirements:
1. Extract DINOv3 intermediate layers [5, 11, 17, 23]. Treat all as native 1/16 token maps.
2. Implement SemanticAnchorMFCE:
   - project each layer from 1024 to 256 channels;
   - compute one spatial score map per layer;
   - softmax over layer dimension;
   - weighted-sum projected features into P4_sem;
   - return layer attention maps.
3. Add LightASPP at P4. Do not create a main P5 branch.
4. Add P4 gated TFCU:
   P4 = P4 + sigmoid(gate4) * TFCU(P4), gate4 init = -3.0.
5. Build top-down decoder P4→P3→P2→P1→logits.
6. Add optional DetailStem from RGB frames; inject only into P3/P2/P1.
7. Add unit tests for shapes, no-main-P5, attention softmax, gate init, and gradients.
8. Add experiment config with num_frames=4, augment_prob=0.75, full LoRA targets, validate_every=20.
```

## 6. 实验顺序

| Exp | Structure | Temporal | Detail | Purpose |
|---|---|---|---|---|
| E1 | SemanticAnchorMFCE + ASPP + decoder | None | Off | test fusion |
| E2 | same | P4 gated | Off | test P4 TFCU |
| E3 | same | P4 gated | On | test boundary/detail |
| E4 | same | P3+P4 gated | On | test local temporal |
| E5 | Temporal-aware MFCE | P4 gated | On | test temporal layer attention |

详细版请查看 DOCX。
