# DINOv3 ViT-L/16 + LoRA 训练说明

> 脚本：`train_val_test_dinov3_lora.py`　｜　配置：`configs/dinov3_vitl16_lora.yml`

---

## 一、模型架构总览

本项目提供 **三种** 模型变体，通过配置文件组合切换：

| 变体 | `use_dpt_fpn` | `use_tfcu_adapter` | 输入形状 | 说明 |
|------|:---:|:---:|---|---|
| **A — 单层 SimpleSegHead** | `false` | `false` | `[B,T,3,H,W]` | DINOv3-IML 风格，单层特征 + 3-conv 头 |
| **B — 多层 DPT+FPN** | `true` | `false` | `[B,T,3,H,W]` | 4 层 DINO 特征融合为 F32，再构建 FPN Decoder 输入 |
| **C — DPT+FPN + TFCU Adapter** | `true` | `true` | `[B,N,T,3,H,W]` | 变体 B + F32 时序 adapter（视频 inpainting 检测） |

下文按模块逐一说明。

---

## 二、编码器 (Encoder)

### 变体 A：单层编码器（`use_dpt_fpn: false`）

```text
clip [B, T, 3, 512, 512]
  → ImageNet 归一化
  → DINOv3 ViT-L/24 backbone，仅提取第 24 层 (block 23) patch tokens
  → [B, T, 1024, 32, 32]
```

### 变体 B / C：多层 DPT 编码器（`use_dpt_fpn: true`）

```text
clip [B, T, 3, 512, 512]          (变体 B)
  or  [B, N, T, 3, 512, 512]      (变体 C, 内部 flatten 为 [BNT, 3, 512, 512])
  → ImageNet 归一化
  → DINOv3 ViT-L/24 backbone，提取 4 层 patch tokens
  → {5: [*,1024,32,32], 11: [...], 17: [...], 23: [...]}
```

参数：`patch_size=16`, `embed_dim=1024`, `depth=24`, `num_heads=16`。

可配置提取层：
```yaml
extract_layers: "5,11,17,23"    # 0-based block 索引；空间分辨率同为 32×32
```

---

## 三、解码器 (Decoder)

### 变体 A：SimpleSegHead（DINOv3-IML 风格）

参考 [DINOv3-IML](https://github.com/Irennnne/DINOv3-IML) (Irennnne et al., 2026)：

```text
[B, 1024,  32,  32]  → Conv3×3(1024→512) → BN → ReLU
[B,  512,  32,  32]  → Conv3×3( 512→256) → BN → ReLU
[B,  256,  32,  32]  → Conv1×1( 256→  1)
                      → bilinear 上采样到 512×512
[B,    1, 512, 512]  mask logits
```

| | SimpleSegHead |
|---|---|
| 卷积层数 | 2×Conv3×3 + 1×Conv1×1 |
| 上采样方式 | 1 次 16× bilinear |
| 归一化 | BatchNorm |
| 激活 | ReLU |
| 参数量 | ~1.6M |

### 变体 B / C：Fused32 Pyramid Neck + FPN Decoder

默认结构使用 `neck_variant: fused32_pyramid`。DINOv3 ViT 的多个 block 输出都是同一 token 分辨率，因此先在 32×32 上融合语义深度，再从 enhanced F32 构建 decoder pyramid：

```text
DINO block 5   → [*,1024,32,32]
DINO block 11  → [*,1024,32,32]
DINO block 17  → [*,1024,32,32]
DINO block 23  → [*,1024,32,32]
        ↓ 1×1 projection 到 neck_channels
        ↓ concat + fusion
F32 [*,256,32,32]
        ↓
P5 = downsample(F32) → [*,256, 16, 16]
P4 = conv(F32)       → [*,256, 32, 32]
P3 = upsample(F32)   → [*,256, 64, 64]
P2 = upsample(F32)   → [*,256,128,128]

FPN top-down fusion:
  P5 → F5
  P4 + upsample(F5) → F4
  P3 + upsample(F4) → F3
  P2 + upsample(F3) → F2

F2 (128×128) → upsample → Conv(256→128) @ 256×256
             → upsample → Conv(128→64)  @ 512×512
             → Conv(64→32) → Conv2d(32→1)  →  [*,1,512,512]
```

旧结构仍可通过 `neck_variant: dpt_reassemble` 回退。旧结构里 `DPTReassembleNeck` 会把不同 block 直接重组为 P2/P3/P4/P5，但这不再是推荐默认。
每个 FPN ConvBlock = `Conv3×3 → GroupNorm → GELU`。

### 变体 D：Semantic-Anchor MFCE + P4 Gated TFCU

新增结构使用 `neck_variant: semantic_anchor_mfce`。它同样承认 DINOv3 ViT 多个 block 都是 32×32 token map，但不再先 concat 成 F32 pyramid，而是用 MFCE-style spatial layer attention 在 1/16 分辨率自适应融合出 P4 semantic anchor：

```text
DINO block 5   → [*,1024,32,32]
DINO block 11  → [*,1024,32,32]
DINO block 17  → [*,1024,32,32]
DINO block 23  → [*,1024,32,32]
        ↓ per-layer 1×1 projection 到 neck_channels
        ↓ per-layer spatial score
        ↓ softmax over layer dimension
P4_sem [*,256,32,32]
        ↓ LightASPP context
P4 [*,256,32,32]
        ↓ gated TFCU residual, gate init = -3.0
enhanced P4
        ↓ top-down decoder
P3 [*,256,64,64] → P2 [*,128,128,128] → P1 [*,64,256,256]
        ↓ upsample
logits [*,1,512,512]
```

这个变体没有主 P5 分支；P5 不再作为 temporal/prompt 主输入。可选 `use_detail_stem=true` 时，RGB detail stem 只注入 P3/P2/P1，用来补充局部边界纹理，默认关闭方便先做结构消融。

### 输出

| 变体 | 输出形状 |
|------|---------|
| A / B | `[B, T, 1, 512, 512]` logits（未经 sigmoid） |
| C / D | `[B, N, T, 1, 512, 512]` logits（未经 sigmoid） |

---

## 四、TFCU-Inpaint Adapter（变体 C 专属）

变体 C 默认在 fused F32 层级插入一个轻量级时序模块，用于视频 inpainting 检测。后续 MaskPromptEncoder 也应接在同一个 F32 插入点，而不是 P4/P5。

### 核心原则

- **不重写 backbone** — 复用现有 DINOv3+LoRA+DPT-FPN，以 residual 方式注入。
- **causal memory** — 当前 clip 只能看到过去的 clip，绝不泄露未来信息。
- **可退化** — 残差系数 α 初始化为 0，模型等价于原始单帧 backbone，避免训练初期崩溃。
- **统一插入点** — temporal/prompt 作用于 F32；P2/P3/P4/P5 由 enhanced F32 统一生成。

### 数据流

```text
video [B, N, T, 3, 512, 512]
  → flatten  [B*N*T, 3, 512, 512]
  → DINOv3 + LoRA → ViTMultiLayerFusionPyramidNeck
  → F32 [BNT, 256, 32, 32]

  ┌─ TFCUInpaintAdapter(F32, B, N, T) ────────────────────┐
  │  1. LocalTemporalDifferenceModule                      │
  │     每帧与前一帧做差 → 门控融合 → 捕捉纹理/边界不一致     │
  │                                                        │
  │  2. InpaintMemoryAttention (causal, cross-clip)        │
  │     clip 0 → mem_0                                     │
  │     clip 1 ← attend(mem_0)  → mem_1                    │
  │     clip 2 ← attend(mem_0, mem_1) → mem_2              │
  │     ...                                                │
  │                                                        │
  │  3. F32_out = F32 + α * temporal_feat  (α 初始 = 0)    │
  └────────────────────────────────────────────────────────┘

  → build P2/P3/P4/P5 from enhanced F32
  → FPNDecoder(P2, P3, P4, P5)
  → logits [B, N, T, 1, 512, 512]
```

### 模块明细

| 模块 | 文件 | 说明 |
|------|------|------|
| `LocalTemporalDifferenceModule` | `my_model/temporal/local_temporal_difference.py` | 计算帧间差分 (x_t − x_{t-1}) → 门控融合 |
| `InpaintMemoryAttention` | `my_model/temporal/memory_attention.py` | 前向历史 memory 多头交叉注意力 + FFN |
| `TFCUInpaintAdapter` | `my_model/temporal/temporal_adapter.py` | 总入口：local diff → causal memory → residual F32 |
| `MaskPromptEncoder` | `my_model/temporal/mask_prompt_encoder.py` | mask/boundary prompt 编码（备用，后续接入 F32） |
| `VideoInpaintTFCU` | `my_model/video_inpaint_tfcu.py` | 顶层 wrapper，封装 base backbone + adapter |

### TFCU 关键配置

```yaml
use_tfcu_adapter: true       # 启用 TFCU adapter
neck_variant: fused32_pyramid
temporal_insert_level: F32
use_image_stem_skip: false   # 可选 RGB stem skip，默认关闭
num_clips: 4                 # 每个视频采样的 clip 数 (N)
num_frames: 4                # 每个 clip 内的帧数 (T)
clip_stride: 1               # clip 内部帧间隔
memory_len: 4                # 保留最近 K 个 clip 的历史 memory
use_memory: true             # 启用 cross-clip memory attention
detach_memory: true          # memory 前 detach（避免跨 clip 梯度爆炸）
use_spatial_pool: false      # memory attention 前 pool 空间维度（OOM 时的备选）
```

---

## 五、LoRA 配置

LoRA 通过 HuggingFace `peft` 库注入，原始 DINOv3 权重冻结。

| 参数 | 当前值 | 说明 |
|---|---|---|
| `use_lora` | `true` | 启用 LoRA |
| `lora_rank` | `32` | 低秩矩阵维度 r |
| `lora_alpha` | `64` | 缩放系数，scaling = α/r = **2.0** |
| `lora_dropout` | `0.1` | LoRA dropout 正则化 |
| `lora_targets` | `attn.qkv,attn.proj,mlp.fc1,mlp.fc2` | 同时适配 Attention 和 MLP 层 |
| `lora_layers` | `all` | 注入所有 24 个 block（也可指定 `"5-23"` 仅深层） |

前向公式：

```text
y = W·x + (α/r) · B(A·x)        ← A ∈ R^(r×d), B ∈ R^(d×r)
```

---

## 六、损失函数

### 全量支持

| 损失 | YAML key | 说明 |
|------|----------|------|
| **Weighted BCE** | `wbce` | 自动正样本加权 `pos_weight = min(neg/pos, max)` |
| **Dice** | `dice` | 软 Dice 损失 |
| **Focal** | `focal` | α=0.25, γ=2.0，抑制简单样本 |
| **IoU** | `iou` | 直接优化 IoU |
| **BCE** | `bce` | 原生 `binary_cross_entropy_with_logits` |
| **Tversky** | `tversky` | Dice 广义形式，α 控 FP、β 控 FN |
| **Edge** | `edge` | 形态学边缘加权 BCE |
| **Boundary** | `boundary` | 🆕 Sobel 边界 L1（预测边界 vs GT 边界） |
| **Temporal Delta** | `temporal_delta` | 🆕 帧间预测差分 L1（时序一致性） |

### 变体 B 推荐配置

```yaml
loss:
  dice:        {weight: 1.0, smooth: 1.0e-6}
  bce:         {weight: 0.5}
  tversky:     {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
```

总损失 = `1.0×Dice + 0.5×BCE + 0.2×Tversky`

### 变体 C 推荐配置

```yaml
loss:
  dice:           {weight: 1.0, smooth: 1.0e-6}
  bce:            {weight: 0.5}
  tversky:        {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
  boundary:       {weight: 0.2}
  temporal_delta: {weight: 0.1}
```

`temporal_delta` 在 `T<=1` 时自动返回 0，兼容单帧模式。

---

## 七、训练配置

### 变体 B（DPT+FPN，单帧）

| 参数 | 值 | 说明 |
|---|---|---|
| `num_frames` | 1 | 单帧训练 |
| `batch_size` | 16 | |
| `learning_rate` | 3e-4 | Adam，统一学习率 |
| `weight_decay` | 1e-4 | |
| `scheduler` | `cosine` | warmup + cosine decay |
| `warmup_epochs` | 20 | |
| `min_lr` | 1e-6 | |
| `n_epochs` | 2000 | |
| `validate_every` | 20 | |
| `amp` | true | 混合精度 |

### 变体 C（TFCU Adapter，多帧多 clip）

| 参数 | 值 | 说明 |
|---|---|---|
| `num_clips` | 4 | 每视频 clip 数 |
| `num_frames` | 4 | 每 clip 帧数 |
| `batch_size` | 1 | TFCU 显存需求高，从小开始 |
| `grad_accum_steps` | 8 | 等效 batch = 1×8 = 8 |
| `learning_rate` | 1e-4 | Adam 基准学习率 |
| `lr_temporal` | 1e-4 | 🆕 temporal adapter 学习率 |
| `lr_decoder` | 1e-4 | 🆕 decoder + neck 学习率 |
| `lr_lora` | 1e-5 | 🆕 LoRA 学习率（应低于 decoder） |
| `weight_decay` | 1e-4 | |
| `scheduler` | `cosine` | |
| `n_epochs` | 2000 | |

> TFCU 模式下 optimizer 使用 **分离学习率**：temporal adapter / decoder / LoRA 各不同。

---

## 八、可训练参数

### 变体 B（DPT+FPN + LoRA）

| 组件 | 参数量 | 状态 |
|---|---|---|
| DINOv3 backbone | ~304M | ❄️ 冻结 |
| LoRA (attn+mlp) | ~6M | 🔥 训练 |
| ViTMultiLayerFusionPyramidNeck | 依 layers/channel 而定 | 🔥 训练 |
| FPNDecoder | ~1.3M | 🔥 训练 |
| **总计** | **~11M** | |

### 变体 A（SimpleSegHead + LoRA）

| 组件 | 参数量 | 状态 |
|---|---|---|
| DINOv3 backbone | ~304M | ❄️ 冻结 |
| LoRA (attn+mlp) | ~6M | 🔥 训练 |
| SimpleSegHead | ~1.6M | 🔥 训练 |
| **总计** | **~7.6M** | |

### 变体 C（DPT+FPN + TFCU + LoRA）

| 组件 | 参数量 | 状态 |
|---|---|---|
| DINOv3 backbone | ~304M | ❄️ 冻结 |
| LoRA (attn+mlp) | ~6M | 🔥 训练 |
| ViTMultiLayerFusionPyramidNeck | 依 layers/channel 而定 | 🔥 训练 |
| FPNDecoder | ~1.3M | 🔥 训练 |
| TFCUInpaintAdapter | ~2.2M | 🆕 🔥 训练 |
| **总计** | **~13.2M** | |

---

## 九、使用命令

### 变体 B：DPT+FPN 训练

```bash
cd /home/wzk/Exp/My_model

python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --use_lora true \
  --gpu_id 0 \
  --save_dir runs/dinov3_vitl16_lora_dpt_fpn
```

### 变体 C：TFCU-Inpaint 训练

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type train \
  --gpu_id 0 \
  --batch_size 1 \
  --grad_accum_steps 8
```

### 多卡训练 / 分布式验证

```bash
# 8 卡 DDP 训练。每卡 batch_size=1，全局等效 batch = 8 × 1 × grad_accum_steps。
torchrun --nproc_per_node=8 train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 1

# 多卡验证/测试也会按 rank 切分样本并 all-reduce 指标。
torchrun --nproc_per_node=8 train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type test \
  --checkpoint runs/dinov3_vitl16_tfcu_inpaint_new_decoder/best_iou.pt
```

### 验证 / 测试

```bash
# 单集测试
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type val \
  --checkpoint runs/dinov3_vitl16_lora_dpt_fpn/best_iou.pt

# 全量测试套件（DVI_20 + CPNET_20 + OPN_20）
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type test \
  --checkpoint runs/dinov3_vitl16_lora_dpt_fpn/best_iou.pt
```

### 常用 CLI 覆盖

```bash
--gpu_id 1                    # 选择 GPU
--dist_backend nccl           # torchrun 多卡后端
--ddp_find_unused_parameters false
--batch_size 8                # 调整 batch size
--grad_accum_steps 2          # 梯度累积（等效 batch = batch_size × grad_accum_steps）
--eval_frame_chunk 5          # 测试时分块前向，降低显存
--threshold 0.3               # 二值化阈值（默认 0.5）
--num_clips 2                 # TFCU：减少 clip 数以降显存
--memory_len 2                # TFCU：减少 memory 长度以降低计算量
--use_memory false            # TFCU：关闭 memory attention（A2 消融）
```

### 形状测试

```bash
python debug/test_tfcu_shapes.py
```

---

## 十、Ablation 实验顺序

为验证 TFCU-Adapter 各模块有效性，建议按以下顺序跑实验：

| 实验 | 配置 | 说明 |
|------|------|------|
| **A0** | 变体 B, `num_frames=1` | 原始单帧 DPT+FPN baseline |
| **A1** | 变体 B, `num_frames=4` | 多帧输入但不加 temporal adapter |
| **A2** | 变体 C, `use_memory=false` | A1 + LocalTemporalDifferenceModule |
| **A3** | 变体 C, `use_memory=true` | A2 + forward MemoryAttention |
| **A4** | A3 + `boundary` + `temporal_delta` loss | A3 + 边界+时序 loss |

关键对比：

| 对比 | 验证问题 |
|------|---------|
| A0 vs A2 | 连续帧差异是否有效？ |
| A2 vs A3 | 历史 memory 是否有效？ |
| A3 vs A4 | 边界和时序 loss 是否提升稳定性？ |

---

## 十一、显存建议

DINOv3 ViT-L/16 在 512×512 下显存较高：

### 变体 B（单帧）

```bash
# 24GB GPU 安全配置
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 16
```

### 变体 C（TFCU，多帧多 clip）

```bash
# 24GB GPU，从小配置开始
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --num_clips 2 \
  --num_frames 4 \
  --memory_len 2

# 防 OOM 环境变量
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_val_test_dinov3_lora.py --config ... --batch_size 1
```

### OOM 降级方案

按优先级依次尝试：

1. `batch_size=1, num_clips=2, num_frames=4, memory_len=2`
2. `use_spatial_pool=true`（memory attention 前 pool 到 16×16）
3. 冻结 LoRA（`use_lora=false`，后续再解冻）
4. 使用 `gradient_checkpointing`

---

## 十二、架构切换与 Checkpoint 兼容

### 配置切换

```yaml
# 变体 A — 单层 SimpleSegHead（DINOv3-IML 风格）
use_dpt_fpn: false
use_tfcu_adapter: false

# 变体 B — 多层 DPT+FPN
use_dpt_fpn: true
use_tfcu_adapter: false
neck_variant: fused32_pyramid
extract_layers: "5,11,17,23"

# 变体 C — DPT+FPN + TFCU Adapter
use_dpt_fpn: true
use_tfcu_adapter: true
neck_variant: fused32_pyramid
temporal_insert_level: F32
num_clips: 4
num_frames: 4

# 变体 D — Semantic-Anchor MFCE + P4 Gated TFCU
use_dpt_fpn: true
use_tfcu_adapter: true
neck_variant: semantic_anchor_mfce
temporal_insert_level: P4
semantic_aspp_rates: "1,2,4,8"
use_detail_stem: false
p4_gate_init: -3.0
num_clips: 2
num_frames: 4
```

### Checkpoint 兼容性

| 源 → 目标 | 兼容？ | 说明 |
|-----------|:---:|------|
| A → B | ❌ | encoder/decoder 结构不同，neck 随机初始化 |
| B → C | ⚠️ 部分 | backbone 权重复用，temporal adapter 随机初始化 |
| C → C | ✅ | 完整恢复（含 temporal adapter 状态） |
| B → B | ✅ | 完整恢复 |

加载时使用 `strict=False`，不匹配的 key 自动跳过并打印 warning。

---

## 十三、文件结构

```text
My_model/
├── train_val_test_dinov3_lora.py          # 主训练脚本（支持变体 A/B/C）
├── train_val_test_convnext_lora.py        # ConvNeXt-Tiny 对比脚本
│
├── configs/
│   ├── dinov3_vitl16_lora.yml             # 变体 A/B 配置
│   └── dinov3_vitl16_lora_tfcu_inpaint.yml # 变体 C 配置
│
├── my_model/
│   ├── __init__.py                        # 模块导出
│   ├── losses.py                          # 所有损失函数（含 Boundary / TemporalDelta）
│   ├── metrics.py                         # IoU/F1 等评估指标
│   ├── dinov3_dpt_fpn.py                  # Fused32/DPT necks / FPNDecoder
│   ├── video_inpaint_tfcu.py              # VideoInpaintTFCU wrapper
│   └── temporal/
│       ├── __init__.py
│       ├── local_temporal_difference.py    # LocalTemporalDifferenceModule
│       ├── memory_attention.py            # InpaintMemoryAttention
│       ├── mask_prompt_encoder.py         # MaskPromptEncoder（备用）
│       └── temporal_adapter.py            # TFCUInpaintAdapter
│
├── zzz_dataset_toolkit/
│   ├── dataset.py                         # VideoInpaintingDataset（支持 num_clips）
│   └── transforms.py
│
├── dinov3/                                # DINOv3 本地 repo
├── flist/                                 # 数据样本列表 (.npy)
├── debug/
│   └── test_tfcu_shapes.py                # TFCU 形状单元测试
└── runs/                                  # 训练输出
```
