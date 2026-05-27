# DINOv3 ViT-L/16 + LoRA 训练说明

> 脚本：`train_val_test_dinov3_lora.py`　｜　配置：`configs/dinov3_vitl16_lora.yml`

---

## 一、模型结构

支持两种解码器架构，通过 `use_dpt_fpn` 配置切换。

### 编码器

#### 模式 A：单层（`use_dpt_fpn: false`）

```text
clip [B, T, 3, 512, 512]
  → ImageNet 归一化
  → DINOv3 ViT-L/24 backbone，仅提取第 24 层 (block 23) patch tokens
  → [B, T, 1024, 32, 32]
```

#### 模式 B：多层 DPT（`use_dpt_fpn: true`，当前启用）

```text
clip [B, T, 3, 512, 512]
  → ImageNet 归一化
  → DINOv3 ViT-L/24 backbone，提取 4 层 (blocks 5, 11, 17, 23) patch tokens
  → {5: [B,T,1024,32,32], 11: [...], 17: [...], 23: [...]}
```

参数：`patch_size=16`, `embed_dim=1024`, `depth=24`, `num_heads=16`

### 解码器

#### 模式 A：SimpleSegHead（`use_dpt_fpn: false`，DINOv3-IML 风格）

参考 [DINOv3-IML](https://github.com/Irennnne/DINOv3-IML) (Irennnne et al., 2026) 的解码器设计：

```text
[B, 1024,  32,  32]  → Conv3×3(1024→512) → BN → ReLU    1/16 尺度
[B,  512,  32,  32]  → Conv3×3( 512→256) → BN → ReLU
[B,  256,  32,  32]  → Conv1×1( 256→  1)
                      → bilinear 上采样到 512×512
[B,    1, 512, 512]  mask logits
```

仅 3 个卷积层（2×Conv3×3 + 1×Conv1×1），在 1/16 尺度上完成所有计算，最后一次性 bilinear 上采样。使用 BatchNorm + ReLU（Kaiming 初始化）。与 ProgressiveDecoder 的对比：

| | ProgressiveDecoder（旧） | SimpleSegHead（新） |
|---|---|---|
| 卷积层数 | 5×Conv3×3 + 1×Conv1×1 | 2×Conv3×3 + 1×Conv1×1 |
| 上采样方式 | 4 次渐进 2× bilinear | 1 次 16× bilinear |
| 归一化 | GroupNorm | BatchNorm |
| 激活 | GELU | ReLU |
| 可训练参数 | ~0.3M | ~1.6M |

#### 模式 B：DPT Reassemble Neck + FPN Decoder（`use_dpt_fpn: true`，当前启用）

```text
DINO block 5  → ReassembleBlock("x4")   → P2 [N,256,128,128]
DINO block 11 → ReassembleBlock("x2")   → P3 [N,256, 64, 64]
DINO block 17 → ReassembleBlock("x1")   → P4 [N,256, 32, 32]
DINO block 23 → ReassembleBlock("down2") → P5 [N,256, 16, 16]

FPN top-down fusion:
  P5 → F5
  P4 + upsample(F5) → F4
  P3 + upsample(F4) → F3
  P2 + upsample(F3) → F2

F2 (128×128) → upsample → Conv(256→128) @ 256×256
             → upsample → Conv(128→64)  @ 512×512
             → Conv(64→32) → Conv2d(32→1)  →  [N,1,512,512]
```

每个 ReassembleBlock = `1×1 Conv → GN → GELU → (upsample+conv 组合)`
每个 FPN ConvBlock = `Conv3×3 → GroupNorm → GELU`

### 输出

```text
[B, T, 1, 512, 512]   logits（未经 sigmoid）
```

---

## 二、LoRA 配置

LoRA 通过 HuggingFace `peft` 库注入，原始 DINOv3 权重冻结。

| 参数 | 当前值 | 说明 |
|---|---|---|
| `use_lora` | `true` | 启用 LoRA |
| `lora_rank` | `32` | 低秩矩阵维度 r |
| `lora_alpha` | `64` | 缩放系数，scaling = α/r = **2.0** |
| `lora_dropout` | `0.1` | LoRA dropout 正则化 |
| `lora_targets` | `attn.qkv, attn.proj, mlp.fc1, mlp.fc2` | 同时适配 Attention 和 MLP 层 |

前向公式：

```text
y = W·x + (α/r) · B(A·x)        ← A ∈ R^(r×d), B ∈ R^(d×r)
```

---

## 三、损失函数

### 当前启用

| 损失 | Weight | 参数 |
|---|---|---|
| **Dice** | 1.0 | smooth=1e-6 |
| **BCE** | 0.5 | 原生 `binary_cross_entropy_with_logits` |
| **Tversky** | 0.2 | α=0.3, β=0.7（偏重惩罚 FN，提升 recall） |

总损失 = `1.0 × Dice + 0.5 × BCE + 0.2 × Tversky`

### 可选用（当前注释）

| 损失 | YAML key | 说明 |
|---|---|---|
| **Weighted BCE** | `wbce` | 自动正样本加权 `pos_weight = min(neg/pos, max)` |
| **Focal** | `focal` | α=0.25, γ=2.0，抑制简单样本 |
| **IoU** | `iou` | 直接优化 IoU |
| **Tversky** | `tversky` | Dice 的广义形式，α 控制 FP 惩罚、β 控制 FN 惩罚 |

配置示例（自由组合，注释即禁用）：

```yaml
loss:
  # wbce:      {weight: 1.0, max_pos_weight: 10.0}
  dice:        {weight: 1.0, smooth: 1.0e-6}
  # focal:     {weight: 1.0, alpha: 0.25, gamma: 2.0}
  # iou:       {weight: 1.0, smooth: 1.0e-6}
  bce:         {weight: 0.5}
  tversky:     {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
```

---

## 四、架构配置

| 参数 | 当前值 | 说明 |
|---|---|---|
| `use_dpt_fpn` | `true` | `true`=多层+DPT+FPN，`false`=单层+ProgressiveDecoder |
| `extract_layers` | `5,11,17,23` | DINOv3 block 索引（0-based），仅 `use_dpt_fpn=true` 时生效 |
| `neck_channels` | `256` | DPT Reassemble 输出 / FPN 通道数 |

## 五、训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| `num_frames` | 1 | 单帧训练 |
| `batch_size` | 16 | |
| `learning_rate` | 1e-4 | Adam |
| `weight_decay` | 1e-4 | L2 正则化 |
| `scheduler` | `plateau` | 基于 val_iou，patience=3，factor=0.5 |
| `min_lr` | 1e-6 | |
| `n_epochs` | 1000 | |
| `validate_every` | 10 | |
| `amp` | true | 混合精度训练 |

---

## 六、使用命令

### 训练

```bash
cd /home/wzk/Exp/My_model

python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --use_lora true \
  --save_dir runs/dinov3_vitl16_lora_dpt_fpn \
  --visualization_dir runs/dinov3_vitl16_lora_dpt_fpn/vis
```

### 验证 / 测试

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type test \
  --checkpoint runs/dinov3_vitl16_lora_dpt_fpn/best_iou.pt
```

### 常用 CLI 覆盖

```bash
--gpu_id 1                    # 选择 GPU
--batch_size 8                # 调整 batch size
--grad_accum_steps 2          # 梯度累积（等效 batch = batch_size × grad_accum_steps）
--eval_frame_chunk 1          # 测试时分块前向，降低显存
--threshold 0.3               # 二值化阈值（默认 0.5）
```

---

## 七、可训练参数

| 组件 | 参数量 | 状态 |
|---|---|---|
| DINOv3 backbone | ~304M | ❄️ 冻结 |
| LoRA (attn+mlp) | ~6M | 🔥 训练 |
| DPTReassembleNeck | ~3.7M | 🔥 训练（仅 DPT+FPN） |
| FPNDecoder | ~1.3M | 🔥 训练（仅 DPT+FPN） |
| SimpleSegHead | ~1.6M | 🔥 训练（仅单层模式） |
| **总计（DPT+FPN）** | **~11M** | |
| **总计（SimpleSegHead）** | **~7.6M** | |

## 八、架构切换

在单层 SimpleSegHead（DINOv3-IML 风格）和多层 DPT+FPN 之间切换：

```yaml
# DPT+FPN（多层特征 + Neck + FPN 解码器）
use_dpt_fpn: true
extract_layers: "5,11,17,23"

# SimpleSegHead（单层特征 + 3-conv 头，DINOv3-IML 风格）
use_dpt_fpn: false
neck_channels: 256

# 单层 ProgressiveDecoder（轻量、显存更低）
use_dpt_fpn: false
```

> 两种架构的 checkpoint 不互通。切换后旧 checkpoint 加载时自动跳过不匹配 key（`strict=False`），neck/decoder 权重随机初始化。

## 九、显存建议

DINOv3 ViT-L/16 在 512×512 下显存较高，24GB GPU 建议：

```bash
# 安全起步配置
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 16

# 防 OOM 环境变量
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_val_test_dinov3_lora.py --config ... --batch_size 1 --grad_accum_steps 16
```
