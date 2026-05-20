# DINOv3 ViT-L/16 对比实验说明

新增脚本：

- `train_val_test_dinov3_lora.py`
- `configs/dinov3_vitl16_lora.yml`

## 模型对应关系

你的权重文件：

```text
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

对应 DINOv3 官方 hub 模型：

```text
dinov3_vitl16
```

也就是 Hugging Face/官方命名里的 `facebook/dinov3-vitl16-pretrain-lvd1689m`。脚本按官方 DINOv3 ViT-L/16 结构加载：`patch_size=16`、`embed_dim=1024`、`depth=24`、`num_heads=16`。

## 准备文件

把权重放到 `My_model/` 下：

```text
My_model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

还需要 DINOv3 官方仓库代码。推荐放在 `My_model/dinov3` 或 `/home/wzk/Exp/dinov3`：

```bash
git clone https://github.com/facebookresearch/dinov3.git My_model/dinov3
```

如果已经放在其他位置，在配置里设置：

```yaml
dinov3_repo: "/path/to/dinov3"
```

或者设置环境变量：

```bash
export DINOV3_REPO=/path/to/dinov3
```

机器能访问 GitHub 时，也可以设置：

```yaml
allow_hub_download: true
```

此时脚本会尝试用 `torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", weights=...)`。

## GPU 选择

通过 `gpu_id` 配置选择使用哪张显卡：

```yaml
# configs/dinov3_vitl16_lora.yml
gpu_id: 1    # 使用 GPU 1，默认 0
```

也可以通过命令行覆盖：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --gpu_id 2
```

如果某张显卡被占用了就用 `CUDA_VISIBLE_DEVICES` 环境变量限制可见 GPU，然后设置 `gpu_id: 0`：

```bash
CUDA_VISIBLE_DEVICES=1 python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train
```

## 训练

默认只冻结 DINOv3 backbone 并训练 decoder：

```bash
cd /home/wzk/Exp/My_model
python train_val_test_dinov3_lora.py --config configs/dinov3_vitl16_lora.yml --type train
```

训练集、验证集和测试集都可以配置为任意个 `.npy` 来源，脚本会按顺序展开并拼接这些样本：

```yaml
train_samples:
  - "./flist/DAVIS-VI_tra_DVI_30.npy"
  - "./flist/DAVIS-VI_tra_CPNET_30.npy"
  - "./flist/DAVIS-VI_tra_OPN_30.npy"
val_samples:
  - "./flist/DAVIS-VI_val_DVI_20.npy"
  - "./flist/DAVIS-VI_val_CPNET_20.npy"
test_samples:
  - "./flist/DAVIS-VI_val_OPN_20.npy"
  - "./flist/DAVIS-VI_val_DVI_20.npy"
```

也可以通过命令行传逗号分隔的列表：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --train_samples ./flist/A.npy,./flist/B.npy
```

验证和测试同理：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type test \
  --test_samples ./flist/A.npy,./flist/B.npy \
  --checkpoint runs/dinov3_vitl16/best_iou.pt
```

当前训练数据处理顺序是：

```text
加载一个或多个 .npy -> 拼接成样本索引列表 -> DataLoader 按样本抽取 video/mask 目录
-> __getitem__ 内随机抽中心帧和邻近帧 -> 读图与 mask -> 对齐 mask 尺寸
-> 训练增强 -> resize 到 input_size -> 转 tensor
```

也就是说，代码不会先把数据复制 4 次再统一预处理；预处理发生在每次样本被抽到之后。

打开 LoRA：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --use_lora true \
  --save_dir runs/dinov3_vitl16_lora \
  --visualization_dir runs/dinov3_vitl16_lora/vis
```

LoRA 默认只加到官方 DINOv3 ViT block 的 attention 线性层：

```yaml
lora_targets: "attn.qkv,attn.proj"
```

这会让原始 DINOv3 权重保持冻结，只训练 decoder 和 LoRA 参数。

LoRA 使用 Hugging Face PEFT 的 `LoraConfig` 和 `inject_adapter_in_model` 注入，不使用项目内手写 wrapper。打开 LoRA 前请先在训练环境安装：

```bash
pip install peft
```

## 验证和测试

验证：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type val \
  --checkpoint runs/dinov3_vitl16/best_iou.pt
```

测试：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type test \
  --checkpoint runs/dinov3_vitl16/best_iou.pt
```

验证和测试会在 `visualization_dir` 下保存：

```text
input frame | probability map | binary prediction | ground truth
```

## 结构

输入仍然沿用本项目数据格式：

```text
[B, T, 3, H, W]
```

### 整体前向流程

```
clip [B, T, 3, H, W]
  ↓
DINOv3 ViT-L/16 backbone (冻结 + 可选 LoRA)
  → get_intermediate_layers(n=[17, 23], reshape=True)
  → layer_17, layer_23: [B*T, 1024, H/16, W/16]
  ↓
┌─ 主分支 (CoarseMaskHead) ──────────────────────────────┐
│  layer_23 → Conv3×3 1024→256 → BN → ReLU               │
│           → Conv3×3 256→128  → BN → ReLU               │
│           → Conv1×1 128→1    → coarse_logits [BT,1,h,w] │
│           → bilinear upsample → coarse_logits_up [BT,1,H,W]
└─────────────────────────────────────────────────────────┘
  ↓
┌─ 门控残差融合 (GatedResidualFusion) ────────────────────┐
│  layer_23 → LayerNorm → Conv1×1 1024→256 → main         │
│  layer_17 → LayerNorm → Conv1×1 1024→256 → aux           │
│  fused = main + sigmoid(alpha) * 0.1 * aux              │
└─────────────────────────────────────────────────────────┘
  ↓
┌─ 残差渐进解码器 (ResidualProgressiveDecoder) ───────────┐
│  concat(fused(256) + coarse_logits(1))                  │
│  → Conv1×1 257→128 → GN → GELU                         │
│  → bilinear×2 → Conv3×3 128→96  → GN → GELU            │
│  → bilinear×2 → Conv3×3 96→64   → GN → GELU            │
│  → bilinear×2 → Conv3×3 64→32   → GN → GELU            │
│  → bilinear×2 → Conv3×3 32→16 → GN → GELU → Conv1×1→1 │
│  → residual_logits [BT, 1, H, W]                        │
└─────────────────────────────────────────────────────────┘
  ↓
final_logits = coarse_logits_up + λ * residual_logits
  ↓
output [B, T, 1, H, W]  (λ = 0.2, 可由 coarse_logits_up 退化)
```

### 核心设计原则

```text
final_logits = coarse_logits_up + lambda_residual * residual_logits
```

- **主分支**：原始强 baseline（DINOv3 last-layer + BN+ReLU coarse head），始终是预测主源
- **门控残差融合**：辅助层(layer 17)通过可学习的 sigmoid gate 初始接近 0，不破坏 baseline
- **残差解码器**：只输出 residual，不直接输出最终 mask
- **Coarse Loss**：在 H/16 分辨率额外监督 coarse_logits，稳定训练

### 多层特征提取

脚本通过 DINOv3 官方接口取指定层特征（默认从 layer 17 和 23）：

```python
layer_outputs = backbone.get_intermediate_layers(
    frames,
    n=[17, 23],  # 默认 2 层，0-indexed
    reshape=True,
    norm=True,
)
```

最后一层（layer 23 / block 24）单独送入 `CoarseMaskHead` 产生主预测。layer 17 通过门控残差融合辅助。

### LoRA 注入位置

DINOv3 ViT-L/16 backbone 本身被冻结：

```python
for param in backbone.parameters():
    param.requires_grad = False
```

打开 `use_lora: true` 后，脚本使用 Hugging Face PEFT：

```python
LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=["attn.qkv", "attn.proj"],
    lora_dropout=lora_dropout,
    bias="none",
)
inject_adapter_in_model(config, backbone)
```

默认 LoRA 只注入 DINOv3 每个 Transformer block 的 self-attention 线性层：

```text
backbone.blocks.0.attn.qkv
backbone.blocks.0.attn.proj
backbone.blocks.1.attn.qkv
backbone.blocks.1.attn.proj
...
backbone.blocks.23.attn.qkv
backbone.blocks.23.attn.proj
```

也就是 24 个 ViT block，每个 block 注入 2 个 Linear，总计 48 个 LoRA 注入点。

每个 block 的 attention 结构可以简化理解为：

```text
x
  -> qkv:  Linear(1024, 3072)  # 一次性生成 Q/K/V
  -> attention
  -> proj: Linear(1024, 1024)
```

LoRA 不改原始权重矩阵本身，而是给目标 Linear 增加一个低秩增量：

```text
y = W x + scale * B(Ax)
scale = lora_alpha / lora_rank
```

其中原始 `W` 冻结，只训练 LoRA 的 `A/B` 矩阵。默认 `lora_rank=4` 时，参数量大致为：

```text
每个 qkv LoRA: 4 * 1024 + 3072 * 4 = 16,384
每个 proj LoRA: 4 * 1024 + 1024 * 4 = 8,192
每个 block LoRA: 24,576
24 个 block 总 LoRA: 589,824
```

训练时可训练参数包括：

```text
CoarseMaskHead + 门控融合投影 + 残差解码器 + (可选) LoRA 参数
```

不开 LoRA 时，可训练参数约 0.5M～1M，具体取决于 selected_layers 数量。

### 关键模块说明

**CoarseMaskHead**：保留原始 baseline 的 BN+ReLU 简单 decoder，直接输入 DINOv3 最后一层 1024 维 raw feature。

**GatedResidualFusion**：对每层做 LayerNorm + Conv1×1(1024→256)。aux 层通过 sigmoid 门控缩放后加到主层上。门控初始化为 `sigmoid(-6) ≈ 0.0025`，训练初期几乎等价于只用主层。

**ResidualProgressiveDecoder**：轻量渐进上采样，使用 GroupNorm + GELU。只输出 residual logits，不直接输出最终 mask。

最终输出：

```text
mask_logits: [B, T, 1, H, W]
coarse_logits: [B, T, 1, h, w]  (H/16 分辨率)
coarse_logits_up: [B, T, 1, H, W]
residual_logits: [B, T, 1, H, W]
```

### Loss 设计

训练总损失：

```text
loss = loss_full + lambda_coarse * loss_coarse
```

- **loss_full**：`SegmentationLoss(final_logits, target)` 在 H×W 分辨率（focal + BCE + IoU）
- **loss_coarse**：`SegmentationLoss(coarse_logits, target_low)` 在 H/16 分辨率，`target_low` 为下采样后的 GT
- **lambda_coarse**：默认 0.5

### BCEWithLogitsLoss

BCE 部分直接使用 PyTorch 的 `binary_cross_entropy_with_logits`。它内部会处理 sigmoid，比先手动 sigmoid 再 BCE 更数值稳定：

```python
bce_loss = F.binary_cross_entropy_with_logits(
    logits.view(logits.size(0), -1),
    target.view(target.size(0), -1),
)
```

这个项主要约束逐像素二分类结果。

### FocalLoss

FocalLoss 用来减轻前景/背景不平衡和简单样本主导的问题。当前参数是：

```text
alpha = 0.25
gamma = 2.0
```

实现逻辑：

```python
probs = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
alpha = torch.where(
    target == 1,
    torch.full_like(probs, 0.25),
    torch.full_like(probs, 0.75),
)
pt = torch.where(target == 1, probs, 1.0 - probs)
ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
focal_loss = alpha * torch.pow(1.0 - pt + 1e-7, 2.0) * ce_loss
focal_loss = focal_loss.mean()
```

直观理解：预测越容易的像素，`pt` 越接近 1，`(1 - pt)^gamma` 越小，loss 权重越低；预测困难的像素会被保留更高权重。

### IoULoss

IoULoss 先对 logits 做 sigmoid 得到概率图，再计算 soft IoU：

```python
probs = torch.sigmoid(logits)
intersection = (probs * target).sum(dim=(2, 3))
union = (probs + target).sum(dim=(2, 3)) - intersection
iou = (intersection + 1e-6) / (union + 1e-6)
iou_loss = 1.0 - iou.mean()
```

这个项直接优化 mask 区域重叠程度，和最终报告的 IoU 指标方向一致。

### 当前源码

当前项目中的核心 loss 代码如下：

```python
class SegmentationLoss(nn.Module):
    """ZZZ_model-style loss: focal + BCE-with-logits + IoU loss."""

    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.iou = IoULoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.float()
        probs = torch.sigmoid(logits)
        focal_loss = self.focal(
            logits.view(logits.size(0), -1),
            target.view(target.size(0), -1),
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.view(logits.size(0), -1),
            target.view(target.size(0), -1),
        )
        iou_loss = self.iou(probs, target)
        loss = focal_loss + bce_loss + iou_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "focal_loss": float(focal_loss.detach().cpu()),
            "bce_loss": float(bce_loss.detach().cpu()),
            "iou_loss": float(iou_loss.detach().cpu()),
        }
```

训练日志 `log.txt` 里会分别记录 `loss`、`focal_loss`、`bce_loss`、`iou_loss`、`coarse_loss`，方便后续画图分析每个分量的变化。

注意：`input_size` 需要能被 16 整除。默认只取 2 层 DINOv3 特征（layer 17, 23），backbone 前向计算量较小。

## 显存建议

DINOv3 ViT-L/16 在 `input_size=512` 时显存占用较高。24GB GPU 上建议从下面的配置开始：

```yaml
batch_size: 1
grad_accum_steps: 16
amp: true
gpu_id: 0
```

**GPU 选择**：如果 GPU 0 被占用，切换到其他空闲显卡：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --gpu_id 1
```

如果仍然 OOM，按优先级尝试：

```text
1. 保持 batch_size: 1
2. 减小 input_size，例如 384 或 256
3. 关闭 LoRA
4. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

测试阶段支持按时间维分块前向：

```yaml
eval_frame_chunk: 1
```

命令行覆盖：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type test \
  --checkpoint runs/dinov3_vitl16_eam_mvp/best_iou.pt \
  --eval_frame_chunk 1
```
