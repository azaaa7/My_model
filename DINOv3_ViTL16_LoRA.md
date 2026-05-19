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

```text
clip [B, T, 3, H, W]
  -> reshape 为逐帧输入 [B*T, 3, H, W]
  -> ImageNet mean/std 归一化
  -> DINOv3 ViT-L/16 backbone
  -> 抽取 4 层 (block 5, 11, 17, 23) patch token 特征
  -> 每层 LayerNorm + Conv1×1 投影到 256 通道
  -> Concat 4 层 → Conv1×1 + Conv3×3 融合 → [B*T, 256, H/16, W/16]
  -> reshape 回 [B, T, 256, H/16, W/16]

高频边界分支 (per-frame):
  RGB + Laplacian + Sobel magnitude [B*T, 5, H, W]
  -> 轻量 CNN 逐步下采样
  -> 输出 1/4 (96ch), 1/8 (128ch), 1/16 (192ch) 多尺度边界特征

频率引导边界解码器 (per-frame):
  1/16: concat(编码器特征(256) + 粗预测mask(1) + 边界1_16(192))
        → Conv1×1→GN→GELU→Conv3×3→GN→GELU → [256, H/16, W/16]
  1/8:  bilinear ×2 + concat(边界1_8) → Conv3×3×2 → [192, H/8, W/8]
  1/4:  bilinear ×2 + concat(边界1_4) → Conv3×3×2 → [128, H/4, W/4]
  Full: bilinear ×2 → [64, H/2] → bilinear ×2 → [32, H] → Conv1×1 → [1, H, W]
  -> mask_logits [B, T, 1, H, W]
  -> (可选) edge_logits [B, T, 1, H, W] 从 1/4 特征预测
```

### 多层特征提取

脚本通过 DINOv3 官方接口取 4 层特征：

```python
layer_outputs = backbone.get_intermediate_layers(
    frames,
    n=[5, 11, 17, 23],  # 4 层，0-indexed，对应 ViT-L/24 的 L/4, L/2, 3L/4, L-1
    reshape=True,
    norm=True,
)
# layer_outputs 是 list，每个元素形状为 [B*T, 1024, H/16, W/16]
```

每层特征通过 LayerNorm + Conv1×1(1024→256) 投影到统一通道，Concat 后经 Conv1×1(4C→2C) → GN → GELU → Conv3×3(2C→C) → GN → GELU 融合为 `[B*T, 256, H/16, W/16]`。

输出 reshape 回视频维度：`[B, T, 256, H/16, W/16]`。

### 高频边界分支

对每帧 RGB 计算边界 cue（5 通道）：RGB + Laplacian + Sobel magnitude。

```python
cues = compute_boundary_cues(frames)  # [B*T, 5, H, W]
```

轻量 CNN 逐步下采样，用 GroupNorm + GELU + 残差块，输出三个尺度的边界特征：

| 尺度 | 通道 | 用途 |
|---|---|---|
| 1/4 | 96 | 解码器 1/4 融合 |
| 1/8 | 128 | 解码器 1/8 融合 |
| 1/16 | 192 | 解码器 1/16 融合 |

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
编码器投影 + 融合参数 + 解码器参数 + 高频边界分支参数 + (可选) LoRA 参数
```

不开 LoRA 时，可训练参数包括：

```text
- 4× LayerNorm + Conv1×1 投影
- 融合 Conv1×1 + Conv3×3
- 高频边界分支 (Stem + 3 stages)
- 频率引导解码器 (coarse_head + fuse_16/8/4 + head + edge_head)
```

### 频率引导边界解码器

解码器是 `FrequencyGuidedBoundaryDecoder`，从编码器融合特征出发，结合多尺度边界特征渐进上采样。

不再使用 BatchNorm + ReLU，改用 GroupNorm + GELU 适配小 batch 训练。

默认 `encoder_dim=256` 时，通道变化如下：

**1/16 → 1/8：**

```text
concat(编码器特征(256) + 粗预测mask(1) + 边界1_16(192))
  -> [B, 449, H/16, W/16]
  -> Conv1×1 449→256 → GN → GELU → Conv3×3 256→256 → GN → GELU
  -> [B, 256, H/16, W/16] → bilinear ×2
  -> concat(边界1_8(128)) → [B, 384, H/8, W/8]
  -> Conv3×3 384→192 → GN → GELU → Conv3×3 192→192 → GN → GELU
  -> [B, 192, H/8, W/8]
```

**1/8 → 1/4：**

```text
bilinear ×2 → concat(边界1_4(96)) → [B, 288, H/4, W/4]
  -> Conv3×3 288→128 → GN → GELU → Conv3×3 128→128 → GN → GELU
  -> [B, 128, H/4, W/4]
```

**1/4 → Full：**

```text
bilinear ×2 → [B, 128, H/2, W/2]
  -> Conv3×3 128→64 → GN → GELU → [B, 64, H/2, W/2]
  -> bilinear ×2 → [B, 64, H, W]
  -> Conv3×3 64→32 → GN → GELU → [B, 32, H, W]
  -> Conv1×1 32→1 → [B, 1, H, W] (mask_logits)
```

**可选 Edge Head (use_edge_head: true)：**

```text
从 1/4 特征 [B, 128, H/4, W/4]
  -> Conv3×3 128→64 → GELU → Conv1×1 64→1
  -> bilinear up to H×W
  -> [B, 1, H, W] (edge_logits)
```

decoder 输出的是未经过 sigmoid 的 logits。训练时直接送入当前项目的 `SegmentationLoss`，推理或可视化时再通过 sigmoid 得到概率图。

最终输出：

```text
mask_logits: [B, T, 1, H, W]
edge_logits: [B, T, 1, H, W] (可选)
```

## Loss

训练使用项目里的 `SegmentationLoss`，定义在 `my_model/losses.py`。总损失是三个部分直接相加：

```text
loss = focal_loss + bce_loss + iou_loss
```

没有额外权重系数。模型输出是未经过 sigmoid 的 `logits`，标签 `target` 是 0/1 mask。

### 输入形状

训练和验证时，如果标签是中心帧 mask，脚本会取中心帧 logits：

```text
logits: [B, 1, H, W]
target: [B, 1, H, W]
```

测试时如果标签包含全视频所有帧，会把时间维展开：

```text
logits: [B*T, 1, H, W]
target: [B*T, 1, H, W]
```

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

训练日志 `log.txt` 里会分别记录 `loss`、`focal_loss`、`bce_loss`、`iou_loss`，方便后续画图分析每个分量的变化。

注意：`input_size` 需要能被 16 整除。多层编码器（4 层 vs 原先 1 层）会增加 backbone 前向计算量，显存需求比单层版本略高。

## 显存建议

DINOv3 ViT-L/16 在 `input_size=512` 时显存占用很高。24GB GPU 上建议从下面的配置开始：

```yaml
batch_size: 1
grad_accum_steps: 16
amp: true
gpu_id: 0
```

这表示每次只放 1 个样本进显存，累计 16 次梯度后再更新一次参数，等效 batch 接近 16，但峰值显存接近 batch 1。

**GPU 选择**：如果 GPU 0 被占用（OOM），切换到其他空闲显卡：

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
2. 减小 input_size，例如 384 或 256，注意必须能被 16 整除
3. 减少 selected_layers 到 3 层（例如 [5, 11, 17]）或 2 层
4. 关闭 LoRA，只训练 decoder + encoder 投影/融合
5. 设置环境变量 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 减少显存碎片影响
```

示例：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 16
```

测试阶段会读取一个视频的所有帧。为了避免一次性把 `[1, T, 3, H, W]` 展开成 `[T, 3, H, W]` 后送入 ViT 导致 OOM，脚本支持按时间维分块前向：

```yaml
eval_frame_chunk: 1
```

`eval_frame_chunk: 1` 表示测试/验证时每次只前向 1 帧，最后把所有分块输出拼回完整视频，再计算全视频所有帧的平均指标。显存足够时可以改成 2、4 或更大来提速。

命令行覆盖：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora.yml \
  --type test \
  --checkpoint runs/dinov3_vitl16/best_iou.pt \
  --eval_frame_chunk 1
```
