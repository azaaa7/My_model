# My Model

这个项目包含一个相对简单的视频 inpainting 检测模型，并保留 `zzz_dataset_toolkit` 作为数据读取工具。

## DINOv3 / TFCU 文档入口

- `DINOv3_ViTL16_LoRA.md`: DINOv3 ViT-L/16、LoRA、FPN/TFCU 的完整说明。
- `Semantic_Anchor_MFCE_agent_implementation_guide.md`: Semantic-Anchor MFCE 的实现约束和验收标准。
- `Semantic_Anchor_MFCE_P4_TFCU_Tutorial.md`: 新结构的使用教程、配置注释说明、训练/验证/测试命令。

## 1. 模型结构

模型使用 `ZZZ_model` 中的 `HRNet` 作为逐帧 backbone，再接一个轻量 decoder 输出 mask logits。

```text
Input clip
[B, T, 3, H, W]
      |
      v
Frame-wise HRNet backbone
      |
      v
Lightweight decoder
      |
      v
Predicted mask logits
[B, T, 1, H, W]
```

如果只输入单帧，令 `T=1` 即可：

```text
[B, 1, 3, H, W] -> [B, 1, 1, H, W]
```

## 2. 模型使用

```python
import torch
from my_model import SimpleHRNetInpaintingDetector

model = SimpleHRNetInpaintingDetector()
clip = torch.randn(2, 1, 3, 256, 256)

logits = model(clip)
print(logits.shape)  # torch.Size([2, 1, 1, 256, 256])
```

`logits` 是未经过 sigmoid 的输出。训练时可以直接配合 `torch.nn.BCEWithLogitsLoss` 使用；推理时可以用：

```python
prob = torch.sigmoid(logits)
mask = prob > 0.5
```

## 3. 实现细节

- `SimpleHRNetInpaintingDetector` 定义在 `my_model/hrnet_detector.py`；
- 默认从 `../ZZZ_model/models/hrnet.py` 加载 `HRNet`；
- 输入 `[B, T, 3, H, W]` 会先 reshape 成 `[B*T, 3, H, W]`；
- `ZZZ_model` 的 HRNet 输出特征为 `[B*T, 32, H/4, W/4]`；
- lightweight decoder 输出 `[B*T, 1, H, W]`；
- 最后 reshape 回 `[B, T, 1, H, W]`。

可以冻结 HRNet backbone：

```python
model = SimpleHRNetInpaintingDetector(freeze_backbone=True)
```

如果 `ZZZ_model` 不在默认相对路径，可以手动传入 HRNet 源码路径：

```python
model = SimpleHRNetInpaintingDetector(
    hrnet_path="/path/to/ZZZ_model/models/hrnet.py",
)
```

## 4. Demo

```bash
python demo_model.py
```

如果输出：

```text
model demo: OK
```

说明模型前向传播和输出 shape 正常。

## 5. 训练 / 验证 / 测试流程

默认配置在 `configs/default.yml`。先把里面的样本索引路径改成自己的 `.npy`：

```yaml
train_samples: "/path/to/train_samples.npy"
val_samples: "/path/to/val_samples.npy"
test_samples: "/path/to/test_samples.npy"
```

训练：

```bash
python train_val_test.py --config configs/default.yml --type train
```

训练时会保存：

- `runs/simple_hrnet/latest.pt`
- `runs/simple_hrnet/best_iou.pt`
- `runs/simple_hrnet/config.json`

验证：

```bash
python train_val_test.py \
  --config configs/default.yml \
  --type val \
  --checkpoint runs/simple_hrnet/best_iou.pt
```

测试：

```bash
python train_val_test.py \
  --config configs/default.yml \
  --type test \
  --checkpoint runs/simple_hrnet/best_iou.pt
```

指标设置参考 `ZZZ_model`：

- loss: `FocalLoss + BCEWithLogitsLoss + IoULoss`
- metrics: `F1`、`IoU`、`precision`、`recall`、`accuracy`
- mask 阈值默认 `0.5`

验证和测试会在 `visualization_dir` 下保存可视化结果，每张图横向拼接：

```text
input frame | probability map | binary prediction | ground truth
```

## 6. 和数据工具配合

`zzz_dataset_toolkit` 的 `build_dataloader` 输出 `frames`，shape 通常是：

```text
[B, num_frames, 3, input_size, input_size]
```

它可以直接作为模型输入：

```python
from my_model import SimpleHRNetInpaintingDetector
from zzz_dataset_toolkit import build_dataloader

loader = build_dataloader(
    samples="/path/to/samples.npy",
    mode="train",
    batch_size=2,
    input_size=512,
    num_frames=1,
)

model = SimpleHRNetInpaintingDetector()

for frames, center_mask, original_h, original_w, sample_name in loader:
    logits = model(frames)
    center_logits = logits[:, logits.shape[1] // 2]
    loss = criterion(center_logits, center_mask)
```

当 `num_frames > 1` 但训练标签只有中心帧 mask 时，可以像上面一样取 `logits[:, T // 2]` 计算中心帧损失。
