# My_model `test_time` 分支 TFCU-Inpaint 修复实施文档

> 目的：让 agent 按照本文档修复当前 `test_time` 分支，使其真正满足之前的 **DINOv3 ViT-L/16 + LoRA + DPT-FPN + TFCU-Inpaint Adapter** 要求。  
> 当前状态：TFCU 模块文件已经部分存在，但训练主流程、Dataset、base model 接口、loss、配置校验还没有真正接通。  
> 修复原则：**不要重写整个项目；保留现有 DINOv3/LoRA 训练脚本，按最小改动把 TFCU 分支真正接入。**

---

## 0. 当前必须修复的问题总览

当前 `test_time` 分支存在以下关键问题：

```text
1. my_model/temporal/ 模块已经存在，但 train_val_test_dinov3_lora.py 没有真正使用它。
2. VideoInpaintTFCU 已存在，但 build_model() 没有按 use_tfcu_adapter=true 构建它。
3. Dataset 已有 num_clips 采样逻辑，但 train/val 仍返回中心帧 mask，而不是完整 [N,T,1,H,W] mask。
4. Dataset 帧排序仍可能是字符串排序，不保证 1,2,10 的正确时序。
5. TFCU 配置里写了 boundary / temporal_delta，但 losses.py 没有实现这些 loss。
6. 配置里写了 lr_temporal / lr_decoder / lr_lora，但 optimizer 没有分参数组。
7. train 脚本仍强制 num_frames 为奇数，和 TFCU 推荐的 num_frames=4 冲突。
8. base model 必须提供 extract_fpn_features() 和 decode_fpn()，但当前训练脚本里的模型接口未确认接通。
9. __init__.py 没有导出 VideoInpaintTFCU 和新 loss。
10. shape tests 有但不一定覆盖训练主流程。
```

修复完成后，必须满足：

```text
输入:  images [B,N,T,3,512,512]
标签:  masks  [B,N,T,1,512,512]
输出:  logits [B,N,T,1,512,512]

DINOv3 + DPT-FPN:
  x [B*N*T,3,512,512]
  → P2 [B*N*T,256,128,128]
  → P3 [B*N*T,256, 64, 64]
  → P4 [B*N*T,256, 32, 32]
  → P5 [B*N*T,256, 16, 16]

TFCU:
  P4 reshape [B,N,T,256,32,32]
  → local temporal difference
  → for n in range(N) 顺序 memory
  → P4_out = P4 + alpha * temporal_delta
  → decoder
```

---

## 1. 修复顺序

必须按下面顺序做，不要跳步：

```text
Step 1: 修 Dataset，保证 TFCU 模式返回 [N,T] images/masks。
Step 2: 修 base model，拆出 extract_fpn_features() / decode_fpn()。
Step 3: 修 VideoInpaintTFCU 和 build_model()，让 use_tfcu_adapter=true 真正生效。
Step 4: 修 loss，增加 Dice/Tversky/Boundary/TemporalDelta，并支持 [B,N,T,1,H,W]。
Step 5: 修 optimizer，使用 temporal/decoder/lora 参数组。
Step 6: 修 config validate，TFCU 模式允许 num_frames=4。
Step 7: 修 train/val/test loop 的 shape 处理。
Step 8: 增加端到端 shape test。
Step 9: 跑 py_compile、shape test、small overfit。
```

---

## 2. Step 1：修 Dataset

目标文件：

```text
zzz_dataset_toolkit/dataset.py
```

### 2.1 增加数字排序函数

必须避免字符串排序：

```python
from pathlib import Path
import re


def _numeric_key(path_or_name):
    name = Path(path_or_name).stem
    nums = re.findall(r"\d+", name)
    if len(nums) > 0:
        return int(nums[-1])
    return name
```

把类似：

```python
frame_list = sorted(frame_list)
mask_list = sorted(mask_list)
```

改成：

```python
frame_list = sorted(frame_list, key=_numeric_key)
mask_list = sorted(mask_list, key=_numeric_key)
```

如果代码里是 `os.listdir()`：

```python
frame_list = sorted(os.listdir(frame_dir), key=_numeric_key)
mask_list = sorted(os.listdir(mask_dir), key=_numeric_key)
```

### 2.2 TFCU 模式必须返回完整 mask

当前问题：train/val 可能返回：

```python
center_mask = masks_out[self.num_clips // 2, self.num_frames // 2]
return frames_out, center_mask, ...
```

这不符合要求。

必须改成：当 `use_tfcu_adapter=True` 或 `num_clips > 1` 时，train/val/test 全部返回：

```python
return frames_out, masks_out, H, W, name
```

其中：

```text
frames_out: [N,T,3,H,W]
masks_out:  [N,T,1,H,W]
```

建议写法：

```python
if self.use_tfcu_adapter or self.num_clips > 1:
    return frames_out, masks_out, H, W, name
else:
    # 保留旧单帧/中心帧逻辑，兼容原 baseline
    center_mask = masks_out[self.num_frames // 2]
    return frames_out, center_mask, H, W, name
```

但如果当前 dataset 的 `frames_out` 在非 TFCU 下是 `[T,3,H,W]`，在 TFCU 下必须是 `[N,T,3,H,W]`。

### 2.3 `__init__` 增加参数

Dataset 必须支持：

```python
num_clips: int = 1
use_tfcu_adapter: bool = False
frame_stride: int = 1
```

示例：

```python
self.num_clips = int(num_clips)
self.num_frames = int(num_frames)
self.use_tfcu_adapter = bool(use_tfcu_adapter)
self.frame_stride = int(frame_stride)
```

### 2.4 TFCU 模式允许偶数帧

旧逻辑如果有：

```python
if num_frames % 2 == 0:
    raise ValueError("num_frames must be odd")
```

必须改成：

```python
if not self.use_tfcu_adapter:
    if self.num_frames <= 0 or self.num_frames % 2 == 0:
        raise ValueError("num_frames must be a positive odd integer in baseline mode")
else:
    if self.num_frames <= 0:
        raise ValueError("num_frames must be positive in TFCU mode")
```

TFCU 第一版推荐：

```yaml
num_clips: 4
num_frames: 4
```

### 2.5 训练采样必须保持时序

TFCU 训练采样函数：

```python
def sample_train_clips(num_total_frames, num_clips, num_frames, stride):
    clip_len = (num_frames - 1) * stride + 1
    max_start = num_total_frames - clip_len

    if max_start < 0:
        # 帧数不足，后续读帧时用 min(idx, last_idx) padding
        starts = [0 for _ in range(num_clips)]
    else:
        if max_start + 1 >= num_clips:
            starts = random.sample(range(max_start + 1), k=num_clips)
        else:
            starts = [random.randint(0, max_start) for _ in range(num_clips)]
        starts = sorted(starts)

    clips = []
    for s in starts:
        clip = [min(s + i * stride, num_total_frames - 1) for i in range(num_frames)]
        clips.append(clip)

    return clips
```

保证：

```text
clip 顺序递增
clip 内 frame 顺序递增
```

### 2.6 验证/测试采样必须顺序

```python
def sample_eval_clips(num_total_frames, num_clips, num_frames, stride):
    step = num_frames * stride
    all_clips = []

    for start in range(0, num_total_frames, step):
        clip = [min(start + i * stride, num_total_frames - 1) for i in range(num_frames)]
        all_clips.append(clip)

    # 如果 dataset 每次只返回一组 N clips，则按 index 取 chunk
    return all_clips
```

如果当前 Dataset 每个 `__getitem__` 代表一个视频，则可以返回前 `num_clips` 个；如果测试要覆盖全视频，应通过 chunk 或 sliding window 处理。

### 2.7 Dataset 验收

在本地写临时检查：

```python
sample = dataset[0]
frames, masks = sample[0], sample[1]
print(frames.shape)
print(masks.shape)
```

TFCU 模式必须输出：

```text
frames: torch.Size([4,4,3,512,512])
masks:  torch.Size([4,4,1,512,512])
```

DataLoader 后：

```text
frames: [B,4,4,3,512,512]
masks:  [B,4,4,1,512,512]
```

---

## 3. Step 2：修 base model 接口

目标文件可能是：

```text
train_val_test_dinov3_lora.py
my_model/dinov3_dpt_fpn.py
my_model/video_inpaint_tfcu.py
```

当前 `VideoInpaintTFCU` 需要 base model 提供：

```python
base_model.use_dpt_fpn == True
base_model.extract_fpn_features(x)
base_model.decode_fpn(P2, P3, P4, P5)
```

必须确保实际训练脚本构建的 DINOv3 模型提供这三个接口。

### 3.1 推荐新增/修复 base model 类

如果当前 DINOv3 模型在 `train_val_test_dinov3_lora.py` 内部定义，建议拆到：

```text
my_model/dinov3_dpt_fpn.py
```

类名建议：

```python
class DINOv3DPTFPNInpaintingDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_dpt_fpn = True
        ...
```

必须实现：

```python
def extract_fpn_features(self, x):
    # x: [BNT,3,512,512]
    # return P2, P3, P4, P5
    ...
    return P2, P3, P4, P5


def decode_fpn(self, P2, P3, P4, P5):
    # return logits [BNT,1,512,512]
    return self.decoder(P2, P3, P4, P5)


def forward(self, clip):
    # 兼容旧输入 [B,T,3,H,W]
    B, T, C, H, W = clip.shape
    x = clip.reshape(B*T, C, H, W)
    P2, P3, P4, P5 = self.extract_fpn_features(x)
    logits = self.decode_fpn(P2, P3, P4, P5)
    return logits.reshape(B, T, 1, H, W)
```

### 3.2 特征尺寸要求

必须满足：

```text
P2: [BNT,256,128,128]
P3: [BNT,256, 64, 64]
P4: [BNT,256, 32, 32]
P5: [BNT,256, 16, 16]
```

如果当前项目只有单层 DINO feature decoder，没有 DPT-FPN，那就先不要声称 `use_dpt_fpn=True`。必须补回 DPT Reassemble + FPN decoder，或修改 `VideoInpaintTFCU` 使其插在现有 32×32 feature 上。

最推荐：继续使用之前文档里的 DPT-FPN 结构。

### 3.3 base model 验收

写临时测试：

```python
x = torch.randn(2, 3, 512, 512).cuda()
P2, P3, P4, P5 = base_model.extract_fpn_features(x)

assert P2.shape == (2, 256, 128, 128)
assert P3.shape == (2, 256, 64, 64)
assert P4.shape == (2, 256, 32, 32)
assert P5.shape == (2, 256, 16, 16)

logits = base_model.decode_fpn(P2, P3, P4, P5)
assert logits.shape == (2, 1, 512, 512)
```

---

## 4. Step 3：修 VideoInpaintTFCU 接入

目标文件：

```text
my_model/video_inpaint_tfcu.py
train_val_test_dinov3_lora.py
my_model/__init__.py
```

### 4.1 VideoInpaintTFCU 必须支持两种输入

```python
def forward(self, video):
    if video.dim() == 5:
        # [B,T,3,H,W] 兼容旧模式
        B, T, C, H, W = video.shape
        N = 1
        video = video[:, None]
        squeeze_n = True
    elif video.dim() == 6:
        # [B,N,T,3,H,W]
        B, N, T, C, H, W = video.shape
        squeeze_n = False
    else:
        raise ValueError(...)
```

输出：

```python
logits = logits.reshape(B, N, T, 1, H, W)

if squeeze_n:
    logits = logits[:, 0]  # [B,T,1,H,W]
return logits
```

### 4.2 确保调用的是 `decode_fpn`

如果当前代码写的是：

```python
logits = self.base.decoder(P2, P3, P4, P5)
```

但 base model 实际方法叫 `decode_fpn`，要统一：

```python
logits = self.base.decode_fpn(P2, P3, P4, P5)
```

### 4.3 `my_model/__init__.py` 导出

增加：

```python
from .video_inpaint_tfcu import VideoInpaintTFCU

__all__ = [
    ...
    "VideoInpaintTFCU",
]
```

---

## 5. Step 4：修 build_model()

目标文件：

```text
train_val_test_dinov3_lora.py
```

### 5.1 导入新模型

增加：

```python
from my_model.video_inpaint_tfcu import VideoInpaintTFCU
from my_model.dinov3_dpt_fpn import DINOv3DPTFPNInpaintingDetector
```

如果 base model 仍在同文件中定义，就只 import `VideoInpaintTFCU`。

### 5.2 build_model 逻辑

必须改成：

```python
def build_model(cfg, device):
    use_tfcu = bool(cfg.get("use_tfcu_adapter", False) or cfg.get("model", {}).get("use_tfcu_adapter", False))

    base_model = DINOv3DPTFPNInpaintingDetector(cfg)

    if use_tfcu:
        model = VideoInpaintTFCU(base_model=base_model, cfg=cfg)
        print("[Model] Using VideoInpaintTFCU with P4 temporal adapter")
    else:
        model = base_model
        print("[Model] Using DINOv3 DPT-FPN baseline")

    model = model.to(device)
    return model
```

注意：配置字段可能在顶层或 `model:` 下，代码必须统一读取。建议写工具函数：

```python
def cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        if "model" in cfg and key in cfg["model"]:
            return cfg["model"][key]
        if "train" in cfg and key in cfg["train"]:
            return cfg["train"][key]
    return default
```

### 5.3 必须打印确认信息

训练开始时必须打印：

```text
[Model] use_tfcu_adapter=True
[Model] model class=VideoInpaintTFCU
[Model] temporal adapter alpha init=0.0
```

否则容易出现“配置写了但没生效”的问题。

---

## 6. Step 5：修 loss

目标文件：

```text
my_model/losses.py
```

### 6.1 必须实现这些 loss

```text
DiceLoss
TverskyLoss
BoundaryLoss
TemporalDeltaLoss
SegmentationLoss
```

### 6.2 通用 flatten 函数

所有像素级 loss 必须支持：

```text
[B,T,1,H,W]
[B,N,T,1,H,W]
[B,1,H,W]
```

建议写：

```python
def flatten_logits_targets(logits, target):
    if logits.dim() == 6:
        B, N, T, C, H, W = logits.shape
        logits = logits.reshape(B * N * T, C, H, W)
        target = target.reshape(B * N * T, C, H, W)
    elif logits.dim() == 5:
        B, T, C, H, W = logits.shape
        logits = logits.reshape(B * T, C, H, W)
        target = target.reshape(B * T, C, H, W)
    elif logits.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

    return logits, target.float()
```

### 6.3 DiceLoss

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        logits, target = flatten_logits_targets(logits, target)
        pred = torch.sigmoid(logits)

        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        inter = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)

        loss = 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)
        return loss.mean()
```

### 6.4 TverskyLoss

```python
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target):
        logits, target = flatten_logits_targets(logits, target)
        pred = torch.sigmoid(logits)

        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)
        fn = ((1 - pred) * target).sum(dim=1)

        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - score).mean()
```

### 6.5 BoundaryLoss

```python
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sobel(x):
        device, dtype = x.device, x.dtype
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, logits, target):
        logits, target = flatten_logits_targets(logits, target)
        pred = torch.sigmoid(logits)
        return F.l1_loss(self.sobel(pred), self.sobel(target.float()))
```

### 6.6 TemporalDeltaLoss

```python
class TemporalDeltaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        # logits/target: [B,N,T,1,H,W] or [B,T,1,H,W]
        if logits.dim() == 5:
            logits = logits[:, None]
            target = target[:, None]

        if logits.dim() != 6:
            return logits.sum() * 0.0

        if logits.shape[2] <= 1:
            return logits.sum() * 0.0

        pred = torch.sigmoid(logits)
        target = target.float()

        dp = pred[:, :, 1:] - pred[:, :, :-1]
        dg = target[:, :, 1:] - target[:, :, :-1]

        return F.l1_loss(dp, dg)
```

注意：`logits.shape[2]` 是 T 维，因为 shape 是 `[B,N,T,1,H,W]`。

### 6.7 SegmentationLoss 读取配置

实现：

```python
class SegmentationLoss(nn.Module):
    def __init__(self, loss_cfg=None):
        super().__init__()
        self.loss_cfg = loss_cfg or {}

        self.dice = DiceLoss(...)
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(...)
        self.boundary = BoundaryLoss()
        self.temporal_delta = TemporalDeltaLoss()

    def forward(self, logits, target):
        total = 0.0
        logs = {}

        if "dice" in self.loss_cfg:
            w = self.loss_cfg["dice"].get("weight", 1.0)
            val = self.dice(logits, target)
            total = total + w * val
            logs["dice"] = val.detach()

        ...

        return total
```

如果当前训练脚本只接受一个 tensor loss，不要返回 dict；可以先只返回 total。需要 logs 时再扩展。

### 6.8 `__init__.py` 导出 loss

```python
from .losses import (
    DiceLoss,
    TverskyLoss,
    BoundaryLoss,
    TemporalDeltaLoss,
    SegmentationLoss,
)
```

---

## 7. Step 6：修 optimizer 参数组

目标文件：

```text
train_val_test_dinov3_lora.py
```

当前问题：所有参数统一 AdamW。

必须改成按组：

```python
def build_optimizer(model, cfg):
    lr_temporal = cfg_get(cfg, "lr_temporal", 1e-4)
    lr_decoder = cfg_get(cfg, "lr_decoder", 1e-4)
    lr_lora = cfg_get(cfg, "lr_lora", 1e-5)
    weight_decay = cfg_get(cfg, "weight_decay", 1e-4)

    param_groups = []

    if hasattr(model, "temporal_adapter"):
        param_groups.append({
            "params": [p for p in model.temporal_adapter.parameters() if p.requires_grad],
            "lr": lr_temporal,
            "weight_decay": weight_decay,
            "name": "temporal",
        })

    # decoder
    decoder = None
    if hasattr(model, "base") and hasattr(model.base, "decoder"):
        decoder = model.base.decoder
    elif hasattr(model, "decoder"):
        decoder = model.decoder

    if decoder is not None:
        param_groups.append({
            "params": [p for p in decoder.parameters() if p.requires_grad],
            "lr": lr_decoder,
            "weight_decay": weight_decay,
            "name": "decoder",
        })

    # LoRA 参数
    lora_params = []
    for name, p in model.named_parameters():
        lname = name.lower()
        if p.requires_grad and ("lora" in lname or "adapter" in lname and "temporal_adapter" not in lname):
            lora_params.append(p)

    if len(lora_params) > 0:
        param_groups.append({
            "params": lora_params,
            "lr": lr_lora,
            "weight_decay": weight_decay,
            "name": "lora",
        })

    # fallback: 如果没有分到任何组，收集剩余可训练参数
    used = set()
    for g in param_groups:
        for p in g["params"]:
            used.add(id(p))

    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
    if len(other_params) > 0:
        param_groups.append({
            "params": other_params,
            "lr": cfg_get(cfg, "learning_rate", 1e-4),
            "weight_decay": weight_decay,
            "name": "other",
        })

    print("[Optimizer] parameter groups:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  - {g['name']}: lr={g['lr']}, params={n/1e6:.3f}M")

    return torch.optim.AdamW(param_groups)
```

注意：如果当前项目不允许 param group 有 `"name"` 字段，可以打印后再删除：

```python
for g in param_groups:
    g.pop("name", None)
```

---

## 8. Step 7：修 config

目标文件：

```text
configs/dinov3_vitl16_lora_tfcu_inpaint.yml
```

推荐最终内容必须包含：

```yaml
model_name: dinov3_vitl16_lora_tfcu_inpaint

use_tfcu_adapter: true
use_dpt_fpn: true

num_clips: 4
num_frames: 4
frame_stride: 1
memory_len: 4
use_memory: true
detach_memory: true
use_mask_prompt: false
use_flow: false
use_historical_review: false

use_lora: true
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.1
lora_targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"

extract_layers: "5,11,17,23"
neck_channels: 256

batch_size: 1
grad_accum_steps: 8
learning_rate: 1.0e-4
lr_temporal: 1.0e-4
lr_decoder: 1.0e-4
lr_lora: 1.0e-5
weight_decay: 1.0e-4
amp: true

loss:
  dice:           {weight: 1.0, smooth: 1.0e-6}
  bce:            {weight: 0.5}
  tversky:        {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
  boundary:       {weight: 0.2}
  temporal_delta: {weight: 0.1}
```

如果项目当前 config 是嵌套结构，也可以放到：

```yaml
model:
  use_tfcu_adapter: true
  ...

train:
  batch_size: 1
  ...

loss:
  ...
```

但代码必须能正确读取。

---

## 9. Step 8：修 validate_config()

目标文件：

```text
train_val_test_dinov3_lora.py
```

当前问题：可能强制 `num_frames` 为奇数。

必须改成：

```python
def validate_config(cfg):
    use_tfcu = cfg_get(cfg, "use_tfcu_adapter", False)
    num_frames = int(cfg_get(cfg, "num_frames", 1))
    num_clips = int(cfg_get(cfg, "num_clips", 1))

    if use_tfcu:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive in TFCU mode")
        if num_clips <= 0:
            raise ValueError("num_clips must be positive in TFCU mode")
    else:
        if num_frames <= 0 or num_frames % 2 == 0:
            raise ValueError("num_frames must be positive odd integer in baseline mode")
```

必须打印：

```python
print(f"[Config] use_tfcu_adapter={use_tfcu}, num_clips={num_clips}, num_frames={num_frames}")
```

---

## 10. Step 9：修 train/val/test loop shape

训练循环必须支持：

```text
frames: [B,N,T,3,H,W]
masks:  [B,N,T,1,H,W]
```

### 10.1 不要错误对齐中心帧

如果当前有：

```python
logits, loss_masks = align_logits_and_masks(logits_all, masks)
```

必须确认它不会把 `[B,N,T]` 压成中心帧。

推荐逻辑：

```python
logits = model(frames)

if logits.shape != masks.shape:
    raise ValueError(f"logits shape {logits.shape} != masks shape {masks.shape}")

loss = criterion(logits, masks)
```

### 10.2 兼容 baseline

如果 baseline 输出 `[B,T,1,H,W]`，mask 也是 `[B,T,1,H,W]`，照样成立。

如果旧 dataset 返回中心帧 mask `[B,1,H,W]`，旧的 baseline 路径可以保留 `align_logits_and_masks()`；但 TFCU 路径必须禁用中心帧对齐。

推荐：

```python
use_tfcu = cfg_get(cfg, "use_tfcu_adapter", False)

if use_tfcu:
    logits = model(frames)
    if logits.shape != masks.shape:
        raise ValueError(...)
    loss = criterion(logits, masks)
else:
    logits_all = model(frames)
    logits, loss_masks = align_logits_and_masks(logits_all, masks)
    loss = criterion(logits, loss_masks)
```

---

## 11. Step 10：修 metrics

如果 metric 当前只支持：

```text
[B,1,H,W]
[B,T,1,H,W]
```

必须支持：

```text
[B,N,T,1,H,W]
```

统一 flatten：

```python
def flatten_for_metric(logits, masks):
    if logits.dim() == 6:
        B, N, T, C, H, W = logits.shape
        logits = logits.reshape(B * N * T, C, H, W)
        masks = masks.reshape(B * N * T, C, H, W)
    elif logits.dim() == 5:
        B, T, C, H, W = logits.shape
        logits = logits.reshape(B * T, C, H, W)
        masks = masks.reshape(B * T, C, H, W)
    return logits, masks
```

---

## 12. Step 11：新增/修 shape test

目标文件：

```text
debug/test_tfcu_shapes.py
```

必须包含：

```python
import torch

from my_model.temporal.local_temporal_difference import LocalTemporalDifferenceModule
from my_model.temporal.memory_attention import InpaintMemoryAttention
from my_model.temporal.temporal_adapter import TFCUInpaintAdapter


def test_local():
    x = torch.randn(2, 4, 4, 256, 32, 32)
    m = LocalTemporalDifferenceModule(256)
    y = m(x)
    assert y.shape == x.shape


def test_memory():
    cur = torch.randn(1, 4, 256, 32, 32)
    mem = torch.randn(1, 4, 4, 256, 32, 32)
    m = InpaintMemoryAttention(256)
    y = m(cur, mem)
    assert y.shape == cur.shape


def test_adapter():
    P4 = torch.randn(1 * 4 * 4, 256, 32, 32)
    m = TFCUInpaintAdapter(channels=256, memory_len=4)
    y = m(P4, B=1, N=4, T=4)
    assert y.shape == P4.shape
    assert abs(float(m.alpha.detach())) < 1e-8


if __name__ == "__main__":
    test_local()
    test_memory()
    test_adapter()
    print("All TFCU shape tests passed.")
```

如果有完整 base model，再增加：

```python
video = torch.randn(1,4,4,3,512,512).cuda()
logits = model(video)
assert logits.shape == (1,4,4,1,512,512)
```

---

## 13. Step 12：新增端到端 dry-run 脚本

推荐新增：

```text
debug/dry_run_tfcu_train_step.py
```

功能：

```text
1. 读取 configs/dinov3_vitl16_lora_tfcu_inpaint.yml
2. build_model()
3. 构造 fake frames/masks
4. forward
5. loss
6. backward
7. optimizer.step()
```

伪代码：

```python
frames = torch.randn(1, 4, 4, 3, 512, 512).cuda()
masks = torch.randint(0, 2, (1, 4, 4, 1, 512, 512)).float().cuda()

model = build_model(cfg, device="cuda")
criterion = SegmentationLoss(cfg["loss"])
optimizer = build_optimizer(model, cfg)

logits = model(frames)
assert logits.shape == masks.shape

loss = criterion(logits, masks)
loss.backward()
optimizer.step()

print("dry-run ok", loss.item())
```

这一步通过后，才允许跑真实数据。

---

## 14. 必须执行的检查命令

修复后必须依次跑：

```bash
python -m py_compile train_val_test_dinov3_lora.py
python -m py_compile zzz_dataset_toolkit/dataset.py
python -m py_compile my_model/video_inpaint_tfcu.py
python -m py_compile my_model/temporal/local_temporal_difference.py
python -m py_compile my_model/temporal/memory_attention.py
python -m py_compile my_model/temporal/temporal_adapter.py
python -m py_compile my_model/losses.py
```

然后跑：

```bash
python debug/test_tfcu_shapes.py
python debug/dry_run_tfcu_train_step.py
```

最后跑真实训练小样本：

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type train \
  --batch_size 1 \
  --grad_accum_steps 8
```

---

## 15. 训练日志必须出现这些信息

训练开始时必须打印：

```text
[Config] use_tfcu_adapter=True
[Config] num_clips=4, num_frames=4
[Model] Using VideoInpaintTFCU
[Model] base_model.use_dpt_fpn=True
[Model] temporal adapter inserted at P4
[Model] temporal adapter alpha init=0.0
[Dataset] TFCU mode: return frames [N,T,3,H,W], masks [N,T,1,H,W]
[Optimizer] temporal lr=1e-4
[Optimizer] decoder lr=1e-4
[Optimizer] lora lr=1e-5
[Loss] Dice + BCE + Tversky + Boundary + TemporalDelta
```

如果没有这些日志，说明 agent 可能只是写了配置，实际没接通。

---

## 16. 最终验收标准

修复完成后，必须满足以下全部条件：

```text
[ ] train_val_test_dinov3_lora.py 能根据 use_tfcu_adapter=true 构建 VideoInpaintTFCU
[ ] Dataset 在 TFCU 模式返回 [N,T,3,H,W] 和 [N,T,1,H,W]
[ ] DataLoader 后 shape 是 [B,N,T,3,H,W] 和 [B,N,T,1,H,W]
[ ] base model 提供 extract_fpn_features()，返回 P2/P3/P4/P5
[ ] base model 提供 decode_fpn()
[ ] P4 经过 TFCUInpaintAdapter
[ ] TFCUInpaintAdapter 中 for n in range(N) 顺序执行
[ ] 当前 clip 只读取 state[-memory_len:] 历史 memory
[ ] 当前 clip 处理完成后才 append 到 state
[ ] alpha 初始化为 0
[ ] 输出 logits shape 为 [B,N,T,1,H,W]
[ ] loss 支持 [B,N,T,1,H,W]
[ ] BoundaryLoss 和 TemporalDeltaLoss 被实际加入总 loss
[ ] optimizer 有 temporal / decoder / lora 参数组
[ ] TFCU 模式允许 num_frames=4
[ ] 帧排序使用数字排序，不是字符串排序
[ ] debug/test_tfcu_shapes.py 通过
[ ] debug/dry_run_tfcu_train_step.py 通过
[ ] 两个视频样本 overfit loss 能下降
```

---

## 17. 不要做的事

Agent 不要做以下事情：

```text
1. 不要把 TFCU 模块只写在 README 里。
2. 不要只创建 temporal 文件但不接入 build_model()。
3. 不要继续返回 center_mask 训练 TFCU。
4. 不要把 [B,N,T] flatten 后丢掉 N 维时序。
5. 不要用双向 attention 偷看未来 clip。
6. 不要第一版加入 optical flow。
7. 不要第一版加入 Historical Review。
8. 不要把 num_frames 强制成奇数。
9. 不要只改 config 不改 train loop。
10. 不要把所有参数用同一个学习率训练。
```

---

## 18. 最小完成版本定义

第一版最小完成版本只需要做到：

```text
DINOv3 DPT-FPN baseline
+
P4-level LocalTemporalDifferenceModule
+
P4-level forward MemoryAttention
+
Dataset [B,N,T]
+
Dice/BCE/Tversky/Boundary/TemporalDelta loss
+
optimizer param groups
```

暂时不需要：

```text
MaskPromptEncoder 接入
Historical Review Module
Optical Flow
跨 forward persistent memory
复杂 sliding window merge
```

先把这个版本跑通，再做后续增强。

---

## 19. 推荐提交信息

修复完成后提交：

```bash
git add .
git commit -m "Fix TFCU inpaint integration for DINOv3 DPT-FPN"
git push origin HEAD:test_time
```

如果远程拒绝：

```bash
git push --force-with-lease origin HEAD:test_time
```

---

## 20. Agent 最终输出要求

Agent 修完后必须在回复中列出：

```text
1. 修改了哪些文件
2. TFCU 是否已被 build_model() 实际使用
3. Dataset 输出 shape
4. Model 输出 shape
5. Loss 组成
6. Optimizer 参数组
7. 通过了哪些命令
8. 如果有未完成项，明确列出
```
