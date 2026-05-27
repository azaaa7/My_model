# Video Inpainting Detection: DINOv3 + LoRA + TFCU-Inpaint Adapter 实施文档

> 目标：在现有 `DINOv3 ViT-L/16 + LoRA + DPT-FPN` 单帧/短视频分割 backbone 上，加入一个最小可行的 TFCU 风格时序模块，用于视频 inpainting 检测。  
> 核心原则：**不要重写 backbone，不要照搬 TFCU 的人脸 landmark 分支；只迁移“连续帧差异 + 历史 memory 前向累积”的机制。**

---

## 1. 当前 backbone 假设

当前项目已有如下能力：

```text
clip [B, T, 3, 512, 512]
  → ImageNet normalize
  → DINOv3 ViT-L/16 + LoRA
  → extract blocks 5, 11, 17, 23
  → DPT Reassemble Neck
  → FPN Decoder
  → logits [B, T, 1, 512, 512]
```

当前启用结构：

```yaml
use_dpt_fpn: true
extract_layers: "5,11,17,23"
neck_channels: 256
num_frames: 1
```

本方案要改成：

```text
video [B, N, T, 3, 512, 512]
  → DINOv3 + LoRA
  → DPT Reassemble Neck
  → P2, P3, P4, P5
  → 在 P4 插入 TFCU-Inpaint Adapter
  → FPN Decoder
  → logits [B, N, T, 1, 512, 512]
```

其中：

```text
B = batch size
N = 一个视频中采样的 clip 数量
T = 每个 clip 内连续帧数
```

第一版推荐：

```yaml
num_clips: 4
num_frames: 4
temporal_insert_level: P4
memory_len: 4
use_historical_review: false
use_flow: false
```

---

## 2. 总体实现顺序

Agent 必须按以下顺序实现，避免一次性改太多导致难以 debug。

```text
Step 1: Dataset 从 [B,T,C,H,W] 扩展到 [B,N,T,C,H,W]
Step 2: Model forward 支持 [B,N,T,3,H,W]
Step 3: 保持原模型输出不变，确保 baseline 跑通
Step 4: 实现 LocalTemporalDifferenceModule
Step 5: 在 P4 上 residual 注入 local temporal cue
Step 6: 实现 InpaintMemoryAttention
Step 7: 按 N 从前到后处理 clip，加入历史 memory
Step 8: 加 mask/boundary prompt encoder
Step 9: 加 temporal_delta_loss 和 boundary_loss
Step 10: 写 shape unit tests 与 overfit small batch 测试
```

---

## 3. 需要新增或修改的文件

推荐文件结构：

```text
models/
  video_inpaint_tfcu.py                # 新主模型 wrapper
  temporal/
    __init__.py
    local_temporal_difference.py       # 连续帧差异模块
    memory_attention.py                # 历史 memory attention
    mask_prompt_encoder.py             # mask/boundary prompt
    temporal_adapter.py                # TFCU-Inpaint Adapter 总入口

losses/
  boundary_loss.py
  temporal_delta_loss.py

datasets/
  video_inpainting_dataset.py          # 如果已有 dataset，则修改已有文件

configs/
  dinov3_vitl16_lora_tfcu_inpaint.yml

tests/
  test_video_inpaint_tfcu_shapes.py
```

如果当前项目不使用 `tests/`，也至少写一个可直接运行的脚本：

```text
debug/test_tfcu_shapes.py
```

---

## 4. Dataset 改造

### 4.1 输入输出格式

Dataset 的 `__getitem__` 必须返回：

```python
{
    "images": Tensor[N, T, 3, 512, 512],
    "masks":  Tensor[N, T, 1, 512, 512],
    "video_id": str,
    "frame_indices": Tensor[N, T]
}
```

DataLoader 后得到：

```python
images: [B, N, T, 3, 512, 512]
masks:  [B, N, T, 1, 512, 512]
```

### 4.2 时序顺序要求

必须按帧号排序：

```python
frame_paths = sorted(frame_paths, key=lambda p: int(Path(p).stem))
```

禁止使用默认字符串排序，因为：

```text
"10.png" 会排在 "2.png" 前面
```

### 4.3 训练采样逻辑

推荐第一版采样：

```python
def sample_train_clips(num_total_frames, num_clips=4, num_frames=4, stride=1):
    clip_len = num_frames * stride
    max_start = num_total_frames - clip_len

    if max_start < 0:
        # 帧数不足时，允许 repeat 最后一帧
        return pad_indices(num_total_frames, num_clips, num_frames)

    starts = random.sample(
        range(0, max_start + 1),
        k=min(num_clips, max_start + 1)
    )
    starts = sorted(starts)

    # 如果可选起点不足 num_clips，则重复采样并排序
    while len(starts) < num_clips:
        starts.append(random.randint(0, max_start))
    starts = sorted(starts)

    clips = []
    for s in starts:
        clip = [s + i * stride for i in range(num_frames)]
        clips.append(clip)

    return clips
```

关键：

```text
一个 sample 内部必须保持从前到后：
clip 0 < clip 1 < clip 2 < clip 3
每个 clip 内部必须保持：
frame t0 < t1 < t2 < t3
```

### 4.4 验证/测试采样逻辑

验证和测试不要随机采样。使用顺序滑窗：

```python
def sample_eval_clips(num_total_frames, num_clips=4, num_frames=4, stride=1):
    clips = []
    step = num_frames

    for start in range(0, num_total_frames, step):
        clip = [min(start + i * stride, num_total_frames - 1) for i in range(num_frames)]
        clips.append(clip)

    # 分块返回，每次最多 num_clips 个 clip
    return chunks(clips, num_clips)
```

推理时同一个视频内必须保持 window 顺序，换视频时清空 memory。

---

## 5. Model forward 改造

### 5.1 支持新输入维度

新增 wrapper：

```python
class VideoInpaintTFCU(nn.Module):
    def forward(self, video):
        # video: [B, N, T, 3, H, W]
        B, N, T, C, H, W = video.shape
        x = video.reshape(B * N * T, C, H, W)

        P2, P3, P4, P5 = self.extract_fpn_features(x)

        # P4: [B*N*T, 256, 32, 32]
        P4_out = self.temporal_adapter(P4, B=B, N=N, T=T)

        logits = self.decoder(P2, P3, P4_out, P5)
        logits = logits.reshape(B, N, T, 1, H, W)

        return logits
```

### 5.2 第一阶段必须保证不改变原始结果

在 temporal adapter 中加入可学习残差系数：

```python
self.alpha = nn.Parameter(torch.tensor(0.0))
```

forward：

```python
P4_out = P4 + self.alpha * temporal_delta
```

初始化时 `alpha=0`，模型等价于原始单帧模型。这样能避免新模块随机初始化导致训练崩溃。

---

## 6. LocalTemporalDifferenceModule

### 6.1 目标

捕捉相邻帧的局部时序不一致：

```text
inpainting 区域在单帧上可能很自然，
但跨帧纹理、边界、运动一致性常出现异常。
```

### 6.2 输入输出

输入：

```python
x: [B, N, T, C, H, W]
```

输出：

```python
out: [B, N, T, C, H, W]
```

### 6.3 推荐实现

文件：`models/temporal/local_temporal_difference.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LocalTemporalDifferenceModule(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, N, T, C, H, W]
        B, N, T, C, H, W = x.shape

        prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
        diff = x - prev
        abs_diff = diff.abs()

        feat = torch.cat([x, diff, abs_diff], dim=3)
        feat = feat.reshape(B * N * T, C * 3, H, W)

        delta = self.fuse(feat)
        gate = self.gate(feat)

        delta = delta * gate
        delta = delta.reshape(B, N, T, C, H, W)

        return x + delta
```

### 6.4 注意事项

第一帧没有前一帧，使用自身作为 prev：

```python
prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
```

这样第一帧 diff 为 0，不会引入随机噪声。

---

## 7. InpaintMemoryAttention

### 7.1 目标

按 clip 顺序从前往后累积历史线索：

```text
clip 0 → 生成 memory_0
clip 1 → 使用 memory_0 → 生成 memory_1
clip 2 → 使用 memory_0, memory_1 → 生成 memory_2
...
```

当前 clip 只能读取过去 memory，不能读取未来。

### 7.2 输入输出

当前 clip：

```python
cur: [B, T, C, H, W]
```

历史 memory：

```python
mem: [B, K, T, C, H, W]
```

输出：

```python
out: [B, T, C, H, W]
```

### 7.3 推荐实现

文件：`models/temporal/memory_attention.py`

```python
import torch
import torch.nn as nn


class InpaintMemoryAttention(nn.Module):
    def __init__(self, channels=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, cur, mem):
        # cur: [B, T, C, H, W]
        # mem: [B, K, T, C, H, W]
        B, T, C, H, W = cur.shape
        K = mem.shape[1]

        # tokens
        q = cur.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        kv = mem.permute(0, 1, 2, 4, 5, 3).reshape(B, K * T * H * W, C)

        qn = self.norm1(q)
        kvn = self.norm1(kv)

        attn_out, _ = self.attn(
            query=self.q_proj(qn),
            key=self.k_proj(kvn),
            value=self.v_proj(kvn),
            need_weights=False,
        )

        x = q + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))

        out = x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        return out
```

### 7.4 显存风险

`T*H*W` token 数可能偏大。

第一版 P4：

```text
T=4, H=W=32 → 4096 query tokens
K=4 → 16384 memory tokens
```

如果 OOM，使用以下降级方案：

```text
方案 A：在 attention 前把 P4 average pool 到 16×16，attention 后上采样回 32×32
方案 B：只用每个 clip 的 temporal summary token 做 memory
方案 C：把 memory attention 换成 ConvGRU
```

优先实现方案 A：

```python
cur_small = F.avg_pool3d_like_or_reshape_pool(cur, spatial_size=16)
mem_small = pool_memory(mem, spatial_size=16)
```

如果 agent 实现时不确定，先保留 32×32，配合 `batch_size=1` 和 AMP 测试。

---

## 8. MaskPromptEncoder

### 8.1 目标

TFCU 原论文使用人脸 landmarks 作为结构提示。视频 inpainting 检测不适合 landmark，因此改成：

```text
历史预测 mask prompt
历史预测 boundary prompt
```

### 8.2 第一版实现策略

第一版可以不让 memory 依赖 ground truth mask，只用预测 mask。训练和推理一致。

输入：

```python
mask_logits: [B, T, 1, 512, 512]
```

下采样：

```python
mask_32 = interpolate(sigmoid(mask_logits), size=(32, 32))
```

边界：

```python
boundary_32 = sobel(mask_32)
```

编码：

```python
prompt = Conv([mask_32, boundary_32]) → [B, T, C, 32, 32]
```

### 8.3 推荐实现

文件：`models/temporal/mask_prompt_encoder.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskPromptEncoder(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    @staticmethod
    def sobel_boundary(mask):
        # mask: [B*T, 1, H, W]
        device = mask.device
        dtype = mask.dtype

        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(mask, kx, padding=1)
        gy = F.conv2d(mask, ky, padding=1)

        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, mask_logits, out_size):
        # mask_logits: [B, T, 1, H_img, W_img]
        # out_size: e.g. (32, 32)
        B, T, _, H_img, W_img = mask_logits.shape

        mask = torch.sigmoid(mask_logits)
        mask = mask.reshape(B * T, 1, H_img, W_img)
        mask = F.interpolate(mask, size=out_size, mode="bilinear", align_corners=False)

        boundary = self.sobel_boundary(mask)

        prompt = torch.cat([mask, boundary], dim=1)
        prompt = self.encoder(prompt)

        C = prompt.shape[1]
        prompt = prompt.reshape(B, T, C, out_size[0], out_size[1])
        return prompt
```

---

## 9. TFCU-Inpaint Adapter 总模块

### 9.1 职责

这个模块接收 P4：

```python
P4: [B*N*T, C, H, W]
```

返回增强后的 P4：

```python
P4_out: [B*N*T, C, H, W]
```

### 9.2 第一版不依赖 mask prompt

先实现最小版本：

```text
LocalTemporalDifferenceModule
+
InpaintMemoryAttention
+
residual alpha
```

### 9.3 第二版加入 mask prompt

需要当前 clip 的初步 mask。实现方式有两种：

```text
方案 A：增加一个 lightweight aux mask head，在 P4 上预测 coarse mask
方案 B：使用上一轮/上一 clip decoder 输出作为历史 mask prompt
```

推荐第一版先用方案 A，简单稳定：

```python
coarse_logits = self.coarse_head(cur_feature)
prompt = self.mask_prompt_encoder(coarse_logits)
memory = feature + prompt
```

### 9.4 推荐实现

文件：`models/temporal/temporal_adapter.py`

```python
import torch
import torch.nn as nn

from .local_temporal_difference import LocalTemporalDifferenceModule
from .memory_attention import InpaintMemoryAttention


class TFCUInpaintAdapter(nn.Module):
    def __init__(
        self,
        channels=256,
        memory_len=4,
        use_memory=True,
    ):
        super().__init__()
        self.channels = channels
        self.memory_len = memory_len
        self.use_memory = use_memory

        self.local = LocalTemporalDifferenceModule(channels=channels)
        self.memory_attn = InpaintMemoryAttention(channels=channels, num_heads=8)

        self.temporal_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def encode_memory(self, cur):
        # cur: [B, T, C, H, W]
        # 第一版直接 detach，避免 memory 过长导致梯度图过大
        return cur.detach()

    def forward(self, P4, B, N, T):
        # P4: [B*N*T, C, H, W]
        _, C, H, W = P4.shape

        x = P4.reshape(B, N, T, C, H, W)

        # 1. local consecutive difference
        x = self.local(x)

        # 2. forward historical memory
        enhanced = []
        state = []

        for n in range(N):
            cur = x[:, n]  # [B, T, C, H, W]

            if (not self.use_memory) or len(state) == 0:
                cur_enhanced = cur
            else:
                mem = torch.stack(state[-self.memory_len:], dim=1)
                cur_enhanced = self.memory_attn(cur, mem)

            enhanced.append(cur_enhanced)
            state.append(self.encode_memory(cur_enhanced))

        temporal = torch.stack(enhanced, dim=1)
        temporal = temporal.reshape(B * N * T, C, H, W)

        temporal = self.temporal_proj(temporal)

        # 3. residual injection
        P4_out = P4 + self.alpha * temporal

        return P4_out
```

### 9.5 为什么 memory 要先 detach

第一版推荐：

```python
return cur.detach()
```

原因：

```text
1. 避免跨多个 clip 反传导致显存爆炸
2. 让 memory 更像推理状态
3. 降低训练不稳定
```

第二版可以加配置：

```yaml
detach_memory: true
```

如果显存允许，再尝试 `detach_memory: false`。

---

## 10. 主模型 wrapper

文件：`models/video_inpaint_tfcu.py`

下面是推荐伪代码，agent 需要根据项目现有类名适配。

```python
import torch
import torch.nn as nn

from models.temporal.temporal_adapter import TFCUInpaintAdapter


class VideoInpaintTFCU(nn.Module):
    def __init__(self, base_model, cfg):
        super().__init__()

        # base_model 是当前已有的 DINOv3+LoRA+DPT-FPN 模型
        self.base = base_model

        self.temporal_adapter = TFCUInpaintAdapter(
            channels=cfg.get("neck_channels", 256),
            memory_len=cfg.get("memory_len", 4),
            use_memory=cfg.get("use_memory", True),
        )

    def extract_fpn_features(self, x):
        # 这里必须复用现有 base_model 的特征提取逻辑
        # 目标返回：
        # P2 [BNT,256,128,128]
        # P3 [BNT,256, 64, 64]
        # P4 [BNT,256, 32, 32]
        # P5 [BNT,256, 16, 16]
        return self.base.extract_fpn_features(x)

    def decode(self, P2, P3, P4, P5):
        return self.base.decoder(P2, P3, P4, P5)

    def forward(self, video):
        # video: [B,N,T,3,512,512]
        if video.dim() == 5:
            # 兼容旧输入 [B,T,3,H,W]
            B, T, C, H, W = video.shape
            N = 1
            video = video[:, None]
        else:
            B, N, T, C, H, W = video.shape

        x = video.reshape(B * N * T, C, H, W)

        P2, P3, P4, P5 = self.extract_fpn_features(x)

        P4 = self.temporal_adapter(P4, B=B, N=N, T=T)

        logits = self.decode(P2, P3, P4, P5)
        logits = logits.reshape(B, N, T, 1, H, W)

        if N == 1:
            logits = logits[:, 0]  # [B,T,1,H,W], 兼容旧训练代码

        return logits
```

如果现有模型没有 `extract_fpn_features()`，agent 需要从当前 `forward()` 中拆出：

```text
DINO feature extraction
DPT Reassemble Neck
FPN feature construction
decoder
```

拆分后原模型和新模型都复用同一套函数，避免重复代码。

---

## 11. Loss 改造

### 11.1 保留现有主损失

继续使用：

```text
Dice + BCE + Tversky
```

### 11.2 新增 BoundaryLoss

文件：`losses/boundary_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sobel(x):
        device = x.device
        dtype = x.dtype

        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, logits, target):
        # logits/target: [B,N,T,1,H,W] or [B,T,1,H,W]
        if logits.dim() == 6:
            B, N, T, C, H, W = logits.shape
            logits = logits.reshape(B * N * T, C, H, W)
            target = target.reshape(B * N * T, C, H, W)
        elif logits.dim() == 5:
            B, T, C, H, W = logits.shape
            logits = logits.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        pred = torch.sigmoid(logits)
        pred_b = self.sobel(pred)
        target_b = self.sobel(target.float())

        return F.l1_loss(pred_b, target_b)
```

### 11.3 新增 TemporalDeltaLoss

文件：`losses/temporal_delta_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDeltaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        # logits/target: [B,N,T,1,H,W] or [B,T,1,H,W]
        if logits.dim() == 5:
            logits = logits[:, None]
            target = target[:, None]

        pred = torch.sigmoid(logits)

        dp = pred[:, :, 1:] - pred[:, :, :-1]
        dg = target[:, :, 1:].float() - target[:, :, :-1].float()

        return F.l1_loss(dp, dg)
```

注意：如果 `T=1`，该 loss 应返回 0：

```python
if logits.shape[2] <= 1:
    return logits.sum() * 0.0
```

agent 实现时必须加这个保护。

### 11.4 推荐总 loss

```yaml
loss:
  dice:           {weight: 1.0, smooth: 1.0e-6}
  bce:            {weight: 0.5}
  tversky:        {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
  boundary:       {weight: 0.2}
  temporal_delta: {weight: 0.1}
```

---

## 12. 配置文件

新增：`configs/dinov3_vitl16_lora_tfcu_inpaint.yml`

```yaml
model:
  name: VideoInpaintTFCU

  backbone: dinov3_vitl16_lora
  use_lora: true
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.1
  lora_targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"

  use_dpt_fpn: true
  extract_layers: "5,11,17,23"
  neck_channels: 256

  temporal_adapter: true
  temporal_insert_level: P4
  num_clips: 4
  num_frames: 4
  memory_len: 4
  use_memory: true
  detach_memory: true
  use_historical_review: false
  use_mask_prompt: false
  use_flow: false

train:
  image_size: 512
  batch_size: 1
  grad_accum_steps: 8
  amp: true

  learning_rate: 1.0e-4
  lr_decoder: 1.0e-4
  lr_temporal: 1.0e-4
  lr_lora: 1.0e-5
  weight_decay: 1.0e-4

  freeze_dino: true
  train_lora: true
  train_decoder: true
  train_temporal: true

  n_epochs: 1000
  validate_every: 10

scheduler:
  name: plateau
  monitor: val_iou
  patience: 3
  factor: 0.5
  min_lr: 1.0e-6

loss:
  dice:           {weight: 1.0, smooth: 1.0e-6}
  bce:            {weight: 0.5}
  tversky:        {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
  boundary:       {weight: 0.2}
  temporal_delta: {weight: 0.1}

eval:
  eval_frame_chunk: 1
  threshold: 0.5
  sliding_window: true
  overlap: 0
```

---

## 13. Optimizer 参数组

必须给 temporal module 单独学习率。

```python
param_groups = [
    {
        "params": model.temporal_adapter.parameters(),
        "lr": cfg.train.lr_temporal,
        "weight_decay": cfg.train.weight_decay,
    },
    {
        "params": model.base.decoder.parameters(),
        "lr": cfg.train.lr_decoder,
        "weight_decay": cfg.train.weight_decay,
    },
]

if cfg.train.train_lora:
    param_groups.append({
        "params": lora_parameters,
        "lr": cfg.train.lr_lora,
        "weight_decay": cfg.train.weight_decay,
    })

optimizer = torch.optim.AdamW(param_groups)
```

推荐：

```text
temporal adapter: 1e-4
decoder:          1e-4
LoRA:             1e-5
DINO base:        frozen
```

---

## 14. 推理逻辑

### 14.1 单视频顺序推理

伪代码：

```python
model.eval()

all_logits = []

for clip_group in sequential_clip_groups(video_frames, num_clips=4, num_frames=4):
    # clip_group: [1,N,T,3,H,W]
    logits = model(clip_group)
    all_logits.append(logits)

final_logits = merge_logits_by_frame_index(all_logits)
```

### 14.2 memory 状态

当前推荐实现中 memory 在 `forward()` 内部建立和清空：

```text
每次 forward 处理 N 个 clips
forward 结束 memory 自动释放
```

这最简单，也最安全。

如果后续想跨 window 传递 memory，需要把 `state` 暴露成：

```python
model.reset_memory()
logits, state = model(video, state=state)
```

第一版不要做跨 forward memory，避免实现复杂。

### 14.3 重叠 window

第一版：

```yaml
overlap: 0
```

第二版可以改：

```yaml
overlap: 2
```

重叠帧 logits 平均：

```python
final_logits[frame_id] = mean(logits_list_for_frame_id)
```

---

## 15. Shape tests

新增测试文件：`tests/test_video_inpaint_tfcu_shapes.py`

至少测试以下情况：

### 15.1 LocalTemporalDifferenceModule

```python
x = torch.randn(2, 4, 4, 256, 32, 32)
m = LocalTemporalDifferenceModule(256)
y = m(x)
assert y.shape == x.shape
```

### 15.2 InpaintMemoryAttention

```python
cur = torch.randn(1, 4, 256, 32, 32)
mem = torch.randn(1, 4, 4, 256, 32, 32)
m = InpaintMemoryAttention(256)
y = m(cur, mem)
assert y.shape == cur.shape
```

### 15.3 TFCUInpaintAdapter

```python
P4 = torch.randn(1 * 4 * 4, 256, 32, 32)
m = TFCUInpaintAdapter(256, memory_len=4)
y = m(P4, B=1, N=4, T=4)
assert y.shape == P4.shape
```

### 15.4 主模型输出

```python
video = torch.randn(1, 4, 4, 3, 512, 512)
logits = model(video)
assert logits.shape == (1, 4, 4, 1, 512, 512)
```

---

## 16. Overfit small batch 测试

在正式训练前，必须做一个 overfit 测试：

```text
取 2 个视频样本
batch_size=1
num_clips=2
num_frames=4
训练 200~500 iterations
```

预期：

```text
loss 明显下降
train IoU 明显上升
预测 mask 从随机噪声变成接近 GT
```

如果无法 overfit，优先排查：

```text
1. mask 是否正确读取
2. images/masks 时序是否对齐
3. logits 与 target shape 是否一致
4. loss 是否对 [B,N,T,1,H,W] 正确 flatten
5. DINO normalize 是否仍然正确
6. alpha 是否一直为 0 没有学习
```

---

## 17. Ablation 实验顺序

为了证明模块有效，按以下顺序跑实验：

```text
A0: 原始 DINOv3 + LoRA + DPT-FPN，num_frames=1
A1: 原始模型支持 [B,N,T]，但不加 temporal adapter
A2: A1 + LocalTemporalDifferenceModule
A3: A2 + forward MemoryAttention
A4: A3 + BoundaryLoss + TemporalDeltaLoss
A5: A4 + MaskPromptEncoder
```

第一篇报告中最重要对比：

```text
A0 vs A2：连续帧差异是否有效
A2 vs A3：历史 memory 是否有效
A3 vs A4：边界和时序 loss 是否提升稳定性
```

---

## 18. 常见错误与修复

### 18.1 OOM

优先降低：

```yaml
batch_size: 1
num_clips: 2
num_frames: 4
memory_len: 2
```

然后再尝试：

```text
把 memory attention 的 H,W 从 32×32 pool 到 16×16
启用 gradient checkpointing
暂时冻结 LoRA
```

### 18.2 输出闪烁

增加：

```yaml
temporal_delta: {weight: 0.1}
```

如果过强导致 mask 迟钝，降到：

```yaml
temporal_delta: {weight: 0.05}
```

### 18.3 recall 低

调高 Tversky 的 FN 惩罚：

```yaml
tversky:
  alpha: 0.3
  beta: 0.7
```

或降低 threshold：

```bash
--threshold 0.3
```

### 18.4 false positive 多

降低 temporal adapter 学习率：

```yaml
lr_temporal: 5.0e-5
```

并考虑降低 boundary loss：

```yaml
boundary: {weight: 0.1}
```

---

## 19. 实现验收标准

第一版完成后必须满足：

```text
1. 支持输入 [B,N,T,3,512,512]
2. 输出 [B,N,T,1,512,512]
3. N 维度按时间从前到后循环处理
4. 当前 clip 只能使用历史 memory
5. 不使用未来 clip 信息
6. alpha 初始化为 0，模型可退化为原始单帧模型
7. Dataset 内部帧顺序严格递增
8. 可以在 2 个样本上 overfit
9. 验证/测试使用顺序采样，不随机打乱时序
```

---

## 20. 最终推荐第一版实现范围

第一版只实现这些：

```text
必须实现：
  - [B,N,T] 数据输入
  - P4-level LocalTemporalDifferenceModule
  - P4-level forward MemoryAttention
  - residual alpha injection
  - BoundaryLoss
  - TemporalDeltaLoss
  - shape tests
  - overfit small batch

暂不实现：
  - Historical Review Module
  - Optical Flow
  - Cross-forward persistent memory
  - 复杂 mask prompt
  - 双向时序 attention
```

这样是当前任务最稳、最小、最容易验证的 TFCU adaptation。

---

## 21. Agent 执行 Checklist

```text
[ ] 备份当前可训练版本
[ ] 新增 config: dinov3_vitl16_lora_tfcu_inpaint.yml
[ ] Dataset 返回 [N,T,3,H,W] 和 [N,T,1,H,W]
[ ] DataLoader 后确认 [B,N,T,3,H,W]
[ ] 拆分 base model: extract_fpn_features() / decoder()
[ ] 新增 LocalTemporalDifferenceModule
[ ] 新增 InpaintMemoryAttention
[ ] 新增 TFCUInpaintAdapter
[ ] 在 P4 插入 adapter
[ ] 主模型 forward 输出 [B,N,T,1,H,W]
[ ] 修改 loss flatten 逻辑
[ ] 新增 BoundaryLoss
[ ] 新增 TemporalDeltaLoss
[ ] 加 optimizer param groups
[ ] 写 shape tests
[ ] 跑 overfit small batch
[ ] 跑 A0/A2/A3 ablation
[ ] 再决定是否加入 MaskPromptEncoder
```
