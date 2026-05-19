# DINOv3 ViT-L/16 + Baseline-Preserving EAM MVP 技术规格（推荐修正版）

目标：在当前效果最好的 **DINOv3 ViT-L/16 + LoRA + 简单 coarse decoder** 基础上，加入可控的多层特征、渐进式上采样、memory/error 模块，但**不能破坏原始强 baseline**。

本修正版的核心原则是：

```text
不要替换原始 DINOv3 + simple decoder。
保留它作为主预测分支。
新增模块只能作为 gated residual / auxiliary refinement。
```

也就是说，本版本不再采用：

```text
multi-layer fusion -> query decoder -> progressive upsampling -> final mask
```

作为唯一主路径。

而是改成：

```text
DINOv3 last-layer feature
    ↓
Original Coarse Mask Head
    ↓
coarse logits at H/16
    ↓
bilinear upsample
    ↓
coarse logits at H
    ↓
final logits main source
```

再并联一个轻量 refinement 分支：

```text
selected DINOv3 multi-layer features
    ↓
gated residual fusion
    ↓
optional memory / error embedding
    ↓
lightweight residual progressive decoder
    ↓
residual logits at H
```

最终输出：

```python
final_logits = coarse_logits_up + lambda_residual * residual_logits
```

其中：

```yaml
lambda_residual: 0.0 -> 0.1 或 0.2 warmup
```

这样模型最差可以退化回原始强 baseline，最好由新模块补充边界和细节。

---

## 0. 当前已验证最强 baseline

当前最强结构如下：

```text
DINOv3 ViT-L/16 as encoder
patch_size = 16
embed_dim = 1024
depth = 24
num_heads = 16

对每一个 ViT block 的 QKV 和线性层做 LoRA 微调
24 个 ViT blocks
48 个 LoRA
rank = 4
```

decoder：

```text
[B*T, 1024, H/16, W/16]
    ↓
Conv2d(1024, 256, kernel_size=3, padding=1, bias=False)
BatchNorm2d(256)
ReLU
    ↓
Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
BatchNorm2d(128)
ReLU
    ↓
Conv2d(128, 1, kernel_size=1)
    ↓
logits [B*T, 1, H/16, W/16]
    ↓
bilinear interpolate
    ↓
logits [B*T, 1, H, W]
    ↓
reshape
    ↓
logits [B, T, 1, H, W]
```

参数量参考：

```text
total parameters:     306,399,105
trainable parameters:   3,244,929
```

训练配置参考：

```yaml
input_size: 512
gt_ratio: 1
num_frames: 1
batch_size: 16
epochs: 1000
```

该 baseline 强的原因：

```text
1. DINOv3 last-layer token 已经有很强区域级异常判别能力。
2. H/16 low-res logits + bilinear upsample 起到了天然正则化作用。
3. 简单 decoder 参数少，优化稳定，不容易过拟合边界噪声。
4. num_frames=1 时，复杂 temporal/memory 模块还不能真正发挥视频优势。
```

因此，后续任何修改都必须保留这一条主分支。

---

## 1. 本修正版最终结构

推荐模型名：

```python
class DINOv3EAMBaselinePreservingMVP(nn.Module):
    ...
```

整体结构：

```text
Input clip [B,T,3,H,W]
    ↓
DINOv3 ViT-L/16 + LoRA
    ↓
取 last layer feature f24: [B*T,1024,h,w]
    ↓
Original Coarse Mask Head
    ↓
coarse_logits: [B*T,1,h,w]
    ↓
bilinear upsample
    ↓
coarse_logits_up: [B*T,1,H,W]
    ↓
                  ┌─────────────────────────────────────┐
                  │                                     │
                  │ main branch                         │ residual branch
                  │                                     │
                  │                                     ↓
                  │        Gated Multi-layer Fusion / optional EAM features
                  │                                     ↓
                  │        Lightweight Residual Progressive Decoder
                  │                                     ↓
                  │        residual_logits: [B*T,1,H,W]
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ↓
          final_logits = coarse_logits_up + lambda_residual * residual_logits
                                     ↓
          output [B,T,1,H,W]
```

---

## 2. 前向接口

```python
def forward(
    self,
    clip: torch.Tensor,                 # [B,T,3,H,W]
    mask: Optional[torch.Tensor] = None, # [B,1,H,W] 或 [B,T,1,H,W]
    update_memory: bool = False,
    return_aux: bool = True,
) -> Dict[str, torch.Tensor]:
    ...
```

输出字典：

```python
{
    "mask_logits": final_logits,          # [B,T,1,H,W]
    "coarse_logits": coarse_logits,       # [B,T,1,h,w]
    "coarse_logits_up": coarse_logits_up, # [B,T,1,H,W]
    "residual_logits": residual_logits,   # [B,T,1,H,W] or zeros
    "fused_feat": fused_feat,             # [B,T,C,h,w]
    "error_map": error_map,               # [B,T,1,h,w] or None
    "accum_error": accum_error,           # [B,T,1,h,w] or None
    "edge_logits": edge_logits,           # [B,T,1,H,W] or None
    "aux": aux_dict
}
```

兼容旧训练脚本：

```python
outputs = model(clip, mask=target, update_memory=model.training)

if isinstance(outputs, dict):
    logits = outputs["mask_logits"]
else:
    logits = outputs
```

---

## 3. DINOv3 特征提取

输入：

```text
clip: [B,T,3,H,W]
frames = clip.reshape(B*T,3,H,W)
```

推荐先取以下层：

```yaml
selected_layers: [17, 23]
```

也就是先只用：

```text
layer 18 + layer 24
```

不要一开始就用 `[5, 11, 17, 23]`。

原因：

```text
layer 5 / layer 11 更偏低级纹理与局部噪声；
它们可能会污染最后一层稳定的区域级异常信息。
```

后续消融再逐步尝试：

```yaml
selected_layers: [11, 17, 23]
selected_layers: [5, 11, 17, 23]
```

DINO 提取：

```python
layer_outputs = backbone.get_intermediate_layers(
    frames,
    n=selected_layers,
    reshape=True,
    norm=True,
)
```

每层输出：

```text
[B*T,1024,h,w], h=H/16, w=W/16
```

必须保证最后一层 `f24` 单独保留，供 coarse head 使用：

```python
f_last = layer_outputs[-1]  # [B*T,1024,h,w]
```

---

## 4. 主分支：Original Coarse Mask Head

必须保留原始 decoder，不要替换。

```python
class CoarseMaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, f_last, out_size):
        coarse_logits = self.net(f_last)  # [BT,1,h,w]
        coarse_logits_up = F.interpolate(
            coarse_logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
        return coarse_logits, coarse_logits_up
```

注意：

```text
coarse_logits_up 是 final_logits 的主来源。
```

---

## 5. 多层特征融合：Gated Residual Fusion

不要直接 concat 4 层后作为唯一 decoder 输入。

推荐做法：

```text
最后一层为主；
其他层只是小权重 residual。
```

### 5.1 输入

```text
f18: [BT,1024,h,w]
f24: [BT,1024,h,w]
```

### 5.2 投影

```python
proj18: Conv1x1(1024 -> 256) + GroupNorm + GELU
proj24: Conv1x1(1024 -> 256) + GroupNorm + GELU
```

### 5.3 门控残差融合

```python
main = proj24(f24)
aux18 = proj18(f18)

gate18 = sigmoid(alpha18) * max_aux_scale
fused_feat = main + gate18 * aux18
```

推荐初始化：

```python
alpha18 = nn.Parameter(torch.tensor(-6.0))
max_aux_scale = 0.1
```

解释：

```text
sigmoid(-6) ≈ 0.0025
gate18 ≈ 0.00025
```

这样训练一开始几乎等价于只用最后一层，避免多层特征一开始破坏 baseline。

如果使用更多层：

```python
fused_feat = main
for layer_i in aux_layers:
    fused_feat = fused_feat + sigmoid(alpha_i) * max_aux_scale * proj_i(f_i)
```

推荐默认：

```yaml
fusion:
  enabled: true
  selected_layers: [17, 23]
  main_layer: 23
  aux_layers: [17]
  encoder_dim: 256
  max_aux_scale: 0.1
  init_alpha: -6.0
```

---

## 6. Residual Progressive Decoder

渐进式上采样不能直接输出最终 mask。

错误设计：

```text
progressive_decoder(...) -> final_logits
```

正确设计：

```text
progressive_decoder(...) -> residual_logits
final_logits = coarse_logits_up + lambda_residual * residual_logits
```

### 6.1 输入

最小版本只输入：

```text
fused_feat:        [BT,256,h,w]
coarse_logits:     [BT,1,h,w]
```

concat：

```text
[BT,257,h,w]
```

如果 memory 开启，再加入：

```text
error_map:         [BT,1,h,w]
accum_error:       [BT,1,h,w]
```

concat：

```text
[BT,259,h,w]
```

### 6.2 推荐轻量结构

不要用太重的 decoder。推荐：

```text
1/16:
  Conv1x1(in_channels -> 128)
  GroupNorm
  GELU

1/8:
  bilinear upsample x2
  Conv3x3(128 -> 96)
  GroupNorm
  GELU

1/4:
  bilinear upsample x2
  Conv3x3(96 -> 64)
  GroupNorm
  GELU

1/2:
  bilinear upsample x2
  Conv3x3(64 -> 32)
  GroupNorm
  GELU

Full:
  bilinear upsample x2
  Conv3x3(32 -> 16)
  GroupNorm
  GELU
  Conv1x1(16 -> 1)
```

输出：

```text
residual_logits: [BT,1,H,W]
```

### 6.3 residual 缩放

最终：

```python
final_logits = coarse_logits_up + lambda_residual * residual_logits
```

推荐配置：

```yaml
residual_decoder:
  enabled: true
  channels: [128, 96, 64, 32, 16]
  lambda_residual: 0.1
  warmup_epochs: 5
  max_lambda_residual: 0.1
```

训练时：

```python
if current_epoch < warmup_epochs:
    lambda_res = max_lambda_residual * current_epoch / warmup_epochs
else:
    lambda_res = max_lambda_residual
```

如果修改训练脚本比较麻烦，先固定：

```yaml
lambda_residual: 0.1
```

如果效果仍下降，改为：

```yaml
lambda_residual: 0.05
```

---

## 7. Normality Memory Bank：改为可选辅助，不参与主路径强控制

Memory 模块可以保留，但不要让它主导 decoder。

### 7.1 默认建议

在当前 `num_frames=1` 阶段，推荐默认：

```yaml
memory.enabled: false
temporal.enabled: false
```

原因：

```text
num_frames=1 时，temporal accumulation 没有真实时序信息；
memory/error 如果直接加入 decoder，容易干扰原始 strong baseline。
```

当 baseline-preserving residual 结构跑通后，再启用：

```yaml
memory.enabled: true
temporal.enabled: false
```

最后当 `num_frames >= 4` 时，再启用：

```yaml
temporal.enabled: true
```

### 7.2 如果启用 memory

memory 只输出辅助图：

```text
error_map: [B,T,1,h,w]
```

不要替代 DINO feature。

推荐融合方式：

```python
residual_input = concat(
    fused_feat,
    coarse_logits,
    detach_or_not(error_map)
)
```

建议初期 detach：

```yaml
memory.detach_error_for_decoder: true
```

也就是：

```python
error_for_decoder = error_map.detach()
```

这样 memory error 不会通过 decoder loss 反向把 memory/readout 拉偏。

### 7.3 Memory bank 基本实现

memory 作为 buffer：

```python
self.register_buffer("memory", torch.randn(num_slots, dim))
self.register_buffer("memory_valid", torch.zeros(num_slots, dtype=torch.bool))
self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
```

memory read：

```python
tokens_n = F.normalize(tokens, dim=-1)
memory_n = F.normalize(memory, dim=-1)
sim = torch.matmul(tokens_n, memory_n.t())     # [BT,N,K]
topv, topi = torch.topk(sim, k=topk, dim=-1)
attn = F.softmax(topv / temperature, dim=-1)
mem_selected = memory[topi]                    # [BT,N,topk,C]
recon = (attn.unsqueeze(-1) * mem_selected).sum(dim=2)
error = 1 - F.cosine_similarity(tokens, recon, dim=-1)
```

输出：

```text
memory_recon: [B,T,C,h,w]
error_map:    [B,T,1,h,w]
```

### 7.4 Memory update

训练时可用 GT normal patch 更新：

```text
mask_low < normal_mask_threshold
```

推理默认不更新：

```python
update_memory = self.training and mask is not None
```

推荐配置：

```yaml
memory:
  enabled: false
  num_slots: 4096
  topk: 16
  temperature: 0.07
  update_momentum: 0.95
  max_update_tokens_per_batch: 2048
  normal_mask_threshold: 0.2
  anomaly_margin: 0.5
  update_in_eval: false
  detach_error_for_decoder: true
```

---

## 8. Temporal Error Accumulation：只在多帧训练时启用

当前 `num_frames=1` 时，不要启用 temporal。

推荐逻辑：

```python
if T == 1 or not self.temporal_enabled:
    accum_error = error_map
else:
    accum_error = self.temporal_accumulator(fused_feat, error_map)
```

只有当：

```yaml
num_frames: 4 或 8
```

时，才开启：

```yaml
temporal.enabled: true
```

### 8.1 Token-correlation alignment

保持原 MVP 逻辑：

```text
prev_feat:    [B,C,h,w]
curr_feat:    [B,C,h,w]
prev_accum:   [B,1,h,w]
error_t:      [B,1,h,w]
```

局部窗口：

```yaml
corr_radius: 2
corr_temperature: 0.07
```

对齐：

```python
prev_feat_patch = F.unfold(prev_feat, kernel_size=5, padding=2)
curr = curr_feat.view(B,C,h*w).unsqueeze(2)
sim = (normalize(curr) * normalize(prev_feat_patch)).sum(dim=1)
weight = F.softmax(sim / corr_temperature, dim=1)

prev_acc_patch = F.unfold(prev_accum, kernel_size=5, padding=2)
aligned_prev = (weight * prev_acc_patch).sum(dim=1).view(B,1,h,w)
```

gated accumulation：

```python
gate = sigmoid(gate_net(concat(error_t, aligned_prev, error_t - aligned_prev)))
accum_t = gate * error_t + (1 - gate) * aligned_prev
```

推荐配置：

```yaml
temporal:
  enabled: false
  use_token_correlation_alignment: true
  corr_radius: 2
  corr_temperature: 0.07
  detach_temporal_state: false
  clamp_max: 2.0
  use_gate: true
```

---

## 9. 可选 Edge Head：只做弱辅助

删除所有频域 cue 后，edge head 只能从 decoder feature 里预测边界。

不要使用：

```text
Sobel
Laplacian
SRM
high-pass residual
RGB + edge cue 5-channel input
```

默认建议关闭：

```yaml
edge_head.enabled: false
loss.lambda_edge: 0.0
```

如果 residual structure 跑通后再打开：

```yaml
edge_head.enabled: true
loss.lambda_edge: 0.05
```

不要一开始设 0.2。

原因：

```text
过强 boundary loss 会让模型过拟合 mask 边缘和标注噪声，损害区域级检测。
```

---

## 10. Loss 设计：必须加入 coarse loss

这是本修正版最关键的训练改动。

总 loss：

```python
loss = loss_full + lambda_coarse * loss_coarse
loss += lambda_residual_reg * loss_residual_reg
loss += lambda_mem * loss_mem
loss += lambda_edge * loss_edge
loss += lambda_temp * loss_temp
```

推荐默认：

```yaml
loss:
  lambda_coarse: 0.5
  lambda_residual_reg: 0.01
  lambda_mem: 0.0
  lambda_edge: 0.0
  lambda_temp: 0.0
```

### 10.1 Full-resolution segmentation loss

```python
loss_full = SegmentationLoss(final_logits, target)
```

其中 `SegmentationLoss` 保持当前工程已有：

```text
FocalLoss + BCEWithLogitsLoss + IoULoss
```

### 10.2 Coarse loss

把 GT 下采样到 H/16：

```python
gt_low = F.interpolate(
    target.float(),
    size=coarse_logits.shape[-2:],
    mode="nearest",
)
loss_coarse = SegmentationLoss(coarse_logits, gt_low)
```

注意：

```text
coarse_logits 是 [B,1,h,w] 或 [B*T,1,h,w]
target 需要对应同一帧。
```

如果中心帧监督：

```python
center_idx = T // 2
coarse_center = outputs["coarse_logits"][:, center_idx]  # [B,1,h,w]
final_center = outputs["mask_logits"][:, center_idx]     # [B,1,H,W]
```

### 10.3 Residual regularization

防止 residual branch 一开始过度修改 coarse 输出：

```python
loss_residual_reg = residual_logits.abs().mean()
```

推荐小权重：

```yaml
lambda_residual_reg: 0.005 或 0.01
```

### 10.4 Memory margin loss

仅当 memory.enabled=true 时启用。

```python
normal_region = mask_low < 0.5
anomaly_region = mask_low >= 0.5

normal_loss = error_map[normal_region].mean()
anomaly_loss = F.relu(memory_margin - error_map[anomaly_region]).mean()
memory_margin_loss = normal_loss + anomaly_loss
```

默认先不开：

```yaml
lambda_mem: 0.0
```

稳定后：

```yaml
lambda_mem: 0.02 ~ 0.05
```

不要一开始 0.1。

### 10.5 Temporal smooth loss

只在 `num_frames >= 4` 时考虑。

默认关闭：

```yaml
lambda_temp: 0.0
```

开启时建议：

```yaml
lambda_temp: 0.01
```

---

## 11. 推荐完整 config

文件名：

```text
configs/dinov3_vitl16_eam_baseline_preserving_mvp.yml
```

推荐初始稳定版本：

```yaml
model:
  name: dinov3_eam_baseline_preserving_mvp
  backbone: dinov3_vitl16
  patch_size: 16
  embed_dim: 1024
  depth: 24
  num_heads: 16
  freeze_backbone: true

dinov3_weights: "My_model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
dinov3_repo: "My_model/dinov3"
allow_hub_download: false

use_lora: true
lora_rank: 4
lora_alpha: 16
lora_dropout: 0.05
lora_targets: "attn.qkv,attn.proj"

input_size: 512
num_frames: 1
batch_size: 16
grad_accum_steps: 1
amp: true
gpu_id: 0

fusion:
  enabled: true
  selected_layers: [17, 23]
  main_layer: 23
  aux_layers: [17]
  encoder_dim: 256
  max_aux_scale: 0.1
  init_alpha: -6.0

coarse_head:
  enabled: true
  in_channels: 1024
  channels: [256, 128]
  norm: "batchnorm"
  activation: "relu"

residual_decoder:
  enabled: true
  in_channels_with_memory: false
  channels: [128, 96, 64, 32, 16]
  lambda_residual: 0.1
  warmup_epochs: 5
  max_lambda_residual: 0.1
  residual_reg: true

memory:
  enabled: false
  num_slots: 4096
  topk: 16
  temperature: 0.07
  update_momentum: 0.95
  max_update_tokens_per_batch: 2048
  normal_mask_threshold: 0.2
  anomaly_margin: 0.5
  update_in_eval: false
  detach_error_for_decoder: true

temporal:
  enabled: false
  use_token_correlation_alignment: true
  corr_radius: 2
  corr_temperature: 0.07
  detach_temporal_state: false
  clamp_max: 2.0
  use_gate: true

edge_head:
  enabled: false
  lambda_edge: 0.0
  edge_radius: 2

loss:
  use_seg_loss: true
  lambda_coarse: 0.5
  lambda_residual_reg: 0.01
  lambda_mem: 0.0
  lambda_edge: 0.0
  lambda_temp: 0.0
  memory_margin: 0.5

eval_frame_chunk: 1

save_dir: "runs/dinov3_vitl16_eam_baseline_preserving_mvp"
visualization_dir: "runs/dinov3_vitl16_eam_baseline_preserving_mvp/vis"
```

---

## 12. Agent 实现顺序

必须按顺序实现，不要一次性把 memory、temporal、query、edge 全部打开。

### Step 0：保留并复现原始 baseline

实现或确认：

```text
DINOv3 last layer f24
Original CoarseMaskHead
coarse_logits
coarse_logits_up
mask_logits = coarse_logits_up
```

验收：

```python
clip = torch.randn(2, 1, 3, 512, 512).cuda()
out = model(clip)
assert out["mask_logits"].shape == (2, 1, 1, 512, 512)
assert out["coarse_logits"].shape == (2, 1, 1, 32, 32)
```

这一阶段输出必须和原始 baseline 逻辑一致。

---

### Step 1：加入 coarse loss

训练脚本加入：

```text
loss_full + 0.5 * loss_coarse
```

验收：

```text
训练能正常跑；
loss_dict 中出现 coarse_loss；
性能不能明显低于原 baseline。
```

---

### Step 2：加入 gated layer18 residual fusion

只加：

```text
layer18 -> proj18 -> small gate -> fused_feat
layer24 -> proj24 -> main
```

但此时还不要启用 residual progressive decoder。

可以临时让 simple decoder 仍使用 f24，不使用 fused_feat。

验收：

```text
确认多层特征提取不会破坏 baseline；
显存可接受；
forward shape 正确。
```

---

### Step 3：加入 residual progressive decoder

开启：

```text
residual_logits = ResidualProgressiveDecoder(fused_feat, coarse_logits)
final_logits = coarse_logits_up + 0.1 * residual_logits
```

验收：

```python
assert out["residual_logits"].shape == (B,T,1,H,W)
assert out["mask_logits"].shape == (B,T,1,H,W)
```

如果性能下降：

```yaml
residual_decoder.lambda_residual: 0.05
loss.lambda_residual_reg: 0.02
fusion.max_aux_scale: 0.05
```

---

### Step 4：可选加入 memory error

开启：

```yaml
memory.enabled: true
loss.lambda_mem: 0.02
```

但保持：

```yaml
temporal.enabled: false
memory.detach_error_for_decoder: true
```

将 `error_map.detach()` 作为 residual decoder 输入之一。

验收：

```python
assert out["error_map"].shape == (B,T,1,32,32)
```

如果性能下降，关闭 memory 输入 decoder，只保留 memory loss 做 auxiliary。

---

### Step 5：num_frames >= 4 后再启用 temporal

只有当训练配置变成：

```yaml
num_frames: 4 或 8
```

再开启：

```yaml
temporal.enabled: true
loss.lambda_temp: 0.01
```

不要在 `num_frames=1` 时打开 temporal。

---

### Step 6：最后再考虑 edge head

开启：

```yaml
edge_head.enabled: true
loss.lambda_edge: 0.05
```

如果边界变细但 IoU/F1 下降，关闭 edge head。

---

## 13. 推荐消融实验顺序

重新设计消融，不要直接从复杂模型开始。

```text
A0: Original baseline
    DINOv3 last feature + simple coarse head + bilinear upsample

A1: A0 + coarse loss
    检查 low-res supervision 是否稳定

A2: A1 + gated layer18 residual fusion
    不启用 progressive decoder

A3: A2 + residual progressive decoder
    final = coarse + 0.1 * residual

A4: A3 + memory error as auxiliary loss only
    memory 不输入 decoder

A5: A3 + memory error as residual decoder input
    error_map detach 后输入 decoder

A6: A5 + num_frames=4 + temporal accumulation

A7: A6 + weak edge head
```

如果 A2 就下降，说明多层特征仍然有害，应改成：

```text
只保留 f24；
不使用多层融合；
直接做 residual progressive decoder。
```

如果 A3 下降，说明 progressive decoder 有害，应改成：

```text
关闭 residual decoder；
保留 coarse head；
只研究 memory auxiliary loss。
```

---

## 14. 关键实现伪代码

```python
def forward(self, clip, mask=None, update_memory=False, return_aux=True):
    B, T, _, H, W = clip.shape
    h, w = H // 16, W // 16

    frames = clip.reshape(B*T, 3, H, W)

    # 1. DINO features
    feats = self.extract_dino_layers(frames)
    # Example:
    # feats[17]: [BT,1024,h,w]
    # feats[23]: [BT,1024,h,w]

    f_last = feats[self.main_layer]  # layer 23 / layer24

    # 2. Original baseline coarse head
    coarse_logits_bt, coarse_logits_up_bt = self.coarse_head(
        f_last,
        out_size=(H, W),
    )

    # 3. Gated residual fusion
    if self.fusion_enabled:
        fused_feat_bt = self.gated_fusion(feats)  # [BT,256,h,w]
    else:
        fused_feat_bt = self.proj_last(f_last)    # [BT,256,h,w]

    fused_feat = fused_feat_bt.view(B, T, -1, h, w)

    # 4. Optional memory
    error_map = None
    accum_error = None

    if self.memory_enabled:
        memory_recon, error_map = self.normal_memory(
            fused_feat,
            mask=mask,
            update=(self.training and mask is not None and update_memory),
        )
    else:
        memory_recon = None
        error_map = torch.zeros(B, T, 1, h, w, device=clip.device, dtype=fused_feat.dtype)

    # 5. Optional temporal
    if self.temporal_enabled and T > 1:
        accum_error = self.temporal_accumulator(fused_feat, error_map)
    else:
        accum_error = error_map

    # 6. Residual progressive decoder
    if self.residual_decoder_enabled:
        residual_logits_bt, edge_logits_bt = self.residual_decoder(
            fused_feat_bt=fused_feat_bt,
            coarse_logits_bt=coarse_logits_bt,
            error_map_bt=error_map.view(B*T, 1, h, w).detach() if self.detach_error else error_map.view(B*T, 1, h, w),
            accum_error_bt=accum_error.view(B*T, 1, h, w).detach() if self.detach_error else accum_error.view(B*T, 1, h, w),
            out_size=(H, W),
        )
    else:
        residual_logits_bt = torch.zeros_like(coarse_logits_up_bt)
        edge_logits_bt = None

    # 7. Baseline-preserving final output
    lambda_res = self.get_lambda_residual()
    final_logits_bt = coarse_logits_up_bt + lambda_res * residual_logits_bt

    # 8. Reshape outputs
    final_logits = final_logits_bt.view(B, T, 1, H, W)
    coarse_logits = coarse_logits_bt.view(B, T, 1, h, w)
    coarse_logits_up = coarse_logits_up_bt.view(B, T, 1, H, W)
    residual_logits = residual_logits_bt.view(B, T, 1, H, W)

    if edge_logits_bt is not None:
        edge_logits = edge_logits_bt.view(B, T, 1, H, W)
    else:
        edge_logits = None

    return {
        "mask_logits": final_logits,
        "coarse_logits": coarse_logits,
        "coarse_logits_up": coarse_logits_up,
        "residual_logits": residual_logits,
        "fused_feat": fused_feat,
        "memory_recon": memory_recon,
        "error_map": error_map,
        "accum_error": accum_error,
        "edge_logits": edge_logits,
    }
```

---

## 15. 训练脚本修改重点

### 15.1 中心帧监督

如果 target 是：

```text
[B,1,H,W]
```

说明只有中心帧监督：

```python
center_idx = T // 2
pred = outputs["mask_logits"][:, center_idx]
coarse = outputs["coarse_logits"][:, center_idx]
residual = outputs["residual_logits"][:, center_idx]
```

如果 target 是：

```text
[B,T,1,H,W]
```

说明全帧监督：

```python
pred = outputs["mask_logits"].reshape(B*T,1,H,W)
coarse = outputs["coarse_logits"].reshape(B*T,1,h,w)
target = target.reshape(B*T,1,H,W)
```

### 15.2 loss 计算

```python
loss_full = criterion(pred, target)

target_low = F.interpolate(
    target.float(),
    size=coarse.shape[-2:],
    mode="nearest",
)
loss_coarse = criterion(coarse, target_low)

loss = loss_full + cfg.loss.lambda_coarse * loss_coarse
```

residual regularization：

```python
if cfg.loss.lambda_residual_reg > 0:
    loss_residual_reg = residual.abs().mean()
    loss = loss + cfg.loss.lambda_residual_reg * loss_residual_reg
```

memory / edge / temporal loss 后续按开关加入。

---

## 16. 特别禁止事项

代码 agent 必须避免以下实现：

```text
1. 不要删除 Original CoarseMaskHead。
2. 不要用 progressive decoder 直接替代 final prediction。
3. 不要把 4 层 DINO feature 直接 concat 后作为唯一输入。
4. 不要默认启用 memory + temporal + edge。
5. 不要在 num_frames=1 时启用 temporal。
6. 不要使用 Sobel / Laplacian / SRM / high-pass / frequency branch。
7. 不要让 memory 在 eval/inference 阶段默认更新。
8. 不要去掉 coarse loss。
9. 不要让 residual_logits 未缩放地加到 final_logits。
10. 不要把 [B*T] 当作连续视频时间维做 temporal accumulation。
```

---

## 17. 最小可运行配置

如果只想先跑一个稳妥版本：

```yaml
fusion.enabled: true
fusion.selected_layers: [17, 23]
residual_decoder.enabled: true
residual_decoder.lambda_residual: 0.05

memory.enabled: false
temporal.enabled: false
edge_head.enabled: false

loss.lambda_coarse: 0.5
loss.lambda_residual_reg: 0.01
loss.lambda_mem: 0.0
loss.lambda_edge: 0.0
loss.lambda_temp: 0.0
```

如果这个版本仍低于原始 baseline，进一步退回：

```yaml
fusion.enabled: false
residual_decoder.enabled: false
```

此时模型应完全等价于：

```text
DINOv3 + original coarse head + bilinear upsample
```

---

## 18. 最终实现目标

代码 agent 最终应该生成一个可训练模型，满足：

```text
1. 完整保留原始 DINOv3 last-layer + simple coarse head baseline。
2. 支持 LoRA 微调 DINOv3 的 QKV 与线性层。
3. final_logits 由 coarse_logits_up 主导。
4. 多层特征只能通过 gated residual fusion 引入。
5. 渐进式上采样只能输出 residual_logits。
6. final_logits = coarse_logits_up + lambda_residual * residual_logits。
7. 必须输出 coarse_logits，并在训练中计算 coarse loss。
8. memory/error/temporal/edge 均为可选模块，默认关闭或弱权重开启。
9. 不包含任何频域、高频、Sobel、Laplacian、SRM 模块。
10. 输入输出接口兼容当前 train/val/test 脚本。
```

---

## 19. 给 agent 的一句话总结

请不要重写成一个复杂的新分割网络。当前最强 baseline 是 DINOv3 ViT-L/16 last-layer feature + simple coarse mask head。新的 MVP 只允许在这个强 baseline 外围加入可控 residual enhancement：多层特征使用 gated residual fusion，渐进上采样只预测 residual logits，最终预测必须是 `coarse_logits_up + λ * residual_logits`，并且训练时必须保留 H/16 coarse loss。
