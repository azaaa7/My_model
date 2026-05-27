# 修改任务说明：将 DINOv3 单层特征 + ProgressiveDecoder 改为多层 DINO 特征 + DPT/FPN Decoder

## 0. 目标

请在现有 `train_val_test_dinov3_lora.py` 和 `configs/dinov3_vitl16_lora.yml` 基础上，只完成第一阶段结构改造：

```text
当前：
DINOv3 ViT-L/16 block 23 单层特征
→ [B*T, 1024, 32, 32]
→ ProgressiveDecoder
→ [B, T, 1, 512, 512]

目标：
DINOv3 ViT-L/16 blocks [5, 11, 17, 23] 多层特征
→ DPT-style Reassemble Neck
→ FPN top-down Decoder
→ [B, T, 1, 512, 512]
```

本次不要引入：

```text
1. 时序模块
2. optical flow
3. ConvLSTM / ConvGRU
4. temporal attention
5. RGB detail branch
6. residual / SRM / high-pass branch
7. edge auxiliary head
8. Mask2Former query decoder
```

只做：

```text
DINO 多层特征提取 + DPT Reassemble + FPN Decoder
```

最终外部接口必须尽量保持不变，训练、验证、测试脚本应能继续使用原来的数据流、loss、metric 和 checkpoint 保存逻辑。

---

## 1. 当前基线结构

当前模型大致为：

```text
clip: [B, T, 3, 512, 512]
  → reshape to [B*T, 3, 512, 512]
  → DINOv3 ViT-L/16 backbone
  → extract last block patch tokens
  → [B*T, 1024, 32, 32]
  → ProgressiveDecoder
  → [B*T, 1, 512, 512]
  → reshape to [B, T, 1, 512, 512]
```

DINOv3 ViT-L/16 参数：

```text
patch_size = 16
embed_dim = 1024
depth = 24
num_heads = 16
input_size = 512
token_map_size = 512 / 16 = 32
```

当前 decoder 只接收一个单尺度特征：

```text
[B*T, 1024, 32, 32]
```

本次修改后，decoder 应接收四个尺度：

```text
P2: [B*T, 256, 128, 128]  # 1/4
P3: [B*T, 256,  64,  64]  # 1/8
P4: [B*T, 256,  32,  32]  # 1/16
P5: [B*T, 256,  16,  16]  # 1/32
```

---

## 2. 推荐最终结构

### 2.1 总体结构

```text
Input: [B, T, 3, 512, 512]
  → reshape: [B*T, 3, 512, 512]

  → DINOv3 ViT-L/16 + LoRA
       extract blocks [5, 11, 17, 23]
       each block patch tokens: [B*T, 1024, 32, 32]

  → DPTReassembleNeck
       block 5  → P2: [B*T, 256, 128, 128]
       block 11 → P3: [B*T, 256,  64,  64]
       block 17 → P4: [B*T, 256,  32,  32]
       block 23 → P5: [B*T, 256,  16,  16]

  → FPNDecoder
       top-down fusion:
       P5 → F4 → F3 → F2
       final upsampling:
       F2 128×128 → 256×256 → 512×512

  → mask logits: [B*T, 1, 512, 512]
  → reshape: [B, T, 1, 512, 512]
```

### 2.2 为什么这样设计

DINO/ViT 的所有 patch tokens 原始分辨率都是 32×32，没有 CNN backbone 那种天然的 1/4、1/8、1/16、1/32 层级特征。

因此需要一个 neck 把不同深度的 token 转成分割 decoder 需要的多尺度 feature maps：

```text
浅层 block 5:
  局部纹理、边缘、低级结构更多
  适合作为高分辨率 P2

中层 block 11:
  局部形状、部件结构更多
  适合作为 P3

中深层 block 17:
  主体语义和空间定位较均衡
  适合作为 P4

深层 block 23:
  全局语义最强
  适合作为 P5 context
```

注意：

```text
P5 = 1/32 不是替代 P4，也不是主输出特征。
P5 只作为全局上下文分支参与 top-down 融合。
精细边界主要依赖 P2/P3。
```

---

## 3. 配置文件修改

在 `configs/dinov3_vitl16_lora.yml` 中添加或修改如下字段。

建议配置：

```yaml
model:
  name: dinov3_vitl16_lora_dpt_fpn

  input_size: 512
  num_frames: 1

  backbone:
    name: dinov3_vitl16
    patch_size: 16
    embed_dim: 1024
    depth: 24
    extract_layers: [5, 11, 17, 23]
    out_indices: [5, 11, 17, 23]
    freeze_backbone: true

  neck:
    type: dpt_reassemble
    in_channels: 1024
    out_channels: 256
    token_hw: [32, 32]
    output_strides: [4, 8, 16, 32]
    norm: groupnorm
    activation: gelu

  decoder:
    type: fpn
    in_channels: 256
    hidden_channels: 256
    out_channels: 1
    norm: groupnorm
    activation: gelu
    upsample_mode: bilinear
    align_corners: false
```

LoRA 配置保持现状：

```yaml
lora:
  use_lora: true
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.1
  lora_targets:
    - attn.qkv
    - attn.proj
    - mlp.fc1
    - mlp.fc2
```

Loss 可以暂时保持当前组合，不要在本次结构改造中同时大改 loss：

```yaml
loss:
  dice:    {weight: 1.0, smooth: 1.0e-6}
  bce:     {weight: 0.5}
  tversky: {weight: 0.2, alpha: 0.3, beta: 0.7, smooth: 1.0e-6}
```

---

## 4. 需要新增的模块

建议新增一个文件，例如：

```text
models/dinov3_dpt_fpn.py
```

或者在现有模型文件中新增以下类：

```text
1. ConvGNAct
2. ReassembleBlock
3. DPTReassembleNeck
4. FPNDecoder
5. DINOv3LoRADPTFPNSegmentationModel
```

如果项目目前没有 `models/` 目录，可以按当前代码风格放到原模型定义附近。

---

## 5. 模块设计细节

### 5.1 ConvGNAct

```python
class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=32,
        activation="gelu",
    ):
        ...
```

要求：

```text
Conv2d bias=False
GroupNorm
GELU
```

注意 GroupNorm 的 `num_groups` 不能大于 channels，也必须整除 channels。

建议写一个 helper：

```python
def get_gn_groups(channels, preferred=32):
    for g in [preferred, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1
```

---

### 5.2 ReassembleBlock

每个 block 把一个 DINO token map `[N, 1024, 32, 32]` 转成目标尺度。

推荐实现：

```python
class ReassembleBlock(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=256,
        scale="x4",  # "x4", "x2", "x1", "down2"
        norm="groupnorm",
        activation="gelu",
    ):
        ...
```

四种 scale：

#### scale = "x4"

用于 block 5：

```text
输入:  [N,1024,32,32]
输出:  [N, 256,128,128]
```

结构：

```text
1×1 Conv 1024→256
ConvTranspose2d 256→256, kernel=2, stride=2  # 32→64
ConvGNAct 256→256
ConvTranspose2d 256→256, kernel=2, stride=2  # 64→128
ConvGNAct 256→256
```

#### scale = "x2"

用于 block 11：

```text
输入:  [N,1024,32,32]
输出:  [N, 256,64,64]
```

结构：

```text
1×1 Conv 1024→256
ConvTranspose2d 256→256, kernel=2, stride=2  # 32→64
ConvGNAct 256→256
```

#### scale = "x1"

用于 block 17：

```text
输入:  [N,1024,32,32]
输出:  [N, 256,32,32]
```

结构：

```text
1×1 Conv 1024→256
ConvGNAct 256→256
```

#### scale = "down2"

用于 block 23：

```text
输入:  [N,1024,32,32]
输出:  [N, 256,16,16]
```

结构：

```text
1×1 Conv 1024→256
ConvGNAct 256→256, stride=2
ConvGNAct 256→256
```

---

### 5.3 DPTReassembleNeck

```python
class DPTReassembleNeck(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=256,
        extract_layers=(5, 11, 17, 23),
    ):
        ...
```

输入是一个 dict 或 list：

```python
features = {
    5:  tensor [N,1024,32,32],
    11: tensor [N,1024,32,32],
    17: tensor [N,1024,32,32],
    23: tensor [N,1024,32,32],
}
```

输出必须是：

```python
{
    "p2": [N,256,128,128],
    "p3": [N,256, 64, 64],
    "p4": [N,256, 32, 32],
    "p5": [N,256, 16, 16],
}
```

建议在 forward 中加入断言：

```python
assert p2.shape[-2:] == (128, 128)
assert p3.shape[-2:] == (64, 64)
assert p4.shape[-2:] == (32, 32)
assert p5.shape[-2:] == (16, 16)
```

这些断言可在 debug 模式启用，正式训练时可关闭。

---

### 5.4 FPNDecoder

```python
class FPNDecoder(nn.Module):
    def __init__(
        self,
        channels=256,
        out_channels=1,
        upsample_mode="bilinear",
        align_corners=False,
    ):
        ...
```

输入：

```python
features = {
    "p2": [N,256,128,128],
    "p3": [N,256, 64, 64],
    "p4": [N,256, 32, 32],
    "p5": [N,256, 16, 16],
}
```

推荐结构：

```text
f5 = ConvBlock(p5)

f4 = ConvBlock(p4 + upsample(f5, size=p4.shape[-2:]))

f3 = ConvBlock(p3 + upsample(f4, size=p3.shape[-2:]))

f2 = ConvBlock(p2 + upsample(f3, size=p2.shape[-2:]))

x = upsample(f2, size=(256,256))
x = ConvBlock(256→128)

x = upsample(x, size=(512,512))
x = ConvBlock(128→64)

mask_logits = Conv3×3 64→32 + GN + GELU + Conv1×1 32→1
```

输出：

```python
mask_logits: [N, 1, 512, 512]
```

注意：

```text
1. 输出 logits，不要 sigmoid。
2. loss 使用 BCEWithLogits / binary_cross_entropy_with_logits。
3. metric 计算时再 sigmoid + threshold。
```

---

## 6. DINO 多层特征提取

### 6.1 目标

当前代码可能只提取最后一层，例如：

```python
tokens = backbone(...)
feat = tokens_to_map(tokens)  # [N,1024,32,32]
```

需要改成提取多个 transformer blocks 的输出：

```python
features = extract_dino_multilayer_features(
    x,
    layers=[5, 11, 17, 23],
)
```

返回：

```python
{
    5:  [N,1024,32,32],
    11: [N,1024,32,32],
    17: [N,1024,32,32],
    23: [N,1024,32,32],
}
```

### 6.2 层索引约定

请确认当前 DINO wrapper 中 block 索引是 0-based 还是 1-based。

推荐使用 0-based：

```text
block 0  = 第 1 个 transformer block
block 23 = 第 24 个 transformer block
```

所以配置：

```yaml
extract_layers: [5, 11, 17, 23]
```

表示提取第 6、12、18、24 个 block 的输出。

如果当前代码使用 1-based，请统一改为 0-based，或者在配置注释中明确：

```yaml
# 0-based transformer block indices
extract_layers: [5, 11, 17, 23]
```

### 6.3 去掉 CLS token

如果 DINO 输出包含 CLS token，必须移除 CLS，只保留 patch tokens。

典型逻辑：

```python
# x: [N, 1 + H*W, C]
patch_tokens = x[:, 1:, :]  # remove cls token
feat = patch_tokens.transpose(1, 2).reshape(N, C, H, W)
```

对于 512 输入、patch_size=16：

```python
H = W = 32
```

必须确保：

```python
patch_tokens.shape[1] == 32 * 32
```

如果没有 CLS token，则不要错误地切掉第一个 patch。

建议写一个 robust helper：

```python
def tokens_to_feature_map(tokens, image_size=512, patch_size=16, has_cls_token=True):
    N, L, C = tokens.shape
    h = w = image_size // patch_size

    if has_cls_token:
        tokens = tokens[:, 1:, :]

    assert tokens.shape[1] == h * w, (
        f"Expected {h*w} patch tokens, got {tokens.shape[1]}"
    )

    return tokens.transpose(1, 2).contiguous().reshape(N, C, h, w)
```

### 6.4 如何从 DINO 里取中间层

具体方法取决于当前 DINOv3 wrapper。

常见情况有三种：

#### 情况 A：backbone 支持 `get_intermediate_layers`

优先使用：

```python
outs = backbone.get_intermediate_layers(
    x,
    n=[5, 11, 17, 23],
    reshape=False,
    return_class_token=False,
)
```

然后把每个输出转成 `[N,C,32,32]`。

注意不同库的 `n` 参数可能表示：

```text
1. 最后 n 层数量
2. 指定层索引列表
```

必须检查当前 DINOv3 实现。

#### 情况 B：backbone forward 可返回 hidden_states

如果 HuggingFace 风格支持：

```python
outputs = backbone(
    pixel_values=x,
    output_hidden_states=True,
)
hidden_states = outputs.hidden_states
```

则提取：

```python
features = {
    5:  tokens_to_feature_map(hidden_states[6], ...),
    11: tokens_to_feature_map(hidden_states[12], ...),
    17: tokens_to_feature_map(hidden_states[18], ...),
    23: tokens_to_feature_map(hidden_states[24], ...),
}
```

注意：

```text
hidden_states[0] 可能是 patch embedding 输出
hidden_states[1] 才是 block 0 输出
```

必须根据实际输出确认索引，避免 off-by-one。

#### 情况 C：只能手动 forward blocks

如果没有中间层接口，需要在 DINO wrapper 里手动遍历 blocks：

```python
x = patch_embed_and_pos_embed(x)

features = {}
for i, blk in enumerate(self.backbone.blocks):
    x = blk(x)
    if i in self.extract_layers:
        features[i] = tokens_to_feature_map(x, ...)
```

注意保留原模型中的：

```text
1. patch embedding
2. positional embedding
3. register tokens / cls token
4. final norm 的处理
```

如果 DINOv3 有 register tokens，必须确认输出 token 的排列：

```text
[cls] + [register tokens] + [patch tokens]
```

这时不能简单 `x[:,1:,:]`，需要移除 cls 和 register tokens。

建议在代码中打印一次 token 长度：

```python
print("token length:", x.shape[1])
```

512 输入、patch 16 的 patch token 数是 1024。若总 token 长度大于 1025，说明可能存在 register tokens。

推荐 helper 支持：

```python
num_prefix_tokens = 1 + num_register_tokens
patch_tokens = tokens[:, num_prefix_tokens:, :]
```

---

## 7. 主模型 forward 接口

主模型建议类似：

```python
class DINOv3LoRADPTFPNSegmentationModel(nn.Module):
    def __init__(self, cfg):
        self.backbone = ...
        self.neck = DPTReassembleNeck(...)
        self.decoder = FPNDecoder(...)

    def forward(self, clip):
        # clip: [B,T,3,H,W] or [B,3,H,W]
        ...
```

必须兼容当前训练脚本的数据格式。

推荐 forward：

```python
def forward(self, x):
    original_ndim = x.ndim

    if x.ndim == 5:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
    elif x.ndim == 4:
        b, c, h, w = x.shape
        t = None
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")

    multi_feats = self.extract_multilayer_dino_features(x)
    pyramid_feats = self.neck(multi_feats)
    logits = self.decoder(pyramid_feats)

    if original_ndim == 5:
        logits = logits.reshape(b, t, 1, h, w)

    return logits
```

必须保证：

```text
1. 输入 [B,T,3,512,512] 时，输出 [B,T,1,512,512]
2. 输入 [B,3,512,512] 时，输出 [B,1,512,512]
3. 输出是 logits，不做 sigmoid
```

---

## 8. LoRA 适配要求

原有 LoRA 注入逻辑尽量不要重写。

保持：

```text
target_modules:
  - attn.qkv
  - attn.proj
  - mlp.fc1
  - mlp.fc2
```

注意事项：

```text
1. DINO backbone 主体仍然冻结。
2. LoRA 参数可训练。
3. neck 和 decoder 参数可训练。
4. 不要错误地把整个 DINO 解冻。
5. 不要把 neck/decoder 冻结。
```

建议初始化后打印：

```text
trainable parameters by module:
  LoRA
  neck
  decoder
  total
```

预期：

```text
LoRA: 约 6M
neck + decoder: 约数 M
总可训练参数应明显大于原来的 6.3M
```

---

## 9. checkpoint 兼容

因为 decoder 结构变了，旧 checkpoint 中的 decoder 权重无法直接加载。

请修改 checkpoint 加载逻辑，使其支持：

```python
strict=False
```

并打印：

```text
missing_keys
unexpected_keys
```

推荐策略：

```text
1. 如果从旧单层 decoder checkpoint 继续训练：
   - 加载 DINO + LoRA 可匹配部分
   - 跳过旧 decoder
   - 新 neck/decoder 随机初始化

2. 如果从头训练：
   - 加载 DINO 预训练权重
   - 注入 LoRA
   - neck/decoder 随机初始化
```

不要因为 decoder key 不匹配而直接报错退出，除非用户显式指定 `strict_load: true`。

---

## 10. 初始化建议

新增 neck/decoder 的卷积层建议使用 Kaiming 初始化：

```python
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

注意 GELU 不完全等价 ReLU，但 Kaiming 作为默认初始化可接受。

---

## 11. shape 检查

请在第一次 forward 时打印或断言以下 shape。

输入：

```text
clip: [B,T,3,512,512]
flat: [B*T,3,512,512]
```

DINO 多层输出：

```text
block 5:  [B*T,1024,32,32]
block 11: [B*T,1024,32,32]
block 17: [B*T,1024,32,32]
block 23: [B*T,1024,32,32]
```

Neck 输出：

```text
p2: [B*T,256,128,128]
p3: [B*T,256, 64, 64]
p4: [B*T,256, 32, 32]
p5: [B*T,256, 16, 16]
```

Decoder 输出：

```text
logits flat: [B*T,1,512,512]
logits clip: [B,T,1,512,512]
```

如果任意 shape 不一致，请优先修 shape，不要通过错误 resize 强行绕过。

---

## 12. loss / metric 不要改

本次结构改造不要求修改 loss。

保留当前训练逻辑：

```text
loss 输入 logits
target 输入 binary mask
BCE 使用 binary_cross_entropy_with_logits
Dice/Tversky 内部需要 sigmoid
metric 阶段再 sigmoid + threshold
```

不要在模型 forward 里做 sigmoid。

正确：

```python
logits = model(images)
loss = loss_fn(logits, masks)
probs = torch.sigmoid(logits)
preds = probs > threshold
```

错误：

```python
return torch.sigmoid(logits)
```

---

## 13. 训练超参数建议

由于新增 neck/decoder，建议区分学习率：

```yaml
optimizer:
  type: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4
```

如果当前代码支持 param groups，推荐：

```text
LoRA:    1e-4
neck:    2e-4
decoder: 2e-4
```

如果不支持 param groups，就先统一：

```text
lr = 1e-4
```

scheduler 继续使用：

```text
monitor = val_iou
mode = max
```

不要因为 val loss 高就提前否定模型，当前任务应优先看：

```text
val IoU
val F1
可视化结果
```

---

## 14. 最小单元测试

请新增或临时运行一个 shape test。

示例：

```python
def test_forward_shape():
    model = DINOv3LoRADPTFPNSegmentationModel(cfg).cuda().eval()
    x = torch.randn(2, 1, 3, 512, 512).cuda()

    with torch.no_grad(), torch.cuda.amp.autocast():
        y = model(x)

    assert y.shape == (2, 1, 1, 512, 512), y.shape
    assert y.dtype in (torch.float16, torch.bfloat16, torch.float32)
```

再测试 T>1，虽然当前不引入时序，但模型应该能按帧处理：

```python
x = torch.randn(1, 3, 3, 512, 512).cuda()
y = model(x)
assert y.shape == (1, 3, 1, 512, 512)
```

---

## 15. 训练前检查清单

完成修改后，请确认：

```text
[ ] 配置中存在 extract_layers: [5, 11, 17, 23]
[ ] DINO 确实返回四层 feature
[ ] 每层 feature 都是 [B*T,1024,32,32]
[ ] Neck 输出 p2/p3/p4/p5 shape 正确
[ ] Decoder 输出 [B*T,1,512,512]
[ ] 主模型输出 [B,T,1,512,512]
[ ] forward 不做 sigmoid
[ ] loss 和 metric 不需要大改
[ ] LoRA 参数仍然可训练
[ ] DINO 原始权重仍然冻结
[ ] neck/decoder 参数可训练
[ ] checkpoint 加载支持 strict=False
[ ] 训练脚本、验证脚本、测试脚本仍然能跑
```

---

## 16. 常见坑

### 16.1 hidden_states 索引 off-by-one

如果使用 HuggingFace 输出：

```python
hidden_states[0]
```

可能是 embedding 输出，不是 block 0 输出。

请确认：

```text
block 5 对应 hidden_states[6] 还是 hidden_states[5]
```

建议打印各层 shape，并用注释写清楚。

---

### 16.2 DINOv3 register tokens

部分 DINO / DINOv2 / DINOv3 实现可能包含 register tokens。

如果 token length 不是：

```text
1 + 32*32 = 1025
```

而是更大，例如：

```text
1 + R + 1024
```

说明有 register tokens。

此时 patch tokens 应该是：

```python
patch_tokens = tokens[:, num_prefix_tokens:, :]
```

其中：

```python
num_prefix_tokens = 1 + num_register_tokens
```

不能只去掉 CLS token。

---

### 16.3 align_corners

所有 bilinear upsample 建议：

```python
F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
```

避免不同尺度下空间偏移。

---

### 16.4 ConvTranspose2d 棋盘格

Reassemble 中如果使用 ConvTranspose2d 后出现棋盘格，可替换为：

```text
bilinear upsample + 3×3 Conv
```

例如：

```python
nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
ConvGNAct(256, 256)
```

如果追求稳定，优先用：

```text
Upsample + Conv
```

而不是 ConvTranspose2d。

---

### 16.5 显存增加

多层 feature 会增加显存。

如果 OOM：

```text
1. batch_size 降低
2. grad_accum_steps 增大
3. AMP 保持开启
4. out_channels 从 256 降到 192 或 128
5. extract_layers 暂时改为 [11, 17, 23]
```

首选不要降输入分辨率，因为 mask 边界精细，512 对任务有价值。

---

## 17. 推荐代码骨架

下面是简化骨架，按项目实际代码风格改写。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gn_groups(channels, preferred=32):
    for g in [preferred, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(get_gn_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ReassembleBlock(nn.Module):
    def __init__(self, in_ch=1024, out_ch=256, scale="x1"):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(get_gn_groups(out_ch), out_ch),
            nn.GELU(),
        ]

        if scale == "x4":
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(out_ch, out_ch),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(out_ch, out_ch),
            ]
        elif scale == "x2":
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(out_ch, out_ch),
            ]
        elif scale == "x1":
            layers += [
                ConvGNAct(out_ch, out_ch),
            ]
        elif scale == "down2":
            layers += [
                ConvGNAct(out_ch, out_ch, stride=2),
                ConvGNAct(out_ch, out_ch),
            ]
        else:
            raise ValueError(f"Unknown scale: {scale}")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DPTReassembleNeck(nn.Module):
    def __init__(self, in_ch=1024, out_ch=256, layers=(5, 11, 17, 23)):
        super().__init__()
        self.layers = list(layers)
        self.reassemble = nn.ModuleDict({
            str(layers[0]): ReassembleBlock(in_ch, out_ch, "x4"),
            str(layers[1]): ReassembleBlock(in_ch, out_ch, "x2"),
            str(layers[2]): ReassembleBlock(in_ch, out_ch, "x1"),
            str(layers[3]): ReassembleBlock(in_ch, out_ch, "down2"),
        })

    def forward(self, feats):
        l0, l1, l2, l3 = self.layers

        p2 = self.reassemble[str(l0)](feats[l0])
        p3 = self.reassemble[str(l1)](feats[l1])
        p4 = self.reassemble[str(l2)](feats[l2])
        p5 = self.reassemble[str(l3)](feats[l3])

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5,
        }


class FPNDecoder(nn.Module):
    def __init__(self, channels=256, out_channels=1):
        super().__init__()

        self.f5 = ConvGNAct(channels, channels)
        self.f4 = ConvGNAct(channels, channels)
        self.f3 = ConvGNAct(channels, channels)
        self.f2 = ConvGNAct(channels, channels)

        self.up1 = ConvGNAct(channels, 128)
        self.up0 = ConvGNAct(128, 64)

        self.head = nn.Sequential(
            ConvGNAct(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, feats):
        p2, p3, p4, p5 = feats["p2"], feats["p3"], feats["p4"], feats["p5"]

        f5 = self.f5(p5)

        f4 = p4 + F.interpolate(
            f5, size=p4.shape[-2:], mode="bilinear", align_corners=False
        )
        f4 = self.f4(f4)

        f3 = p3 + F.interpolate(
            f4, size=p3.shape[-2:], mode="bilinear", align_corners=False
        )
        f3 = self.f3(f3)

        f2 = p2 + F.interpolate(
            f3, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        f2 = self.f2(f2)

        x = F.interpolate(
            f2, size=(256, 256), mode="bilinear", align_corners=False
        )
        x = self.up1(x)

        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=False
        )
        x = self.up0(x)

        logits = self.head(x)
        return logits
```

---

## 18. 最终验收标准

本次修改完成后，必须满足：

```text
1. `python train_val_test_dinov3_lora.py --type train ...` 可以启动训练
2. 第一次 forward shape 全部正确
3. loss 可以正常 backward
4. 验证和测试脚本不需要大改即可运行
5. 输出仍然是 logits
6. checkpoint 能保存和加载
7. 旧 checkpoint 加载时 decoder 不匹配不会导致程序崩溃
8. val_iou 正常计算
```

建议提交时附带一次 dry-run 日志：

```text
Input:          [B,T,3,512,512]
DINO block 5:   [B*T,1024,32,32]
DINO block 11:  [B*T,1024,32,32]
DINO block 17:  [B*T,1024,32,32]
DINO block 23:  [B*T,1024,32,32]
P2:             [B*T,256,128,128]
P3:             [B*T,256,64,64]
P4:             [B*T,256,32,32]
P5:             [B*T,256,16,16]
Logits flat:    [B*T,1,512,512]
Logits final:   [B,T,1,512,512]
```

---

## 19. 不要做的事情

请不要在本次修改中做以下事情：

```text
1. 不要加入时序模块
2. 不要改 dataset
3. 不要改 mask label 格式
4. 不要在模型 forward 里 sigmoid
5. 不要把 DINO 主干全部解冻
6. 不要删除 LoRA 逻辑
7. 不要大改训练循环
8. 不要同时更换 loss 组合
9. 不要把输入分辨率从 512 改成其他大小
10. 不要把输出改成低分辨率再外部 resize
```

---

## 20. 一句话任务总结

把当前：

```text
single DINO block 23 feature + lightweight progressive decoder
```

改成：

```text
DINO blocks [5,11,17,23] multi-layer features
+ DPT-style reassemble neck
+ FPN top-down decoder
```

保持输入输出接口、LoRA、loss、metric、训练脚本逻辑基本不变。
