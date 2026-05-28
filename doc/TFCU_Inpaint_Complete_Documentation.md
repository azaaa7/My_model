# TFCU-Inpaint：视频 Inpainting 检测完整文档

> 项目：`/home/wzk/Exp/My_model`  
> 主脚本：`train_val_test_dinov3_lora.py`  
> 配置：`configs/dinov3_vitl16_lora_tfcu_inpaint.yml`  
> 基于 DINOv3 ViT-L/16 + LoRA + DPT-FPN，在 FPN P4 层级注入轻量时序 Adapter，用于视频 inpainting 伪造检测。

---

## 一、模型架构总览

```text
输入 video [B, N, T, 3, 512, 512]
  │
  ├─ 若 [B, T, 3, 512, 512]（兼容旧格式） → 内部补 N=1
  │
  ├─ flatten → [B*N*T, 3, 512, 512]
  │
  ├─ encoder_chunk 分流（VRAM 控制）
  │   ├─ chunk=2 → 每次 2 帧过 DINOv3，共 (BNT/2) 趟
  │   └─ chunk=1 → 每次 1 帧（当前配置，最省显存）
  │
  ├─ DINOv3 ViT-L/16 backbone（冻结）
  │   └─ LoRA 注入：attn.qkv, attn.proj, mlp.fc1, mlp.fc2
  │       Blocks: 11, 15, 19, 23 → 4 层 patch tokens [1024-dim, 32×32]
  │
  ├─ DPT Reassemble Neck
  │   ├─ block 11 → ReassembleBlock("x4")    → P2 [BNT, 256, 128, 128]
  │   ├─ block 15 → ReassembleBlock("x2")    → P3 [BNT, 256,  64,  64]
  │   ├─ block 19 → ReassembleBlock("x1")    → P4 [BNT, 256,  32,  32]
  │   └─ block 23 → ReassembleBlock("down2") → P5 [BNT, 256,  16,  16]
  │
  ├─ ═══════ TFCU-Inpaint Adapter @ P4 ═══════
  │   │
  │   ├─ P4 reshape → [B, N, T, 256, 32, 32]
  │   │
  │   ├─ [1] LocalTemporalDifferenceModule
  │   │     每帧与前一帧差分 (x_t − x_{t-1}) → 门控融合
  │   │     第一帧 diff=0（prev = self）
  │   │
  │   ├─ [2] for n in range(N) 因果循环：
  │   │       ├─ cur = x[:, n]                          [B, T, 256, 32, 32]
  │   │       ├─ 若 n=0（无历史）→ cur_enhanced = cur
  │   │       ├─ 若 n≥1 → InpaintMemoryAttention
  │   │       │     cur(query) attend to mem(key/value)
  │   │       │     mem = stack(state[-memory_len:])   [B, K, T, 256, 32, 32]
  │   │       ├─ state.append(cur_enhanced.detach())    # 第一版 detach
  │   │       └─ enhanced.append(cur_enhanced)
  │   │
  │   ├─ temporal = stack(enhanced) → reshape [BNT, 256, 32, 32]
  │   ├─ temporal = temporal_proj(temporal)              # 1×1 Conv
  │   └─ P4_out = P4 + α * temporal                      # α 初始 = 0
  │
  ├─ FPNDecoder
  │   ├─ P5 → F5 [BNT, 256, 16, 16]
  │   ├─ P4_out + upsample(F5) → F4 [BNT, 256, 32, 32]
  │   ├─ P3 + upsample(F4) → F3 [BNT, 256, 64, 64]
  │   ├─ P2 + upsample(F3) → F2 [BNT, 256, 128, 128]
  │   ├─ F2 → upsample(256²) → Conv(256→128)
  │   ├─       upsample(512²) → Conv(128→64)
  │   └─       Conv(64→32) → Conv2d(32→1)
  │
  └─ output logits [B, N, T, 1, 512, 512]
       （若 squeeze_n 则 → [B, T, 1, 512, 512]）
```

### 关键设计原则

| 原则 | 实现 |
|------|------|
| **不重写 backbone** | VideoInpaintTFCU wrapper 包裹现有 DINOv3ViTL16InpaintingDetector |
| **可退化** | `α=0` 时 P4_out = P4，模型等价于原始单帧 DPT-FPN |
| **因果 memory** | clip n 只能 attend clip 0..n-1，禁止偷看未来 |
| **detach memory** | memory 前 detach，避免跨 clip 梯度爆炸 |

---

## 二、文件结构与模块入口

```text
My_model/
├── train_val_test_dinov3_lora.py          ★ 主训练/验证/测试脚本
├── train_val_test_convnext_lora.py        ConvNeXt-Tiny 对比脚本（含共享工具函数）
│
├── configs/
│   ├── dinov3_vitl16_lora.yml             DPT-FPN baseline 配置
│   └── dinov3_vitl16_lora_tfcu_inpaint.yml ★ TFCU-Inpaint 配置
│
├── my_model/
│   ├── __init__.py                        模块导出
│   ├── losses.py                          损失函数（9 种）
│   ├── metrics.py                         IoU/F1/Precision/Recall
│   ├── dinov3_dpt_fpn.py                  DPTReassembleNeck / FPNDecoder
│   ├── video_inpaint_tfcu.py              ★ VideoInpaintTFCU wrapper
│   └── temporal/
│       ├── __init__.py
│       ├── local_temporal_difference.py    ★ LocalTemporalDifferenceModule
│       ├── memory_attention.py             ★ InpaintMemoryAttention
│       ├── mask_prompt_encoder.py          MaskPromptEncoder（备用，未启用）
│       └── temporal_adapter.py             ★ TFCUInpaintAdapter（总入口）
│
├── zzz_dataset_toolkit/
│   ├── dataset.py                         VideoInpaintingDataset（支持 N clips）
│   └── transforms.py                      数据增强
│
├── dinov3/                                DINOv3 本地 repo（facebookresearch/dinov3）
├── flist/                                 样本列表 (.npy)
├── debug/
│   ├── test_tfcu_shapes.py                9 个单元测试
│   └── dry_run_tfcu_train_step.py         端到端 dry-run
└── runs/dinov3_vitl16_tfcu_inpaint/       ★ 训练输出目录
```

---

## 三、各模块详解

### 3.1 `DINOv3ViTL16InpaintingDetector`（Base Model）

**文件**：`train_val_test_dinov3_lora.py`（类定义在该脚本内）  
**依赖**：`my_model/dinov3_dpt_fpn.py`（DPTReassembleNeck, FPNDecoder）

| 组件 | 参数量 | 状态 |
|------|--------|------|
| DINOv3 ViT-L/24 backbone | ~304M | ❄️ 冻结 |
| LoRA（r=32, α=64） | ~6M | 🔥 训练 |
| DPTReassembleNeck | ~3.7M | 🔥 训练 |
| FPNDecoder | ~1.3M | 🔥 训练 |

**关键方法**：

```python
# 训练/测试统一入口（兼容旧格式）
def forward(self, clip: [B,T,3,H,W]) -> [B,T,1,H,W]:
    frames = clip.reshape(B*T, 3, H, W)
    P2, P3, P4, P5 = self.extract_fpn_features(frames)
    logits = self.decode_fpn(P2, P3, P4, P5)
    return logits.reshape(B, T, 1, H, W)

# TFCU wrapper 调用的接口
def extract_fpn_features(self, frames: [B_flat, 3, H, W]) -> (P2, P3, P4, P5):
    # 输入：flat batch（B_flat = B*N*T 或 chunk）
    # 内部调 DinoMultiLayerEncoder → DPTReassembleNeck
    # 输出：P2 [B_flat,256,128,128] P3 [B_flat,256,64,64]
    #       P4 [B_flat,256,32,32]   P5 [B_flat,256,16,16]

def decode_fpn(self, P2, P3, P4, P5) -> [B_flat, 1, 512, 512]:
    # 内部调 FPNDecoder
```

**LoRA 注入**：通过 `peft.LoraConfig` + `inject_adapter_in_model`，冻结 backbone 后显式将 `lora_*` 参数设回 `requires_grad=True`。

---

### 3.2 `VideoInpaintTFCU`（顶层 Wrapper）

**文件**：`my_model/video_inpaint_tfcu.py`

```python
class VideoInpaintTFCU(nn.Module):
    def __init__(self, base_model: DINOv3ViTL16InpaintingDetector, cfg: dict):
        self.base = base_model          # 复用 backbone + neck + decoder
        self.temporal_adapter = TFCUInpaintAdapter(...)
        self._encoder_chunk = cfg.get("encoder_chunk", 0)

    def forward(self, video: [B,N,T,3,H,W] or [B,T,3,H,W]) -> logits:
        # 1. 兼容旧格式：5D → 补 N=1
        # 2. flatten → [BNT,3,H,W]
        # 3. encoder_chunk 分流：每 chunk 帧过 extract_fpn_features → concat
        # 4. P4 过 temporal_adapter(B,N,T)
        # 5. decode_fpn(P2, P3, P4_out, P5)
        # 6. reshape → [B,N,T,1,H,W]
```

**encoder_chunk 机制**（解决 OOM）：
- `encoder_chunk=0`：全部帧一次过 DINOv3（最快，可能 OOM）
- `encoder_chunk=2`：每次 2 帧，峰值 VRAM ~3 GiB
- `encoder_chunk=1`：每次 1 帧，峰值 VRAM ~1.5 GiB（当前配置）

---

### 3.3 `TFCUInpaintAdapter`（时序 Adapter 总入口）

**文件**：`my_model/temporal/temporal_adapter.py`  
**参数量**：~1.9M

```python
class TFCUInpaintAdapter(nn.Module):
    def forward(self, P4: [BNT, C, H, W], B, N, T) -> [BNT, C, H, W]:
        x = P4.reshape(B, N, T, C, H, W)

        # [1] 局部时序差分（clip 内帧间）
        x = self.local(x)

        # [2] 前向历史 memory（cross-clip, causal）
        enhanced = []
        state = []          # FIFO memory buffer
        for n in range(N):
            cur = x[:, n]
            if len(state) == 0:
                cur_enhanced = cur
            else:
                mem = stack(state[-memory_len:])
                cur_enhanced = self.memory_attn(cur, mem)
            enhanced.append(cur_enhanced)
            state.append(cur_enhanced.detach())  # detach 防梯度爆炸

        # [3] 残差注入
        temporal = stack(enhanced).reshape(BNT, C, H, W)
        temporal = self.temporal_proj(temporal)
        return P4 + self.alpha * temporal
```

| 属性 | 说明 |
|------|------|
| `self.alpha` | `nn.Parameter(torch.tensor(0.0))` — 从零学习，避免初期干扰 |
| `self.local` | `LocalTemporalDifferenceModule` |
| `self.memory_attn` | `InpaintMemoryAttention` |
| `self.temporal_proj` | `nn.Conv2d(C, C, 1)` |

---

### 3.4 `LocalTemporalDifferenceModule`（帧间差分）

**文件**：`my_model/temporal/local_temporal_difference.py`  
**参数量**：~1.05M

```text
输入 x: [B, N, T, C, H, W]

  prev = [x[:,:,:1], x[:,:,:-1]]     # 第一帧 prev=自身 → diff=0
  diff = x - prev
  abs_diff = |diff|

  feat = cat([x, diff, abs_diff], dim=3)    # [B,N,T,3C,H,W]
  feat = flatten [BNT, 3C, H, W]

  delta = fuse(feat)      # Conv1×1(3C→C) → GN → GELU → Conv3×3 → Conv1×1
  gate  = gate(feat)      # Conv1×1(3C→C) → Sigmoid

  delta = delta * gate
  delta = reshape [B,N,T,C,H,W]

输出: x + delta     [B,N,T,C,H,W]
```

捕获相邻帧纹理/边界/运动不一致 —— inpainting 区域在单帧上可能自然，跨帧常出现异常。

---

### 3.5 `InpaintMemoryAttention`（历史 Memory 交叉注意力）

**文件**：`my_model/temporal/memory_attention.py`  
**参数量**：~0.79M  
**注意**：使用**手动 multi-head attention**（非 `nn.MultiheadAttention`），避免 PyTorch 2.7 `scaled_dot_product_attention` 在 `torch.no_grad()` 下的 sparse tensor 兼容性问题。

```text
输入 cur: [B, T, C, H, W]      mem: [B, K, T, C, H, W]

  可选 spatial_pool: 32×32 → 16×16（省 VRAM）

  q  = cur → permute → [B, T*H*W, C]    (query tokens)
  kv = mem → permute → [B, K*T*H*W, C]  (key/value tokens)

  Pre-norm → Q/K/V 投影 → reshape [B, 8 heads, seq, 32 dim]

  attn = softmax(Q @ K^T / √32) @ V    (手动 scaled dot-product)

  Reshape → out_proj → residual + FFN(4×C)

输出: [B, T, C, H, W]
```

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `num_heads` | 8 | |
| `head_dim` | 32 | 256 / 8 |
| `use_spatial_pool` | true | memory attn 前 pool 到 16×16，省 ~75% tokens |

**旧 checkpoint 兼容**：`_load_from_state_dict` 自动将 `nn.MultiheadAttention` 的 `in_proj_weight` [3C,C] 拆分为 q_proj/k_proj/v_proj 各 [C,C]。

---

### 3.6 `MaskPromptEncoder`（备用）

**文件**：`my_model/temporal/mask_prompt_encoder.py`  
当前 **未启用**（`use_mask_prompt: false`）。功能：将预测 mask + Sobel 边界编码为空间 prompt，后续版本可 fuse 到 temporal features。

---

## 四、损失函数

**文件**：`my_model/losses.py`  
**总入口**：`SegmentationLoss(loss_cfg)`，通过 YAML 配置自由组合。

### 当前 TFCU 启用

| 损失 | Weight | 说明 |
|------|--------|------|
| **Dice** | 1.0 | 软 Dice 损失 |
| **BCE** | 0.5 | 逐像素二值交叉熵 |
| **Tversky** | 0.2 | α=0.3(FP), β=0.7(FN)，偏重惩罚漏检 |
| **Boundary** | 0.2 | 🆕 Sobel 边界 L1 损失 |
| **Temporal Delta** | 0.1 | 🆕 帧间预测差分 L1 损失（T≤1 时返回 0） |

总损失 = `1.0×Dice + 0.5×BCE + 0.2×Tversky + 0.2×Boundary + 0.1×TemporalDelta`

### 全量支持（9 种）

`wbce`, `dice`, `focal`, `iou`, `bce`, `tversky`, `edge`, `boundary`, `temporal_delta`

### 维度处理

`SegmentationLoss.forward()` 自动处理 4D/5D/6D 输入：
- BCE/Dice/Tversky/IoU/Edge → 自动 flatten 到 4D `[*, C, H, W]`
- Boundary/TemporalDelta → 保留原始维度以访问空间/时序信息

---

## 五、Dataset

**文件**：`zzz_dataset_toolkit/dataset.py`  
**类**：`VideoInpaintingDataset`

### TFCU 模式（`num_clips > 1`）

| 模式 | 返回 |
|------|------|
| 训练 | `frames [N,T,3,512,512]`, `masks [N,T,1,512,512]`, H, W, name |
| 验证 | 同上 |
| 测试 | 同上 |

### 关键特性

- **数字排序**：帧名按末尾数字排序（`10.png` 在 `2.png` 之后），不等同字符串排序
- **训练采样**：随机起点，clip 间 chronologically sorted，clip 内帧序递增
- **验证/测试采样**：顺序滑窗，非重叠
- **偶数帧**：TFCU 模式允许 `num_frames=4`（baseline 模式仍需奇数）

---

## 六、训练流程

### 6.1 启动命令

```bash
cd /home/wzk/Exp/My_model

# 防 OOM 环境变量（推荐）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type train \
  --gpu_id 0 \
  --batch_size 1 \
  --grad_accum_steps 8
```

### 6.2 训练循环（`run_epoch`）

```text
每个 epoch:
  for batch in train_loader:
    frames [1,N,T,3,512,512], masks [1,N,T,1,512,512]

    if use_tfcu (frames.ndim==6):
      logits = model(frames)                    # [1,N,T,1,512,512]
      loss_masks = masks                         # [1,N,T,1,512,512]
    else:
      logits = model(frames)
      logits, loss_masks = align_logits_and_masks(logits, masks)

    loss = criterion(logits, loss_masks)
    (loss / grad_accum_steps).backward()

    if step % grad_accum_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
```

### 6.3 Optimizer 参数组

TFCU 模式使用**分离学习率**：

| 参数组 | 学习率 | 包含 |
|--------|--------|------|
| temporal | 1e-4 | `model.temporal_adapter.*` |
| decoder | 1e-4 | `model.base.decoder.*`, `model.base.neck.*` |
| lora | 1e-5 | 剩余可训练参数（含 LoRA） |

否则使用统一 `learning_rate`（baseline 模式）。

### 6.4 TFCU 模式 vs Baseline 模式关键差异

| | Baseline | TFCU |
|---|---|---|
| 输入 | `[B,T,3,H,W]` | `[B,N,T,3,H,W]` |
| 帧数要求 | 奇数 | 无限制 |
| 训练 mask | 中心帧 `[B,1,H,W]` | 完整 `[B,N,T,1,H,W]` |
| loss 处理 | `align_logits_and_masks` 取中心帧 | 直接传完整 tensor |
| optimizer | 单组 | temporal/decoder/lora 三组 |
| eval_frame_chunk | 按配置 | 强制 0（不切 T） |
| encoder chunking | 无 | 按 `encoder_chunk` 分流 |

---

## 七、测试流程

### 7.1 单集验证

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type val \
  --gpu_id 0 \
  --checkpoint runs/dinov3_vitl16_tfcu_inpaint/best_iou.pt
```

### 7.2 全量测试套件（3 个子集）

```bash
python train_val_test_dinov3_lora.py \
  --config configs/dinov3_vitl16_lora_tfcu_inpaint.yml \
  --type test \
  --gpu_id 0 \
  --checkpoint runs/dinov3_vitl16_tfcu_inpaint/best_iou.pt
```

测试套件自动运行 3 个子集并打印汇总：

| 子集 | flist |
|------|-------|
| DVI_20 | `DAVIS-VI_val_DVI_20.npy` |
| CPNET_20 | `DAVIS-VI_val_CPNET_20.npy` |
| OPN_20 | `DAVIS-VI_val_OPN_20.npy` |

### 7.3 测试流程（`evaluate` → `run_epoch`）

```text
build_model(cfg) → VideoInpaintTFCU + load checkpoint
make_loader(cfg, "test") → DataLoader

for batch in test_loader:
  frames [1,N,T,3,512,512], masks [1,N,T,1,orig_H,orig_W]

  with torch.no_grad():
    logits = forward_in_frame_chunks(model, frames, eval_chunk=0)
    # TFCU: eval_chunk=0 → model(frames) 直接前向

    if logits.shape[-2:] != masks.shape[-2:]:
      logits = interpolate(logits, size=masks.shape[-2:])

    loss = criterion(logits, masks)

  metrics = binary_metrics_from_logits(logits, masks, threshold=0.5)
  save_visualization(前 50 个样本)
```

---

## 八、检查点兼容性

| 源 → 目标 | 兼容？ | 说明 |
|-----------|:---:|------|
| TFCU (旧 `nn.MHA`) → TFCU (新手动 attn) | ✅ | `_load_from_state_dict` 自动拆分 `in_proj_weight` |
| TFCU → TFCU | ✅ | 完整恢复 |
| Baseline DPT-FPN → TFCU | ⚠️ | backbone/neck/decoder 权重加载，temporal adapter 随机初始化（α=0 可安全训练） |

加载时 `strict=False`，不匹配 key 自动跳过并打印日志。

---

## 九、显存与速度

### 显存

| encoder_chunk | 峰值 VRAM（估算） | 速度 |
|:---:|:---:|:---:|
| 1（当前） | ~8 GiB | 慢（8× DINOv3 调用/sample） |
| 2 | ~12 GiB | 中（4× 调用） |
| 0 | ~20+ GiB | 快（1× 调用） |

### OOM 降级顺序

1. `encoder_chunk: 1`（最安全）
2. `num_clips: 2`（当前）
3. `num_frames: 3`
4. `use_spatial_pool: true`
5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 十、消融实验

| 实验 | 配置 | 验证问题 |
|------|------|---------|
| A0 | Baseline, num_frames=1 | 单帧 baseline |
| A1 | Baseline, num_frames=4 | 多帧不加 adapter |
| A2 | TFCU, use_memory=false | 仅 local temporal diff 是否有效？ |
| A3 | TFCU, use_memory=true | 历史 memory 是否有效？ |
| A4 | A3 + boundary + temporal_delta loss | 边界/时序 loss 是否提升？ |

---

## 十一、常用 CLI 覆盖

```bash
--gpu_id 1                    # 选择 GPU
--num_clips 2                 # 减少 clip 数
--num_frames 3                # 减少每 clip 帧数
--encoder_chunk 2             # 提高 encoder 吞吐
--use_memory false            # 关闭 memory attention（A2 消融）
--use_spatial_pool false      # 关闭 spatial pool
--threshold 0.3               # 二值化阈值
--lr_temporal 5e-5            # 降低 temporal adapter 学习率
```

---

## 十二、单元测试

```bash
# 形状测试（9 项）
python debug/test_tfcu_shapes.py

# 端到端 dry-run（config + loss + adapter + backward + dataset）
python debug/dry_run_tfcu_train_step.py
```
