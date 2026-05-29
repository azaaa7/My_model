# Semantic-Anchor MFCE + P4 Gated TFCU 使用教程

本文档面向实际训练和测试。设计约束可以继续看 `doc/Semantic_Anchor_MFCE_agent_implementation_guide.md`，这里主要说明新结构怎么工作、配置怎么改、命令怎么启动。

## 1. 结构总览

新结构名为：

```yaml
neck_variant: semantic_anchor_mfce
temporal_insert_level: P4
```

核心思想是：DINOv3 ViT 的多个 block 输出都是同一个 token 分辨率，512 输入下都是 `32x32`，所以不要把浅层 block 硬当 P2、深层 block 硬当 P5。新结构先在 `32x32` 上做 semantic anchor 融合，再从 P4 自顶向下恢复 mask。

```text
video [B, N, T, 3, 512, 512]
        |
        v
flatten frames [B*N*T, 3, 512, 512]
        |
        v
DINOv3 ViT-L/16 + LoRA
        |
        +-- layer 5  [BNT, 1024, 32, 32]
        +-- layer 11 [BNT, 1024, 32, 32]
        +-- layer 17 [BNT, 1024, 32, 32]
        +-- layer 23 [BNT, 1024, 32, 32]
        |
        v
SemanticAnchorMFCE
  per-layer 1x1 projection
  per-layer spatial score
  softmax over layer dimension
  weighted sum
        |
        v
P4_sem [BNT, 256, 32, 32]
        |
        v
LightASPP context at P4
        |
        v
P4 [BNT, 256, 32, 32]
        |
        v
P4 = P4 + sigmoid(gate4) * (TFCU(P4) - P4)
        |
        v
SemanticAnchorDecoder
P4 32x32 -> P3 64x64 -> P2 128x128 -> P1 256x256 -> logits 512x512
        |
        v
logits [B, N, T, 1, 512, 512]
```

注意这里没有主 P5 分支。P5 不作为 temporal/prompt 主输入，后续如果接 MaskPromptEncoder，也应该接在 P4 semantic anchor 这条线上。

## 2. 代码入口

主要实现文件：

```text
my_model/dinov3_dpt_fpn.py
  SemanticAnchorMFCE
  LightASPP
  SemanticAnchorDetailStem
  SemanticAnchorDecoder

my_model/video_inpaint_tfcu.py
  GatedTemporalInjector
  NoOpTemporalInjector
  VideoInpaintTFCU semantic_anchor_mfce forward path

train_val_test_dinov3_lora.py
  DINOv3ViTL16InpaintingDetector semantic_anchor_mfce 分支
  extract_semantic_anchor_features()
  decode_semantic_anchor()
  build_model(), validate_config(), CLI 参数

configs/semantic_anchor_mfce_p4_tfcu.yaml
  推荐主实验配置
```

## 3. 推荐配置

主实验使用：

```yaml
use_dpt_fpn: true
neck_variant: semantic_anchor_mfce
extract_layers: "5,11,17,23"
neck_channels: 256
semantic_aspp_rates: "1,2,4,8"

use_tfcu_adapter: true
temporal_insert_level: P4
p4_gate_init: -3.0
use_memory: true
use_spatial_pool: true

use_lora: true
lora_targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
```

关键配置含义：

| 配置 | 含义 | 建议 |
|---|---|---|
| `neck_variant` | neck 结构选择，`semantic_anchor_mfce` 启用新结构 | 主实验用 `semantic_anchor_mfce` |
| `extract_layers` | 抽取的 DINO block | 固定 `"5,11,17,23"` |
| `semantic_aspp_rates` | P4 上下文 ASPP dilation | 默认 `"1,2,4,8"` |
| `temporal_insert_level` | TFCU 插入位置 | 新结构用 `P4` |
| `p4_gate_init` | P4 TFCU gate 初值 | `-3.0`，初始影响小 |
| `use_detail_stem` | 是否用 RGB detail stem 补边界纹理 | 第一轮 `false`，消融再开 |
| `use_spatial_pool` | memory attention 前把 32x32 降到 16x16 | 显存紧张和多卡训练建议 `true` |
| `resume_optimizer` | 恢复 checkpoint 时是否恢复 optimizer | 换卡数或改结构时用 `false` |
| `cuda_visible_devices` | 参与训练的物理 GPU id | 数量要和 `nproc_per_node` 匹配 |
| `nproc_per_node` | 单机 DDP 进程数 | 不要超过可见 GPU 数 |

完整注释版配置见：

```text
configs/semantic_anchor_mfce_p4_tfcu.yaml
```

## 4. 实验开关

E1: 只测 semantic anchor 融合，不用 TFCU。

```yaml
use_tfcu_adapter: true
neck_variant: semantic_anchor_mfce
temporal_insert_level: NONE
use_detail_stem: false
```

E2: 主实验，P4 gated TFCU。

```yaml
use_tfcu_adapter: true
neck_variant: semantic_anchor_mfce
temporal_insert_level: P4
use_detail_stem: false
```

E3: 加 RGB detail stem，观察边界和局部纹理。

```yaml
use_tfcu_adapter: true
neck_variant: semantic_anchor_mfce
temporal_insert_level: P4
use_detail_stem: true
detail_stem_gated: true
detail_gate_init: -3.0
```

## 5. 启动训练

配置里已经写了：

```yaml
auto_torchrun: true
cuda_visible_devices: "2,3,4,5,6,7"
nproc_per_node: 6
```

因此可以直接用普通 python 启动，脚本会自动 relaunch 到 torchrun：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type train
```

如果想手动 torchrun，可以临时关闭 `auto_torchrun`：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --standalone --nproc_per_node=6 \
  train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type train \
  --auto_torchrun false
```

## 6. 断点恢复训练

从最新 checkpoint 继续：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type train \
  --checkpoint runs/semantic_anchor_mfce_p4_tfcu/latest.pt
```

如果换了 GPU 数量、改了 LoRA targets、改了结构，建议跳过 optimizer state：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type train \
  --checkpoint runs/semantic_anchor_mfce_p4_tfcu/latest.pt \
  --resume_optimizer false
```

如果 checkpoint 的 epoch 已经大于等于 `n_epochs`，脚本会直接退出。此时需要提高：

```yaml
n_epochs: 3000
```

## 7. 验证和测试

快速验证：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type val \
  --checkpoint runs/semantic_anchor_mfce_p4_tfcu/best_iou.pt
```

完整视频验证，把 padding 帧用 `valid_mask` 过滤：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type val \
  --checkpoint runs/semantic_anchor_mfce_p4_tfcu/best_iou.pt \
  --val_full_video true
```

测试默认完整视频 window：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python train_val_test_dinov3_lora.py \
  --config configs/semantic_anchor_mfce_p4_tfcu.yaml \
  --type test \
  --checkpoint runs/semantic_anchor_mfce_p4_tfcu/best_iou.pt
```

## 8. 多卡注意事项

1. `nproc_per_node` 必须小于等于 `cuda_visible_devices` 的数量。
2. 如果写 `cuda_visible_devices: "2,3,4,5,6,7"`，进程内部看到的是 `cuda:0..cuda:5`，不要再写物理 id 作为 local rank。
3. 出现 `invalid device ordinal`，优先检查 `nproc_per_node` 是否超过可见卡数。
4. 从 8 卡换 6 卡恢复时，保留 `resume_optimizer: false`，否则 optimizer parameter group 可能不匹配。
5. 如果某个 rank 自动退出但主日志不明显，看 `torchrun_log_dir` 下对应 rank 的日志。

## 9. 结构自检

运行 shape/debug tests：

```bash
/home/wzk/anaconda3/envs/dinov3/bin/python debug/test_tfcu_shapes.py
```

预期最后输出：

```text
21/21 tests passed
```

也可以只做语法编译检查：

```bash
python -m py_compile \
  my_model/dinov3_dpt_fpn.py \
  my_model/video_inpaint_tfcu.py \
  train_val_test_dinov3_lora.py \
  my_model/__init__.py \
  debug/test_tfcu_shapes.py
```

## 10. 常见调参建议

显存不够时按顺序尝试：

1. 保持 `use_spatial_pool: true`。
2. 减小 `encoder_chunk`，例如从 `2` 改成 `1`。
3. 减小 `num_clips` 或 `test_max_clips`。
4. 减小 `grad_accum_steps` 只会降低等效 batch，不一定省单步显存；优先调上面几项。

想提升边界细节时：

```yaml
use_detail_stem: true
detail_stem_gated: true
detail_gate_init: -3.0
```

想测试没有 TFCU 的纯 semantic anchor：

```yaml
temporal_insert_level: NONE
```

想测试更完整 LoRA 学习能力：

```yaml
lora_targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
lora_layers: "all"
```
