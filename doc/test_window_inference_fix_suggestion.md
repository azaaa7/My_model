# Test 阶段顺序 Window 推理修复建议

## 目标

本次只修复 **test 阶段** 的采样与推理覆盖问题，暂时不要修改 train / val / model / loss / temporal adapter。

当前需要解决的问题是：

```text
test 阶段长视频仍然会被压缩成固定 num_clips 个 clip，
导致大量中间帧没有参与测试。
```

正确目标是：

```text
一个长视频 → 拆成多个顺序 window
每个 window → [N, T, 3, H, W]
所有 window 合起来 → 覆盖完整视频所有帧
最后 padding 的重复帧 → 不重复计入指标
```

当前不要求跨 window persistent memory。也就是说，每个 window 内部有 clip-to-clip memory 即可，不要求 window 1 读取 window 0 的 memory。

---

## 一句话修复原则

不要通过增大 `actual_num_clips` 或 `test_max_clips` 来修复 test 覆盖问题。

正确做法是：

```text
video → sequential windows → dataset items
```

即：**一个视频在 test dataset 中展开成多个 window item**。

---

## 当前问题

如果当前代码里存在类似逻辑：

```python
if len(clips) > num_clips:
    # evenly subsample
    ...
```

或者：

```python
actual_num_clips = min(..., test_max_clips)
```

那么长视频仍然会被截断或抽样。

例如：

```text
video_length = 40
num_frames = 4
num_clips = 4
stride = 1
```

错误逻辑可能最终只保留：

```text
[0,1,2,3]
[12,13,14,15]
[24,25,26,27]
[36,37,38,39]
```

导致这些帧没有被测试：

```text
4~11, 16~23, 28~35
```

这不是真正的完整视频测试。

---

## 修改范围

本次只允许修改：

```text
zzz_dataset_toolkit/dataset.py
train_val_test_dinov3_lora.py 中 test batch 读取/valid_mask 过滤部分
新增 debug/test_test_windows.py
```

不要修改：

```text
train 采样逻辑
val 采样逻辑
model forward
temporal adapter
loss
optimizer
配置主体
```

---

## 1. 新增 test window 采样函数

在 `zzz_dataset_toolkit/dataset.py` 中新增独立函数，不要复用训练用的 `_sample_multi_clip_indices()`。

```python
def _sample_test_windows(
    video_length: int,
    num_clips: int,
    num_frames: int,
    stride: int = 1,
):
    """
    Build sequential fixed-size windows for test-time inference.

    Returns:
        windows: List[Dict]
        each item:
            {
                "frame_indices": List[List[int]],  # [N, T]
                "valid_mask": List[List[bool]],    # [N, T]
            }
    """
    assert video_length > 0
    assert num_clips > 0
    assert num_frames > 0
    assert stride > 0

    all_clips = []
    all_valid = []

    # Non-overlap windows by default.
    # Example: num_frames=4, stride=1:
    # clip0 = 0,1,2,3
    # clip1 = 4,5,6,7
    step = num_frames * stride

    for start in range(0, video_length, step):
        clip = []
        valid = []

        for i in range(num_frames):
            idx = start + i * stride

            if idx < video_length:
                clip.append(idx)
                valid.append(True)
            else:
                # Pad only at the end of video.
                clip.append(video_length - 1)
                valid.append(False)

        all_clips.append(clip)
        all_valid.append(valid)

    windows = []

    for i in range(0, len(all_clips), num_clips):
        window = all_clips[i:i + num_clips]
        valid_window = all_valid[i:i + num_clips]

        # Pad the last window to fixed [N, T].
        while len(window) < num_clips:
            window.append(window[-1])
            valid_window.append([False] * num_frames)

        windows.append({
            "frame_indices": window,
            "valid_mask": valid_window,
        })

    return windows
```

---

## 2. 在 Dataset 初始化时展开 test items

在 `VideoInpaintingDataset.__init__()` 末尾添加 test 专用展开逻辑。

伪代码如下：

```python
self.eval_items = None

if self.mode == "test" and self.num_clips > 1:
    self.eval_items = []

    for sample_idx, (video_dir, mask_dir) in enumerate(self.samples):
        frame_list = sorted(
            [p for p in os.listdir(video_dir) if is_image_file(p)],
            key=_numeric_key,
        )

        video_length = len(frame_list)
        if video_length <= 0:
            continue

        windows = _sample_test_windows(
            video_length=video_length,
            num_clips=self.num_clips,
            num_frames=self.num_frames,
            stride=self.clip_stride,
        )

        name = _derive_sample_name(video_dir)

        for window_id, item in enumerate(windows):
            self.eval_items.append({
                "sample_idx": sample_idx,
                "video_id": name,
                "window_id": window_id,
                "frame_indices": item["frame_indices"],
                "valid_mask": item["valid_mask"],
                "is_last_window": window_id == len(windows) - 1,
            })
```

如果项目里没有 `_derive_sample_name(video_dir)`，可以先用已有的 name 生成逻辑，例如：

```python
name = Path(video_dir).name
```

但要保持同一个视频所有 window 的 `video_id` 一致。

---

## 3. 修改 `__len__()`

当前如果 `__len__()` 仍然是：

```python
return len(self.samples) * self.dataset_repeat
```

说明 test 阶段一个视频仍然只对应一个 item，这是不对的。

改成：

```python
def __len__(self) -> int:
    if self.mode == "test" and self.num_clips > 1 and self.eval_items is not None:
        return len(self.eval_items)

    return len(self.samples) * self.dataset_repeat
```

验收例子：

```text
video_length = 40
num_clips = 4
num_frames = 4
stride = 1
```

test dataset 中这个视频应该产生：

```text
3 个 window item
```

而不是 1 个 item。

---

## 4. 修改 test 的 `__getitem__()`

在 `__getitem__()` 开头添加 test-window 分支：

```python
def __getitem__(self, idx: int):
    if self.mode == "test" and self.num_clips > 1 and self.eval_items is not None:
        item = self.eval_items[idx]

        sample_idx = item["sample_idx"]
        video_dir, mask_dir = self.samples[sample_idx]

        frame_list = sorted(
            [p for p in os.listdir(video_dir) if is_image_file(p)],
            key=_numeric_key,
        )
        mask_list = sorted(
            [p for p in os.listdir(mask_dir) if is_image_file(p)],
            key=_numeric_key,
        )

        return self._get_multi_clip_by_indices(
            video_dir=video_dir,
            mask_dir=mask_dir,
            frame_list=frame_list,
            mask_list=mask_list,
            clip_indices=item["frame_indices"],
            video_id=item["video_id"],
            window_id=item["window_id"],
            valid_mask=item["valid_mask"],
            is_last_window=item["is_last_window"],
        )

    # 原来的 train / val / baseline 逻辑保持不变
    ...
```

---

## 5. 新增 `_get_multi_clip_by_indices()`

该函数只按传入的 `clip_indices` 取帧，不允许在里面重新采样。

```python
def _get_multi_clip_by_indices(
    self,
    video_dir,
    mask_dir,
    frame_list,
    mask_list,
    clip_indices,
    video_id,
    window_id,
    valid_mask,
    is_last_window,
):
    all_frames = []
    all_masks = []
    original_h, original_w = None, None

    for clip in clip_indices:
        frames = []
        masks = []

        for frame_idx in clip:
            frame_path = str(Path(video_dir) / frame_list[frame_idx])
            mask_path = str(Path(mask_dir) / mask_list[frame_idx])

            frame = _read_image(frame_path)
            mask = _read_image(mask_path)

            mask = _align_mask_to_frame(mask, frame)
            mask = threshold_mask(mask)

            if original_h is None:
                original_h, original_w = frame.shape[:2]

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks.append(mask)

        # Test 阶段可以保留确定性预处理。
        # 如果 robust noise / jpeg 是测试鲁棒性设置，可以保留；
        # 如果是训练增强，test 阶段不要使用随机增强。
        if self.robust_noise_snr > 0:
            frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]

        if 1 <= self.robust_jpeg_quality <= 100:
            frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

        frame_tensors = []
        mask_tensors = []

        for img, mask in zip(frames, masks):
            img = cv2.resize(img, (self.input_size, self.input_size))
            mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0

            frame_tensors.append(self.to_tensor(img).unsqueeze(0))

            if mask.ndim == 2:
                mask = mask[:, :, None]

            mask_tensors.append(
                torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0)
            )

        all_frames.append(torch.cat(frame_tensors, dim=0))  # [T,3,H,W]
        all_masks.append(torch.cat(mask_tensors, dim=0))    # [T,1,H,W]

    frames_out = torch.stack(all_frames, dim=0)  # [N,T,3,H,W]
    masks_out = torch.stack(all_masks, dim=0)    # [N,T,1,H,W]

    return {
        "images": frames_out,
        "masks": masks_out,
        "video_id": video_id,
        "window_id": window_id,
        "frame_indices": torch.tensor(clip_indices, dtype=torch.long),
        "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
        "is_last_window": is_last_window,
        "original_h": original_h,
        "original_w": original_w,
        "name": video_id,
    }
```

注意：如果项目中已有 resize / normalize / albumentations 逻辑，尽量复用已有函数，避免和原数据处理不一致。上面代码是结构示例，agent 需要按项目现有实现适配。

---

## 6. 修改 test loop，支持 dict batch

在 `train_val_test_dinov3_lora.py` 的 `run_epoch()` 或 test loop 中，支持 dict batch。

原逻辑可能是：

```python
frames, masks = batch[0].to(device), batch[1].to(device)
```

改成兼容形式：

```python
if isinstance(batch, dict):
    frames = batch["images"].to(device)
    masks = batch["masks"].to(device)

    valid_mask = batch.get("valid_mask", None)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    frame_indices = batch.get("frame_indices", None)
    video_id = batch.get("video_id", None)
    window_id = batch.get("window_id", None)
else:
    frames = batch[0].to(device)
    masks = batch[1].to(device)
    valid_mask = None
    frame_indices = None
    video_id = None
    window_id = None
```

推理后，如果是 test 并且有 `valid_mask`，需要过滤 padding 帧：

```python
logits_all = forward_with_optional_chunk(model, frames, eval_frame_chunk)

if masks.ndim == 6:
    b, n, t, c, h, w = masks.shape

    logits = logits_all.reshape(
        b * n * t,
        1,
        logits_all.shape[-2],
        logits_all.shape[-1],
    )
    masks_flat = masks.reshape(b * n * t, c, h, w)

    if logits.shape[-2:] != masks_flat.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks_flat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if valid_mask is not None:
        valid_flat = valid_mask.reshape(b * n * t)
        logits = logits[valid_flat]
        masks_flat = masks_flat[valid_flat]

    loss = criterion(logits, masks_flat)
    metrics = binary_metrics_from_logits(
        logits,
        masks_flat,
        threshold=threshold,
    )
```

这一步的核心是：**最后一个 window 的 padding 帧不能重复计入 loss/metrics**。

---

## 7. 更推荐的指标聚合方式

最小修复可以先用 `valid_mask` 过滤 padding。

更严格的 test 指标应该按：

```text
video_id + frame_idx
```

聚合，确保每个真实帧只统计一次。

伪代码：

```python
pred_store = defaultdict(dict)
mask_store = defaultdict(dict)

for batch in test_loader:
    logits = model(images)
    probs = torch.sigmoid(logits)

    for b in range(B):
        vid = video_ids[b]

        for n in range(N):
            for t in range(T):
                if not valid_mask[b, n, t]:
                    continue

                frame_idx = int(frame_indices[b, n, t])

                pred_store[vid][frame_idx] = probs[b, n, t].detach().cpu()
                mask_store[vid][frame_idx] = masks[b, n, t].detach().cpu()

# After loop:
for vid in pred_store:
    frame_ids = sorted(pred_store[vid].keys())
    preds = torch.stack([pred_store[vid][i] for i in frame_ids], dim=0)
    gts = torch.stack([mask_store[vid][i] for i in frame_ids], dim=0)

    # compute per-video or global metrics
```

如果当前项目还没有 video-level 聚合，可以先不做这一步，但 `valid_mask` 过滤必须做。

---

## 8. 新增测试脚本

新增：

```text
debug/test_test_windows.py
```

内容至少包括：

```python
from zzz_dataset_toolkit.dataset import _sample_test_windows


def collect_valid_frames(windows):
    covered = set()

    for w in windows:
        frame_indices = w["frame_indices"]
        valid_mask = w["valid_mask"]

        for clip, valid_clip in zip(frame_indices, valid_mask):
            for idx, valid in zip(clip, valid_clip):
                if valid:
                    covered.add(idx)

    return covered


def test_sample_test_windows_cover_all_frames():
    windows = _sample_test_windows(
        video_length=40,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    covered = collect_valid_frames(windows)
    assert covered == set(range(40))


def test_sample_test_windows_short_video():
    windows = _sample_test_windows(
        video_length=10,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    covered = collect_valid_frames(windows)
    assert covered == set(range(10))


def test_window_shape_is_fixed():
    windows = _sample_test_windows(
        video_length=10,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    for w in windows:
        assert len(w["frame_indices"]) == 4
        assert len(w["valid_mask"]) == 4

        for clip in w["frame_indices"]:
            assert len(clip) == 4

        for valid_clip in w["valid_mask"]:
            assert len(valid_clip) == 4


def test_window_order_is_monotonic():
    windows = _sample_test_windows(
        video_length=40,
        num_clips=4,
        num_frames=4,
        stride=1,
    )

    previous_last = -1

    for w in windows:
        for clip, valid_clip in zip(w["frame_indices"], w["valid_mask"]):
            valid_indices = [idx for idx, valid in zip(clip, valid_clip) if valid]
            if not valid_indices:
                continue

            assert valid_indices[0] > previous_last
            previous_last = valid_indices[-1]
```

运行：

```bash
python -m pytest debug/test_test_windows.py
```

---

## 9. 验收标准

修复完成后，必须满足以下标准。

### Case 1

```text
video_length = 40
num_clips = 4
num_frames = 4
stride = 1
```

期望：

```text
window 0:
  clip 0: 0,1,2,3
  clip 1: 4,5,6,7
  clip 2: 8,9,10,11
  clip 3: 12,13,14,15

window 1:
  clip 0: 16,17,18,19
  clip 1: 20,21,22,23
  clip 2: 24,25,26,27
  clip 3: 28,29,30,31

window 2:
  clip 0: 32,33,34,35
  clip 1: 36,37,38,39
  clip 2: padding, valid_mask=False
  clip 3: padding, valid_mask=False
```

最终有效帧：

```text
0~39 全部覆盖一次
```

### Case 2

```text
video_length = 10
num_clips = 4
num_frames = 4
stride = 1
```

期望：

```text
window 0:
  clip 0: 0,1,2,3
  clip 1: 4,5,6,7
  clip 2: 8,9,9,9，其中 valid_mask = True, True, False, False
  clip 3: padding, valid_mask=False
```

最终有效帧：

```text
0~9 全部覆盖一次
```

### Case 3

test loader 的 batch 输入仍然是：

```text
images: [B,N,T,3,H,W]
masks:  [B,N,T,1,H,W]
```

model forward 不需要修改。

---

## 10. 最终结论

本次修复的核心不是让 memory 跨完整视频传播，而是让 test 阶段完整覆盖视频帧。

当前允许：

```text
window 0 内部 clip 之间有 memory
window 1 内部 clip 之间有 memory
window 2 内部 clip 之间有 memory
```

当前不要求：

```text
window 1 读取 window 0 的 memory
window 2 读取 window 1 的 memory
```

只要所有 window 的有效帧都参与测试，当前 window-level memory 设计在性能和工程复杂度上是合理的。
