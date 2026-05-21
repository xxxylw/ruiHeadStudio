# 2026-05-21 Temporal Window Sampling and Loss Design

## 背景

当前 RuiHeadStudio 的训练 dataloader 已经支持多姿态源加载。训练配置里有两个 pose source：

```yaml
talkshow_train_paths:
  - name: talkshow
    path: ./talkshow/collection/chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg__68891-00_01_40-00_01_46_exp.npy
    weight: 1.0
  - name: talkvid
    path: ./talkshow/collection/talkvid/talkvid_tracker_exp.npy
    weight: 1.0
```

每个 source 加载成一个 `PoseSource`：

```python
PoseSource(
    name=name,
    weight=weight,
    sequences=sequences,
)
```

每个 `.npy` 文件里保存的是 sequence list。每条 sequence 是一个视频片段或 tracking clip，包含：

- `expression`
- `jaw_pose`
- `leye_pose`
- `reye_pose`
- `neck_pose`
- `video_name`
- `clip_name`
- 其他 source metadata

当前训练采样逻辑是：

```text
按 source weight 随机选 source
在 source 内随机选 sequence
在 sequence 内随机选一帧
```

这适合单帧随机姿态训练，但不适合做 temporal loss。因为相邻 training step 之间模型参数已经更新，不能干净地比较同一个模型状态下相邻 FLAME frame 的几何变化。

## 目标

本轮设计目标是把训练采样改成 temporal window 形式，并在 system 里加入可开关的 temporal geometry loss。

核心目标：

- source 仍然按配置权重随机选，保留 talkshow/talkvid 的数据分布控制能力。
- 每个 source 内部维护独立 sequence/frame cursor。
- 被选中的 source 从自己的 cursor 顺序取 sequence 和 frame。
- 每个 training step 返回一个连续 frame window。
- SDS/render 主路径仍只使用 window 的 primary frame，避免显存翻倍。
- temporal loss 使用 window 内连续 FLAME 参数计算几何状态。
- 第一版 temporal loss 聚焦 motion-following 和 normalized scale stability。

## 非目标

第一版不做以下内容：

- 不让 source 顺序轮转；source 仍然 weighted random。
- 不做 full temporal window rendering。
- 不对 window 中每一帧都算 SDS。
- 不做 image/opacity temporal consistency。
- 不做 local offset temporal loss。
- 不做 opacity mask 或 region mask。
- 不做 variable-length window。
- 不做 tail padding。
- 不 shuffle sequence order。

这些能力后续可以加，但第一版先保证语义清楚、改动面小、能稳定跑起来。

## Dataloader 设计

### Source 选择

source 继续使用当前权重随机逻辑：

```python
source = rng.choices(
    sources,
    weights=[item.weight for item in sources],
    k=1,
)[0]
```

当前 `talkshow` 和 `talkvid` 权重都是 `1.0`，所以 source 层面大致 50/50。后续可以通过 config 调整权重。

保留 weighted random 的原因：

- source 权重是当前配置中明确的数据分布控制手段。
- talkshow/talkvid 之间没有 temporal 连续关系，不应该跨 source 建 temporal window。
- 如果 source 也顺序走，训练可能长时间集中在同一 source，造成阶段性分布偏移。

### 每个 source 独立 cursor

每个 source 维护自己的 cursor：

```python
source_cursor = {
    "sequence_index": 0,
    "frame_index": 0,
}
```

实际实现建议按 source list index 存，而不是按 name 存，避免 source name 重名：

```python
self.pose_source_cursors = [
    {"sequence_index": 0, "frame_index": 0},
    {"sequence_index": 0, "frame_index": 0},
]
```

当某个 source 被 weighted random 选中时，从该 source 自己的 cursor 继续往后取 window。其他 source 的 cursor 不受影响。

这样采样序列可能是：

```text
step 1: talkshow -> talkshow seq0 frames [0,1]
step 2: talkvid  -> talkvid  seq0 frames [0,1]
step 3: talkshow -> talkshow seq0 frames [1,2]
step 4: talkshow -> talkshow seq0 frames [2,3]
step 5: talkvid  -> talkvid  seq0 frames [1,2]
```

### Sequence 顺序

每个 source 内 sequence 固定顺序循环：

```text
seq0 -> seq1 -> seq2 -> ... -> seq0
```

第一版不 shuffle。原因：

- 更可复现。
- 更容易 debug 某个 sequence 的具体问题。
- 当前训练已有 camera/SDS/source sampling 随机性，不需要额外引入 sequence shuffle。

如果后续发现源文件排序造成偏差，可以再加：

```yaml
temporal_sequence_shuffle: true
```

### Frame window

每个 step 取一个连续 frame window：

```text
[t, t+1, ..., t+T-1]
```

窗口内部 frame 固定连续，不支持 inside-window stride。只支持窗口起点 stride：

```yaml
temporal_window_stride: 1
```

例如 `temporal_window_length=2`，`temporal_window_stride=1`：

```text
[0,1] -> [1,2] -> [2,3]
```

如果 `temporal_window_stride=2`：

```text
[0,1] -> [2,3] -> [4,5]
```

### Sequence 尾部处理

当 sequence 剩余帧数不足一个完整 window 时，丢弃尾部不足窗口，切到该 source 的下一个 sequence。

例如：

```text
sequence length = 100
temporal_window_length = 4
最后一个合法 window = [96,97,98,99]
```

如果 cursor 到 97，则不返回：

```text
[97,98,99]
[97,98,99,99]
```

而是切到下一个 sequence 的开头。

原因：

- 固定 shape 更容易实现和测试。
- 不做 repeat padding，避免给 temporal loss 引入假静止信号。
- `T` 通常较小，最多丢掉 `T-1` 帧。

## Config 设计

建议新增 data 配置：

```yaml
data:
  temporal_window_enabled: false
  temporal_window_length: 2
  temporal_window_stride: 1
  temporal_primary_index: 0
  temporal_same_camera: true
```

含义：

- `temporal_window_enabled`：是否启用 temporal window 采样。默认 `false`，保持旧训练行为。
- `temporal_window_length`：每个 step 使用多少连续帧。第一版建议实验用 2。
- `temporal_window_stride`：窗口起点每次前进多少帧。
- `temporal_primary_index`：窗口中哪一帧用于 render/SDS。默认 0。
- `temporal_same_camera`：window 内所有 frame 是否共享同一组 camera/light/fov。第一版固定使用 true。

建议新增 system 配置：

```yaml
system:
  temporal_loss_start_step: 2400

  loss:
    lambda_temporal_motion: 0.0
    lambda_temporal_scale_ratio: 0.0
```

含义：

- `temporal_loss_start_step`：temporal loss 从第几步开始。默认 2400，和 prune-only / soft surface constraint 对齐。
- `lambda_temporal_motion`：motion-following loss 权重。
- `lambda_temporal_scale_ratio`：normalized scale stability loss 权重。

所有 temporal loss 默认权重为 0，需要实验时显式打开。

## Batch Contract

第一版不把 `T * B` 全部送进 renderer。原因是当前 1024 分辨率、batch size 8、双 ControlNet 已经接近 3090 显存上限。把 window 全部 render 会显著增加 OOM 风险。

因此 batch 保持现有主路径字段：

```python
expression
jaw_pose
leye_pose
reye_pose
neck_pose
flame_conds
```

这些字段来自 window 的 primary frame，默认是 `temporal_primary_index=0`。

同时新增 temporal 字段：

```python
temporal_enabled
temporal_source_name
temporal_source_index
temporal_sequence_index
temporal_frame_indices
temporal_primary_index
temporal_window_length
temporal_expression
temporal_jaw_pose
temporal_leye_pose
temporal_reye_pose
temporal_neck_pose
```

shape 建议：

```text
temporal_expression: [T, expression_dim]
temporal_jaw_pose:   [T, 3]
temporal_leye_pose:  [T, 3]
temporal_reye_pose:  [T, 3]
temporal_neck_pose:  [T, 3]
```

primary frame 字段保持当前 shape：

```text
expression: [1, expression_dim]
jaw_pose:   [1, 3]
...
```

## Camera 策略

window 内所有 frame 共用同一组 camera/light/fov。

第一版只用 primary frame 生成 `flame_conds` 并参与 render/SDS。后续 frame 不生成 ControlNet 条件，也不参与 SDS。

这样一个 step 的监督关系是：

```text
primary frame:
  render + ControlNet pose/depth + SDS

window frames:
  temporal geometry loss only
```

这种设计的优点：

- 不增加主渲染 batch 大小。
- 不改 ControlNet/SDS 主流程。
- temporal loss 的变量更干净：同一 sequence、连续 pose、同一 camera。

## Temporal State Helper

dataloader 只负责返回 FLAME 参数，不负责几何计算。

几何状态由 `GaussianFlameModel` 或 system helper 根据 window 中每一帧 FLAME 参数计算。

原因：

- dataloader 不应该知道 Gaussian 绑定 faces、当前 shape、scale、center 等模型内部状态。
- Gaussian shape 和 local parameters 是训练中变化的，dataloader 预计算几何会过期。
- `GaussianFlameModel` 已经拥有 `get_xyz`、`get_trans_matrix`、`get_scaling` 等几何能力。

helper 需要对每个 frame 计算：

```text
xyz
triangle_centroid
triangle_area
scaling
scale_ratio
```

其中：

```text
scale_ratio = get_scaling / sqrt(triangle_area)
```

用于 normalized scale temporal loss。

helper 必须内部 save/restore 当前 Gaussian pose。不能让 helper 计算完 window 后，把 `self.gaussian._expression/_jaw_pose/...` 停留在 window 最后一帧。

temporal pose tensors 是外部 condition，应 detach。temporal loss 不训练 pose 数据本身，但不能把整个 helper 包在 `torch.no_grad()` 里，因为 temporal loss 需要回传到 Gaussian 参数和可训练 shape。

## Temporal Loss 设计

### Motion-following loss

相邻 frame 中，绑定三角面 centroid 会随 FLAME pose 移动。Gaussian 世界坐标也应该跟随绑定面片运动，而不是自己乱漂。

定义：

```text
delta_gaussian = xyz[t+1] - xyz[t]
delta_triangle = triangle_centroid[t+1] - triangle_centroid[t]
loss_temporal_motion = mean(||delta_gaussian - delta_triangle||)
```

这个 loss 约束的是：

```text
Gaussian 的相邻帧位移应该接近它绑定 FLAME 面片的相邻帧位移。
```

### Normalized scale stability loss

直接约束 world scale 不理想，因为 world scale 会受到绑定三角形面积变化影响。更合理的是约束局部归一化 scale：

```text
scale_ratio = get_scaling / sqrt(triangle_area)
```

然后对相邻 frame 做稳定性约束：

```text
loss_temporal_scale_ratio = mean(|scale_ratio[t+1] - scale_ratio[t]|)
```

这个 loss 约束的是：

```text
Gaussian 相对绑定面片的局部大小不要在相邻帧异常抖动。
```

这是第一版 temporal loss 的重点实验项。

### 不做 local offset loss

不单独做 local offset temporal loss。

原因是 `_xyz` 本身是 canonical local parameter，在一个 step 内不随 frame 变。如果把 world offset 反投影回每帧 local coordinates，理论上也会接近同一个 `_xyz`，很容易变成重复或无效约束。

第一版先聚焦：

- motion-following
- normalized scale stability

## Mask 策略

第一版 temporal loss 对所有 Gaussian 计算，不做 opacity mask，也不做 region mask。

原因：

- 实现简单，减少额外超参。
- `temporal_loss_start_step=2400` 已经避开早期最不稳定阶段。
- loss 权重默认 0，实验时可以从小权重开始。

如果后续发现头发、头盔、衣领等区域被过度约束，再考虑增加：

```yaml
temporal_loss_opacity_threshold
temporal_loss_region_mask
```

## 实现顺序建议

建议下一步用 `superpowers:writing-plans` 写实现计划，然后按计划执行。

实现应拆成三块：

1. Dataloader cursor/window contract
   - 加配置字段。
   - 加每 source 独立 cursor。
   - 输出 temporal window metadata。
   - 保持默认 `temporal_window_enabled=false` 时旧行为不变。

2. Gaussian temporal state helper
   - 根据一帧 FLAME 参数计算 geometry state。
   - 支持 window 内多帧。
   - 内部 save/restore 当前 pose。
   - 确保 temporal pose detach，但 Gaussian 参数可接收梯度。

3. System temporal losses
   - 在 `training_step` 中检测 temporal batch。
   - step >= `temporal_loss_start_step` 且 loss 权重大于 0 时计算。
   - 加 `train/loss_temporal_motion` 和 `train/loss_temporal_scale_ratio` 日志。
   - 默认权重 0。

## 当前共识摘要

```text
source:
  weighted random

sequence:
  per-source independent cursor
  fixed cyclic order

frame:
  contiguous temporal window
  configurable window length
  configurable window-start stride
  drop incomplete tail window

SDS/render:
  primary frame only
  default primary_index = 0

camera:
  same camera/light/fov inside window

temporal loss v1:
  motion-following
  normalized scale-ratio stability
  all Gaussians
  no opacity/region mask
  start at step 2400
  default weights 0
```
