# 周报：Temporal Window 与局部 Scale 约束小改动

## 本周工作

1. 默认启用 Temporal Window 训练采样
   - 将 `configs/headstudio.yaml` 中的 `data.temporal_window_enabled` 默认改为 `true`。
   - 将 `temporal_window_length` 从 2 提升到 3，保持 `temporal_window_stride=1`、`temporal_primary_index=0`，即每个训练 step 从同一段表情序列里取连续三帧，但主渲染和 SDS 仍只使用第一帧。
   - 这样不会把渲染 batch 扩大成 `T * B`，显存开销基本保持和单帧训练一致。
   - temporal window 的完整 FLAME 参数只用于几何约束，目标是让同一个模型状态下的连续表情帧具有更稳定的 Gaussian 局部绑定状态。

2. 将 Temporal Loss 升级到局部绑定坐标约束
   - 保留旧的一阶 `lambda_temporal_motion` 和 `lambda_temporal_scale_ratio` 代码路径，但默认权重设为 `0.0`。
   - 新增 `lambda_temporal_local_offset`，约束 Gaussian 在绑定三角形局部坐标中的 offset 跨帧稳定。
   - 新增 `lambda_temporal_local_offset_accel`，对局部 offset 做二阶平滑，减少连续三帧内的抖动。
   - 新增 `lambda_temporal_scale_ratio_accel`，对局部 scale ratio 做二阶平滑，减少局部尺度在动画帧间突然闪动。
   - 新配置：

```yaml
data:
  temporal_window_length: 3

system:
  loss:
    lambda_temporal_motion: 0.0
    lambda_temporal_scale_ratio: 0.0
    lambda_temporal_local_offset: 0.5
    lambda_temporal_local_offset_accel: 0.2
    lambda_temporal_scale_ratio_accel: 0.1
```

   - 代码内部会对这些权重乘 `0.01`，所以实际进入总 loss 的权重仍然较小。
   - `temporal_loss_start_step` 保持 `2400`，避免在早期 densify/prune 不稳定阶段过早约束动画几何。

3. 新增默认局部 Gaussian 约束
   - 将旧 `lambda_position` 和 `lambda_scaling` 默认设为 `0.0`，默认训练不再计算旧 position/scale loss。
   - 新增 `lambda_local_position`，复用原有局部位置约束公式，但使用新的配置名和日志名，作为默认位置约束。
   - 新增局部 scale ratio 约束，作为默认尺度约束。
   - 新配置：

```yaml
system:
  scale_ratio_threshold: 0.5

  loss:
    lambda_position: 0.0
    lambda_scaling: 0.0
    lambda_local_position: 20.0
    lambda_scale_ratio: 5.0
```

   - 新 loss 的核心计算是：

```python
scale_ratio = scaling / (tris_scaling.unsqueeze(-1) + 1e-10)
scale_ratio_excess = F.relu(scale_ratio - scale_ratio_threshold)
loss_scale_ratio = (scale_ratio_excess ** 2).mean()
```

   - 这相当于约束每个 Gaussian 的尺度不要超过其绑定 FLAME 三角面的局部尺度比例。
   - 使用平方惩罚是为了更重点地压制少量特别大的 scale outlier，而不是平均压小所有 Gaussian。
   - 第一版没有做 opacity weighting，但代码中保留了注释，后续如果可见大椭球仍然存在，可以尝试对该 loss 乘 detached opacity。

## 改动动机

近期 Thor temporal 训练结果中可以看到少量大 scale 椭球导致的高亮 streak。检查 `last.ply` 后发现，虽然大多数 Gaussian scale 正常，但仍有少量明显 outlier：

```text
exp(scale).p99:  0.876
exp(scale).p999: 1.956
exp(scale).max:  31.484
exp(scale) > 4:  66
exp(scale) > 8:  14
```

原有 `loss_scaling` 会惩罚超过局部三角面尺度的 Gaussian，但它直接对 world scale 做 L1 约束。新增的 Local Scale Ratio Loss 更贴近当前位置约束的思路：都以绑定 FLAME 三角面为局部参考系，限制 Gaussian 不要相对自己的绑定三角面异常膨胀。

同时，默认位置约束也切到新的 `lambda_local_position` 入口。它第一版复用原有 position 公式，仍从 `prune_only_start_step=2400` 后生效，但默认权重提高到 `20.0`。代码内部会乘 `0.01`，所以有效权重为 `0.2`，用于承担旧 position/scale 默认关闭后的主要 GS 稳定约束。

## 当前状态

- Temporal window 默认开启，但仍只渲染 primary frame，不额外增加 SDS 渲染成本。
- 旧的一阶 Temporal motion / scale-ratio loss 默认关闭；新的 local offset / 二阶平滑 temporal loss 默认开启。
- 旧 `lambda_position` 和 `lambda_scaling` 默认关闭；只有命令行显式设置为正值时才会计算和记录旧 loss。
- `lambda_local_position` 默认开启，用于约束 Gaussian 不要离绑定三角形局部中心太远。
- Local Scale Ratio Loss 默认开启，用于压制少量大尺度 Gaussian outlier。

## 验证

本轮改动做了轻量源码契约测试和语法检查：

```bash
conda run -n ruiheadstudio-bnbfix python tests/test_scale_ratio_loss_source.py
conda run -n ruiheadstudio-bnbfix python tests/test_temporal_window_training_source.py
conda run -n ruiheadstudio-bnbfix python tests/test_uncond_rand_exp_multi_source_loader.py
conda run -n ruiheadstudio-bnbfix python -m py_compile threestudio/systems/Head3DGSLKs.py tests/test_scale_ratio_loss_source.py tests/test_temporal_window_training_source.py
git diff --check -- configs/headstudio.yaml threestudio/systems/Head3DGSLKs.py tests/test_scale_ratio_loss_source.py tests/test_temporal_window_training_source.py
```

以上检查均通过。

## 后续计划

1. 用相同 prompt 重新跑一组 Thor temporal 训练，对比头盔高亮 streak 和大 scale 椭球是否减少。
2. 重点观察 `train/loss_scale_ratio`、`train/loss_local_position`、`train/loss_temporal_local_offset` 和测试视角中的金属高光区域。
3. 如果仍有明显可见 outlier，下一步尝试 opacity-weighted scale ratio penalty。
4. 如果动画仍有抖动，再考虑提高 `lambda_temporal_local_offset_accel` 或加入渲染层 opacity/depth temporal consistency。
