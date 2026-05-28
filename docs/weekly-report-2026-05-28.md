# 周报：Thor Temporal 训练复盘与约束问题定位

## 本周工作

1. 梳理当前训练损失构成
   - 重新检查 `Head3DGSLKsRig.training_step()` 中的默认 loss 组合，明确当前训练主要由四类目标构成：
     - SDS / ControlNet guidance：让渲染结果符合文本 prompt 和 FLAME pose/depth 条件。
     - Sparsity / scale ratio：控制 Gaussian 不要糊满空间，也不要相对绑定 FLAME 三角面异常膨胀。
     - Local position：约束 Gaussian 不要离自己的绑定三角面太远。
     - Temporal losses：约束连续表情帧下 Gaussian 的局部绑定状态不要滑动、抖动或忽大忽小。
   - 默认实际进入总 loss 的主要项为：

```text
loss =
  1.0   * loss_sds
+ 1.0   * loss_sparsity
+ 0.05  * loss_scale_ratio
+ 0.2   * loss_local_position              # step >= 2400
+ 0.005 * loss_temporal_local_offset       # step >= 2400
+ 0.002 * loss_temporal_local_offset_accel # step >= 2400
+ 0.001 * loss_temporal_scale_ratio_accel  # step >= 2400
```

   - 旧的 `lambda_position`、`lambda_scaling`、surface constraint、shape 和 opaque loss 当前默认关闭，部分仍会记录日志但不影响优化。

2. 复跑 Thor temporal 训练
   - 使用 `scripts/headstudio.sh` 中 Thor 的 prompt 启动后台训练：

```text
a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation
```

   - 训练通过 `setsid nohup` 在宿主环境启动，避免 Codex 会话断开后进程被沙箱带走。
   - 启动脚本与日志位置：

```text
logs/run_headstudio_thor_20260527.sh
logs/headstudio_thor_marvel_20260527.log
```

   - 训练输出目录：

```text
outputs/headstudio/headstudio_thor_marvel_20260527@20260527-184725/
```

   - 训练进入正常 step 后，GPU 显存稳定在约 23GB / 24GB，未出现 CUDA OOM。说明本次失败不是显存不足。

3. 定位训练中断原因
   - 训练没有跑满 `10000` step，而是在约 `2678` step 中断。
   - 最后一次可用 validation 输出在 `2600` step，保存了：

```text
outputs/headstudio/headstudio_thor_marvel_20260527@20260527-184725/save/last.ply
outputs/headstudio/headstudio_thor_marvel_20260527@20260527-184725/save/it2600-0.png
outputs/headstudio/headstudio_thor_marvel_20260527@20260527-184725/save/it2600-1.png
outputs/headstudio/headstudio_thor_marvel_20260527@20260527-184725/save/it2600-2.png
outputs/headstudio/headstudio_thor_marvel_20260527@20260527-184725/save/it2600-3.png
```

   - `last.ply` 大小约 16MB，保存时间为 2026-05-27 20:08。
   - 退出错误为：

```text
torch._C._LinAlgError: linalg.inv: ... the input matrix is singular.
```

   - 错误发生在 temporal loss 计算路径：

```text
Head3DGSLKsRig.training_step()
  -> compute_temporal_losses()
    -> GaussianFlameModel.get_temporal_surface_states()
      -> triangle_basis.inverse()
```

4. 分析 temporal 约束的几何假设
   - 当前每个 Gaussian 都绑定到一个 FLAME triangle。
   - Temporal local offset 约束的目标是：在连续表情帧中，Gaussian 相对自己绑定三角面的局部坐标位置保持稳定。
   - 该约束需要用 FLAME triangle 的三个顶点构造 TBN 局部坐标系，并对 `triangle_basis` 求逆，把 world-space offset 转到局部坐标：

```text
local_offset = inverse(triangle_basis) * normalized_world_offset
```

   - 这隐含了一个强假设：每个绑定 FLAME triangle 在所有 expression / jaw pose 下都能提供稳定、可逆的局部坐标系。
   - 实际 TalkSHOW / TalkVid 表情窗口中，某些 triangle 可能在特定表情帧里退化或近退化，例如面积接近 0、边长过短、三个顶点近似共线，导致 TBN basis 不可逆。
   - 因此，本次问题不是随机训练崩溃，而是 temporal local offset 约束缺少 triangle quality 检查和降级策略。

## 当前结果

- Thor 训练已成功跑过 ControlNet 加载、densify/prune 早期阶段和 temporal loss 启用点。
- 训练不是因为显存不足中断，而是在 temporal local offset loss 里遇到 FLAME triangle basis 奇异矩阵。
- 当前仍有可检查的中间结果，最有价值的是 `2600` step 的 validation 图和 `last.ply`。
- Temporal window 默认开启后，训练能够跑到 temporal loss 阶段，但当前局部坐标约束对退化三角面不够鲁棒。

## 问题总结

这次问题是在引入 temporal window 和 temporal local surface constraint 后暴露的。

原本希望做到的是：同一个 Gaussian 绑定到 FLAME 表面后，在连续表情帧里跟随绑定三角面运动，并保持相对三角面的局部位置和尺度变化平滑。这样可以减少动画时的滑动、抖动和 scale 闪烁。

问题在于 local offset 约束需要对每个绑定三角面的 TBN basis 求逆。如果三角面在某个表情帧退化，局部坐标系就没有稳定逆矩阵。当前代码没有识别这种情况，而是直接求逆，因此一个坏三角面就会让整次训练中断。

## 后续计划

1. 给 temporal surface state 增加 triangle quality 检查
   - 统计每个绑定 triangle 的面积、边长、basis determinant 或 condition 指标。
   - 对不可逆或近退化 triangle 生成 valid mask。
   - 记录 `temporal_invalid_ratio`、`temporal_min_area`、`temporal_valid_count`，确认问题是少量坏 face 还是系统性绑定问题。

2. 对 temporal local offset 做 mask 和 fallback
   - 健康 triangle 继续使用 local offset / local offset acceleration 约束。
   - 退化 triangle 不参与 local offset inverse 相关 loss。
   - 必要时对退化区域降级为 centroid motion 或 world-space motion 约束，避免完全失去 temporal 稳定性。

3. 避免单纯依赖 `pinv`
   - `torch.linalg.pinv` 可以避免 crash，但退化 triangle 上的局部坐标本身不可靠。
   - 更稳妥的策略是先 mask 不可信几何，再用伪逆作为最后兜底，而不是让所有退化 face 继续贡献强梯度。

4. 重新跑 Thor temporal 对照实验
   - 修复后用同一 Thor prompt 复跑。
   - 重点观察训练是否能越过 `2678` step，并继续到 `10000` step。
   - 对比 `loss_temporal_local_offset`、`loss_temporal_local_offset_accel`、`loss_temporal_scale_ratio_accel` 的日志稳定性。
   - 检查输出中的动画抖动、表面滑动和大 scale Gaussian outlier 是否改善。
