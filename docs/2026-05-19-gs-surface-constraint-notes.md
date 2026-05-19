# 2026-05-19 GS Surface Constraint Notes

## 背景

当前 HeadStudio 分支在 `d85d396 Checkpoint before GS position constraint changes` 上做了回退点，用来标记修改 Gaussian position constraints 前的状态。此前代码已有两类约束：

- `lambda_position`：在 prune-only 阶段后，惩罚 Gaussian 中心离绑定 FLAME 三角面中心过远。
- `lambda_scaling`：惩罚 Gaussian 尺度超过绑定三角面的局部尺度。

这两个约束能减少点云跑飞和局部膨胀，但它们不是严格的三角面几何约束。旧的位置约束只看 Gaussian 到三角形 centroid 的距离，不判断该点是否仍在三角形内部，也不判断它是否离开了三角形平面。

## 本次修改目标

采用软约束方案 B：保留现有 Gaussian 的自由 3D offset 参数化，不修改 PLY 存储、resume、densify/split/clone 逻辑，只在训练 loss 中增加额外几何诊断和软惩罚。

目标是让每个 Gaussian 更贴近它绑定的 FLAME 三角面，同时避免一开始就把 Gaussian 硬锁死在三角形平面内，给头发、衣领、胡子、头盔等可能离开 FLAME 表面的结构保留表达空间。

## 几何思路

每个 Gaussian 已经通过 `self._faces` 绑定到一个 FLAME 三角形。新逻辑会取出该 Gaussian 绑定的三角形顶点：

```text
A, B, C
```

然后对当前 Gaussian 世界坐标 `P` 做两件事：

1. 将 `P` 投影到三角形平面，得到 `P_projected`。
2. 对 `P_projected` 计算 barycentric 坐标：

```text
P_projected = u * A + v * B + w * C
u + v + w = 1
```

如果 `u/v/w` 都大于等于 0，则投影点在三角形内部。若任一值小于 0，则说明投影点已经跑到三角形外侧。

同时计算：

```text
normal_offset = dot(P - P_projected, triangle_normal)
```

这个值表示 Gaussian 离绑定三角形平面的有符号距离。

## 新增代码

### `gaussiansplatting/scene/gaussian_flame_model.py`

新增：

- `get_bound_triangles()`：集中计算每个 Gaussian 当前绑定的 FLAME 三角形顶点。
- `get_surface_constraint_terms()`：返回：
  - `barycentric`: shape `[num_gs, 3]`
  - `normal_offset`: shape `[num_gs]`

同时 `get_tris_scaling()` 和 `get_trans_matrix()` 改为复用 `get_bound_triangles()`，减少重复的 FLAME triangle 计算代码。

### `threestudio/systems/Head3DGSLKs.py`

新增配置项：

```python
surface_constraint_start_step: int = 2400
```

训练中新增两个 loss：

```python
loss_barycentric_inside = F.relu(-barycentric).mean()
loss_normal_offset = torch.abs(normal_offset).mean()
```

含义：

- `loss_barycentric_inside`：只惩罚小于 0 的 barycentric 分量，拉回跑到三角形外侧的 Gaussian。
- `loss_normal_offset`：惩罚离三角形平面过远的 Gaussian。

这两个 loss 只有在达到 `surface_constraint_start_step` 且对应权重大于 0 时才会计算，避免默认配置下增加训练开销。

### `configs/headstudio.yaml`

新增：

```yaml
surface_constraint_start_step: 2400

loss:
  lambda_barycentric_inside: 0.0
  lambda_normal_offset: 0.0
```

默认权重设为 0，表示默认训练行为不变。后续实验可以通过命令行打开，例如：

```bash
system.loss.lambda_barycentric_inside=1.0 \
system.loss.lambda_normal_offset=1.0
```

## 为什么先做软约束

没有直接把 Gaussian 参数改成 `barycentric + normal_offset` 的硬参数化，主要原因是硬改会影响范围更大：

- 初始化逻辑要改。
- PLY save/load 字段要改。
- resume 兼容性要处理。
- densify/split/clone 生成新点的逻辑要重新设计。
- 可能限制头发、衣物、角色头盔等表面外结构。

本次方案只增加 loss，不改变现有参数存储和训练主流程，风险更低，方便快速做消融实验。

## 验证

新增测试：

```text
tests/test_gaussian_surface_constraint_source.py
```

已跑过：

```text
python tests/test_gaussian_surface_constraint_source.py
python tests/test_disable_eye_neck_pose_source.py
python tests/test_headstudio_ply_resume_source.py
python tests/test_uncond_rand_exp_multi_source_loader.py
python tests/test_convert_talkvid_to_ruiheadstudio.py
python -m py_compile gaussiansplatting/scene/gaussian_flame_model.py threestudio/systems/Head3DGSLKs.py tests/test_gaussian_surface_constraint_source.py
```

## 后续实验建议

第一组实验建议保持默认 2400 step 后开启，尝试较小权重：

```bash
system.loss.lambda_barycentric_inside=0.5 \
system.loss.lambda_normal_offset=0.2
```

如果点云仍明显漂出 FLAME 表面，可逐步提高：

```bash
system.loss.lambda_barycentric_inside=1.0 \
system.loss.lambda_normal_offset=1.0
```

观察重点：

- 正脸和侧脸是否减少浮点/毛刺。
- 头发、头盔、衣领是否被过度压回 FLAME 表面。
- 表情驱动时 Gaussian 是否更稳定贴合面部拓扑。
- 是否牺牲角色身份细节或夸张外形。
