# 2026-05-21 Soft GS Surface Constraint Method

## 目标

这次改动的目标是让 HeadStudio 训练出来的 3D Gaussian 更稳定地贴合它绑定的 FLAME 头部拓扑，减少训练后期常见的点云漂移、浮点、毛刺和局部膨胀，同时保留角色外形的表达自由度。

我们没有把 Gaussian 硬改成“只能在 FLAME 表面上”的参数化，而是在训练 loss 中增加一组软几何约束。这样可以继续支持头发、头盔、胡子、衣领等本来就可能离开 FLAME 表面的结构，也不需要改 PLY 存储、resume、densify、split、clone 等训练基础逻辑。

## 旧约束方法的问题

修改前主要有两类和 Gaussian 空间稳定性相关的约束。

第一类是 `lambda_position`。它在 prune-only 阶段后生效，计算 Gaussian 当前中心离绑定三角形局部中心的距离。如果距离超过基于三角形尺度得到的阈值，就用 Smooth L1 loss 把它往局部中心附近拉。

第二类是 `lambda_scaling`。它限制 Gaussian 的尺度不要明显超过绑定三角面的局部尺度，避免局部点云膨胀成过大的 blob。

这两个约束能减少一部分跑飞和膨胀，但它们只使用了比较粗的几何信息：

- `lambda_position` 主要看 Gaussian 离三角形 centroid 有多远。
- `lambda_scaling` 主要看 Gaussian 自身尺度是否超过局部三角形尺度。

它们没有回答两个更精确的问题：

1. Gaussian 投影到绑定三角面所在平面后，是否还在这个三角形内部？
2. Gaussian 是否已经沿法线方向离开绑定三角面太远？

所以旧约束能控制“离中心太远”和“尺度太大”，但不能清楚地区分“沿三角面切向跑出边界”和“沿法线方向飘离表面”。这就是本次新增软表面约束要补上的部分。

## 新约束的几何思路

每个 Gaussian 已经通过 `self._faces` 绑定到一个 FLAME 三角形。新增方法先取出当前 FLAME 状态下每个 Gaussian 绑定的三角形：

```text
A, B, C
```

然后取 Gaussian 当前世界坐标：

```text
P
```

我们对 `P` 做两步几何诊断。

第一步，计算三角形法线，把 `P` 投影到三角形所在平面：

```text
P_projected = P - dot(P - A, normal) * normal
```

这里的法线方向距离就是：

```text
normal_offset = dot(P - P_projected, normal)
```

它表示 Gaussian 沿绑定三角面的法线方向离开了多远。

第二步，对 `P_projected` 计算 barycentric 坐标：

```text
P_projected = u * A + v * B + w * C
u + v + w = 1
```

如果 `u`、`v`、`w` 都大于等于 0，投影点就在三角形内部。如果任意一个分量小于 0，说明投影点已经越过三角形边界，跑到绑定面片外侧。

这样我们把“Gaussian 是否贴着绑定面片”拆成两个更明确的问题：

- 切向是否跑出三角形边界：看 barycentric 是否有负数。
- 法向是否离表面太远：看 `normal_offset` 的绝对值。

## 代码实现

### 1. 统一获取绑定三角形

在 `gaussiansplatting/scene/gaussian_flame_model.py` 中新增 `get_bound_triangles()`。

它会根据当前可训练的 FLAME shape、expression、jaw pose、eye pose、neck pose 重新得到 FLAME 顶点，然后完成和原训练流程一致的坐标变换：

- 减去 `self.center`
- 乘以 `self.scale`
- 交换 y/z 坐标，把 OpenGL 坐标系转成 Blender 风格
- 按 `flame_scale` 缩放
- 用 `self.get_faces` 取出每个 Gaussian 当前绑定的三角形顶点

返回结果是：

```text
tris: [num_gaussians, 3, 3]
```

其中每个 Gaussian 对应一个三角形，每个三角形有 3 个顶点，每个顶点是 3D 坐标。

同时，原来的 `get_trans_matrix()` 改为复用 `get_bound_triangles()`，避免多个地方重复计算 FLAME 三角形。

### 2. 计算表面约束项

新增 `get_surface_constraint_terms()`。

这个方法做的事情是：

1. 调用 `get_bound_triangles()` 得到每个 Gaussian 绑定的三角形。
2. 调用 `self.get_xyz` 得到 Gaussian 当前世界坐标。
3. 对每个三角形计算法线。
4. 把 Gaussian 点投影到绑定三角形平面。
5. 返回：
   - `barycentric`: `[num_gaussians, 3]`
   - `normal_offset`: `[num_gaussians]`

关键代码对应：

```python
signed_offset = torch.sum((points - a) * normal, dim=-1)
projected = points - signed_offset.unsqueeze(-1) * normal
normal_offset = torch.sum((points - projected) * normal, dim=-1)
```

以及：

```python
v = (d11 * d20 - d01 * d21) / denom
w = (d00 * d21 - d01 * d20) / denom
u = 1.0 - v - w
barycentric = torch.stack([u, v, w], dim=1)
```

这里没有改变 Gaussian 的可训练参数。它只根据当前参数算出两个诊断量，供训练 loss 使用。

### 3. 在训练 loss 中接入

在 `threestudio/systems/Head3DGSLKs.py` 中新增配置：

```python
surface_constraint_start_step: int = 2400
```

训练时先读取两个权重：

```python
lambda_barycentric_inside = self.C(self.cfg.loss.lambda_barycentric_inside)
lambda_normal_offset = self.C(self.cfg.loss.lambda_normal_offset)
```

只有满足两个条件才计算新约束：

- 当前 step 已达到 `surface_constraint_start_step`
- 至少一个新 loss 权重大于 0

实际 loss 是：

```python
loss_barycentric_inside = F.relu(-barycentric).mean()
loss_normal_offset = torch.abs(normal_offset).mean()
```

`F.relu(-barycentric)` 的含义是：只惩罚负的 barycentric 分量。也就是说，如果投影点已经在三角形内部，就不额外惩罚；只有投影点越过边界时，才把它拉回去。

`torch.abs(normal_offset)` 的含义是：不关心点在三角面正面还是背面，只惩罚离平面太远。

最后加到总 loss：

```python
loss += loss_barycentric_inside * lambda_barycentric_inside
loss += loss_normal_offset * lambda_normal_offset
```

## 配置方式

默认配置保持关闭：

```yaml
surface_constraint_start_step: 2400

loss:
  lambda_barycentric_inside: 0.0
  lambda_normal_offset: 0.0
```

这样旧命令不会改变行为，也不会增加默认训练开销。

实际实验时通过命令行打开，例如 Thor 第一组实验：

```bash
system.loss.lambda_barycentric_inside=0.5 \
system.loss.lambda_normal_offset=0.2
```

这组参数是温和约束：主要防止点云明显跑出绑定三角形和明显离开表面，但不会强行把所有 Gaussian 压死在 FLAME mesh 上。

## 为什么从 2400 step 后开始

`surface_constraint_start_step` 默认设为 2400，是为了和原来的 prune-only 阶段对齐。

训练早期 Gaussian 还在快速建立整体形状、视角覆盖和身份特征。如果一开始就强约束贴面，可能会限制角色外形生成，尤其是头发、头盔、肩颈轮廓这类不完全贴合 FLAME 的结构。

2400 step 后，基础结构已经有一定稳定性，再加入表面约束，更像是训练后期的几何整理，而不是一开始就限制生成空间。

## 相比旧约束的改进

### 更精确地区分漂移方向

旧 `lambda_position` 只看点离三角形中心有多远。一个点可能离中心不算特别远，但已经越过三角形边界；也可能离中心较远，但仍沿着局部表面合理延展。

新约束把问题拆开：

- barycentric 约束负责判断是否跑出三角形边界。
- normal offset 约束负责判断是否离开表面平面。

这样比单一距离阈值更贴近三角面几何本身。

### 只惩罚真正越界的投影

`loss_barycentric_inside` 使用 `F.relu(-barycentric)`。当投影点在三角形内部时，loss 为 0；只有有分量为负时才惩罚。

这比直接把点吸向 centroid 更温和。它允许 Gaussian 在三角形内部自由移动，不会把所有点都往面片中心挤。

### 保留表面外结构的表达空间

我们没有把 Gaussian 参数改成硬性的 barycentric 坐标，也没有强制 `normal_offset=0`。新的 normal loss 是软惩罚，权重可以调。

这意味着头发、头盔、胡子、衣领等结构仍然可以离开 FLAME 表面，只是离得太远会付出 loss。对于文本生成角色，这比硬贴网格更适合。

### 不破坏训练和存储兼容性

新方法只增加训练 loss，不改变底层 Gaussian 存储。

因此不需要改：

- PLY save/load 字段
- resume 逻辑
- densify/split/clone 逻辑
- 已有 checkpoint 的基本兼容方式

这让它适合作为低风险实验开关，而不是一次大范围参数化重构。

### 和旧约束互补

新约束不是完全替代 `lambda_position` 和 `lambda_scaling`。

更合理的理解是：

- `lambda_scaling` 控制 Gaussian 不要局部膨胀过大。
- `lambda_position` 控制 Gaussian 不要整体离绑定区域太远。
- `lambda_barycentric_inside` 控制投影不要跑出绑定三角形边界。
- `lambda_normal_offset` 控制点不要沿法线方向飘离表面太远。

旧约束是粗粒度稳定器，新约束是面片级几何整理器。

## 当前实验结论

Thor 实验使用：

```bash
system.loss.lambda_barycentric_inside=0.5
system.loss.lambda_normal_offset=0.2
```

已经完整跑到 10000 step，并完成 test 渲染。训练结果说明这组软约束可以接入完整训练流程，不会破坏训练闭环。

后续比较重点应放在：

- 是否减少正脸和侧脸的浮点/毛刺。
- 表情驱动时面部 Gaussian 是否更稳定贴合 FLAME 拓扑。
- 头发、头盔、衣领是否被过度压回。
- 身份细节是否被更强约束削弱。

如果需要更强几何控制，可以继续试：

```bash
system.loss.lambda_barycentric_inside=1.0 \
system.loss.lambda_normal_offset=0.5
```

或者：

```bash
system.loss.lambda_barycentric_inside=1.0 \
system.loss.lambda_normal_offset=1.0
```

但如果强约束导致外轮廓变平或角色特征被压掉，应优先保留 `0.5 / 0.2` 这组温和参数。
