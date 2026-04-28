# 2026-04-27 两阶段训练问题复盘与后续路径

这篇文档记录当前 `two-stage` 分支走到的位置：之前主要面对什么问题，我们对问题的判断是什么，以及接下来准备沿什么路径继续推进。

## 当前问题分层

之前的问题可以分成两层：

1. 链路问题
2. 质量问题

链路问题现在已经基本解决，当前主要矛盾已经转移到生成质量上。

## 1. 链路层问题

最早的问题不是单个 loss 或 prompt，而是数据和训练链路本身还不完整：

- 训练主要依赖原始 `TalkSHOW`，数据分布偏窄。
- `TalkVid` 真实视频还没有接进 RuiHeadStudio 的 FLAME 参数流。
- loader 偏单一路径采样，不能同时管理 `talkshow / synthetic_aug / talkvid`。
- `stage1 -> stage2` 的训练交接不稳定，早期用 `resume=` 会连 optimizer state 一起恢复，densify / prune 后参数形状对不上。
- 环境链路也有版本兼容问题，`torch / bitsandbytes / tinycudann / pytorch3d / triton` 必须对齐。

对应的解决已经落地：

- 接入 `TalkVid`，完成 tracking 和 `.npy` 转换。
- 增加 `synthetic_aug`，补中间姿态区域。
- 做三源联合分布分析，确认训练池不只是变大，而且覆盖更广。
- 改 multi-input loader，支持 group 权重、source 采样和 metadata。
- 加 difficulty bucket 和 curriculum sampling。
- 拆出 `stage1` / `stage2` 两阶段配置和脚本。
- `stage2` 改用 `system.weights=` 接 `stage1`，只继承模型权重，不继承旧 optimizer state。

所以当前主要问题已经不是“能不能跑起来”。

## 2. 当前质量问题

最新实验暴露的问题更集中在生成质量：

- 早期结果偏透明，正脸能看到后脑勺。
- opacity / alpha 修复后，透明问题明显缓解。
- 但 `stage2` 又出现外观优化过强的问题，例如彩色噪声、高频斑点、鼻子和脖子附近局部不稳定。
- prompt 里如果出现衣领、`turtleneck`、强摄影高频词，例如 `skin pores`、`shallow depth of field`，模型会把这些 2D 语义硬塞到当前 head-only 3DGS 几何上。

当前判断是：

`stage2` 的 2D diffusion 外观驱动力，已经超过了当前 3D 几何可以稳定承载的范围。

也就是说，现在不是系统跑不起来，而是 `stage2` 太容易为了追单帧外观，把没有几何支持的区域长成噪声点或不稳定纹理。

## 3. 设想的解决路径

后续路径分三步。

### 第一步：先让 Stage2 更保守

这一轮已经把 `stage2` 收紧：

- 默认关闭 `use_nfsd`
- `lambda_sds` 降到 `0.6`
- 给 `stage2` 加轻量稳定项：
  - `lambda_anchor: 0.2`
  - `lambda_temporal_xyz: 0.02`
- 默认 prompt 限制在当前几何能表达的范围内：
  - `head and neck only`
  - `no clothing`
  - `no collar`
  - 少用皮肤毛孔、景深、镜头感这类高频诱导词

目标是先验证：在不引入衣服和复杂 torso 语义的情况下，头、脸、鼻子、脖子能不能稳定、干净、多视角一致。

### 第二步：如果还有噪声，加更明确的几何和 Alpha 约束

如果保守 `stage2` 后仍然有明显噪声，就不应该继续只调 prompt 和 loss 权重。

下一步更应该考虑：

- silhouette / alpha coverage 约束
- 更直接的前景 alpha 覆盖目标
- 对脖子下方或边界区域的 densification 做限制
- 对异常高颜色值或高频点云增长做统计和抑制

因为当前噪声的本质是：没有足够几何支持的区域，被 SDS 逼着长出纹理。

### 第三步：Head-only 稳定后，再考虑衣服和身份

当前路线重点是 FLAME head，不应该急着让 prompt 生成衣领、衣服或复杂 torso。

如果后续要做衣服、领子或更强身份一致性，可能需要：

- torso / collar 几何
- semantic mask
- 分区 loss
- 身份参考图约束

但这不是当前最优先。当前更重要的是先把头和脖子的动态稳定、厚实、不透明、多视角一致做好。

## 阶段性结论

当前进展已经从“链路能不能跑”推进到“`stage2` 如何不过度追 2D 外观，同时保持 3D 稳定”的阶段。

近期最合理的方向是：

1. 先复跑保守版 `stage2`，看是否减少彩色噪声和局部闪烁。
2. 如果仍然不够，再引入更显式的 alpha / silhouette / 区域约束。
3. 在 head-only 路线稳定之前，不把衣服和复杂身份语义作为主目标。
