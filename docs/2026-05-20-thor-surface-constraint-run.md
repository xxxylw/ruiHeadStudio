# 2026-05-20 Thor GS Surface Constraint Run

## 背景

上一轮已经把 Gaussian 绑定到 FLAME 三角面的软表面约束接入训练，但默认权重保持为 0，目的是不改变原始 HeadStudio 训练行为。本次实验把这组约束实际打开，在 Thor prompt 上跑一轮完整 10000 step，用来观察它对点云贴面稳定性、浮点毛刺和表情驱动一致性的影响。

## 本轮代码变化

本轮代码对应提交：

```text
643648d Add soft GS surface constraints
```

主要变化如下：

- 在 `gaussiansplatting/scene/gaussian_flame_model.py` 中新增 `get_bound_triangles()`，统一取每个 Gaussian 当前绑定的 FLAME 三角形顶点。
- 在 `gaussiansplatting/scene/gaussian_flame_model.py` 中新增 `get_surface_constraint_terms()`，计算：
  - Gaussian 投影到绑定三角形平面后的 barycentric 坐标。
  - Gaussian 相对绑定三角形平面的 normal offset。
- 在 `threestudio/systems/Head3DGSLKs.py` 中新增两项 loss：
  - `loss_barycentric_inside`：惩罚小于 0 的 barycentric 分量，把投影跑到三角形外侧的点拉回三角形内。
  - `loss_normal_offset`：惩罚 Gaussian 离绑定三角形平面过远。
- 在 `configs/headstudio.yaml` 中新增配置：

```yaml
surface_constraint_start_step: 2400

loss:
  lambda_barycentric_inside: 0.0
  lambda_normal_offset: 0.0
```

默认权重仍为 0，因此旧训练命令默认行为不变。需要通过命令行显式打开。

## 本次训练配置

使用 Thor 原始 prompt：

```text
a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation
```

本次打开的表面约束权重：

```bash
system.loss.lambda_barycentric_inside=0.5
system.loss.lambda_normal_offset=0.2
```

训练 tag：

```text
headstudio_thor_surface_bary05_norm02_20260520
```

输出目录：

```text
outputs/headstudio/headstudio_thor_surface_bary05_norm02_20260520@20260520-111150
```

日志文件：

```text
outputs/run_logs/thor_surface_bary05_norm02_20260520_tmux.log
```

配置已在 `parsed.yaml` 中确认：

```yaml
surface_constraint_start_step: 2400
lambda_barycentric_inside: 0.5
lambda_normal_offset: 0.2
```

## 启动与环境处理

第一次后台启动使用裸 `python`，因不在项目 conda 环境中失败：

```text
ModuleNotFoundError: No module named 'pytorch_lightning'
```

随后改用项目环境：

```text
/home/rui/miniconda3/envs/ruiheadstudio-bnbfix/bin/python
```

第二次启动进入 CUDA 和 ControlNet 初始化，但因为环境变量 `HF_ENDPOINT=https://hf-mirror.com` 与当前 `huggingface_hub==0.17.3` 行为不兼容，加载 `lllyasviel/control_v11p_sd15_openpose` 时失败。处理方式是：

- 取消 `HF_ENDPOINT`。
- 通过官方 Hugging Face 补齐本地 ControlNet snapshot/cache。
- 重启训练时设置离线缓存读取：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1
```

最终训练在 `thor_surface_20260520_r3` tmux session 中正常跑通。

## 训练结果

训练完成：

```text
Trainer.fit stopped: max_steps=10000 reached.
Test results saved to ./outputs/headstudio/headstudio_thor_surface_bary05_norm02_20260520@20260520-111150/save
```

耗时和速度：

```text
10000 steps
6:26:14
0.43 it/s
```

训练过程中每 100 step 输出 validation 图片，末尾已生成：

```text
it9900-0.png
it9900-1.png
it9900-2.png
it9900-3.png
last.ply
```

训练结束后 test 阶段完成 180 帧渲染，结果保存到同一个 `save` 目录。

## 当前判断

这次实验说明软表面约束可以在不改 PLY 格式、不改 densify/split/clone 逻辑的情况下接入完整训练流程，并且 Thor 10000 step 能正常收敛和完成测试渲染。

从当前观察看，`lambda_barycentric_inside=0.5`、`lambda_normal_offset=0.2` 是一个可继续比较的温和约束组合。下一步应与原始 Thor run 做并排对比，重点看：

- 正脸和侧脸是否减少漂浮 Gaussian、局部毛刺和异常膨胀。
- 头发、头盔、衣领等可能离开 FLAME 表面的结构是否被过度压回。
- 表情驱动时面部区域是否更稳定贴合拓扑。
- 是否牺牲 Thor 身份细节或夸张外形。

## 后续建议

建议保留本轮结果作为第一组 surface-constraint baseline。后续可以继续跑两组消融：

```bash
system.loss.lambda_barycentric_inside=1.0
system.loss.lambda_normal_offset=0.5
```

以及：

```bash
system.loss.lambda_barycentric_inside=1.0
system.loss.lambda_normal_offset=1.0
```

如果更强约束导致头发、头盔或肩颈外轮廓变平，则优先保留本轮较温和参数。
