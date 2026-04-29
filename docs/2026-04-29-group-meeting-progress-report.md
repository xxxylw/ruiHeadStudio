# 2026-04-29 组会报告：文本驱动高斯头部的两阶段训练与 Reference 保真路线

汇报时间范围：2026-04-22 至 2026-04-28。

本周主要围绕 RuiHeadStudio 的文本驱动高斯头部训练做了三件事：第一，把训练数据和采样方式从单一 TalkSHOW 推进到多源数据和课程采样；第二，把原来单阶段文本优化拆成 Stage1 头部先验训练和 Stage2 文本外观优化；第三，在 Stage2 里引入文生图 reference sheet，开始尝试解决身份、脸部纹理和衣服质感不稳定的问题。

## 1. 本周目标

之前系统的主要问题是：文本先验很强，但 3D 头部动态先验不够稳。直接用 SDS / NFSD 从零同时学几何、外观和可驱动性，容易出现：

- 侧脸和大姿态不稳定；
- 表情、嘴部、眼部和脖子在连续驱动中抖动；
- 前脸不够厚实，正面能隐约看到后脑勺；
- Stage2 文本优化容易把没有几何支撑的语义变成高频噪声；
- 具体身份和衣服纹理很难只靠 prompt 锁住。

所以本周的目标不是单纯追一个更好看的单帧，而是把训练路线改成更可控的 pipeline：先学稳定可驱动头部，再做文本外观定制，最后尝试用文生图 reference 提供身份和材质锚点。

## 2. 做了什么

### 2.1 多源数据接入

训练入口从单一 `talkshow_train_path` 扩展成三源 FLAME 数据：

- `talkshow`
- `synthetic_aug`
- `talkvid`

对应工作包括：

- 支持多个训练输入路径；
- 支持 `group` 标签；
- 支持不同数据源的加权采样；
- 修复多源 loader 启动和显存问题。

这样做的原因是：TalkSHOW 比较干净，但姿态和动作分布偏窄；TalkVid 真实视频覆盖更多头部姿态、表情幅度和脖子运动；synthetic_aug 用来补中间姿态区域。

### 2.2 课程采样和难度 metadata

只把数据混在一起随机抽样不够，因为模型不知道哪些样本简单、哪些样本困难。因此本周加入了离线 difficulty metadata：

- 新增 `scripts/build_pose_difficulty_metadata.py`
- 按序列统计姿态、表情和相邻帧变化；
- 给序列分成 `easy / medium / hard / rare`；
- dataloader 根据训练 step 按 curriculum schedule 抽不同难度桶。

当前使用的难度指标包括：

- jaw open；
- neck rotation；
- head yaw / pitch / roll；
- expression norm；
- jaw / neck / expression 的相邻帧变化量。

采样流程从原来的随机抽帧，变成：

```text
difficulty bucket -> group -> source -> sequence -> frame
```

这一步的意义是把“更多数据”变成“可调度的数据”。

### 2.3 两阶段训练 pipeline

本周把训练显式拆成两阶段：

```text
TalkSHOW / synthetic_aug / TalkVid
  -> pose difficulty metadata
  -> curriculum sampling
  -> Stage1 head prior
  -> Stage2 text refinement
```

对应配置和脚本：

- `configs/headstudio_stage1_prior.yaml`
- `configs/headstudio_stage2_text.yaml`
- `scripts/run_stage1_prior.sh`
- `scripts/run_stage2_text.sh`
- `scripts/run_two_stage.sh`

Stage1 的目标是先学稳定的可驱动头部先验。它使用更中性的 prompt、更保守的 SDS 强度，并加入 anchor / temporal 约束，重点是头部体积、姿态连续性和基础几何稳定。

Stage2 的目标是在 Stage1 权重基础上做文本外观定制。它使用 `system.weights=<stage1 ckpt>` 继承模型权重，而不是 `resume=`，避免把 Stage1 的 optimizer state 一起恢复导致 densify / prune 后张量尺寸不匹配。

### 2.4 透明和厚实度问题修复

上周后半段实验里，比较明显的问题是人物偏透明：正脸能隐约看到后脑勺，头部整体不够厚实。

针对这个问题，本周做了：

- 降低 `lambda_sparsity`，减少过强稀疏约束；
- 提高 `lambda_opaque`，鼓励更高前景不透明度；
- 把 densify / prune 的 opacity 阈值从硬编码改成配置项；
- 两个阶段都使用更保守的 prune 阈值；
- 调整 Stage2 默认 prompt，减少衣领、毛孔、景深等高频词对几何的干扰。

这部分使透明问题比早期有所缓解，但没有完全解决。当前结果里，后脑勺区域仍然有一点透明、虚、薄的感觉，尤其在转头或正侧过渡时更明显。

### 2.5 文生图 Reference 保真路线

在两阶段训练能跑通后，本周进一步尝试解决身份和纹理不稳定的问题。

当前方法仍然保持文本驱动，不要求用户提供参考图。但我们允许先用文生图模型把文本 prompt 生成成一个固定 reference sheet，再用它监督 Stage2。

本周以 C 罗作为第一组实验：

- 生成 reference sheet；
- 保存 metadata；
- 在 Stage2 中加入 reference fidelity loss；
- 完整跑通两阶段训练和冷启动 eval。

reference 资产路径：

- `outputs/reference_sheets/cristiano_ronaldo_v1/reference_sheet.png`
- `outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json`

训练输出：

- Stage1: `outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage1-prior/`
- Stage2: `outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/`
- 最终 checkpoint: `outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/ckpts/epoch=0-step=10000.ckpt`
- 最终视频: `outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/save/it10000-test.mp4`
- 冷启动 eval: `outputs/cristiano_ronaldo_ref_v1_eval@20260428-185102/headstudio-stage2-text/save/it0-test.mp4`

当前 reference fidelity MVP 加入了：

- `train/loss_ref_person`
- `train/loss_ref_face`
- `train/loss_ref_temporal_face`

这一步把方法从 `SDS-only` 推进成：

```text
text prompt
  -> generated reference sheet
  -> SDS + reference fidelity joint supervision
  -> animated Gaussian head sequence
```

## 3. 怎么做的

工程上，本周主要做了四类改造。

### 3.1 数据侧

把原来单一路径读取改成多源输入，加入 group label 和 group weight。离线构建 metadata 后，训练时可以根据当前 step 从不同 difficulty bucket 里抽样。

### 3.2 训练侧

拆出 Stage1 / Stage2 两套配置。Stage1 更重视几何和动态先验，Stage2 更重视文本外观。两阶段之间通过 `system.weights` 传模型权重，避免恢复 optimizer。

### 3.3 稳定性侧

加入或加强了：

- anchor loss；
- temporal xyz loss；
- opaque loss；
- 更保守 prune；
- Stage2 prompt 约束。

这些改动主要是为了减少透明、漂浮点和高频噪声。

### 3.4 Reference 侧

新增 reference sheet metadata 读取和校验逻辑。训练时读取 face crop 和 person crop，计算颜色/纹理统计，并和当前渲染结果对齐。当前版本是轻量 MVP，目的是先验证 reference-guided 的闭环是否可行。

## 4. 当前效果

### 4.1 已经变好的地方

整体链路已经从“方案设计”进入“完整实验可跑通”：

- 多源数据可以进入训练；
- curriculum sampling 可以工作；
- Stage1 / Stage2 可以自动串起来；
- Stage1 能跑到 4000 step；
- Stage2 能跑到 10000 step；
- 训练结束能保存 checkpoint、PLY、PNG 和 MP4；
- 冷启动 eval 已经可以从 checkpoint 恢复并生成视频。

视觉效果上：

- 头部轮廓比早期更完整；
- 正脸前景不透明度比最早版本好；
- C 罗实验里短发、脸型、下颌、衣领方向已经能被拉起来；
- reference loss 确实能参与训练，并对颜色、脸部区域和衣服区域产生约束。

### 4.2 仍然存在的问题

目前效果还没有达到最终目标，主要问题是：

1. 后脑勺仍然有点透明、虚、薄。
   - 早期是正脸明显能看到后脑勺；
   - 现在有所缓解，但头部背面/侧后方仍不够实；
   - 说明当前 opacity 和 prune 约束还不够，或者几何厚度/覆盖约束还需要更显式。

2. Stage2 仍然容易产生高频斑点。
   - C 罗实验里，脸和头发有明显彩色颗粒；
   - 这说明当前 reference 统计损失会把“纹理细节”和“噪声纹理”混在一起；
   - 只靠颜色均值/方差还不足以约束真实皮肤纹理。

3. 身份保真还不够强。
   - 当前结果是朝 C 罗方向靠近；
   - 但还没有真正锁住 identity；
   - 主要原因是 MVP reference loss 还不是身份特征 loss。

4. 衣服和头部一起训练仍然有压力。
   - 用户目标是脸和衣服一起看；
   - 但当前系统本质仍是文本驱动高斯头部，torso / collar 几何表达能力有限；
   - 如果 prompt 里衣服语义太强，Stage2 容易把衣领和布料纹理变成边界噪声。

## 5. 本周结论

本周的核心进展是把方法从单阶段文本优化推进到更完整的训练体系：

```text
单源随机采样 + 单阶段 SDS
  -> 多源数据 + curriculum sampling
  -> Stage1 头部先验 + Stage2 文本外观
  -> 文生图 reference-guided fidelity
```

这条路线现在已经跑通，说明工程链路成立。但质量上还有两个主要矛盾：

- 头部几何和 opacity 还不够厚实，后脑勺仍有透明感；
- Reference fidelity 目前还只是弱统计约束，能拉颜色和大致方向，但不能真正锁身份和自然纹理。

## 6. 下一步计划

下一步建议分两条线推进。

### 6.1 先解决头部厚实度和后脑勺透明

优先方向：

1. 增加更显式的 alpha / silhouette coverage 约束；
2. 对背面和侧后方视角单独检查 opacity 分布；
3. 统计 prune 后不同区域的高斯数量，确认后脑勺是否被过度剪掉；
4. 增加 head mask 或 region-aware opacity regularization；
5. 必要时单独提高 Stage1 对后脑勺和大 yaw 视角的采样比例。

目标是先让 head-only 的几何和不透明度稳定下来。

### 6.2 再升级 reference fidelity

当前 reference loss 是 MVP，下一步应该从统计约束升级到特征约束：

1. 加 face identity embedding loss，用于真正锁身份；
2. 加 DINO / CLIP / VGG perceptual loss，用于稳定脸部结构、皮肤和衣服材质；
3. 加 face crop temporal feature consistency，用于减少视频序列里的纹理闪烁；
4. 对高频点云增长加正则，避免把噪声当成细节强化。

阶段性目标是：保持文本驱动属性，同时让生成序列在身份、脸部纹理、衣服质感和多视角一致性上更稳定。

## 7. 组会可以重点讲的三句话

1. 本周把训练从单阶段 SDS 推进成了多源数据、课程采样和 Stage1 / Stage2 两阶段 pipeline。
2. 现在完整链路已经能跑通，并进一步接入了文生图 reference sheet，用于尝试身份和材质保真。
3. 当前主要问题不是链路跑不起来，而是结果质量：后脑勺仍有透明虚薄感，reference 统计损失也会带来高频斑点，下一步要加强 alpha/silhouette 约束和身份/感知特征 loss。

## 8. 2026-04-29 追加实验观察与修正计划

今天继续按 C 罗 prompt 跑了一次带 reference fidelity 和 opacity 修复开关的两阶段完整实验：

- 输出目录：`outputs/cristiano_ronaldo_opacity_v120260429-120114/`
- Stage1 输出：`headstudio-stage1-prior/`
- Stage2 输出：`headstudio-stage2-text/`
- Stage2 最终 checkpoint：`headstudio-stage2-text/ckpts/epoch=0-step=10000.ckpt`
- Stage2 最终视频：`headstudio-stage2-text/save/it10000-test.mp4`
- 最终测试序列：`headstudio-stage2-text/save/it10000-test/`，共 180 帧
- 诊断输出：`headstudio-stage2-text/diagnostics/opacity_thickness/`

这次实验确认链路可以稳定跑完，但视觉上仍然有明显透明感。它不是单纯的后脑勺局部问题，而是头部、脸部和衣服区域整体都有偏薄、偏玻璃、偏贴片的感觉。正脸和侧脸都能看到皮肤高光、五官和衣领像低 alpha 叠在黑底上，导致结果虽然更像 C 罗，但实体感不够。

### 8.1 现象定位

为避免只凭主观判断，今天做了几项检查：

- 从最终测试视频抽取多帧 contact sheet，观察不同角度的整体透明感；
- 对比 Stage2 的 `2000 / 4000 / 6000 / 8000 / 10000` 验证图；
- 对比 Stage1 和 Stage2 的同视角结果；
- 对最终 `last.ply` 跑 opacity thickness 诊断；
- 检查最终测试 PNG 是否带 alpha 通道。

结论是：

1. 最终 PNG 是 RGB，不存在导出 alpha 通道导致播放器透明的问题。
2. 透明感来自渲染时累计 opacity 不够，和黑色背景混合后显得发暗、发虚。
3. Stage2 文本外观优化后，整体 opacity 比 Stage1 下降。
4. 当前 test 视频主要覆盖正面到侧正面，还没有充分覆盖真正后脑勺视角，所以这次只能确认“整体偏薄”，不能完全评估“后脑勺是否修好”。

定量结果：

```text
Stage1:
  gaussian count: 95647
  opacity mean: 0.110852
  scaling max mean: 0.228641

Stage2:
  gaussian count: 82999
  opacity mean: 0.088519
  opacity p50: 0.069998
  opacity p90: 0.171022
  scaling max mean: 0.224680
```

这说明 Stage2 在拉身份、脸型、头发和衣服纹理时，同时把点数和整体 opacity 削薄了。当前 `lambda_opacity_coverage=0.02`，训练末尾 `train/loss_opacity_coverage` 仍然在较高区间，说明这个约束虽然接进来了，但强度和作用方式都还不够。

