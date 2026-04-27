# 2026-04-24 本周总览：多源数据、课程采样、两阶段训练与当前实验状态

这篇文档的目标是把这周在 `ruiHeadStudio` 里做的事情用一篇相对通俗、可回看的方式讲清楚。重点不是列所有改动明细，而是回答下面几个问题：

- 这周到底做了哪些实质性工作？
- 现在训练流程和上周相比有什么变化？
- `stage1` 和 `stage2` 分别在做什么？
- 三类数据是怎么接进训练里的？
- 中间遇到了哪些问题，最后是怎么修的？
- 到今天为止，实验实际推进到哪里了？

如果后面隔几天再回来，只看这一篇，应该就能较快接上当前状态。

---

## 1. 先说结论：这周我们把什么做出来了

这周最重要的变化，是把原来偏单阶段、偏手工串命令的训练流程，推进成了一条更完整、更稳定的两阶段训练链路：

`TalkSHOW / synthetic_aug / TalkVid -> pose difficulty metadata -> curriculum sampling -> stage1 头部先验 -> stage2 文本外观优化`

已经落地的核心内容包括：

1. 把三类 FLAME 数据统一接进训练入口：
   - `talkshow`
   - `synthetic_aug`
   - `talkvid`
2. dataloader 支持多源加权采样，不再只从单一路径随机抽。
3. 增加了离线难度元数据脚本，可以给序列打 `easy / medium / hard / rare` 标签。
4. 训练时支持 curriculum sampling，能按阶段调整难度分布。
5. 拆出了两套明确的训练配置：
   - `configs/headstudio_stage1_prior.yaml`
   - `configs/headstudio_stage2_text.yaml`
6. 拆出了两个阶段脚本，并新增了一键总控脚本：
   - `scripts/run_stage1_prior.sh`
   - `scripts/run_stage2_text.sh`
   - `scripts/run_two_stage.sh`
7. 修通了两阶段真实训练里最关键的几类问题：
   - 多源 loader 启动与环境兼容问题
   - `bitsandbytes / torch / tinycudann / pytorch3d / triton` 版本链路问题
   - `stage1 -> stage2` checkpoint 交接问题
   - 输出目录命名和两阶段共享 `tag/timestamp` 的路径问题
8. 在今天又进一步针对“人物偏透明、前脸能看到后脑勺”的问题，增加了更偏保真的不透明度约束和更保守的 prune 阈值。

也就是说，到今天为止，我们已经不只是有“方案文档”，而是已经有：

- 能跑的两阶段训练脚本
- 能复现的目录结构
- 跑通过的 stage1 / stage2 真实实验
- 正在继续迭代质量问题的训练配置

---

## 2. 这周按时间顺序做了什么

为了避免把数据、训练、环境、实验混在一起，这里按实际推进顺序串一下。

### 2.1 先把真实数据接进来

本周最早的工作重点，是把 `TalkVid` 真正接进 RuiHeadStudio 的 FLAME 参数链路里。

在这一步里，我们完成了：

- `TalkSHOW` 参数转换到 RuiHeadStudio 可用格式
- `TalkSHOW synthetic_aug` 合成增广数据生成
- `TalkVid` metadata 整理、clip 下载、tracking、逐 clip `.npy` 转换
- 三类数据的联合分布分析

这部分结果已经记录在：

- [docs/experiment-progress/2026-04-17-ruiheadstudio-talkvid-progress.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/experiment-progress/2026-04-17-ruiheadstudio-talkvid-progress.md:1)
- [docs/experiment-progress/2026-04-20-ruiheadstudio-talkvid-analysis-progress.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/experiment-progress/2026-04-20-ruiheadstudio-talkvid-analysis-progress.md:1)
- [docs/2026-04-21-weekly-report.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/2026-04-21-weekly-report.md:1)

这一步解决的是“数据从哪里来”的问题。

### 2.2 再把多源数据真正接进训练

只有数据文件还不够，关键是训练入口要能吃进去。

于是我们做了多源数据入口改造，把训练从单一 `talkshow_train_path` 推成了多输入、多 group、带权重采样的 loader。现在训练池默认包含三类输入：

- `./collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`
- `./collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug`
- `./collection/ruiheadstudio/flame_collections/talkvid/per_clip`

同时引入：

- `train_pose_inputs`
- `train_pose_group_labels`
- `train_pose_group_weights`
- `source_sampling_mode`

这一步完成后，训练不再只是“从单个文件随机拿一帧”，而是能在三类数据之间做统一采样。

### 2.3 把“更多数据”变成“可调度的数据”

接下来做的是 curriculum。原因很简单：如果只是把三类数据堆在一起，模型依然不知道哪些样本简单、哪些样本极端、哪些姿态是稀有边界情况。

所以本周中段的工作重点，是加上离线 difficulty metadata 和 curriculum sampling：

- 新增 `scripts/build_pose_difficulty_metadata.py`
- 用启发式统计量给序列打上：
  - `easy`
  - `medium`
  - `hard`
  - `rare`
- dataloader 读取 metadata，并按 `curriculum_schedule` 控制 bucket 采样概率

也就是说，训练现在已经不是“完全随机”，而是可以显式控制：

`difficulty bucket -> group -> source -> sequence -> frame`

这一步解决的是“怎么更有节奏地喂数据”的问题。

### 2.4 正式拆成 Stage1 / Stage2 两阶段

当三源数据和 curriculum 都接好以后，单一配置已经不够用了。

于是本周把训练显式拆成两个阶段：

- `stage1`：更偏稳定动态头部先验
- `stage2`：在已有头部先验基础上继续做文本外观优化

对应文件是：

- [configs/headstudio_stage1_prior.yaml](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/configs/headstudio_stage1_prior.yaml:1)
- [configs/headstudio_stage2_text.yaml](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/configs/headstudio_stage2_text.yaml:1)

以及：

- [scripts/run_stage1_prior.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage1_prior.sh:1)
- [scripts/run_stage2_text.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage2_text.sh:1)

这一步解决的是“训练目标混在一起不好学”的问题。

### 2.5 修环境、修交接、修目录

真正开始跑两阶段后，这周后半段主要在修真实训练链路里的坑。

踩到的问题主要有：

1. 环境版本链路不一致
   - `torch`
   - `torchvision`
   - `pytorch3d`
   - `tinycudann`
   - `bitsandbytes`
   - `triton`
2. `stage2` 一开始用 `resume=` 继续 `stage1`，结果把旧 optimizer state 一起恢复了，导致张量尺寸不匹配
3. stage 脚本环境隔离不够彻底，`bitsandbytes` 会错误扫到 base conda 的 CUDA runtime
4. 两阶段串行运行时，`timestamp` 和输出根目录拼接容易出错
5. 输出目录顺序不顺手，于是把规则改成了 prompt-first 布局

这部分最后修成了：

- `stage2` 用 `system.weights=` 接 `stage1` 权重，而不是 `resume=`
- 脚本里强制固定训练环境的 `CONDA_PREFIX`
- 补了一键脚本 [scripts/run_two_stage.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_two_stage.sh:1)
- 输出目录改成：
  - `outputs/<prompt_or_tag><timestamp>/headstudio-stage1-prior`
  - `outputs/<prompt_or_tag><timestamp>/headstudio-stage2-text`

这一步解决的是“能不能稳定跑起来”的问题。

### 2.6 今天又针对结果质量继续调了一轮

今天比较明确暴露出来的问题，是生成头部整体偏薄、偏透明：

- 正面视角会透出后脑勺高斯
- 前脸覆盖不够实
- 前景高斯的 opacity 看起来偏低

对应地，今天又追加了这批改动：

- 降低 `lambda_sparsity`
- 提高 `lambda_opaque`
- 把 densify/prune 时的 `min_opacity` 从硬编码改成配置项
- 两个阶段都把 prune 阈值调得更保守

对应最新 commit 是：

- `ac2935e` 增强两阶段头部不透明度约束并补充一键训练脚本

这一步解决的是“结果看起来太透、不够厚实”的问题。

---

## 3. 现在的整体流程是什么

截至今天，推荐的训练流程已经比较明确了：

### 3.1 数据准备

训练使用三类 FLAME 数据：

- `talkshow`
- `synthetic_aug`
- `talkvid`

这三类数据会一起组成统一训练池。

### 3.2 生成难度 metadata

训练前先生成：

- `collection/ruiheadstudio/flame_collections/curriculum/train_pose_metadata.json`

它记录每条序列的难度 bucket。

### 3.3 跑 Stage1

`stage1` 的任务是先学更稳定的动态头部先验。

它的特点是：

- prompt 更中性
- `lambda_sds` 更低
- 有 `anchor loss`
- 有相邻帧 temporal loss
- curriculum 更偏从简单样本开始

### 3.4 跑 Stage2

`stage2` 的任务是：

- 继承 `stage1` 的头部先验
- 接回更强的文本引导
- 把外观往目标人物或目标风格上拉

它的特点是：

- 继续使用三类数据
- curriculum 起点更偏中高难
- `use_nfsd: True`
- 使用 `system.weights=<stage1 ckpt>` 做权重交接

### 3.5 当前推荐入口

现在推荐直接用：

- [scripts/run_two_stage.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_two_stage.sh:1)

它会自动：

1. 共享 `tag` 和 `timestamp`
2. 跑 `stage1`
3. 检查 `stage1` 的 `last.ckpt`
4. 自动接上 `stage2`

---

## 4. Stage1 和 Stage2 现在分别怎么做

### 4.1 Stage1

`stage1` 不是几何-only，但它比 `stage2` 更偏“先把头学稳”。

当前策略大致是：

- 中性 prompt
- 较低 `lambda_sds`
- 开启 `anchor loss`
- 开启相邻帧 `temporal xyz loss`
- 降低 `lambda_sparsity`
- 给一部分 `lambda_opaque`

它的作用不是先把人物做得特别像，而是先学出：

- 更稳定的头部结构
- 更自然的动态驱动
- 更不容易被文本先验拉坏的初始化状态

### 4.2 Stage2

`stage2` 在 `stage1` 基础上继续往目标外观走。

当前策略大致是：

- 用 `system.weights` 接入 `stage1` checkpoint
- prompt 更具体
- 文本外观驱动力更强
- `use_nfsd: True`
- 保留 curriculum
- 进一步降低 `lambda_sparsity`
- 提高 `lambda_opaque`

它的目标是：

- 保留前一阶段已经学到的头部几何和驱动稳定性
- 在此基础上再做更强的外观、身份和质感优化

---

## 5. 这周遇到的关键问题和修法

### 5.1 环境问题

这周最花时间的一类问题，不是训练逻辑，而是训练环境版本链路不一致。

最后稳定下来的组合是：

- `torch 2.0.1+cu118`
- `torchvision 0.15.2+cu118`
- `bitsandbytes 0.38.1`
- `triton 2.0.0`
- `tinycudann` 重新按当前环境编译

这部分的经验是：

- 不能只修某一个包
- 要按整条 ABI 链路去对齐

### 5.2 Stage2 权重交接问题

最初 `stage2` 用 `resume=stage1.ckpt`，结果会恢复：

- 模型参数
- optimizer state

但 `stage1` 里点数已经 densify / prune 过了，optimizer state 的张量形状和 `stage2` 新初始化的参数不一致，所以第一步 Adam 就会炸。

最后修法是：

- `stage2` 改成只用 `system.weights=...`
- 即只继承模型权重，不继承旧 optimizer

### 5.3 两阶段串行脚本问题

本周还踩到了一个比较隐蔽但很典型的问题：

- `stage1` 跑完了
- 但 `stage2` 没接上
- 原因不是训练挂了，而是 checkpoint 路径拼错了

最后补了总控脚本，并把：

- `tag`
- `timestamp`
- `output_root`
- `stage1_ckpt`

都统一进了一个地方，避免手工拼路径出错。

### 5.4 人物偏透明问题

今天暴露出来的质量问题是：

- 前脸高斯太薄
- 能从正面看到后脑勺

当前判断是：

- `lambda_sparsity` 原来太强
- `lambda_opaque` 原来没开
- prune 的 `min_opacity` 也偏激进

所以今天已经把这三刀一起改了。

---

## 6. 当前实验结果到哪里了

到今天为止，可以把实验结果分成三类来看。

### 6.1 这周早些时候已经跑通的一轮两阶段实验

本周早些时候已经成功跑通过一轮：

- `stage1` 跑到 `4000 step`
- `stage2` 跑到 `8000 step`

它证明的是：

- 两阶段链路是能真实跑通的
- `stage1 -> stage2` 的 checkpoint 交接已经成立
- 输出目录结构、save、ckpt、metrics 都能正确落盘

### 6.2 昨天开始的 `silver_haired_scientist_portrait`

这一轮里：

- `stage1` 已经完整跑完
- `stage2` 后来也被重新拉起来，并进入了训练循环

它主要用于验证：

- 新的一键链路
- `stage2` 的手工接力修复

### 6.3 今天的新实验：`charcoal_jacket_portrait`

今天在改完“更厚实不透明”的配置后，又重新起了一轮新的两阶段实验：

- 根目录：
  - [outputs/charcoal_jacket_portrait20260424-105240](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/charcoal_jacket_portrait20260424-105240:1)

当前状态是：

- `stage1` 已经在跑
- 今天抓到的最新 step 已经过了 `278`

这一轮的目标不是只验证链路，而是验证：

- 降低 `lambda_sparsity`
- 提高 `lambda_opaque`
- 降低 prune `min_opacity`

之后能不能让人物更厚实、不再那么透。

---

## 7. 当前怎么跑，怎么找结果

### 7.1 推荐怎么启动

现在推荐的方式是直接在 worktree 里运行：

```bash
cd /home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline
bash scripts/run_two_stage.sh
```

如果要自定义 prompt 或步数，可以通过环境变量覆盖，例如：

```bash
RUN_TAG=my_run \
RUN_TS=$(date +%Y%m%d-%H%M%S) \
STAGE1_PROMPT="a neutral photorealistic human head portrait, realistic skin, natural face, studio lighting" \
STAGE2_PROMPT="a realistic studio portrait of a weathered middle aged man, short dark hair with subtle gray, natural skin tone, defined cheekbones, calm focused expression, clean realistic face, matte skin, soft even studio lighting, plain dark gray background, head and neck only, no clothing, no collar" \
STAGE1_MAX_STEPS=4000 \
STAGE2_MAX_STEPS=10000 \
bash scripts/run_two_stage.sh
```

### 7.2 结果去哪看

每次实验会落到：

`outputs/<tag><timestamp>/`

里面一般会有：

- `headstudio-stage1-prior/`
- `headstudio-stage2-text/`

重点看：

- `csv_logs/version_0/metrics.csv`
- `ckpts/last.ckpt`
- `save/`

### 7.3 当前工作位置

到写这篇文档的时候，当前开发工作不是在主仓库 `main`，而是在 worktree 里：

- worktree 路径：
  [/.worktrees/feat/prior-curriculum-pipeline](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline:1)
- 当前分支：
  `two-stage`

这意味着最近这批两阶段训练、课程采样和质量修复改动，都是在 `two-stage` 这条线上继续推进的。

---

## 8. 这周的阶段性判断

如果只给一个阶段性判断，我会这样总结：

这周最大的进展，不是“多跑了一次实验”，而是把整个系统从“数据接进来但训练链路还很脆弱”，推进成了“多源数据、课程采样、两阶段训练、目录约定和一键脚本都已经建立起来”的状态。

到今天为止，这套系统已经具备：

- 三源数据统一训练
- 难度元数据和课程采样
- `stage1 -> stage2` 的可重复训练流程
- 对结果质量问题做更有针对性的配置迭代

换句话说，现在最主要的问题已经不再是“能不能跑起来”，而是：

- 怎么让头部更厚实、更保真
- 怎么进一步稳定动态驱动
- 怎么继续提高侧脸、边缘和遮挡质量

这其实是一个好信号，因为说明链路层面的阻塞已经基本被清掉了，接下来可以更集中地优化质量。

---

## 9. 下周建议

下周建议集中在三件事上：

1. 继续跟踪今天这轮“更不透明、更厚实”的实验结果
   - 看它是否真正减少正面透视后脑勺的问题
2. 如果这轮有效，把 opacity / sparsity / prune 阈值进一步整理成更系统的配置组
   - 而不是继续靠一次次手工试值
3. 如果“偏透明”问题还没有明显缓解，下一步应当考虑增加更明确的 silhouette / alpha coverage 约束
   - 而不只是继续调现有的 sparsity / opaque 比例

简化地说，下周的重点会从“把两阶段训练做出来”，转成“把两阶段训练调得更稳、更像、更厚实”。

---

## 10. 追加：`opacity_fix_weathered_architect` 实验复盘和 Stage2 收敛策略

今天后面又基于 opacity / alpha 修复拉起了一轮完整两阶段实验：

- 输出目录：
  [outputs/opacity_fix_weathered_architect20260424-143620](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/opacity_fix_weathered_architect20260424-143620:1)
- `stage2` 最终结果：
  [headstudio-stage2-text/save](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/opacity_fix_weathered_architect20260424-143620/headstudio-stage2-text/save:1)

这轮实际使用的 `stage2` prompt 是：

```text
a photorealistic DSLR portrait of a weathered architect in his forties, short dark hair with subtle gray at the temples, solid natural skin texture, defined cheekbones, calm focused expression, matte cotton charcoal turtleneck collar, realistic skin pores, soft studio key light, clean gray backdrop, 85mm lens, shallow depth of field
```

结果说明两件事：

1. opacity / alpha 修复方向是有效的，人物整体不再像之前那样明显半透明。
2. `stage2` 外观优化过强，导致保真度下降。衣领和脖子区域出现类似电视噪声的彩色高斯，鼻子附近也有局部闪烁。

点云统计也支持这个判断：

- `stage1` 最终约 `96k` 点，颜色 DC 最大值约 `2.97`
- `stage2` 最终约 `297k` 点，颜色 DC 最大值约 `29.21`

也就是说，这轮的主要问题已经不是单纯的 opacity，而是 `stage2` 在没有足够几何支持和稳定约束的区域里，把 2D diffusion 的高频纹理强行写进了 3DGS。

具体根因：

- 当前 FLAME / head 3DGS 主要支持头和脖子，没有真实衣服或领子几何；`turtleneck collar` 这类词会把衣服语义压到脖子和下方稀疏高斯上。
- `realistic skin pores`、`85mm lens`、`shallow depth of field` 会鼓励强摄影高频细节，在 SDS 训练里容易变成彩色斑点。
- 旧 `stage2` 默认是 `lambda_sds=1.0`、`use_nfsd=True`，同时没有 `lambda_anchor` / `lambda_temporal_xyz`，比 `stage1` 更容易追逐单帧外观而牺牲多视角和动态一致性。

因此后续默认策略改成更保守的 `stage2`：

- 默认关闭 `use_nfsd`
- `lambda_sds` 从 `1.0` 降到 `0.6`
- 给 `stage2` 加回轻量稳定项：
  - `lambda_anchor: 0.2`
  - `lambda_temporal_xyz: 0.02`
- 默认 prompt 限制在当前几何能表达的范围内：

```text
a realistic studio portrait of a weathered middle aged man, short dark hair with subtle gray, natural skin tone, defined cheekbones, calm focused expression, clean realistic face, matte skin, soft even studio lighting, plain dark gray background, head and neck only, no clothing, no collar
```

这不是说后面永远不能做衣服，而是当前这条 FLAME-head 路线应该先把头和脖子稳定做保真。等头部、鼻子、脖子在动态测试里稳定后，再考虑引入 torso / collar 几何或语义 mask，而不是只靠 prompt 让高斯自己长出衣服。
