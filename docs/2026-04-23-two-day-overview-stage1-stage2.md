# 这两天的修改总览：多源数据、课程采样、Stage1 / Stage2 两阶段训练

这篇文档的目标不是记录每一行代码改了什么，而是用一篇相对通俗、可回顾的说明，把这两天我们在 `feat/prior-curriculum-pipeline` 这条线里到底做了什么、为什么这么做、现在跑到了哪里，讲清楚。

如果过两天再回来，只看这一篇，应该就能快速回答下面几个问题：

- 这两天到底做了哪些实质性改动？
- 为什么训练会从单阶段，变成 `stage1 + stage2`？
- `stage1` 和 `stage2` 各自负责什么？
- 三个数据集现在是怎么被用起来的？
- 我们中途踩了哪些坑，最后是怎么修好的？
- 现在结果在哪，下一步从哪里接着做？

---

## 1. 先说结论：这两天我们把什么做出来了

如果只用一句话概括，这两天我们把原来“单阶段、单数据入口、随机采样”的训练流程，推进成了下面这条更完整的链路：

`三源 FLAME 数据 -> 离线难度 metadata -> curriculum 采样 -> stage1 先验训练 -> stage2 文本继续优化`

更具体一点，已经落地的内容包括：

1. 训练数据入口从单一 `talkshow_train_path` 扩展成了三源输入：
   - `talkshow`
   - `synthetic_aug`
   - `talkvid`
2. dataloader 支持按 group 做加权采样，不再只是从一个文件里随机抽。
3. 增加了离线难度元数据脚本，能给训练序列打上 `easy / medium / hard / rare` 标签。
4. dataloader 能读取这些难度标签，并按 curriculum schedule 动态采样。
5. 拆出了两份阶段配置：
   - `configs/headstudio_stage1_prior.yaml`
   - `configs/headstudio_stage2_text.yaml`
6. 增加了两个阶段启动脚本：
   - `scripts/run_stage1_prior.sh`
   - `scripts/run_stage2_text.sh`
7. 修通了 `stage2` 训练中遇到的两个关键问题：
   - `bitsandbytes / CUDA` 环境兼容问题
   - `resume checkpoint` 导致的 optimizer state 尺寸不匹配问题
8. 实际跑通了一次完整的两阶段实验：
   - `stage1` 跑到 `4000 step`
   - `stage2` 跑到 `8000 step`

也就是说，现在不是“方案写出来了但还没落地”，而是最小可运行版本已经真正跑完了一轮。

---

## 2. 这两天按时间顺序做了什么

为了不把所有事情混在一起，这里按时间顺序串一下。

### 2.1 第一步：把三类 FLAME 数据组织成统一训练入口

在更早一点的工作里，我们已经完成了：

- `TalkSHOW` 原始参数转换
- `TalkSHOW synthetic_aug` 合成增广数据生成
- `TalkVid` 真实视频 clip 的 FLAME tracking 与 `.npy` 转换

这些内容前面已经记录在：

- [docs/experiment-progress/2026-04-17-ruiheadstudio-talkvid-progress.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/experiment-progress/2026-04-17-ruiheadstudio-talkvid-progress.md:1)
- [docs/experiment-progress/2026-04-20-ruiheadstudio-talkvid-analysis-progress.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/experiment-progress/2026-04-20-ruiheadstudio-talkvid-analysis-progress.md:1)
- [docs/2026-04-21-weekly-report.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/2026-04-21-weekly-report.md:1)

真正进入这两天主题的起点，是把这些分散的数据源正式接到训练入口里。

对应提交是：

- `e91409f` 扩展训练数据入口以支持多源表情参数输入
- `03ec12e` 实现多源表情参数数据加载与加权采样训练
- `4710fc9` 修复多源训练环境兼容性并调整批大小以消除首步显存溢出

这一步做完以后，训练不再只吃一个 `talkshow` 文件，而是可以同时从三类数据里抽样。

### 2.2 第二步：把“更多数据”变成“可调度的数据”

只有三类数据还不够，因为如果只是简单混起来随机抽，模型仍然不知道：

- 哪些序列比较简单
- 哪些序列是大姿态、大表情、难样本
- 哪些片段比较稀有，应该在后期多看一点

所以第二步做的是：先给数据打上难度标签，再让 dataloader 按难度采样。

对应提交是：

- `eb97f4a` 添加姿态难度元数据构建脚本
- `5c15e22` 让训练语料加载器支持姿态难度元数据
- `951bab6` 为训练数据采样增加课程调度能力
- `c1fcce3` 补充头部姿态统计量
- `0c1965f` 补充相邻帧运动幅度统计量
- `b8a1473` 改进姿态难度分桶规则并记录验证结果

这一步的核心变化是：训练时不再只是“随机看数据”，而是能按课程学习节奏逐步增加难度。

### 2.3 第三步：把训练拆成两个阶段

当数据入口和 curriculum 都有了以后，训练逻辑就不再适合继续用一个单一配置解决所有问题了。

于是我们拆出了两个显式阶段：

- `stage1`: 先把稳定的动态头部先验学出来
- `stage2`: 在 stage1 基础上再做文本外观优化

对应提交是：

- `732e1c7` 拆分先验训练与文本微调配置
- `26ab315` 补充两阶段训练启动脚本与交接说明
- `391c4fc` 打通两阶段训练的元数据接线与阶段交接

这一步完成后，整个流程终于不再是“手工改 yaml，反复试命令”，而是：

1. 跑 `stage1`
2. 拿到 `stage1` 的 checkpoint
3. 跑 `stage2`

### 2.4 第四步：把真实训练踩坑修通

方案有了、脚本有了、配置也有了，但真正开始训练时又遇到两类实际问题。

第一类问题是环境兼容：

- `bitsandbytes`
- CUDA 路径
- `diffusers / accelerate`

第二类问题是阶段交接逻辑本身：

- 一开始 `stage2` 通过 `resume=stage1.ckpt` 来接力
- 这会把 `stage1` 的 optimizer state 也一起恢复
- 但 `stage1` 训练过程中点数会 densify / prune，导致 optimizer 里张量尺寸已经不是初始大小
- `stage2` 又按初始点数重建参数，结果第一步 Adam 更新就报尺寸不匹配

这部分后面会详细讲，因为它正好解释了为什么最后 `stage2` 改成 `system.weights=`，而不是继续用 `resume=`

---

## 3. 现在整体训练流程是什么

现在这套训练链路，可以用一句人话版来理解：

先把三类数据整理成统一训练池，再给每条序列打上难度标签，然后用课程学习方式先训练一个更稳定的动态头部先验，最后在这个先验基础上接回文本引导，做目标外观的优化。

如果写成更工程化一点的流程，就是：

1. 准备三类 FLAME 数据
   - `talkshow`
   - `synthetic_aug`
   - `talkvid`
2. 运行离线脚本，生成 pose difficulty metadata
3. `stage1` 用 curriculum 采样训练动态头部先验
4. `stage2` 读取 `stage1` 权重，继续做文本阶段优化
5. 在验证点保存中间图像，在训练结束后保存 `mp4` 和 `ply`

相关核心文件是：

- 配置：
  - [configs/headstudio_stage1_prior.yaml](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/configs/headstudio_stage1_prior.yaml:1)
  - [configs/headstudio_stage2_text.yaml](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/configs/headstudio_stage2_text.yaml:1)
- 启动脚本：
  - [scripts/run_stage1_prior.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage1_prior.sh:1)
  - [scripts/run_stage2_text.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage2_text.sh:1)
- 训练入口：
  - [launch.py](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/launch.py:1)
- dataloader：
  - [threestudio/data/uncond_rand_exp.py](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/threestudio/data/uncond_rand_exp.py:1)
- 元数据脚本：
  - [scripts/build_pose_difficulty_metadata.py](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/build_pose_difficulty_metadata.py:1)

---

## 4. 三个数据集现在是怎么用的

这是最近最容易搞混的一个点，所以单独解释。

### 4.1 现在不是“某个数据集只给 stage1，另一个数据集只给 stage2”

当前实现里，`stage1` 和 `stage2` 都会使用这三类训练输入：

- `talkshow`
- `synthetic_aug`
- `talkvid`

也就是说，两个阶段共享同一个三源训练池。

### 4.2 真正的区别，不在“用哪个数据集”，而在“怎么采样”

两个阶段的主要区别不是数据源切分，而是：

1. 难度分布不同
2. 文本引导强度不同
3. 阶段目标不同

当前配置里，三类数据通过 `train_pose_inputs` 接入，配合：

- `train_pose_group_labels`
- `train_pose_group_weights`
- `pose_metadata_inputs`
- `difficulty_sampling_mode`
- `curriculum_schedule`

来控制采样逻辑。

### 4.3 难度 metadata 是什么

为了让训练知道“哪些序列难、哪些序列简单”，现在先离线生成了一份 JSON：

- [collection/ruiheadstudio/flame_collections/curriculum/train_pose_metadata.json](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/collection/ruiheadstudio/flame_collections/curriculum/train_pose_metadata.json:1)

这份 JSON 不是按帧打标签，而是按**序列级别**打标签。

也就是说：

- 一条 sequence 里可能有很多帧
- metadata 里每一条 entry 代表一段序列
- bucket 是根据整段序列的统计量判断出来的

目前用到的统计量包括：

- `jaw_open_max`
- `neck_rot_max`
- `head_pitch_max`
- `head_yaw_max`
- `head_roll_max`
- `expression_norm_max`
- `jaw_delta_max`
- `neck_delta_max`
- `expression_delta_max`

最后会给每条序列分到一个 bucket：

- `easy`
- `medium`
- `hard`
- `rare`

### 4.4 dataloader 现在怎么抽样

当前训练采样逻辑可以简化理解成：

`difficulty bucket -> group -> source -> sequence -> frame`

意思是：

1. 先根据当前训练 step 和 curriculum schedule，决定这一步更倾向抽哪一档难度
2. 再从该难度里按 group 权重选 `talkshow / synthetic_aug / talkvid`
3. 再选具体 source
4. 再选 sequence
5. 最后从 sequence 里选 frame

所以这三类数据已经不只是“堆在一个目录里”，而是已经被纳入了一个分阶段、分难度的可调度采样流程里。

---

## 5. Stage1 是怎么做的

### 5.1 Stage1 的目的是什么

`stage1` 的任务不是尽快生成一个“像某个人”的头，而是先学出一个更稳定的、可驱动的动态头部表示。

如果说得更直白一点：

`stage1` 更关心“这个头能不能稳定地动起来”，而不是“这个头像不像 Elon Musk”。

它要先把下面这些基础能力学稳：

- 头部几何整体是否稳定
- 侧脸和大姿态会不会崩
- 表情和转头驱动是否连续
- 多视角下是不是还保持合理结构

### 5.2 Stage1 用什么配置

配置文件是：

- [configs/headstudio_stage1_prior.yaml](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/configs/headstudio_stage1_prior.yaml:1)

启动脚本是：

- [scripts/run_stage1_prior.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage1_prior.sh:1)

这个脚本会做两件事：

1. 如果 metadata 不存在，先生成：
   - `collection/ruiheadstudio/flame_collections/curriculum/train_pose_metadata.json`
2. 用 `headstudio_stage1_prior.yaml` 启动训练

### 5.3 Stage1 的训练风格是什么

从设计上说，`stage1` 更偏“先验阶段”，所以它有几个特点：

1. 使用 curriculum sampling
2. 从相对容易的 bucket 开始，逐渐引入更难的姿态和动作
3. 文本引导相对克制
4. 更关注把几何和驱动学稳

在测试里也有一条明确区分：

- [tests/test_stage_configs.py](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/tests/test_stage_configs.py:7)

这里可以看到：

- `stage1.data.difficulty_sampling_mode == "curriculum"`
- `stage1.system.guidance.use_nfsd == False`

也就是说，`stage1` 并不是纯文本阶段，它更像是“先把基础结构和动态规律打牢”。

### 5.4 Stage1 实际跑出来了什么

这次我们实际跑了一轮 `Elon Musk` 提示词的 stage1：

- 输出目录：
  [stage1 输出](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/headstudio-stage1-prior/Elon_Musk,_photorealistic_DSLR_portrait,_realistic_skin_texture,_subtle_beard_shadow,_natural_facial_asymmetry,_cinematic_rim_light,_ultra-detailed_face,_85mm_lens,_shallow_depth_of_field,_studio_background@20260422-190918)

这一轮跑到了 `4000 step`，可以看到：

- `ckpts/epoch=0-step=1000.ckpt`
- `ckpts/epoch=0-step=2000.ckpt`
- `ckpts/epoch=0-step=3000.ckpt`
- `ckpts/epoch=0-step=4000.ckpt`
- `ckpts/last.ckpt`
- `save/it4000-test.mp4`
- `save/last.ply`

这说明 `stage1` 不是 smoke test 级别地“刚启动就停掉”，而是真正完整跑完了一次阶段训练。

---

## 6. Stage2 是怎么做的

### 6.1 Stage2 的目的是什么

`stage2` 不是重新从头训练，而是在 `stage1` 结果的基础上继续做文本驱动外观优化。

如果用一句简单的话来说：

`stage1` 负责把头做稳，`stage2` 负责把头往目标外观上推。

### 6.2 Stage2 用什么配置

配置文件是：

- [configs/headstudio_stage2_text.yaml](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/configs/headstudio_stage2_text.yaml:1)

启动脚本是：

- [scripts/run_stage2_text.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage2_text.sh:1)

这个脚本同样会先检查 metadata 是否存在，不存在就自动生成，然后再进入 `stage2` 训练。

### 6.3 Stage2 和 Stage1 的主要区别是什么

虽然两个阶段都共享三源数据和 curriculum metadata，但 `stage2` 和 `stage1` 在目标和配置上还是有明显区别。

最重要的区别有三个：

1. `stage2` 的任务更偏文本外观定制
2. `stage2` 会启用更强的文本侧引导
3. `stage2` 的 curriculum 起点更靠后，更偏中高难 bucket

在配置测试里，这个差别也被明确写进去了：

- `stage1.system.guidance.use_nfsd == False`
- `stage2.system.guidance.use_nfsd == True`

这意味着：

- `stage1` 更像“结构与驱动基础课”
- `stage2` 更像“外观定制与强化课”

### 6.4 Stage2 最重要的交接逻辑：现在为什么用 `system.weights`

这是这两天最关键的技术细节。

一开始，`stage2` 是想直接用：

- `resume=/path/to/stage1/last.ckpt`

来接 `stage1`。

看起来这很自然，因为 Lightning 的 `resume` 本来就是“继续训练”。

但这里有一个隐藏问题：

- `resume` 不只恢复模型权重
- 它还会恢复 optimizer state

而在我们的系统里，`stage1` 训练过程会 densify / prune 点，因此：

- `stage1` 最后保存的 optimizer state 里，某些参数张量大小已经不是初始值了
- 实际检查 checkpoint 时，能看到像 `99902` 这样的 optimizer 张量尺寸
- 但 `stage2` 启动时还是按 `pts_num=100000` 去初始化模型和 optimizer

结果就是：

- 模型表面上能启动
- 但第一步 Adam 更新就会报 tensor size mismatch

这个问题最后不是通过“改小学习率”或者“重装环境”解决的，而是通过**改变交接方式**解决的：

- 不再用 `resume=...`
- 改成 `system.weights=...`

这代表：

1. 只加载 `stage1` 学到的模型权重
2. 不恢复 `stage1` 的 optimizer state
3. 让 `stage2` 自己新建一个干净的 optimizer

这就是现在 [scripts/run_stage2_text.sh](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/scripts/run_stage2_text.sh:42) 里使用 `system.weights=${STAGE1_CKPT}` 的原因。

这也是这两天里最重要的一次“从根因出发修逻辑”的改动。

### 6.5 Stage2 这次实际跑到了哪里

这次 `Elon Musk` 相关实验的 `stage2` 输出目录是：

- [stage2 输出](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/headstudio-stage2-text/Elon_Musk,_photorealistic_DSLR_portrait,_realistic_skin_texture,_subtle_beard_shadow,_natural_facial_asymmetry,_cinematic_rim_light,_ultra-detailed_face,_85mm_lens,_shallow_depth_of_field,_studio_background@20260423-103413)

这次已经不只是“启动成功”，而是真正跑完了一轮 `8000 step`：

- `ckpts/epoch=0-step=2000.ckpt`
- `ckpts/epoch=0-step=4000.ckpt`
- `ckpts/epoch=0-step=6000.ckpt`
- `ckpts/epoch=0-step=8000.ckpt`
- `ckpts/last.ckpt`
- `save/it2000-*.png`
- `save/it4000-*.png`
- `save/it6000-*.png`
- `save/it8000-*.png`
- `save/it8000-test.mp4`
- `save/last.ply`

`metrics.csv` 也已经写到了 `7999`，说明训练实际上已经跑到了阶段末尾。

所以从实验状态上说：

- `stage1` 已经跑完
- `stage2` 也已经跑完
- 当前不是“还在设计阶段”，而是“两阶段流程已经完成了一次真实闭环”

---

## 7. 这两天到底踩了哪些坑

这一部分很重要，因为它解释了为什么很多现象表面上看起来像“没跑”“没保存”“路径不对”，其实根因不是这些。

### 7.1 坑一：多源训练环境兼容问题

在多源 loader 刚接进训练时，训练一开始就遇到过环境兼容和显存问题。

这部分最终通过以下手段处理掉了：

- 调整 batch size
- 补 NumPy 兼容补丁
- 清理训练环境入口

也就是提交：

- `4710fc9`

这一步的意义是：先让训练入口至少能稳定进入 `Epoch 0`，不在第一步就死掉。

### 7.2 坑二：`bitsandbytes / CUDA` 导致 stage2 起不来

后面真正跑 `stage2` 时，最先遇到的是环境问题，不是训练逻辑问题。

当时的现象是：

- `stage2` 目录会被创建
- `cmd.txt` 和 `parsed.yaml` 也会生成
- 但没有 `metrics.csv`
- 也没有 `ckpt`
- 也没有 `save/`

乍一看像是“脚本没执行”或者“路径错了”，但实际不是。

真正原因是：

- `stage2` 走到了 `diffusers -> accelerate -> bitsandbytes`
- `bitsandbytes` 与当前 CUDA 环境组合不兼容
- 导致它在真正进入训练循环前就退出了

这一步修的不是训练逻辑，而是脚本和环境边界：

- 固定训练解释器
- 固定 `BNB_CUDA_VERSION`
- 用更干净的 `env -i` 环境启动

相关回归测试在：

- [tests/test_stage_run_scripts.py](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/tests/test_stage_run_scripts.py:1)

### 7.3 坑三：`resume checkpoint` 导致 optimizer 尺寸不匹配

这才是后来真正卡住 `stage2` 的根因。

表面错误是：

- 第一轮训练步开始时，Adam 报 tensor size mismatch

但根因不是某个 loss 写错，也不是 prompt 问题，而是：

- `stage2` 用 `resume=stage1.ckpt`
- 连 optimizer state 一起恢复了
- stage1 已经 densify / prune 过
- optimizer 里保存的是变化后的参数尺寸
- stage2 又按初始点数建了一套新的参数结构

最终修复方式是：

- `resume` 改成 `system.weights`

也就是：

- 保留模型权重
- 丢掉旧 optimizer state
- 让 stage2 用自己的新 optimizer 接着训

这个修复对应的计划文档在：

- [docs/superpowers/plans/2026-04-23-stage2-optimizer-handoff-fix.md](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/docs/superpowers/plans/2026-04-23-stage2-optimizer-handoff-fix.md:1)

### 7.4 坑四：为什么有时候你看不到 `save/`

这个问题也反复出现过，所以这里一起说明。

`save/` 目录不是 trial 一创建就一定出现，它通常是在第一次真正触发保存动作时才生成。

保存的触发时机在代码里很明确：

- `validation_step()` 会写 `it{step}-{index}.png`
- `on_test_epoch_end()` 会写 `it{step}-test.mp4`

所以有没有 `save/`，取决于：

1. 训练是不是已经真的进入循环
2. 有没有跑到验证点
3. 有没有跑完整个阶段并进入 test/export

这也是为什么之前有些失败的 `stage2` trial 会有：

- `cmd.txt`
- `configs/`
- `hparams.yaml`

但就是没有 `save/`

因为它们在训练前就崩了，根本没有走到保存逻辑。

---

## 8. 当前实验结果在哪看

如果现在要看这一轮完整的两阶段实验，最重要的目录就是下面两个。

### 8.1 Stage1 结果

- [stage1 输出目录](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/headstudio-stage1-prior/Elon_Musk,_photorealistic_DSLR_portrait,_realistic_skin_texture,_subtle_beard_shadow,_natural_facial_asymmetry,_cinematic_rim_light,_ultra-detailed_face,_85mm_lens,_shallow_depth_of_field,_studio_background@20260422-190918)

重点看：

- `ckpts/last.ckpt`
- `save/it4000-test.mp4`
- `save/last.ply`

### 8.2 Stage2 结果

- [stage2 输出目录](/home/rui/of_work/code/ruiHeadStudio/.worktrees/feat/prior-curriculum-pipeline/outputs/headstudio-stage2-text/Elon_Musk,_photorealistic_DSLR_portrait,_realistic_skin_texture,_subtle_beard_shadow,_natural_facial_asymmetry,_cinematic_rim_light,_ultra-detailed_face,_85mm_lens,_shallow_depth_of_field,_studio_background@20260423-103413)

重点看：

- `ckpts/last.ckpt`
- `save/it8000-test.mp4`
- `save/last.ply`
- `csv_logs/version_0/metrics.csv`

如果只是快速判断训练有没有正常跑：

1. 看 `metrics.csv` 是否持续增长
2. 看 `ckpts/` 是否按验证点生成
3. 看 `save/` 是否出现 `it2000-*`、`it4000-*`、`it6000-*`、`it8000-*`
4. 看最后有没有 `it8000-test.mp4`

---

## 9. 现在的阶段性判断

截至这篇文档写下来的时间点，可以给出一个比较明确的阶段性判断：

1. 三源数据接入已经完成，不再是单数据入口训练。
2. 课程采样不是只停留在文档里，已经接进 dataloader 并真实用于训练。
3. 两阶段训练不是概念验证，而是已经完成了一轮真实运行。
4. `stage2` 最大的工程性阻塞已经修掉，当前交接方式应当以 `system.weights` 为准，而不是 `resume`
5. 现在最值得做的，不再是“证明这条路能不能跑通”，而是开始比较：
   - 不同 step 分配效果如何
   - 不同 curriculum 策略效果如何
   - `stage1 -> stage2` 相比单阶段到底提升了多少

也就是说，我们现在已经从“搭流程”和“排故障”阶段，正式进入了“做实验对比和看效果”阶段。

---

## 10. 下一步建议

如果继续往前推，我建议优先做下面几件事。

### 10.1 先固定一版可复现实验配方

比如固定：

- prompt
- `stage1.max_steps`
- `stage2.max_steps`
- `val_check_interval`
- group weights
- curriculum schedule

这样后面做比较时，不会一边改逻辑一边改实验口径。

### 10.2 做真正的对比实验

最值得比较的不是更多 prompt，而是这几组：

1. 单阶段 baseline
2. 两阶段但不做 curriculum
3. 两阶段 + curriculum

这样才能看清楚：

- 两阶段本身是否有效
- curriculum 是否有额外收益

### 10.3 重新审视 difficulty bucket 的口径

现在 metadata 是按 sequence-level 打标，不是 frame-level。

这很务实，也方便先落地，但它也可能带来一个问题：

- 只要一段长序列里出现过一次大姿态或大表情
- 整段就容易被打成 `hard` 或 `rare`

后面如果发现 bucket 分布长期不理想，可以考虑：

- 更细的 window-level bucket
- 或者重新校准阈值

### 10.4 开始关注“效果”而不只是“是否跑通”

现在最应该做的，不是再证明一次训练能启动，而是去回答：

- `stage1` 是否真的让几何更稳？
- `stage2` 是否真的带来更好的文本外观？
- 两阶段相比单阶段，提升主要出现在正脸、侧脸，还是连续驱动稳定性？

这会决定后面是继续深挖 curriculum，还是把重点转向更强的 identity / appearance 约束。

---

## 11. 一句话版本总结

这两天我们做的事情，本质上是把 RuiHeadStudio 从“单阶段随机训练一个文本驱动头”的工作流，推进成了“用三源数据和课程采样先学稳定动态头部，再接文本优化做外观定制”的两阶段训练流程；而且这条流程现在已经不只是设计，而是已经被真正跑通了一轮。
