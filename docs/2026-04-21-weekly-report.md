# 2026-04-21 周报

## 本周目标

本周的核心目标是把新的真实视频数据源 `TalkVid` 接入 RuiHeadStudio 的 FLAME 参数数据流，并与现有的 `TalkSHOW` 原始参数池、`TalkSHOW` 合成增广参数池放在同一分析框架下，判断新数据是否不仅增加了样本量，而且真实拓宽了训练数据的参数分布覆盖范围。

## 本周完成的主要工作

### 1. 完成 RuiHeadStudio 运行环境与依赖打通

本周重新整理并验证了 RuiHeadStudio 所需的本地运行环境，补齐了训练和渲染相关的核心依赖，确认项目主入口与关键 import 可以正常运行。这一步的作用是保证后续参数转换、分布分析和真实视频 tracking 可以在同一套环境中稳定执行。

### 2. 跑通 TalkSHOW 到 RuiHeadStudio 的参数转换

在已有 TalkSHOW 数据基础上，完成了项目内 `.pkl` 资产向 RuiHeadStudio 兼容 FLAME 参数格式的转换，产出：

- `collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`

转换后已确认包含训练所需的主要字段，包括：

- `expression`
- `jaw_pose`
- `leye_pose`
- `reye_pose`
- `neck_pose`

这一步建立了后续所有比较实验的基准数据池。

### 3. 构建 TalkSHOW 合成增广数据

为了在原始 TalkSHOW 参数池之外补充中间过渡区域，本周新增了合成增广脚本，并生成了 `10` 个增广 `.npy` 文件，输出位于：

- `collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug/`

增广策略主要是对原始参数做扰动混合和适度外扩，重点覆盖表情、下巴和颈部姿态的连续变化。这样做的目的不是简单复制原始样本，而是尝试补齐原始池中较稀疏的过渡区域。

### 4. 扩展多源 FLAME 分布分析工具

本周进一步增强了 `scripts/analyze_pose_distribution.py`，使其支持：

- 同时输入多个 `.npy` 数据源
- 按组打标签，例如 `talkshow`、`synthetic_aug`、`talkvid`
- 使用统一采样与联合嵌入做对比
- 在共享低维空间中做聚类和可视化

这使得后续不再只是看单一数据集内部结构，而是可以系统比较多个数据源之间的重叠、补充和新增覆盖关系。

### 5. 完成 TalkVid 真实视频数据接入

本周完成了 `TalkVid` 数据流的关键接入工作，包括：

- 整理和筛选可用的 `TalkVid` metadata
- 验证视频下载与 clip 截取流程
- 配置并修复外部 `flame-head-tracker`
- 使用 `photometric_fitting=True` 跑通真实视频逐帧 tracking
- 将 tracking 输出转换为 RuiHeadStudio 可用的逐 clip `.npy`

最终本周完成了 `20` 个 `TalkVid` clip 的 FLAME tracking 与转换，输出位于：

- `collection/ruiheadstudio/flame_collections/talkvid/per_clip/`

这一部分是本周最重要的新增数据来源，也是后续判断数据集是否真正被拓宽的关键。

### 6. 统一数据与分析目录结构

本周还对数据与分析产物做了统一整理，当前主要 FLAME 数据源已经集中到：

- `collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`
- `collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug/`
- `collection/ruiheadstudio/flame_collections/talkvid/per_clip/`

分析结果则统一输出到：

- `collection/ruiheadstudio/analysis/pose_distribution/`

这样后续继续补更多真实数据或继续做定量分析时，目录组织会更稳定，也更利于复现实验。

## 本周是怎么做的

整体流程上，本周采用的是“先打通数据链路，再做统一比较分析”的策略，而不是先盲目扩大数据量。

具体来说，先把 `TalkSHOW` 原始参数转换成 RuiHeadStudio 兼容格式，作为基准池；然后基于这个基准池生成一批合成增广数据，用来补中间过渡区域；接着把 `TalkVid` 真实视频从 metadata、下载、裁剪、tracking、参数转换这一整条链路全部跑通，形成第三类真实参数源。最后把这三类数据统一送入同一个 FLAME 分布分析脚本中，在共享嵌入空间下比较它们的覆盖关系和新增区域。

这种做法的好处是，最后得到的结论不是“多加了一批数据”，而是可以更具体地回答“新增的数据到底补在什么地方，是否只是重复采样，还是确实扩展了训练池覆盖范围”。

## 当前结果

本周已完成一次三组正式联合分析，比较对象为：

- `talkshow`
- `synthetic_aug`
- `talkvid`

正式分析输出目录为：

- `collection/ruiheadstudio/analysis/pose_distribution/talkshow_synthetic_aug_vs_talkvid/formal_frame_train_like/`

当前分析规模如下：

- `talkshow`：`630` 帧
- `synthetic_aug`：`6300` 帧
- `talkvid`：`8390` 帧
- 总序列数：`53`
- 总帧数：`15320`

其中，`synthetic_aug` 由 `10` 个增广文件构成，`talkvid` 当前已包含 `20` 个逐 clip 转换结果。

本周产出的正式分析文件包括：

- `summary.json`
- `frame_metadata.csv`
- `frame_umap_embedding.npy`
- `umap_frame_by_group.png`
- `umap_frame_by_source.png`
- `umap_frame_by_cluster.png`

## 是否有证据说明数据集被拓宽了

当前可以说，有初步而且比较明确的证据支持“数据集覆盖范围被拓宽了”，但这个证明目前主要来自联合嵌入可视化和覆盖统计，还不是最终版的严格定量证明。

现有证据主要包括两类。

第一类是分布形态上的证据。在三组数据的联合 UMAP 可视化中，`talkshow`、`synthetic_aug`、`talkvid` 并不是简单重合。当前观察到的结构大致是：

- `talkshow` 更偏外圈，保留了一些较极端区域
- `synthetic_aug` 更多分布在中间环带，起到补中间区域的作用
- `talkvid` 更多集中在中心高密度区域，但并不是已有数据的简单重复，而是形成了相对独立且连续的一块覆盖

第二类是覆盖统计上的证据。根据现有实验进展文档中的总结：

- 相比 `talkshow`，`synthetic_aug` 带来了约 `29.5%` 的自身新增网格覆盖
- 相比 `talkshow`，`talkvid` 带来了约 `75.3%` 的自身新增网格覆盖
- 即使相对 `talkshow + synthetic_aug` 的并集，`talkvid` 仍然保留了约 `48.6%` 的自身新增网格覆盖

这说明 `talkvid` 并不是只在已有分布上做重复采样，而是确实补充了此前 `talkshow` 和 `synthetic_aug` 尚未覆盖到的一部分区域。换句话说，本周的新增工作不仅提升了样本量，也在一定程度上扩大了训练池的有效覆盖范围。

不过，需要说明的是，当前这部分“拓宽”的证据仍主要建立在低维嵌入可视化与网格覆盖统计上。它已经足够支持阶段性结论，但如果要把结论写得更硬，后续最好补充原始特征空间中的定量指标，例如：

- `kNN coverage`
- 最近邻距离分布
- `MMD` 或其他组间差异度量

## 阶段性结论

截至本周，RuiHeadStudio 已经具备了将 `TalkSHOW` 原始参数、`TalkSHOW` 合成增广参数以及 `TalkVid` 真实视频参数放入统一分析框架中的能力。实验结果表明，三类数据并不是简单堆叠关系，而是分别覆盖了外圈极端区域、中间过渡区域和中心真实高频区域。

因此，目前可以给出的阶段性判断是：训练池整体不只是“变多了”，而且“变广了”。尤其是 `TalkVid` 的加入，为原有基于 `TalkSHOW` 的参数池补充了以前覆盖不足的一部分真实分布区域。

## 下周建议

下周建议继续沿着“扩大真实数据覆盖 + 提高证明强度”两个方向推进：

1. 继续补充更多可用 `TalkVid` clip，扩大真实视频来源。
2. 在现有联合嵌入分析之外，补充原始特征空间中的定量覆盖指标。
3. 进一步检查 `TalkVid` 中不同 clip 的质量差异，筛掉异常 tracking 结果。
4. 评估三类数据合并后，作为训练池输入时对最终模型表现是否带来可观察提升。
