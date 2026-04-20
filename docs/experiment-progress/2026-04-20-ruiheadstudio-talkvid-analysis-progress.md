# 2026-04-20 RuiHeadStudio 实验进展

## 今日目标

今天的核心目标是把一批真实 `TalkVid` 视频完整接入 RuiHeadStudio 的 FLAME 参数数据流，并与现有 `TalkSHOW` 原始参数池、`TalkSHOW` 扰动增广池一起做正式的分布可视化分析，判断新的真实视频数据是否真正扩展了训练池的覆盖范围。

## 今日完成事项

### 1. 完成 20 个 TalkVid clip 的 FLAME tracking

- 恢复并继续执行原先的 20 条 `TalkVid` manifest。
- 使用 `scripts/run_talkvid_flame_tracking.py` 跑完全部 20 个 clip。
- 输出目录位于：
  - `data/talkvid/flame_tracker/`
- 最终确认 20 个目标 clip 都完成了逐帧 tracking 输出。

### 2. 将 20 个 TalkVid tracking 结果转换为 RuiHeadStudio 可用的 `.npy`

- 将每个 clip 独立转换成一个 `.npy` 文件，而不是只生成一个总文件。
- 输出目录位于：
  - `collection/ruiheadstudio/flame_collections/talkvid/per_clip/`
- 最终得到 20 个逐视频 `.npy` 文件，便于后续逐条检查和多源分析。

### 3. 统一 RuiHeadStudio 的 FLAME collection 目录结构

- 新的数据集合根目录统一整理为：
  - `collection/ruiheadstudio/flame_collections/`
- 当前主要数据源为：
  - `collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`
  - `collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug/`
  - `collection/ruiheadstudio/flame_collections/talkvid/per_clip/`
- 将原来散落在 `talkshow/collection/...` 下的训练参数集合路径，逐步切换到新的目录体系。

### 4. 统一分析输出目录结构

- 将分析输出统一整理到：
  - `collection/ruiheadstudio/analysis/pose_distribution/`
- 历史的 `combined_distribution_test` 也迁移到了新的分析目录体系中。
- 更新了 `scripts/analyze_pose_distribution.py` 的默认输出逻辑，使新分析结果优先落到上述目录下。

### 5. 跑通三组正式分布分析

- 本次正式对比的三组数据为：
  - `talkshow`
  - `synthetic_aug`
  - `talkvid`
- 对应输入为：
  - `collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`
  - `collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug/`
  - `collection/ruiheadstudio/flame_collections/talkvid/per_clip/`
- 正式分析输出目录：
  - `collection/ruiheadstudio/analysis/pose_distribution/talkshow_synthetic_aug_vs_talkvid/formal_frame_train_like/`

### 6. 本次正式分析的主要产物

- 统计汇总：
  - `summary.json`
- 帧级特征与元数据：
  - `frame_features.npy`
  - `frame_metadata.csv`
  - `frame_umap_embedding.npy`
- 可视化结果：
  - `umap_frame_by_group.png`
  - `umap_frame_by_source.png`
  - `umap_frame_by_cluster.png`

## 本次分析规模

- `talkshow`：`630` 帧
- `synthetic_aug`：`6300` 帧
- `talkvid`：`8390` 帧
- 总序列数：`53`
- 总帧数：`15320`
- 为了平衡可视化采样，本次三组在分析时都采样到了 `10000` 个点用于联合嵌入。

## 实验结果解读

从本次 `UMAP by group` 的三组可视化结果看，三类数据在低维嵌入空间里并不是简单重合关系，而是呈现出比较清晰的层次结构。

- `talkshow` 原始池更偏向外圈，覆盖了较多边缘和极端区域。
- `synthetic_aug` 主要分布在中间环带，起到从原始池向中间区域扩张和补洞的作用。
- `talkvid` 更多集中在中心高密度区域，但并不是对原始池或增广池的简单重复，而是在中心区域形成了自己相对稳定且连续的覆盖。

从补充的定量统计看，这个判断也基本成立：

- 相比 `talkshow`，`synthetic_aug` 带来了约 `29.5%` 的自身新增网格覆盖。
- 相比 `talkshow`，`talkvid` 带来了约 `75.3%` 的自身新增网格覆盖。
- 即使相对 `talkshow + synthetic_aug` 的并集，`talkvid` 仍然保留了约 `48.6%` 的自身新增网格覆盖。

这说明 `talkvid` 并不是只在已有分布上做重复采样，而是确实补充了此前未被 `talkshow` 和 `synthetic_aug` 覆盖到的一部分区域。

## 结论

当前可以给出一个比较明确的阶段性结论：

本次把 `TalkVid` 真实视频参数、`TalkSHOW` 原始参数池、以及 `TalkSHOW` 扰动增广池放在一起分析后，可以看到训练池不只是样本数量增加了，更重要的是参数分布的覆盖范围变得更完整了。

- `TalkSHOW` 原始池提供了较多外圈和极端区域。
- `TalkSHOW` 扰动增广池主要扩展了中间过渡带。
- `TalkVid` 则补充了中心高密度、真实驱动更强的一部分区域，并且这部分区域并不完全包含在前两者之内。

因此，从当前这版可视化和覆盖统计来看，把这三类数据合并后，训练池整体上是“更大了”，同时也是“更广了”。更准确地说，不只是总样本数变多，而是从外圈极端区域、中间过渡区域，到中心真实高频区域，这三个层级的覆盖都更完整了。

## 下一步建议

为了把这个结论写得更硬一些，下一步建议在可视化之外，再补一版原始特征空间的定量指标，例如：

- `kNN coverage`
- 到基池的最近邻距离分布
- `MMD` 或其他组间分布差异度量

这样就能把“看起来分布更广”进一步提升为“在原始特征空间里也能量化证明覆盖变大”。
