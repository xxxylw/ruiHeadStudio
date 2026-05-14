# 周报：RuiHeadStudio 角色训练与约束实验

## 本周工作

1. 完成版本回退和基线整理
   - 将训练流程退回到当前可复现实验的稳定版本，避免继续在不稳定改动上叠加调参。
   - 保留并验证了从已有 Gaussian PLY 初始化训练的能力，支持通过 `system.gaussian_init_ply` 载入已有点云状态，并通过 `system.gaussian_init_step` 对齐续训步数。
   - 默认关闭 eye pose 和 neck pose 对 Gaussian 状态的直接写入，减少 TalkSHOW 姿态噪声对头部生成稳定性的影响。

2. 跑通三组角色实验
   - 在修改前的版本上完成 Thor 和灭霸两组角色实验，用于观察原始流程下的身份保持、头部几何和表情驱动效果。
   - 在完成版本回退、训练数据组织和 GS 位置/尺度约束等修改后，继续跑了美队角色实验，用于对比修改前后的稳定性变化。
   - 训练仍以 HeadStudio 的 3D Gaussian + FLAME rig 流程为主，使用 ControlNet 的 pose/depth 条件约束生成视角。
   - 重点观察角色身份保持、头部几何稳定性、表情驱动后正面与侧面的一致性。

3. 将训练数据组织为视频序列粒度
   - 新增 TalkSHOW 到 RuiHeadStudio 训练格式的转换流程：`scripts/convert_talkshow_to_ruiheadstudio.py`。
   - 转换后的 `.npy` 不再只保存单帧数据，而是以视频片段/序列为单位保存，每条序列包含：
     - `expression`
     - `jaw_pose`
     - `leye_pose`
     - `reye_pose`
     - `neck_pose`
     - `video_name`
     - `clip_name`
     - `source_file`
     - `source_path`
   - 训练时从序列集合中随机采样一个视频片段，再从该片段中随机采样一帧表情/姿态，用于当前 batch 的 FLAME 条件生成。

4. 梳理并保留 GS 位置相关约束
   - 当前配置中保留了 `system.loss.lambda_position` 和 `system.loss.lambda_scaling`，用于约束 Gaussian 的空间位置和尺度。
   - 对比近期 commit，GS 位置损失主体不是这次新加的，而是沿用了已有基线中的约束逻辑；本轮主要是在回退后的稳定版本中保留该约束，并结合角色实验继续观察它对点云稳定性的影响。
   - 位置约束逻辑在 prune-only 阶段后生效：计算每个 Gaussian 点相对其所属 FLAME 三角面变换中心的偏移，如果偏移超过三角面尺度阈值，则加入 Smooth L1 位置损失。
   - 尺度约束会惩罚超过局部三角面尺度的 Gaussian，降低点云异常膨胀、漂移和局部毛刺的概率。
   - 这部分约束的目标是让 GS 更贴合 FLAME 头部拓扑，减少训练后期点云跑飞，同时保留局部表达能力。

## 当前结果

- 训练入口、环境依赖和数据路径已经可以支撑继续做角色实验；修改前已跑 Thor 和灭霸，修改后已跑美队作为对照实验。
- TalkSHOW 数据已经能转成 RuiHeadStudio 需要的序列集合格式，并接入随机表情/姿态采样。
- PLY 初始化能力可以用于从已有结果回退/续训，便于比较不同约束和 prompt 设置的影响。
- GS 位置与尺度约束已经纳入训练损失，可用于控制 Gaussian 点云的稳定性。

## 下周计划

1. 继续对 Thor、灭霸和美队三组角色结果做可视化对比，重点比较修改前后在正脸、侧脸、背面和表情变化下的稳定性。
2. 调整 `lambda_position`、`lambda_scaling` 和 prune-only 阶段参数，观察 GS 约束强度对细节和稳定性的影响。
3. 扩充 TalkSHOW 序列数据，优先加入表情变化更明显、头部姿态更干净的视频片段。
4. 整理角色训练结果，包括关键 checkpoint、PLY 输出、测试视频和失败案例，方便后续复现实验。
