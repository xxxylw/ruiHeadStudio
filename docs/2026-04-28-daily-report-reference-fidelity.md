# 2026-04-28 日报：文生图 Reference 保真训练闭环

今天的工作把 RuiHeadStudio 的两阶段路线从 `SDS-only` 进一步推进到 `text -> generated reference sheet -> reference-guided Stage2`。核心不是放弃文本驱动，而是用文生图模型先把身份、脸、衣服和质感压成一个固定视觉锚点，再让 3DGS 训练在 SDS 之外受到弱参考监督。

## 今天完成了什么

### 1. 明确了 C 罗实验路线

今天选定 C 罗作为第一组身份保真实验对象，并把目标从单纯的“写实头像”推进为：

- 保持文本驱动高斯头部的主任务不变。
- 允许先用先进文生图模型生成 reference sheet。
- reference sheet 不作为用户上传参考图，而作为文本提示词生成出来的中间监督目标。
- 脸、脖子、衣服一起看，训练内部再区分 face crop 和 person crop 权重。

对应的 reference 资产已经落到：

- `outputs/reference_sheets/cristiano_ronaldo_v1/reference_sheet.png`
- `outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json`

### 2. 写入并落地 reference fidelity 计划

新增了文档：

- `docs/2026-04-28-text-generated-reference-fidelity.md`

这份文档明确了当前路线和原 SDS-only 方法的差异：

- SDS 继续负责文本语义、写实先验和整体可生成性。
- Reference loss 负责把脸、肤色、衣服和照片质感拉向一个固定视觉目标。

这使后续质量优化不再只靠 prompt 和 SDS 权重，而是开始有一个稳定的视觉锚点。

### 3. 接入 Stage2 reference fidelity MVP

今天完成并提交了文生图 reference 监督的第一版工程闭环：

- 增加 reference sheet metadata 读取和校验。
- `configs/headstudio_stage2_text.yaml` 增加 `system.reference_fidelity` 配置。
- `run_stage2_text.sh` 和 `run_two_stage.sh` 支持传入 `REFERENCE_METADATA`。
- Stage2 训练中加入：
  - `train/loss_ref_person`
  - `train/loss_ref_face`
  - `train/loss_ref_temporal_face`
- 增加固定评估入口 `scripts/render_fidelity_eval.sh`。

相关提交：

- `bdfc5d5 接入文生图参考保真训练`
- `fe6c7e2 修复参考保真训练启动问题`
- `8d4f50d 修复参考评估冷启动恢复`

### 4. 完整跑通 C 罗 reference 实验

完整跑通了一次两阶段实验：

- Stage1 输出：`outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage1-prior/`
- Stage2 输出：`outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/`
- Stage2 最终 checkpoint：`outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/ckpts/epoch=0-step=10000.ckpt`
- Stage2 最终视频：`outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/save/it10000-test.mp4`
- 冷启动 eval 视频：`outputs/cristiano_ronaldo_ref_v1_eval@20260428-185102/headstudio-stage2-text/save/it0-test.mp4`

训练过程中确认 reference fidelity loss 正常记录并参与优化。`6000 / 8000 / 10000` step 都顺利产出 checkpoint、PNG、PLY 和 test 视频。

### 5. 修复了冷启动评估链路

今天还暴露并修掉了一个重要评估问题：训练结束后原进程 test 可以成功，但从 `last.ckpt` 冷启动 eval 时，`GaussianFlameModel` 缺少 FLAME face binding 和中性 frame 信息。

修复后：

- `render_fidelity_eval.sh` 使用和 Stage2 训练一致的 clean env / CUDA 环境。
- eval 脚本内补了 `chumpy` 对新版 numpy 的旧别名兼容。
- `GaussianFlameModel.load_ply()` 会重建中性 FLAME frame。
- `Head3DGSLKsRig.post_configure()` 会在冷启动 ckpt 缺少 face bindings 时，自动读取同一次 run 的 `save/last.ply`。

这让训练后的 checkpoint 可以真正独立用于评估，而不是只能依赖训练进程的内存状态。

## 今天把方法带到了什么方向

昨天的判断是：当前主要矛盾已经从“链路能不能跑”转移到“Stage2 如何不过度追 2D 外观，同时保持 3D 稳定”。

今天进一步把方向往前推进了一步：

```text
昨天：SDS-only Stage2 需要更保守，避免 2D diffusion 把高频噪声塞进 3DGS。
今天：在保守 Stage2 之外，引入由文本生成的 reference sheet，让身份和材质有固定视觉锚点。
```

这条路线的意义是：

- 仍然保持文本驱动，不要求用户提供真人参考图。
- 但不再只让 SDS 从随机噪声里“猜一个人”。
- 先用文生图模型把身份、衣服和皮肤质感生成成固定 reference。
- 再用 reference loss 把 3DGS 往同一个视觉目标收束。

换句话说，方法从“文本直接监督 3D”变成了：

```text
text prompt
  -> generated identity reference sheet
  -> SDS + reference fidelity joint supervision
  -> animated Gaussian head sequence
```

## 当前观察

这次 C 罗实验说明链路是通的，但质量还没到最终目标。

好的地方：

- 短发、脸型、下颌、衣领方向已经被拉起来。
- reference loss 能稳定参与训练。
- Stage2 能完整跑到 10000 step。
- 最终 checkpoint 可以冷启动评估。

主要问题：

- 面部和头发出现明显高频斑点。
- 当前身份还只是“朝 C 罗方向靠近”，不是强 identity lock。
- MVP 的 reference loss 还是颜色/纹理统计，不能可靠区分真实皮肤纹理和噪声纹理。

## 下一步建议

下一步不应该继续只调 prompt。当前最值得推进的是把 reference fidelity 从统计约束升级成特征约束：

1. 加 face identity embedding loss，用于真正锁身份。
2. 加 DINO / CLIP / VGG perceptual loss，用于稳定脸部结构、皮肤和衣服材质。
3. 加 face crop temporal feature consistency，用于减少视频序列里的纹理闪烁。
4. 对高频点云增长加入更明确的正则，避免 reference 统计损失把噪声当作“细节”强化。

阶段性结论：今天已经把方法从 SDS-only 的文本驱动头像，推进到文生图 reference-guided 的身份和材质保真路线。接下来质量提升的关键，不是再证明链路能跑，而是把 reference loss 从弱统计锚点升级成真正的身份和感知特征锚点。
