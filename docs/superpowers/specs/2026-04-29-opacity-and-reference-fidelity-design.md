# Opacity 厚实度修复与 Reference Fidelity 升级设计

日期：2026-04-29

## 背景

当前 RuiHeadStudio 的文本驱动高斯头部路线已经从单阶段 SDS 推进到：

```text
TalkSHOW / synthetic_aug / TalkVid
  -> pose difficulty metadata
  -> curriculum sampling
  -> Stage1 head prior
  -> Stage2 text refinement
  -> optional generated reference sheet fidelity
```

链路已经可以完整跑通，C 罗 reference 实验也能跑到 Stage2 10000 step，并能冷启动 eval。当前主要问题已经从“能不能跑”转移到“生成质量是否稳定”。

目前最明显的质量问题有两类：

1. 头部厚实度不足：后脑勺仍然有透明、虚、薄的感觉，尤其在侧后方或转头视角更明显。
2. Reference 保真不足：当前 reference fidelity MVP 是颜色/纹理统计 loss，能把结果往目标人脸和衣服方向拉，但会把真实纹理和高频噪声混在一起，身份也没有真正锁住。

这两个问题需要分阶段处理。后脑勺透明是几何和 opacity 基础问题，优先级高于身份和衣服纹理；否则更强的 reference loss 只会把外观监督压到一个仍然不够厚实的头部上。

## 目标

本设计目标是把下一步工作拆成可验证的两条线：

1. 先诊断并修复 head opacity / thickness 问题，让后脑勺和侧后方视角更实。
2. 再把 reference fidelity 从统计约束升级成特征约束，提高身份、脸部纹理、衣服材质和视频一致性。

目标不是一次性解决所有视觉问题，而是建立一套可诊断、可对比、可逐步加 loss 的实验路径。

## 非目标

本轮不做以下事情：

- 不重构整体训练框架。
- 不引入新的 torso / full-body 几何。
- 不把衣服单独建模成独立 mesh。
- 不把用户输入模式改成必须上传真人参考图。
- 不在没有诊断指标的情况下盲目调大 SDS 或 reference loss 权重。

## 总体方案

整体按三个阶段推进：

```text
Phase A: opacity / thickness diagnostics
Phase B: head opacity coverage repair
Phase C: reference fidelity feature upgrade
```

Phase A 先让系统能回答：“后脑勺为什么虚？”  
Phase B 基于诊断结果增加更明确的厚实度约束。  
Phase C 在厚实度稳定后，再升级 reference identity 和 perceptual fidelity。

## Phase A：Opacity 和后脑勺诊断

### 目的

先区分后脑勺透明到底来自哪里：

- Stage1 初始头部先验就不够厚；
- Stage2 文本优化后把后脑勺剪薄；
- prune 阈值过强导致后脑勺高斯数量不足；
- 后侧视角训练采样不足；
- opacity 已经有，但 alpha compositing 结果仍覆盖不足；
- 高斯数量足够，但颜色/尺度/opacity 分布导致视觉上虚。

### 需要新增的诊断输出

新增一个轻量评估脚本或扩展现有 eval，输出固定视角的诊断图和统计表：

1. 固定视角渲染：
   - front
   - left 3/4
   - right 3/4
   - left side
   - right side
   - rear 3/4
   - rear

2. 每个视角保存：
   - RGB
   - opacity map
   - depth map
   - alpha threshold mask

3. 统计当前高斯：
   - 总高斯数量；
   - 平均 opacity；
   - opacity 分位数；
   - scaling 分位数；
   - 按 FLAME face normal 或 face centroid 粗分 front / side / rear 后的高斯数量和 opacity 分布。

4. 对比点：
   - Stage1 checkpoint；
   - Stage2 checkpoint；
   - reference-guided Stage2 checkpoint。

### 产物

输出目录建议为：

```text
outputs/<run>/diagnostics/opacity_thickness/
  views/
    front_rgb.png
    front_opacity.png
    ...
  gaussian_region_stats.json
  summary.md
```

### 验收标准

Phase A 完成后，至少能明确回答：

- 后脑勺区域高斯数量是否明显少于前脸；
- 后脑勺区域 opacity 是否明显低；
- Stage2 相比 Stage1 是否让后脑勺更薄；
- 透明问题是局部区域问题还是整体 opacity 偏低。

## Phase B：Head Opacity Coverage 修复

### 原则

修复时先加 head-only 稳定约束，不急着增强身份和衣服。目标是让几何和 alpha 先稳定。

### 候选约束

按推荐顺序实现。

#### B1. Opacity Coverage Loss

对训练视角的 `out["opacity"]` 增加前景覆盖目标。

第一版不依赖外部 mask，使用当前渲染的有效区域和 FLAME 投影粗 mask。目标不是让整张图全不透明，而是让头部区域的 opacity 不低于阈值。

配置项：

```yaml
system.loss.lambda_opacity_coverage: 0.0
system.opacity_coverage:
  enabled: false
  min_alpha: 0.85
  mode: head_region
```

第一版默认关闭，通过实验命令打开。

#### B2. Rear Region Opacity Regularization

根据每个高斯绑定的 FLAME face，利用 face normal 或 face centroid 把高斯粗分为 front / side / rear。对 rear 区域加入更保守的 opacity 下限或更弱的 prune。

配置项：

```yaml
system.rear_opacity:
  enabled: false
  min_mean_opacity: 0.35
  lambda_rear_opacity: 0.0
```

#### B3. Region-Aware Prune Guard

在 prune 时允许 rear 区域使用更低的 prune opacity threshold，避免后脑勺被过早剪掉。

配置项：

```yaml
system.prune_region_guard:
  enabled: false
  rear_min_opacity_scale: 0.5
```

#### B4. Back / Large-Yaw Sampling Boost

如果诊断发现不是 prune，而是后侧视角训练不足，则在 curriculum 或 camera sampling 上提高 rear / large yaw 视角出现概率。

这一项放在 B1-B3 之后做，因为它会影响训练分布，需要更谨慎对比。

### 验收标准

至少跑一组 ablation：

1. baseline：当前 Stage1 + Stage2；
2. opacity coverage only；
3. opacity coverage + rear prune guard；
4. 如果需要，再加 back-view sampling boost。

成功标准：

- 后脑勺和侧后方 opacity map 更连续；
- 正脸看不到明显后脑勺透影；
- RGB 中后脑勺虚薄感下降；
- 高频斑点没有明显增加；
- 头部轮廓没有被强行糊成一团。

## Phase C：Reference Fidelity 特征升级

### 目的

当前 reference fidelity MVP 用 face/person crop 的颜色和纹理统计。这个约束太弱，不能真正表达身份，也容易把噪声当成纹理。Phase C 的目标是把 reference 从“颜色锚点”升级成“身份和感知特征锚点”。

### C1. Region-Aware Reference Targets

保留当前 reference sheet metadata，但训练内部明确区分：

- face crop：身份、五官、皮肤；
- neck crop：肤色连续性；
- clothing/person crop：衣服颜色、材质、整体风格；
- global crop：整体一致性。

用户目标仍然是脸和衣服一起生成，但 loss 内部必须分区，避免衣服纹理污染脸。

### C2. Perceptual Feature Loss

第一版优先使用较容易接入的感知特征，不直接上很重的身份模型。

候选：

- DINO feature loss：更偏结构和语义一致；
- VGG / LPIPS feature loss：更偏图像质感；
- CLIP image feature loss：更偏全局语义和文本/图像一致。

推荐第一版先做 DINO 或 CLIP image embedding，因为 reference 来自文生图，语义一致性比像素一致性更重要。

配置项：

```yaml
system.reference_fidelity:
  feature_loss:
    enabled: false
    backbone: dino
    lambda_face_feature: 0.0
    lambda_person_feature: 0.0
```

### C3. Identity Embedding Loss

身份 loss 单独作为第二步，不和 C2 同时第一次接入。原因是身份模型依赖和裁脸质量会引入额外不确定性。

配置项：

```yaml
system.reference_fidelity:
  identity_loss:
    enabled: false
    backbone: arcface
    lambda_identity: 0.0
```

### C4. Temporal Feature Consistency

对同一训练 batch 内相邻或相近表情帧的 face crop feature 做一致性约束，减少视频序列里脸部纹理闪烁。

第一版可以只对 face crop 的低维 feature 做 SmoothL1，不做复杂 optical flow。

### C5. 高频噪声约束

为了避免 feature loss 进一步强化斑点，需要同时监控或约束：

- RGB 局部高频能量；
- 高 opacity 小尺度高斯的异常增长；
- face crop 内颜色方差异常升高。

第一版可以先只做统计 logging，不直接加 loss。

### 验收标准

Phase C 完成后，对 C 罗 reference 实验至少比较：

1. 统计 reference loss baseline；
2. + DINO/CLIP feature loss；
3. + identity loss；
4. + temporal feature consistency。

成功标准：

- 身份方向更稳定；
- face crop 高频斑点减少或不增加；
- 衣服颜色和整体风格更稳定；
- 多视角视频里脸部纹理闪烁下降。

## 实验顺序

推荐执行顺序：

1. 对现有 C 罗结果和一个非 reference baseline 跑 Phase A 诊断。
2. 根据诊断选择 B1/B2/B3 中最小必要改动。
3. 跑短程 Stage2 ablation，先看 opacity map 和后脑勺。
4. 跑完整 Stage2，确认厚实度问题是否缓解。
5. 厚实度稳定后，再接 C2 feature loss。
6. C2 稳定后，再接 C3 identity loss。
7. 最后加 temporal feature consistency 和噪声统计。

## 测试策略

### 单元测试

需要覆盖：

- 诊断脚本能找到 checkpoint / ply 并生成输出路径；
- 高斯 region 分类函数能处理空 faces 和正常 faces；
- 新增配置默认关闭，不改变现有训练；
- reference feature loss 配置解析正确；
- eval 冷启动仍能从 `last.ckpt + save/last.ply` 恢复。

### 集成测试

需要至少跑：

- `python -m unittest tests/test_reference_fidelity_config.py tests/test_stage_run_scripts.py -v`
- 新增 diagnostics 相关测试；
- 一个极短 eval smoke test，如果 GPU 环境允许。

### 实验验证

实验验证以固定输出为准：

- 固定 seed；
- 固定 Stage1 ckpt；
- 固定 Stage2 prompt；
- 固定 reference metadata；
- 固定 eval views。

## 风险和缓解

### 风险 1：Opacity coverage 让头变糊

缓解：coverage loss 默认关闭；先短程 ablation；只约束 head region，不约束整张图。

### 风险 2：Rear opacity 破坏正脸

缓解：rear 区域只根据 FLAME face binding 作用于后侧高斯；先 logging，再启用 loss。

### 风险 3：Feature loss 增加显存和训练时间

缓解：先只在固定 crop / 较低分辨率上跑；backbone 默认关闭；支持降低 feature loss 频率。

### 风险 4：Identity loss 依赖裁脸质量

缓解：identity loss 不作为第一步；先把 face crop 和 eval crop 稳定后再接入。

### 风险 5：衣服区域污染脸部

缓解：loss 分区；face crop 权重和 clothing/person crop 权重独立配置。

## 预期结果

短期结果：

- 能明确诊断后脑勺透明原因；
- 能量化 Stage1 / Stage2 的 opacity 变化；
- 后脑勺和侧后方透明感下降。

中期结果：

- reference-guided Stage2 的身份方向更稳定；
- 脸部和衣服纹理不再主要依赖 prompt 猜测；
- 视频序列里的脸部纹理闪烁减少。

最终方向：

```text
text prompt
  -> generated reference sheet
  -> Stage1 stable head prior
  -> Stage2 SDS + opacity coverage + region-aware reference feature supervision
  -> thicker, more identity-preserving animated Gaussian head
```
