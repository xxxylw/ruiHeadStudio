# 2026-04-29 Identity-Aware Two-Stage 改进计划

## 目标

把两阶段训练从：

```text
Stage1: 中性头部先验
Stage2: 在中性头上强行拉身份和外观
```

改成：

```text
Stage1: 带目标身份信息的稳定 FLAME-driven Gaussian head prior
Stage2: 在身份底座上做 face reference 和 temporal refinement
```

这样避免 Stage2 被 anchor / position / temporal loss 限制后，只能用贴图、高频纹理和颜色去“画出”目标人物。

## 改动策略

### 1. Stage1 引入角色 prompt

两阶段脚本默认启用：

```bash
IDENTITY_AWARE_STAGE1_ENABLED=true
```

当用户没有显式传 `STAGE1_PROMPT` 时，Stage1 会直接使用 `STAGE2_PROMPT`：

```bash
STAGE1_PROMPT="${STAGE1_PROMPT:-$STAGE2_PROMPT}"
```

也就是说，C 罗实验里 Stage1 不再是纯中性脸，而是从第一阶段就学习：

- C 罗的大脸型；
- 短发体积；
- 下颌线；
- 颧骨；
- 脖子和黑色训练夹克的大轮廓；
- 在 FLAME 序列下稳定运动的绑定关系。

如果需要旧行为，可以显式设置：

```bash
IDENTITY_AWARE_STAGE1_ENABLED=false
```

### 2. Stage1 / Stage2 降低 sparsity，提高 opaque

为了减少透明、薄片和过强稀疏化，默认配置调整为：

```text
Stage1:
  lambda_sparsity = 0.1
  lambda_opaque = 0.1

Stage2:
  lambda_sparsity = 0.05
  lambda_opaque = 0.2
```

脚本也暴露了环境变量：

```bash
STAGE1_LAMBDA_SPARSITY
STAGE1_LAMBDA_OPAQUE
STAGE2_LAMBDA_SPARSITY
STAGE2_LAMBDA_OPAQUE
```

这样后续可以做 ablation，而不需要改 YAML。

### 3. Stage2 默认保留 face 和 temporal reference，关闭 person reference

当前 `loss_ref_person` 是大 crop 的 RGB mean/std 统计，会把脸、衣服、脖子和背景混在一起，监督太粗。先把它默认关掉：

```text
lambda_ref_person = 0.0
```

保留：

```text
lambda_ref_face = 0.2
lambda_ref_temporal_face = 0.02
```

对应含义：

- `loss_ref_face`：让脸部颜色/纹理统计靠近 reference face crop；
- `loss_ref_temporal_face`：让 batch 内相邻脸部 crop 不要闪烁；
- `loss_ref_person`：暂时不作为默认监督，后续等有更好的 cloth/person feature loss 再打开。

## 第一组建议实验

```bash
RUN_TAG=cristiano_ronaldo_identity_stage1_v1 \
IDENTITY_AWARE_STAGE1_ENABLED=true \
STAGE2_PROMPT="a realistic coherent portrait of Cristiano Ronaldo, short dark hair, athletic face, defined jawline, strong cheekbones, natural skin texture, black athletic training jacket, coherent face neck and upper clothing together, soft studio lighting" \
REFERENCE_FIDELITY_ENABLED=true \
REFERENCE_METADATA=outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json \
REFERENCE_LAMBDA_REF_PERSON=0.0 \
REFERENCE_LAMBDA_REF_FACE=0.2 \
REFERENCE_LAMBDA_REF_TEMPORAL_FACE=0.02 \
STAGE1_LAMBDA_SPARSITY=0.1 \
STAGE1_LAMBDA_OPAQUE=0.1 \
STAGE2_LAMBDA_SPARSITY=0.05 \
STAGE2_LAMBDA_OPAQUE=0.2 \
OPACITY_COVERAGE_ENABLED=true \
LAMBDA_OPACITY_COVERAGE=0.05 \
PRUNE_REGION_GUARD_ENABLED=true \
bash scripts/run_two_stage.sh
```

## 判断指标

这次实验重点看：

- Stage1 结束时是否已经有 C 罗大脸型和短发轮廓；
- Stage2 是否减少高频斑点和贴图感；
- Stage2 opacity mean 是否不再明显低于 Stage1；
- 正脸、侧脸和衣服边界是否更厚实；
- FLAME 序列驱动下脸部是否仍然稳定；
- person reference 关闭后，衣服是否仍能靠 prompt / SDS 保持大轮廓。

## 当前实现状态

已完成：

- `scripts/run_two_stage.sh` 支持 identity-aware Stage1 prompt；
- `scripts/run_stage1_prior.sh` 支持 Stage1 sparsity / opaque override；
- `scripts/run_stage2_text.sh` 支持 Stage2 sparsity / opaque override；
- `configs/headstudio_stage1_prior.yaml` 更新 Stage1 默认 sparsity / opaque；
- `configs/headstudio_stage2_text.yaml` 更新 Stage2 默认 sparsity / opaque 和 reference 权重；
- `tests/test_stage_run_scripts.py` 增加回归测试。

验证命令：

```bash
python -m unittest tests.test_stage_run_scripts tests.test_opacity_alpha_pipeline -v
bash -n scripts/run_two_stage.sh
bash -n scripts/run_stage1_prior.sh
bash -n scripts/run_stage2_text.sh
```
