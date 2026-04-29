# 2026-04-29 Opacity 与 Reference Fidelity 实现状态

## 已完成

- Phase A CPU 诊断工具：`scripts/diagnose_opacity_thickness.py`
- Gaussian opacity/scaling 统计：`threestudio/utils/opacity_diagnostics.py`
- FLAME face binding 区域分类接口：`GaussianFlameModel.get_gaussian_region_labels`
- 默认关闭的 opacity coverage / rear opacity / prune region guard 配置
- Stage2 与 two-stage 脚本 opacity 修复实验开关
- Reference feature / identity loss 的默认关闭配置表面

## 当前诊断结果

对现有 C 罗 Stage2 输出运行了 PLY-only 诊断：

```bash
conda run -n ruiheadstudio python scripts/diagnose_opacity_thickness.py \
  --ply outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/save/last.ply \
  --output outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/diagnostics/opacity_thickness
```

结果文件：

- `outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/diagnostics/opacity_thickness/gaussian_region_stats.json`
- `outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/diagnostics/opacity_thickness/summary.md`

关键数值：

- Gaussian 数量：78250
- opacity mean：0.096888
- opacity p50：0.077838
- opacity p90：0.184769
- scaling max mean：0.232224

这个结果支持当前判断：透明、虚的问题不只是局部纹理问题，整体 Gaussian opacity 分布偏低。当前 CLI 只读 PLY，不加载 FLAME/渲染状态，所以 front/side/rear 区域标签暂时是 `unknown`。

## 尚未完成

- GPU 固定视角 RGB / opacity / depth 渲染诊断
- PLY 诊断中的 model-bound front/side/rear 区域统计
- DINO / CLIP feature loss 实际 backbone
- ArcFace identity loss
- temporal feature consistency

## 下一步实验

先做一组保守 opacity 修复实验，只打开 coverage 与 rear prune guard：

```bash
OPACITY_COVERAGE_ENABLED=true \
LAMBDA_OPACITY_COVERAGE=0.02 \
PRUNE_REGION_GUARD_ENABLED=true \
bash scripts/run_stage2_text.sh ...
```

如果后脑勺仍然虚，再加入 rear mean opacity 约束：

```bash
REAR_OPACITY_ENABLED=true \
LAMBDA_REAR_OPACITY=0.01 \
bash scripts/run_stage2_text.sh ...
```
