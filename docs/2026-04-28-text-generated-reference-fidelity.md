# 2026-04-28 文生图 Reference 保真训练路线

这条路线仍然保持 RuiHeadStudio 的文本驱动属性。区别是：先用文生图模型把文本压成一组稳定的整体人物 reference sheet，再让 Stage2 在 SDS 之外受到这组图的弱监督。

## 和当前 SDS-only 的区别

当前 Stage2 主要依赖 SDS。它能把结果推向“像一张写实人物照片”的分布，但不容易锁住一个具体身份，也容易把衣服、毛孔、镜头词变成不稳定的 3DGS 高频噪声。

Reference fidelity 多了一层固定视觉目标：

```text
SDS: 维持文本语义、写实先验和整体可生成性
Reference loss: 把脸、肤色、衣服和整体照片感拉向同一个稳定目标
```

## C 罗实验 reference sheet

当前生成并保存了一张三视角 reference sheet：

- `outputs/reference_sheets/cristiano_ronaldo_v1/reference_sheet.png`
- `outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json`

metadata 使用 `identity_mode: target_person`，包含 front、left 3/4、right 3/4 三个 crop。用户层面的目标仍然是一个完整人物：脸、脖子和黑色训练夹克放在一起看。训练内部会给 face crop 更高权重，避免衣服纹理污染脸部身份。

## 当前实现范围

这次先实现 MVP：

- 本地 reference sheet metadata 读取和校验。
- Stage2 配置 `system.reference_fidelity`。
- `run_stage2_text.sh` 和 `run_two_stage.sh` 支持传入 `REFERENCE_METADATA`。
- Stage2 启用后读取 reference crop 的 face/person 颜色和纹理统计。
- 训练中记录并施加：
  - `train/loss_ref_person`
  - `train/loss_ref_face`
  - `train/loss_ref_temporal_face`
- 固定评估入口 `scripts/render_fidelity_eval.sh`。

## 运行方式

校验 reference sheet：

```bash
python scripts/validate_reference_sheet.py outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json
```

启动两阶段训练：

```bash
RUN_TAG=cristiano_ronaldo_ref_v1 \
REFERENCE_FIDELITY_ENABLED=true \
REFERENCE_METADATA=outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json \
STAGE2_PROMPT="a realistic coherent portrait of Cristiano Ronaldo, short dark hair, athletic face, defined jawline, natural skin texture, black athletic training jacket, face neck and clothing together, soft studio lighting" \
bash scripts/run_two_stage.sh
```

后续如果只接一个已有 Stage1 checkpoint，也可以直接跑 Stage2：

```bash
STAGE1_CKPT="$(find outputs -path '*/headstudio-stage1-prior/ckpts/last.ckpt' | sort | tail -n 1)" \
REFERENCE_FIDELITY_ENABLED=true \
REFERENCE_METADATA=outputs/reference_sheets/cristiano_ronaldo_v1/metadata.json \
bash scripts/run_stage2_text.sh \
  "system.prompt_processor.prompt=a realistic coherent portrait of Cristiano Ronaldo, short dark hair, athletic face, defined jawline, natural skin texture, black athletic training jacket, face neck and clothing together, soft studio lighting"
```

## 后续升级

MVP 的 reference loss 是轻量统计约束，只用于先打通链路和观察是否能减少明显漂移。下一步应该把 face/person 统计约束替换或补充为更强的特征约束：

- face identity embedding loss，用于真正锁身份。
- DINO/CLIP/VGG perceptual loss，用于稳定皮肤、鼻子、嘴唇和衣服质感。
- face crop temporal feature consistency，用于减少视频序列里的闪烁。

这三项上去以后，reference sheet 才会从“颜色和纹理锚点”升级成“身份和材质锚点”。
