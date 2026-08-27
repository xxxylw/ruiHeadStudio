# 2026-08-27 Text-GS Alignment Nightly

## Goal

Continue RuiHeadStudio from the HeadStudio baseline toward a publishable text-to-3D Gaussian head avatar method. The immediate target is to beat the supplied HeadStudio baseline metrics while adding a method contribution beyond simple hyperparameter tuning.

## Baseline

| Metric | HeadStudio baseline | Direction |
| --- | ---: | --- |
| ViT-L/14 CLIP | 0.2784 | Higher |
| ViT-B/16 CLIP | 0.3130 | Higher |
| ViT-B/32 CLIP | 0.3131 | Higher |
| PIQE | 59.93 | Lower |
| MUSIQ | 51.36 | Higher |

## Method Hypothesis

The HeadStudio baseline uses text diffusion guidance, but the final 3D Gaussian head can still miss semantic identity and view-specific text cues. Directly optimizing one CLIP score is too blunt: it can overfit the full frame, damage perceptual quality, or improve only one view.

This nightly adds a multi-component alignment objective:

`L_text_gs = lambda_clip * (w_global L_global + w_foreground L_foreground + w_view L_view) / sum(w)`

- `L_global`: full-frame render against the original text prompt.
- `L_foreground`: opacity-cropped head render against the original text prompt.
- `L_view`: opacity-cropped render against front/side/back view-conditioned prompts.

The method is backward compatible. If all component weights are zero, RuiHeadStudio keeps the previous single CLIP loss behavior.

## Fixed Data and Prompt

Prompt:

`a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation`

Training data/config:

- Repository: `/home/huangqirui/Projects/ruiHeadStudio`
- Branch: `codex/text-gs-alignment-nightly`
- Conda env: `ruiheadstudio`
- Config: `configs/headstudio.yaml`
- Base guidance: `system.guidance.guidance_scale=25`
- Training length: `trainer.max_steps=10000`
- Main output root: `outputs/text_gs_alignment_20260827`

## Overnight Ablations

| Tag | GPU | Purpose | Key flags |
| --- | ---: | --- | --- |
| `global_clip_warm` | 0 | Test whether late full-frame text alignment improves final CLIP without destabilizing geometry. | `lambda_clip=0.001`, `clip_start_step=8500`, `clip_global_weight=1.0` |
| `foreground_view_clip` | 1 | Focus alignment on the rendered head and view-conditioned text. | `lambda_clip=0.003`, `clip_start_step=7500`, `clip_foreground_weight=0.65`, `clip_view_weight=0.35` |
| `text_gs_multicomponent` | 2 | Proposed method: balance global identity, head foreground, and view-aware alignment. | `lambda_clip=0.0025`, `clip_start_step=7500`, `clip_global_weight=0.45`, `clip_foreground_weight=0.35`, `clip_view_weight=0.20` |

## Evaluation

Each successful run is evaluated with:

```bash
python3 evaluation/run_evaluation.py \
  --batch-root <variant_batch_root> \
  --output-dir <variant_batch_root>/eval/all_metrics \
  --device cpu \
  --metrics all
```

The dashboard is generated with:

```bash
python3 scripts/summarize_alignment_dashboard.py \
  --output-dir outputs/text_gs_alignment_20260827/dashboard \
  outputs/text_gs_alignment_20260827/*/eval/all_metrics/summary.json
```

## Success Gate

Primary success means improving ViT-L/14 CLIP over `0.2784`. A stronger result should also exceed ViT-B/16 `0.3130` and ViT-B/32 `0.3131`, keep PIQE close to or below `59.93`, and keep MUSIQ above `51.36`.

The most paper-useful result is not only the best score, but an ablation pattern where the multi-component objective improves semantic metrics without destroying no-reference perceptual quality.

## Artifacts

- Training logs: `outputs/text_gs_alignment_20260827/<tag>/<tag>.train.log`
- Manifests: `outputs/text_gs_alignment_20260827/<tag>/manifest.tsv`
- Metrics: `outputs/text_gs_alignment_20260827/<tag>/eval/all_metrics`
- Dashboard: `outputs/text_gs_alignment_20260827/dashboard/README.md`
- CSV: `outputs/text_gs_alignment_20260827/dashboard/metrics_comparison.csv`
- SVG: `outputs/text_gs_alignment_20260827/dashboard/metrics_bars.svg`
