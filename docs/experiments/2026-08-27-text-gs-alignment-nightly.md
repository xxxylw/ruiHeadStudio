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

## Round 1 Results (2026-08-28)

All three runs completed 10,000 training steps on GPUs 0, 1, and 2. Evaluation used the repository's offline CLIP, PIQE, and MUSIQ implementations on the four final views. The evaluator initially rejected the generated TSV header; `evaluation/src/dataset.py` now explicitly skips the standard header while continuing to validate every data row.

| Run | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `global_clip_warm` | 0.270631 | 0.302638 | 0.293552 | 69.448099 | 55.540452 |
| `foreground_view_clip` | 0.277507 | 0.307239 | 0.298511 | 68.591249 | 54.452843 |
| `text_gs_multicomponent` | 0.271995 | 0.300687 | 0.301576 | 67.922038 | 55.929020 |

Round 1 improves MUSIQ for every variant, with `text_gs_multicomponent` giving the best MUSIQ (+4.569020). It does not yet pass the primary CLIP or PIQE gates. The likely cause is that the CLIP objective is active only during the last 1,500--2,500 steps, after the SDS geometry and appearance have mostly converged. This motivates the continuation/refinement round below rather than treating the first round as a final result.

## Round 2 Plan

Continue from the best Round 1 Gaussian state (`text_gs_multicomponent/runs/text_gs_multicomponent/save/last.ply`) for 3,000 steps with `gaussian_init_step=7000`, so the final checkpoint remains named `it10000`. The alignment loss will be active throughout the continuation, with foreground and view-conditioned components receiving more weight. This tests whether semantic alignment needs a longer, low-amplitude refinement window instead of a late pulse.

## Round 2 Results (2026-08-28)

All three continuation runs completed 3,000 refinement steps from the Round 1 `text_gs_multicomponent` checkpoint and were evaluated on the same four final views.

| Run | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `refine_global` | 0.274353 | 0.303470 | 0.306624 | 64.574665 | 55.785665 |
| `refine_multicomponent` | 0.273504 | 0.312926 | 0.309419 | 65.032274 | 57.048752 |
| `refine_semantic` | 0.268077 | 0.312228 | 0.304467 | 64.750737 | 56.273251 |

Round 2 improves MUSIQ over the baseline for every variant. The proposed `refine_multicomponent` reaches `0.312926` versus `0.313000` on ViT-B/16, but it still misses all three CLIP gates and PIQE is worse by `+5.102274`. The result supports the multi-component semantic hypothesis while showing that the foreground mask and loss amplitude need better control.

During round 2, foreground alignment still used legacy depth-normalized opacity because the rasterized-alpha correction was committed after the processes started. The next round reuses the best round-2 PLY with the committed `alpha_3dgs` foreground mask.

## Round 3 Plan

Run the same 3,000-step continuation launcher from `refine_multicomponent/runs/refine_multicomponent/save/last.ply` with the committed rasterized-alpha foreground correction active. Preserve the exact prompt, four-view evaluation protocol, and three-way ablation in a separate output root.

## Round 3 Results (2026-08-28)

The alpha-corrected `refine_global` and `refine_semantic` runs completed 3,000 steps and were evaluated on the same four views. Both now exceed the two smaller CLIP gates and substantially improve MUSIQ, but neither reaches the ViT-L/14 or PIQE gate.

| Run | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `refine_global` alpha | 0.269375 | 0.313725 | 0.315584 | 62.266065 | 56.774312 |
| `refine_semantic` alpha | 0.270986 | 0.315448 | 0.315254 | 61.285742 | 57.011314 |

The strongest current trade-off is `refine_semantic` alpha: ViT-B/16 improves by `+0.002448`, ViT-B/32 by `+0.002154`, and MUSIQ by `+5.651314`. The multi-component alpha branch first failed on an incorrect batch shape, then exposed a 24GB GPU memory limit when using batch 4. Its reproducible batch-1 retry is running from the same checkpoint with the exact prompt in `configs/headstudio_retry.yaml`.

## Current Status and Next Decision

The proposed contribution is now implemented as differentiable multi-component text-GS alignment with rasterized-alpha foreground crops and view-conditioned prompts. The current evidence is promising but does not yet satisfy the full success gate because ViT-L/14 remains below `0.2784` and PIQE remains above `59.93`. After the batch-1 multi-component result, the next focused experiment should reduce the CLIP loss after a quality checkpoint or add a no-reference quality constraint, rather than increasing CLIP weight blindly.

## Artifacts

- Training logs: `outputs/text_gs_alignment_20260827/<tag>/<tag>.train.log`
- Manifests: `outputs/text_gs_alignment_20260827/<tag>/manifest.tsv`
- Metrics: `outputs/text_gs_alignment_20260827/<tag>/eval/all_metrics`
- Dashboard: `outputs/text_gs_alignment_20260827/dashboard/README.md`
- CSV: `outputs/text_gs_alignment_20260827/dashboard/metrics_comparison.csv`
- SVG: `outputs/text_gs_alignment_20260827/dashboard/metrics_bars.svg`
- Round 2 metrics: `outputs/text_gs_alignment_refine_20260828/<tag>/eval/all_metrics/summary.json`
- Round 3 output root: `outputs/text_gs_alignment_refine_alpha_20260828`
- Round 3 alpha metrics: `outputs/text_gs_alignment_refine_alpha_20260828/<tag>/eval/all_metrics/summary.json`
- Current batch-1 retry: `outputs/text_gs_alignment_refine_alpha_retry5_20260828`
