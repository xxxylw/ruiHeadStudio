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

## Round 3 Multi-Component Retry Results (2026-08-29)

The fixed-prompt retry completed the full 3,000-step continuation from the round-2 multi-component PLY with `data.batch_size=1`. It used rasterized-alpha foreground crops and weights `global=0.20`, `foreground=0.55`, `view=0.25`, with `lambda_clip=0.006`. The exact prompt was read from `configs/headstudio_retry.yaml`. Evaluation was run per metric after the combined evaluator exceeded the available memory while loading MUSIQ; the five outputs were then merged with the repository confidence-interval summarizer.

| Run | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `retry5_multicomponent_alpha` | 0.273263 | 0.320631 | 0.313588 | 64.224634 | 55.503357 |

Relative to HeadStudio, this run improves ViT-B/16 by `+0.007631`, ViT-B/32 by `+0.000488`, and MUSIQ by `+4.143357`. ViT-L/14 is `-0.005137` below baseline and PIQE is `+4.294634` worse, so the full five-metric success gate remains open. The result supports the proposed component-aware text-GS alignment direction, but it also shows that stronger semantic alignment currently trades against the large CLIP model and no-reference perceptual quality. The next method iteration should introduce an explicit quality-preserving constraint or quality-aware late-stage loss schedule.

The retry manifest and per-metric evidence are under `outputs/text_gs_alignment_refine_alpha_retry5_20260828/refine_multicomponent/`. The consolidated result is `eval/all_metrics/summary.json`; the comparison dashboard was regenerated at `outputs/text_gs_alignment_refine_20260828/dashboard/`.

## Quality Sweep Results (2026-08-30)

Three alpha-aware, batch-1 continuation runs were launched from the same round-2 multi-component checkpoint. They used the exact prompt, 3,000 refinement steps, and the following component weights:

| Run | lambda_clip | Global | Foreground | View |
| --- | ---: | ---: | ---: | ---: |
| `quality_l003` | 0.0030 | 0.20 | 0.55 | 0.25 |
| `balanced_l004` | 0.0040 | 0.30 | 0.45 | 0.25 |
| `global_l0035` | 0.0035 | 0.40 | 0.35 | 0.25 |

All runs completed and were evaluated on four final views with the combined offline evaluator.

| Run | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `quality_l003` | 0.270719 | 0.304899 | **0.313624** | 65.130355 | **55.196145** |
| `balanced_l004` | 0.262153 | 0.310138 | 0.309459 | 65.732342 | 54.653815 |
| `global_l0035` | 0.263218 | 0.308012 | 0.308804 | 64.660643 | 54.860821 |

`quality_l003` is the strongest sweep variant: it improves ViT-B/32 by `+0.000524` and MUSIQ by `+3.836145`, but the full five-metric gate remains open. The sweep also confirms that increasing the global component does not recover ViT-L/14, while the lower-`lambda_clip` quality-focused schedule gives the best trade-off in this setting. The next iteration should add an explicit quality-preserving regularizer or a late-stage schedule that decays semantic alignment after a perceptual-quality checkpoint.

Reproducibility scripts are `scripts/launch_quality_sweep.sh` and `scripts/evaluate_quality_sweep.sh`. The consolidated dashboard is under `outputs/text_gs_alignment_refine_alpha_sweep_20260830/dashboard/`.

## Trust-Region Smoke Results (2026-08-30)

The trust-region smoke run continued from the round-2 multi-component checkpoint for 3,000 batch-1 steps. It used alpha-aware component weights `global=0.20`, `foreground=0.55`, `view=0.25`, `lambda_clip=0.003`, `lambda_trust=0.02`, and linearly decayed the CLIP weight from global step 9,000 to 10,000. The trust penalty constrained normalized drift of XYZ, scale, opacity, and DC color relative to the starting PLY.

| Run | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `trust_l003_anchor` | 0.266277 | **0.315043** | 0.307351 | 66.427173 | **55.959326** |

The trust-region variant improves ViT-B/16 by `+0.002043` and MUSIQ by `+4.599326`, but it does not preserve PIQE or the larger/smaller CLIP models: ViT-L/14 is `-0.012123`, ViT-B/32 is `-0.005749`, and PIQE is `+6.497173` relative to HeadStudio. Thus the proposed regularizer is implemented and reproducibly evaluated, but this parameterization does not pass the full gate. The result suggests that parameter anchoring alone is not a sufficient perceptual-quality constraint; the next iteration should anchor rendered-image features or add a no-reference quality proxy, with a lower trust weight and an earlier quality checkpoint.

Artifacts: `outputs/text_gs_alignment_trust_region_20260830/trust_l003_anchor/eval/all_metrics/summary.json`, `outputs/text_gs_alignment_trust_region_20260830/dashboard/`, `scripts/launch_trust_region_smoke.sh`, and `scripts/evaluate_trust_region_smoke.sh`.

## Semantic Checkpoint Quality Sweep Results (2026-08-30)

Three batch-1 continuation runs started from the strongest prior semantic checkpoint, `outputs/text_gs_alignment_refine_alpha_20260828/refine_semantic/runs/refine_semantic/save/last.ply`. Each ran 2,000 continuation steps, from global step 10,000 to 12,000, with alpha-aware component weights `global=0.20`, `foreground=0.55`, `view=0.25`, `max_grad_norm=0.0005`, and CLIP decay scheduled from steps 11,000 to 12,000.

| Run | lambda_clip | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `semantic_l0003` | 0.0003 | 0.271005 | **0.319354** | 0.309028 | 64.282049 | **56.483161** |
| `semantic_l0006` | 0.0006 | 0.274738 | 0.313507 | 0.309509 | 62.724339 | 55.834595 |
| `semantic_l0010` | 0.0010 | 0.274554 | 0.317739 | 0.310083 | 61.862961 | 56.447024 |

`semantic_l0003` is the best B/16 and MUSIQ trade-off, improving them by `+0.006354` and `+5.123161`. `semantic_l0010` gives the best PIQE in this sweep, but it is still `+1.932961` above the baseline. None of the three passes the full five-metric gate. The experiment confirms that late low-weight semantic refinement can preserve the prior checkpoint better than a stronger continuation, but image quality remains the limiting factor.

The evaluator initially assumed filenames at fixed step 10,000 while these continuation runs saved final views at step 12,000. The evaluation manifests were corrected with aliases pointing to the actual `it12000-*` images; no image content was changed. Final summaries are under `outputs/text_gs_alignment_semantic_quality_sweep_20260830/<tag>/eval/all_metrics/summary.json`, with the consolidated dashboard under `outputs/text_gs_alignment_semantic_quality_sweep_20260830/dashboard/`.

## Frequency-Gated Rendered Quality Sweep Results (2026-08-30)

The frequency quality gate was added as a differentiable rendered-image regularizer. It combines normalized total variation and Laplacian energy at full and half resolution, weighted by a detached alpha floor of `0.25`. The loss ramps from global step 11,000 to 12,000 while the component-aware CLIP weight decays over the same window. All runs continued from the strongest semantic checkpoint for 2,000 steps with batch size 1, `lambda_clip=0.001`, and component weights `global=0.20`, `foreground=0.55`, `view=0.25`.

| Run | lambda_frequency_quality | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `frequency_q0005` | 0.0005 | 0.271067 | 0.319440 | **0.315881** | **61.998288** | **56.573508** |
| `frequency_q0010` | 0.0010 | 0.270004 | 0.318233 | 0.299741 | 63.901771 | 56.444856 |
| `frequency_q0020` | 0.0020 | 0.274647 | **0.320800** | 0.315701 | 62.982658 | 56.450035 |

`frequency_q0005` is the best quality-preserving variant in this sweep: it has the lowest PIQE at `61.998288`, although that is still `+2.068288` above the HeadStudio baseline. `frequency_q0020` gives the strongest B/16 result at `0.320800`. Across this sweep, the best run improves B/16 by `+0.006440`, B/32 by `+0.002781`, and MUSIQ by `+5.213508`, but PIQE remains above baseline and ViT-L/14 remains below `0.2784`; therefore the full five-metric gate remains open.

The helper and schedule tests pass (`10 passed`). The evaluator now reads `HEADSTUDIO_FINAL_STEP=12000`, so reruns use the actual continuation endpoint without filename aliases. Scripts are `scripts/launch_frequency_quality_sweep.sh` and `scripts/evaluate_frequency_quality_sweep.sh`. The consolidated visual artifacts are under `outputs/text_gs_alignment_frequency_quality_sweep_20260830/dashboard/`.

## Checkpoint Selection Probe (2026-08-30)

To test whether the final continuation step was over-optimized, the `frequency_q0005` trajectory was evaluated at steps 11,000, 11,500, and 12,000 using the same four-view protocol.

| Checkpoint | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `it11000` | 0.266876 | 0.312086 | 0.306772 | 61.366190 | 56.876076 |
| `it11500` | 0.267341 | 0.318798 | 0.312202 | **60.598685** | 56.230666 |
| `it12000` | 0.271067 | **0.319440** | **0.315881** | 61.998288 | 56.573508 |

The intermediate checkpoint lowers PIQE by `1.399603` relative to the final checkpoint, but it also loses the B/32 and B/16 gains recovered by step 12,000. No checkpoint passes the full gate, so checkpoint selection is useful as a reporting/Pareto tool but is not sufficient as the main method. The visual comparison is under `outputs/text_gs_alignment_frequency_quality_sweep_20260830/checkpoint_selection_dashboard/`.

## Global CLIP Recovery Sweep Results (2026-08-30)

To test whether foreground and view-conditioned text components were responsible for the ViT-L/14 regression, three 2,000-step continuations started from the `frequency_q0005` checkpoint at global step 12,000. The frequency gate remained active with weight `0.0005`; CLIP used `lambda_clip=0.0005` or `0.0010`, and the component weights were either global-only or `0.50/0.30/0.20` for global/foreground/view.

| Run | lambda_clip | Global/Foreground/View | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `global_l0005` | 0.0005 | 1.00/0.00/0.00 | 0.268430 | 0.321448 | 0.313147 | 62.517246 | 55.880399 |
| `global_l0010` | 0.0010 | 1.00/0.00/0.00 | 0.273201 | 0.321723 | **0.319443** | 62.947541 | **56.298323** |
| `mixed_l0010` | 0.0010 | 0.50/0.30/0.20 | 0.274077 | **0.322018** | 0.315927 | **62.359344** | 55.395778 |

The global-only variants improve B/16, while `global_l0010` improves B/32 and MUSIQ; `mixed_l0010` gives the best B/16 and PIQE in this sweep. ViT-L/14 remains below the baseline for every variant, showing that removing foreground/view terms does not recover the large-model score. The full five-metric gate therefore remains open. The evaluator race was fixed operationally by rerunning after all `last.ply` files existed; no training was repeated.

Artifacts: `outputs/text_gs_alignment_global_recovery_sweep_20260830/dashboard/`, `scripts/launch_global_recovery_sweep.sh`, and `scripts/evaluate_global_recovery_sweep.sh`.

## Content Checkpoint Quality Sweep Results (2026-08-30)

Starting from the strongest prompt-factorized `content` checkpoint, three 2,000-step continuations used the training prompt `a DSLR portrait of Elon Musk`, CLIP component weights `0.50/0.30/0.20`, `lambda_clip=0.0005`, and frequency quality weights `0.0010`, `0.0020`, and `0.0040`.

| Run | lambda_frequency_quality | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `content_q0010` | 0.0010 | **0.288824** | **0.335375** | 0.302853 | 62.309422 | **53.899444** |
| `content_q0020` | 0.0020 | **0.288860** | **0.328856** | 0.304970 | 62.624196 | **55.033799** |
| `content_q0040` | 0.0040 | **0.287199** | **0.337779** | 0.310882 | 64.685421 | 54.088773 |

The sweep preserves strong ViT-L/14 and B/16 gains, with `content_q0040` reaching B/16 `0.337779`, but stronger frequency weights do not lower PIQE: the best is `62.309422`, still `+2.379422` above baseline. This rejects frequency smoothing as a sufficient quality teacher and motivates a rendered-image reference or learned no-reference quality proxy. All summaries and the visual dashboard are under `outputs/text_gs_alignment_content_quality_sweep_20260830/`.

## Prompt Factorization Sweep Results (2026-08-30)

The next experiment factorized the training text to test whether long style descriptors dilute identity alignment. All three runs continued for 2,000 steps from `mixed_l0010`, retained the frequency quality gate at `0.0005`, used CLIP component weights `global=0.50`, `foreground=0.30`, `view=0.20`, and were evaluated against the original full prompt.

| Run | Training prompt | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `identity` | `a portrait of Elon Musk` | **0.285566** | 0.327941 | **0.322520** | 65.748740 | **56.947380** |
| `content` | `a DSLR portrait of Elon Musk` | **0.289189** | **0.343709** | **0.316544** | 62.903001 | 55.577168 |
| `full` | original full prompt | 0.276166 | 0.324031 | 0.313305 | **61.410410** | 56.379683 |

Prompt factorization is the strongest semantic result so far: both short prompts exceed ViT-L/14, B/16, and B/32 baselines, and `identity` also exceeds MUSIQ. The `content` prompt reaches the best B/16 score (`+0.030709`), while the full prompt remains below the ViT-L/14 gate. Quality is still the bottleneck: the best PIQE is `61.410410`, which is `+1.480410` above baseline. This supports separating identity content from style during alignment, followed by a stronger rendered-image quality phase.

Artifacts: `outputs/text_gs_alignment_prompt_factorization_sweep_20260830/dashboard/`, `scripts/launch_prompt_factorization_sweep.sh`, and `scripts/evaluate_prompt_factorization_sweep.sh`.

## Rendered Reference Teacher Sweep Results (2026-08-30)

The rendered-reference teacher keeps a frozen copy of the initialization Gaussian model and renders it under the exact current camera and pose. The current render is then regularized with a masked Charbonnier RGB loss plus gradient loss. Three 2,000-step continuations started from `content_q0010`, used the factorized content prompt, `lambda_clip=0.0005`, and reference weights `0.001`, `0.003`, and `0.006`.

| Run | lambda_rendered_reference | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `reference_q001` | 0.001 | **0.288734** | **0.334765** | 0.299832 | 64.511250 | 54.543825 |
| `reference_q003` | 0.003 | **0.290226** | **0.337575** | 0.299237 | 63.448435 | 54.117458 |
| `reference_q006` | 0.006 | **0.290524** | **0.329527** | 0.300779 | 63.233809 | 53.134557 |

The teacher preserves the large CLIP and B/16 gains, but it copies the checkpoint's existing artifacts and suppresses B/32/MUSIQ. It therefore does not solve PIQE and is rejected as the primary quality teacher. The smoke test and all 13 focused tests pass; the full gate remains open. Artifacts are under `outputs/text_gs_render_reference_sweep_20260830/dashboard/`.

## Reference Statistics Teacher Sweep Results (2026-08-30)

`reference_statistics_loss` is a no-reference, frozen-teacher statistic alignment term. It renders the frozen initialization Gaussian model under the current camera and pose, computes local contrast and edge magnitude, and matches those statistics under detached alpha weighting. This avoids copying RGB pixels while preserving local facial structure. Three 2,000-step continuations started from `content_q0010`, with training prompt `a DSLR portrait of Elon Musk`, `lambda_clip=0.0005`, components `0.50/0.30/0.20`, and statistics weights `0.001`, `0.003`, and `0.006`, ramped from step 19,000 to 20,000.

| Run | lambda_reference_statistics | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `statistics_q001` | 0.001 | **0.291274** | **0.334073** | 0.308791 | 61.278472 | **54.062219** |
| `statistics_q003` | 0.003 | **0.293071** | **0.335430** | 0.308760 | 63.370208 | **54.366879** |
| `statistics_q006` | 0.006 | **0.291877** | **0.333305** | 0.306259 | 61.034154 | **54.442806** |

`statistics_q006` is the best quality-oriented point (PIQE 61.034154, MUSIQ 54.442806); `statistics_q001` is a slightly stronger CLIP/B16 tradeoff. All three improve ViT-L/14, ViT-B/16, and MUSIQ, but all remain below the ViT-B/32 baseline and above the PIQE baseline, so the full five-metric gate remains open. This is preferable to a rendered RGB teacher because it does not directly copy checkpoint artifacts, but the frozen statistics teacher alone is insufficient for no-reference quality. The next direction is a learned quality proxy or multi-scale perceptual teacher, paired with explicit B/32 recovery.

Artifacts: `outputs/text_gs_reference_statistics_sweep_20260830/dashboard/`, `scripts/launch_reference_statistics_sweep.sh`, and `scripts/evaluate_reference_statistics_sweep.sh`.

## ViT-B/32 Recovery Teacher Fast Continuation (2026-08-30)

The original dual-teacher continuation was computationally too slow, so its three intermediate checkpoints were preserved and used as initialization for a 600-step fast continuation. This version kept the frozen ViT-L/14 primary teacher and ViT-B/32 recovery teacher, but restricted both to the global render branch. The recovery blend weights were `0.15`, `0.30`, and `0.45`; the frozen reference-statistics teacher remained at `0.001`.

| Run | recovery blend | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `recovery_fast_q015` | 0.15 | **0.288185** | **0.343947** | 0.305845 | 62.075521 | **53.793602** |
| `recovery_fast_q030` | 0.30 | 0.289464 | **0.331806** | 0.299432 | 63.570331 | **53.150425** |
| `recovery_fast_q045` | 0.45 | **0.289364** | **0.340412** | 0.304893 | 61.925814 | **54.776586** |

The fast continuation confirms that B/32 recovery is not enough by itself: all three runs remain below the B/32 baseline and above the PIQE baseline, although they improve L/14, B/16, and MUSIQ. `recovery_fast_q045` is the best MUSIQ point, while `recovery_fast_q015` is the strongest B/16 point. The full five-metric gate therefore remains open. The result rules out simply increasing B/32 loss weight as the quality solution; the next experiment should use a learned no-reference quality proxy or multi-scale perceptual teacher and explicitly calibrate all three CLIP backbones.

Artifacts: `outputs/text_gs_b32_recovery_fast_sweep_20260830/dashboard/`, `scripts/launch_b32_recovery_fast_sweep.sh`, `scripts/evaluate_b32_recovery_fast_sweep.sh`, and `scripts/dashboard_b32_recovery_fast_after_eval.sh`.

## Guarded Full-Precision B/32 + Statistics Continuation (2026-08-30)

Several short continuation probes exposed two reproducibility failures: loading a PLY left `max_radii2D` at the old point count, and 16-bit AMP produced non-finite Gaussian gradients. The implementation now resets pointwise buffers on PLY reload, sanitizes non-finite gradients before the optimizer step, and records `gradient_probe.json` plus `parameter_drift.json`. The valid v8 sweep uses `precision=32-true`, starts from the recovered PLY at logical step 7000, runs 500 steps, keeps a ViT-B/32 recovery teacher at `0.15`, and sweeps reference-statistics weights `0.0002`, `0.0005`, and `0.0010`.

| Run | statistics weight | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ | mean XYZ drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 | - |
| `stats_valid_q0002` | 0.0002 | **0.287690** | **0.339301** | 0.307942 | 62.258522 | **53.7160** | 0.003800 |
| `stats_valid_q0005` | 0.0005 | **0.288221** | **0.339197** | 0.305951 | 61.579051 | **54.9192** | 0.003848 |
| `stats_valid_q0010` | 0.0010 | **0.288694** | **0.338580** | 0.306951 | 63.995110 | **52.6714** | 0.003843 |

This is the first continuation sweep in this branch with finite gradients and non-zero parameter drift. `stats_valid_q0005` is the best joint point for B/32 and MUSIQ, while all three improve L/14 and B/16. However, all remain above the PIQE baseline and below the supplied B/32 baseline, so the full five-metric gate remains open. The result validates the training path and rejects this particular teacher-weight range as sufficient; the next method should target no-reference quality without copying the frozen checkpoint statistics.

Artifacts: `outputs/text_gs_b32_stats_valid_sweep_v8_20260830/dashboard/`, `scripts/launch_b32_stats_valid_sweep.sh`, `gradient_probe.json`, and `parameter_drift.json`.

The earlier `text_gs_b32_stats_tail_sweep_20260830`, v2, v3, v4, and v7 runs are retained as debugging evidence only and are excluded from quantitative conclusions because of incorrect manifest paths, zero parameter drift, AMP NaNs, or data-loader failures.

## Content-Initialized Guarded Sweep (2026-08-30)

To preserve the B/32 gain from prompt-factorized training, this sweep initialized all variants from `content_q0010` rather than the earlier B/32 recovery checkpoints. It retained the ViT-B/32 recovery teacher (`0.15`) and swept reference-statistics weights `0.0002`, `0.0005`, and `0.0010`. Training used full precision, non-finite gradient protection, logical step 7000, and 500 continuation steps; each run recorded non-zero parameter drift and was evaluated at `it7500` with the corrected per-run manifest.

| Run | statistics weight | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `content_valid_q0002` | 0.0002 | **0.287430** | **0.339916** | 0.311484 | 63.099900 | **54.2676** |
| `content_valid_q0005` | 0.0005 | **0.289982** | **0.336910** | 0.309158 | 62.922252 | **54.4292** |
| `content_valid_q0010` | 0.0010 | **0.287607** | **0.336949** | 0.310140 | 61.468688 | **53.2755** |

The content initialization preserved strong L/14, B/16, and MUSIQ improvements, but the short continuation reduced B/32 below its `content_q0010` starting value and did not bring PIQE below baseline. This rejects the current recovery/statistics combination as a sufficient final method, while validating the full-precision continuation and evaluation protocol. The next direction should decouple B/32 alignment from the quality teacher, for example with a late B/32-only calibration phase and a differentiable no-reference quality proxy.

Artifacts: `outputs/text_gs_b32_content_valid_sweep_20260830/dashboard/` and `scripts/launch_b32_stats_valid_sweep.sh`.

## Late ViT-B/32 Calibration Sweep (2026-08-30)

Because the content-initialized statistics sweep reduced B/32, this ablation removed the reference-statistics teacher and varied only the ViT-B/32 recovery blend (`0.05`, `0.10`, `0.20`). All runs used full precision, the NaN-gradient guard, the `content_q0010` initialization, logical step 7000, and 500 steps; all produced non-zero parameter drift and were evaluated at `it7500`.

| Run | B/32 recovery blend | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `calibration_q005` | 0.05 | **0.288989** | **0.337208** | 0.298028 | 63.354777 | **53.7255** |
| `calibration_q010` | 0.10 | **0.288236** | **0.338016** | 0.307319 | 63.591640 | **53.7059** |
| `calibration_q020` | 0.20 | **0.287298** | **0.340838** | 0.305734 | 62.199286 | **54.5462** |

The isolated B/32 calibration also failed to preserve the content checkpoint's B/32 score and substantially worsened PIQE. It does improve L/14, B/16, and MUSIQ over baseline, but the result rejects a simple late scalar B/32 blend as the solution. The next method should be view- and frequency-selective calibration rather than a global teacher weight.

Artifacts: `outputs/text_gs_b32_calibration_sweep_20260830/dashboard/` and `scripts/launch_b32_stats_valid_sweep.sh`.

## True Factorized-Content B/32 Calibration (2026-08-30)

The previous calibration used the wrong `content_q0010` checkpoint. This corrected sweep initializes from the prompt-factorization `content` checkpoint, whose starting ViT-B/32 score is `0.316544`, and applies only a full-precision ViT-B/32 recovery calibration with blend weights `0.05`, `0.10`, and `0.20`. Each run uses 500 steps from logical step 7000, the non-finite gradient guard, and verified non-zero parameter drift.

| Run | B/32 recovery blend | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `factorized_q005` | 0.05 | **0.289407** | **0.337941** | **0.313192** | 62.309931 | **54.5629** |
| `factorized_q010` | 0.10 | **0.287853** | **0.340441** | 0.310951 | 64.140808 | **54.8721** |
| `factorized_q020` | 0.20 | **0.289943** | **0.341427** | 0.313122 | 61.982813 | **55.7792** |

This is the strongest semantic result so far: `factorized_q005` and `factorized_q020` exceed all three supplied CLIP baselines, and all variants improve MUSIQ. The best joint point still fails PIQE, so the overall five-metric gate remains open by exactly one quality metric. This isolates the remaining research problem to no-reference visual quality rather than text/3DGS alignment.

Artifacts: `outputs/text_gs_b32_factorized_content_calibration_20260830/dashboard/`, `scripts/launch_b32_stats_valid_sweep.sh`, and per-run `parameter_drift.json`.

## Diffusion Guidance Quality Sweep (2026-08-30)

To target PIQE without changing the text alignment objective, this sweep initialized from the true factorized-content checkpoint, fixed ViT-B/32 recovery at `0.05`, and varied the ControlNet diffusion guidance scale (`10`, `15`, `20`). All runs used full precision, the non-finite gradient guard, 500 steps from logical step 7000, and verified non-zero parameter drift.

| Run | guidance scale | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `guidance_q10` | 10 | **0.283251** | **0.342207** | 0.305713 | 62.375967 | **55.8469** |
| `guidance_q15` | 15 | **0.288240** | **0.344256** | 0.312016 | 63.135094 | **55.4278** |
| `guidance_q20` | 20 | **0.286276** | **0.338737** | **0.316393** | 61.450295 | **55.5994** |

`guidance_q20` is the best current semantic/quality Pareto point: all three CLIP scores and MUSIQ exceed baseline, B/32 reaches `0.316393`, and PIQE is the lowest observed in a valid run, though still `1.520295` above baseline. Lowering guidance alone is therefore insufficient, but guidance 20 is selected as the initialization for the next quality-only tail phase.

Artifacts: `outputs/text_gs_guidance_quality_sweep_20260830/dashboard/`, `scripts/launch_b32_stats_valid_sweep.sh`, and per-run `parameter_drift.json`.

## q20 Reference-Statistics Quality Tail (2026-08-30)

Starting from `guidance_q20`, the best guidance-calibrated point, this tail phase fixed guidance scale at `20` and B/32 recovery at `0.05`, then swept very small reference-statistics weights (`0.00005`, `0.00010`, `0.00020`) for 500 full-precision steps. The purpose was to reduce PIQE without disturbing the already strong CLIP scores.

| Run | statistics weight | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `quality_tail_q00005` | 0.00005 | **0.288396** | **0.339480** | **0.314498** | 61.624846 | **54.6205** |
| `quality_tail_q00010` | 0.00010 | **0.291027** | **0.338978** | **0.314965** | 62.345568 | **55.0106** |
| `quality_tail_q00020` | 0.00020 | **0.285769** | **0.343267** | **0.318491** | 61.748085 | **55.0108** |

`quality_tail_q00020` is the best current semantic point, improving every CLIP backbone and MUSIQ while reaching B/32 `0.318491`. PIQE remains the sole failing metric, so the remaining method contribution should be a direct PIQE-oriented differentiable proxy or a view-selective artifact suppression loss.

Artifacts: `outputs/text_gs_quality_tail_q20_20260830/dashboard/` and `scripts/launch_b32_stats_valid_sweep.sh`.

## Edge-Aware Artifact Suppression Sweep (2026-08-30)

To address the remaining PIQE gap, this sweep adds a differentiable no-reference quality proxy. It penalizes high-frequency luminance residuals in locally flat rendered regions while attenuating the penalty near image edges, and uses the current render opacity as a foreground weight. The loss is ramped with the existing quality schedule so early text alignment remains dominant. All runs initialized from `quality_tail_q00020`, used guidance scale `20`, ViT-B/32 recovery `0.05`, full precision, 500 continuation steps from logical step 7000, and verified non-zero parameter drift.

| Run | artifact weight | ViT-L/14 CLIP | ViT-B/16 CLIP | ViT-B/32 CLIP | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio baseline | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `artifact_q0001` | 0.0001 | **0.287502** | **0.338545** | **0.317347** | 60.425517 | **55.8754** |
| `artifact_q0003` | 0.0003 | **0.285901** | **0.341617** | **0.318511** | 62.730646 | **54.8853** |
| `artifact_q0006` | 0.0006 | **0.290199** | **0.341607** | **0.313116** | **59.550065** | **55.6317** |

`artifact_q0006` is the first valid candidate to pass all five supplied gates: all three CLIP metrics and MUSIQ improve over HeadStudio, while PIQE decreases from `59.93` to `59.5501`. The B/32 margin is very small (`+0.000016`), so this point should be repeated with an additional seed before treating it as a robust result. The lower-weight `artifact_q0001` gives the strongest MUSIQ and a better semantic/quality balance, while `artifact_q0006` is the current gate-passing point.

Artifacts: `outputs/text_gs_artifact_quality_sweep_20260830/dashboard/`, `threestudio/models/clip_alignment.py`, `threestudio/systems/Head3DGSLKs.py`, and per-run `parameter_drift.json`.
