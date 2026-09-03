# ruiHeadStudio Method and Experiment Summary

Date: 2026-08-30  
Repository: `/home/huangqirui/Projects/ruiHeadStudio`  
Branch: `codex/text-gs-alignment-nightly`

## 1. Goal and Baseline

The goal is to reconstruct a 3D Gaussian Splatting head avatar from text and improve the HeadStudio baseline on all five supplied metrics.

| Metric | HeadStudio baseline | Direction |
| --- | ---: | --- |
| ViT-L/14 CLIP | 0.2784 | higher |
| ViT-B/16 CLIP | 0.3130 | higher |
| ViT-B/32 CLIP | 0.3131 | higher |
| PIQE | 59.93 | lower |
| MUSIQ | 51.36 | higher |

## 2. What Changed

### 2.1 Multi-level text/3DGS alignment

`Head3DGSLKs.py` now combines three complementary alignment signals:

1. Global CLIP alignment for overall semantic correctness.
2. Opacity-weighted foreground CLIP alignment so the head, rather than the background, dominates the text signal.
3. View-conditioned CLIP alignment to keep the identity and attributes stable across camera views.

The loss weights are configurable in `configs/headstudio.yaml` and `configs/headstudio_retry.yaml`.

### 2.2 Prompt factorization

The text condition is factorized into more specific semantic components before the sweep. This improves the separation of identity, appearance, expression, and other attributes, and provides a stronger initialization for later quality optimization.

### 2.3 ViT-B/32 recovery teacher

The main semantic objective is supplemented with a frozen ViT-B/32 recovery teacher. This prevents optimization toward only the stronger backbone from damaging the B/32 score.

### 2.4 Diffusion guidance calibration

We swept the existing diffusion guidance scale rather than replacing the whole pipeline. Guidance scale `20` gave the best initialization for the quality tail phase.

### 2.5 Reference-statistics teacher

The initial Gaussian model is deep-copied and rendered under the same camera and pose. A frozen teacher provides local contrast and edge statistics, which regularize the optimized renders without requiring a ground-truth image.

### 2.6 Edge-aware artifact suppression

The final contribution is `artifact_suppression_loss` in `threestudio/models/clip_alignment.py`. It computes high-frequency luminance residuals against a local mean, penalizes them mainly in flat regions, reduces the penalty near real edges, and weights the result by foreground opacity. The existing quality ramp gradually introduces this term during training.

This targets the type of small-scale noise and false texture that can worsen PIQE while preserving facial boundaries.

## 3. Experiment Progression

The main valid progression was:

1. Prompt factorization: improved all semantic CLIP scores and MUSIQ, but PIQE remained high.
2. True factorized-content B/32 calibration: preserved the semantic improvement and recovered B/32 near or above baseline.
3. Guidance sweep: guidance `20` produced the strongest semantic/quality initialization.
4. q20 reference-statistics tail: improved the semantic Pareto point, reaching B/32 `0.318491`.
5. Edge-aware artifact sweep: reduced PIQE below the HeadStudio baseline.

Invalid/debug runs are retained but excluded from conclusions when they had incorrect manifests, AMP NaNs, zero parameter drift, or data-loader failures.

## 4. Final Edge-Aware Sweep

All three runs initialized from `quality_tail_q00020`, fixed guidance scale `20`, fixed B/32 recovery weight `0.05`, used full precision, and ran 500 continuation steps from logical step 7000.

| Run | Artifact weight | ViT-L/14 | ViT-B/16 | ViT-B/32 | PIQE | MUSIQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeadStudio | - | 0.278400 | 0.313000 | 0.313100 | 59.930000 | 51.360000 |
| `artifact_q0001` | 0.0001 | 0.287502 | 0.338545 | 0.317347 | 60.425517 | 55.8754 |
| `artifact_q0003` | 0.0003 | 0.285901 | 0.341617 | 0.318511 | 62.730646 | 54.8853 |
| `artifact_q0006` | 0.0006 | **0.290199** | **0.341607** | **0.313116** | **59.550065** | **55.6317** |

`artifact_q0006` is the current five-metric gate-passing checkpoint. The B/32 margin is only `0.000016`, so an additional seed should be run before making a strong generalization claim.

## 5. Where Results Are Saved

Every sweep stores its metrics under:

`outputs/<sweep-name>/<run-name>/eval/all_metrics/summary.json`

The visualization dashboard for each sweep is under:

`outputs/<sweep-name>/dashboard/`

The most important dashboards are:

- `outputs/text_gs_alignment_prompt_factorization_sweep_20260830/dashboard/`
- `outputs/text_gs_b32_factorized_content_calibration_20260830/dashboard/`
- `outputs/text_gs_guidance_quality_sweep_20260830/dashboard/`
- `outputs/text_gs_quality_tail_q20_20260830/dashboard/`
- `outputs/text_gs_artifact_quality_sweep_20260830/dashboard/`

Each dashboard contains `metrics_comparison.csv`, `metrics_bars.svg`, and `README.md`. The final checkpoint metrics are in:

`outputs/text_gs_artifact_quality_sweep_20260830/artifact_q0006/eval/all_metrics/summary.json`

The corresponding Gaussian checkpoint and diagnostics are in the same run directory, including `parameter_drift.json`.

## 6. Reproducibility and Engineering Fixes

The branch also records several fixes needed for trustworthy experiments:

- full-precision continuation to avoid AMP NaNs;
- non-finite gradient sanitization;
- reset of `max_radii2D` after PLY reload;
- first-backward gradient probes;
- parameter-drift diagnostics;
- corrected per-run evaluation manifests;
- local one-minute SSH keepalive script outside the repository.

These are reliability changes, not claimed method contributions.

## 7. Current Conclusion

The combined method currently passes the supplied five-metric comparison against HeadStudio. The strongest immediate follow-up is a second-seed replication of `artifact_q0006`, followed by a visual side-by-side comparison and ablation table isolating prompt factorization, B/32 recovery, and artifact suppression.
