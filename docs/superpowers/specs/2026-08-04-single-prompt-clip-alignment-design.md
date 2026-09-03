# Single-Prompt CLIP Alignment Experiment

Date: 2026-08-04
Status: approved design; awaiting written-spec review

## Goal

Improve final image-text alignment for the fixed Elon Musk prompt while preserving visual quality. This is a single-prompt study, so the primary semantic metric is mean frozen CLIP ViT-L/14 cosine similarity. Retrieval rank is intentionally excluded: with one unique prompt it is always rank 1 and has no discriminative value.

## Fixed protocol

- Prompt: `a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation`
- Views: the same 180 test renders for every run.
- Semantic metric: mean and per-view CLIP ViT-L/14 matched cosine similarity using the existing evaluator preprocessing.
- Quality metrics: mean PIQE (lower is better) and MUSIQ (higher is better).
- Acceptance gate: semantic score must exceed the baseline. PIQE may increase by at most 2% and MUSIQ may decrease by at most 1%; report all three metrics and per-view distributions.
- Controls: fixed seed, prompt, pose/test views, number of steps, checkpoint selection, and evaluator version.

## Method: dual-scale CLIP loss

The existing loss aligns a foreground crop with a view-dependent prompt. Add an independent full-frame CLIP loss against the original prompt, then combine them after the existing CLIP warm-up:

`L_clip = lambda_clip * (alpha_global * L_global + (1 - alpha_global) * L_foreground_view)`

Where `L_global` uses the full rendered image and base prompt, while `L_foreground_view` retains the existing opacity crop and view prompt behavior. `alpha_global` is configurable and defaults to zero, preserving prior behavior.

This targets the final evaluator directly without discarding the foreground/view-aware signal. It also avoids changing the frozen CLIP model or test protocol.

## Experiment sequence

1. Run the formal one-prompt evaluation for the current `elon_cfg25_fgclip003` checkpoint; store it as the baseline artifact.
2. Add the configurable dual-scale loss and unit tests for component selection, weighting, and backward-compatible defaults.
3. Run a short smoke training to detect regressions.
4. Run one controlled full training with `lambda_clip=0.003` and `alpha_global=0.35`; this is the sole proposed new training run.
5. Run the same final evaluator, compare against baseline, and accept only if the quality gate passes.

## Failure handling

- If semantic similarity does not increase, do not tune based on the same test result; record the result and propose the next ablation separately.
- If either quality guardrail fails, mark the run rejected even if CLIP rises.
- If evaluation dependencies or model weights are unavailable, record the exact failure and do not substitute a different CLIP model.

## Reproducibility artifacts

Each accepted or rejected run retains its launch command, configuration snapshot, checkpoint path, per-image metric CSV, summary JSON, and a short comparison report.