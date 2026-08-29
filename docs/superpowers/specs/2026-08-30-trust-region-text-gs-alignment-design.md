# Trust-Region Text-GS Alignment Design

## Goal

Improve text alignment without repeating the observed quality trade-off: the current multi-component CLIP loss raises ViT-B/32 and MUSIQ, but lowers ViT-L/14 and worsens PIQE. The next experiment keeps the existing differentiable global, rasterized-alpha foreground, and view-conditioned alignment method, while constraining late-stage Gaussian drift.

## Method

At initialization, snapshot the trainable Gaussian tensors needed for a quality-preserving trust region: XYZ, scaling, opacity, and SH/color features. During continuation, add a normalized parameter-drift penalty after `clip_start_step`:

`L_trust = mean(|xyz - xyz0| / (scale0 + eps)) + mean(|scale - scale0| / (scale0 + eps)) + mean(|opacity - opacity0|) + color_weight * mean(|features - features0|)`.

The anchor is detached and remains fixed. The loss is applied only to the continuation checkpoint, so it does not interfere with the original HeadStudio formation stage. A configurable `lambda_trust` controls its strength, and logging exposes both the total trust loss and its components.

CLIP keeps the alpha-aware component weights `global=0.20`, `foreground=0.55`, `view=0.25`. Its effective weight linearly decays during the final 1,000 continuation steps, preserving semantic gains early and reducing late perceptual damage. The decay is configurable and disabled when the end step is not set.

## Experiment

Continue from `outputs/text_gs_alignment_refine_20260828/refine_multicomponent/runs/refine_multicomponent/save/last.ply`, with batch size 1, 3,000 steps, `lambda_clip=0.003`, `lambda_trust=0.02`, and the above component weights. Keep the exact prompt and four-view evaluation protocol. Use one GPU first to validate behavior; expand to an ablation sweep only after the smoke run completes.

## Verification

- Unit-test the trust loss for zero drift, finite gradients, shape mismatch, and detached anchors.
- Run the existing focused CLIP/alpha tests.
- Verify the training log contains trust-loss and CLIP-decay values.
- Evaluate ViT-L/14, ViT-B/16, ViT-B/32, PIQE, and MUSIQ.
- Update the experiment Markdown, CSV/SVG dashboard, and commit each reproducible change.

## Success Gate

The primary target is to exceed HeadStudio on all five supplied metrics: ViT-L/14 `0.2784`, ViT-B/16 `0.3130`, ViT-B/32 `0.3131`, PIQE below `59.93`, and MUSIQ above `51.36`. Partial success is reported explicitly, with ablation evidence and no claim of a full win unless every gate is met.
