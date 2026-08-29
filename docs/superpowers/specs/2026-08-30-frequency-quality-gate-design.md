# Frequency-Gated Text-GS Alignment Design

## Goal

Improve text-to-3D Gaussian head alignment while reducing the quality regression observed in PIQE and the ViT-L/14 CLIP score. The method must remain differentiable, reproducible, and compatible with the existing multi-component global/foreground/view CLIP loss.

## Current Evidence

The current multi-component CLIP alignment improves smaller CLIP backbones and MUSIQ, but the best semantic checkpoint still has PIQE 61.285742 and ViT-L/14 0.270986 versus the HeadStudio baseline of PIQE 59.93 and ViT-L/14 0.2784. Parameter-space trust-region anchoring did not preserve rendered quality, so the next constraint should act on rendered images.

## Proposed Method

Add a `frequency_quality_loss` computed from the rendered RGB image and the detached foreground mask. The loss combines:

1. A normalized horizontal and vertical total-variation term to discourage isolated pixel noise.
2. A normalized Laplacian energy term to discourage high-frequency ringing and speckle artifacts.

The two terms are computed at the native render resolution and at a half-resolution image. The foreground mask is detached only for weighting, so RGB gradients remain differentiable. The quality loss is enabled after the semantic checkpoint, while CLIP alignment is linearly decayed over the same late window. This creates a frequency budget for semantic changes instead of freezing Gaussian parameters.

The total objective is:

`L = L_SDS + lambda_clip(t) * L_component_CLIP + lambda_quality(t) * L_frequency + existing regularizers`

The first sweep will use `lambda_quality` values `0.0005`, `0.0010`, and `0.0020`, with `lambda_clip=0.0010`, quality starting at global step 11000, and quality ramping to full weight at step 12000. These values are intentionally small because the image gradients are dense and are measured in a different scale from CLIP.

## Files and Data Flow

- `threestudio/models/clip_alignment.py`: add a pure helper that computes the multi-scale frequency loss and validates image/mask shapes.
- `threestudio/systems/Head3DGSLKs.py`: add configuration fields, compute the loss after CLIP alignment, log component terms, and add it to the training objective.
- `configs/headstudio_retry.yaml`: expose safe defaults with the new loss disabled unless explicitly configured.
- `tests/test_clip_alignment.py`: test zero-frequency behavior, gradients, mask handling, and shape validation.
- `scripts/launch_frequency_quality_sweep.sh`: launch three continuation runs from the strongest semantic checkpoint on available GPUs.
- `scripts/evaluate_frequency_quality_sweep.sh`: wait for jobs, use the configurable final-step evaluator, and write manifests and summaries.
- `docs/experiments/2026-08-27-text-gs-alignment-nightly.md`: record commands, exact parameters, five metrics, and gate status.
- `outputs/text_gs_alignment_frequency_quality_sweep_20260830/dashboard/`: publish CSV, SVG, and README artifacts.

## Testing and Acceptance

Unit tests must pass in the `ruiheadstudio` environment before launch. The sweep must preserve the exact prompt, four-view evaluation protocol, and five metrics: ViT-L/14, ViT-B/16, ViT-B/32, PIQE, and MUSIQ. A run is a success only if all five beat the HeadStudio baseline; partial improvements remain documented as ablations. The evaluator must use the actual continuation final step rather than a filename alias.

## Risks and Mitigations

- Over-smoothing may improve PIQE while hurting CLIP. Mitigation: three small weights and a late ramp.
- Background pixels may dominate the loss. Mitigation: use detached alpha weighting with a small uniform background floor.
- Laplacian scale may vary with resolution. Mitigation: normalize by mean absolute image intensity and test finite gradients.
- CPU evaluation is memory-sensitive. Mitigation: evaluate each run separately in the configured environment and retain per-metric summaries.
