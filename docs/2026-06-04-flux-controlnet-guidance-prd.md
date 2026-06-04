# PRD: FLUX ControlNet Guidance for RuiHeadStudio

## Problem Statement

RuiHeadStudio currently uses the SD1.5 ControlNet Baseline for pose and depth guidance. That baseline can train a FLAME-bound 3D Avatar, but it limits image prior quality and keeps the Guidance Backend tied to a classic SD1.5 UNet/DDPM-style stack.

The next experiment should replace the SD1.5 ControlNet Baseline with FLUX ControlNet Guidance while preserving the core RuiHeadStudio training shape: Multi-View Supervision, FLAME-derived Pose Condition and Depth Condition inputs, Rendered RGB Sample optimization, and existing local geometry and temporal regularization.

## Solution

Build a new experimental Guidance Backend based on FLUX.1-dev and Shakker-Labs FLUX.1-dev ControlNet Union Pro 2.0. The first implementation should run in an isolated conda environment so the known-good SD1.5 training environment remains available.

The new Guidance Backend will consume:

- Rendered RGB Sample
- text prompt
- independent Pose Condition
- independent Depth Condition
- camera metadata already present in the multi-view batch, including azimuth, elevation, and distance

The implementation should first prove the FLUX Smoke Test path at 512-pixel resolution:

- load FLUX base and FLUX ControlNet Union
- run ordinary 2D FLUX ControlNet generation with pose and depth conditions
- encode a Rendered RGB Sample to latent space
- sample timestep/sigma and produce a single conditional FLUX prediction
- construct an SDS-like gradient for the Rendered RGB Sample
- confirm nonzero gradient flows back to RGB
- connect the Guidance Backend to the existing training step shape
- run a small-batch, small-step smoke test before any full training

## User Stories

1. As a RuiHeadStudio researcher, I want to replace the SD1.5 ControlNet Baseline with FLUX ControlNet Guidance, so that the 3D Avatar can benefit from a stronger modern image prior.
2. As a RuiHeadStudio researcher, I want the FLUX experiment isolated in a separate environment, so that the known-good SD1.5 baseline remains reproducible.
3. As a RuiHeadStudio researcher, I want Pose Condition and Depth Condition passed as independent inputs, so that pose and depth strength can be tuned separately.
4. As a RuiHeadStudio researcher, I want control guidance schedules to be independent for pose and depth, so that each Control Condition can stop influencing generation at the appropriate denoising phase.
5. As a RuiHeadStudio researcher, I want Multi-View Supervision preserved, so that the optimized avatar remains a 3D-consistent head rather than a single-view portrait.
6. As a RuiHeadStudio researcher, I want camera azimuth, elevation, and distance preserved in the batch contract, so that view-dependent behavior remains debuggable and controllable.
7. As a RuiHeadStudio researcher, I want the Rendered RGB Sample treated as the optimization sample, so that gradients update the 3D Avatar instead of preserving the current render as a reference image.
8. As a RuiHeadStudio researcher, I want the first FLUX guidance implementation to use single conditional prediction with FLUX Guidance Scale, so that the initial gradient path is simpler to validate.
9. As a RuiHeadStudio researcher, I want true classifier-free guidance deferred to a later ablation, so that the first implementation does not mix two guidance mechanisms.
10. As a RuiHeadStudio researcher, I want a 512-pixel FLUX Smoke Test first, so that memory and gradient correctness are validated before increasing resolution.
11. As a RuiHeadStudio researcher, I want ordinary 2D FLUX ControlNet generation tested before SDS-like guidance, so that pose/depth condition compatibility is proven independently.
12. As a RuiHeadStudio researcher, I want RGB latent encoding tested separately, so that VAE compatibility and tensor shape assumptions fail early.
13. As a RuiHeadStudio researcher, I want timestep/sigma sampling tested separately, so that FLUX flow-matching assumptions are explicit and inspectable.
14. As a RuiHeadStudio researcher, I want the SDS-like loss to produce a nonzero RGB gradient, so that the Guidance Backend can actually drive 3DGS optimization.
15. As a RuiHeadStudio researcher, I want the FLUX Guidance Backend to expose a narrow interface compatible with the current training step, so that geometry losses and temporal losses do not need to be rewritten.
16. As a RuiHeadStudio researcher, I want the existing local scale and local position regularizers preserved, so that FLUX guidance does not reintroduce large Gaussian outliers or unstable geometry.
17. As a RuiHeadStudio researcher, I want the existing temporal-window regularizers preserved, so that expression-driven avatar animation remains smooth.
18. As a RuiHeadStudio researcher, I want logs to distinguish SD1.5 baseline runs from FLUX runs, so that experimental comparisons are not ambiguous.
19. As a RuiHeadStudio researcher, I want control scales to start near pose 0.9 and depth 0.8, so that initial behavior follows the FLUX ControlNet Union model guidance while remaining tunable.
20. As a RuiHeadStudio researcher, I want control guidance end to start near pose 0.65 and depth 0.8, so that structural pose guidance can relax earlier than depth guidance.
21. As a RuiHeadStudio researcher, I want smoke tests to run with small batches and few steps, so that integration mistakes are caught before overnight training.
22. As a RuiHeadStudio researcher, I want full training attempted only after the FLUX Smoke Test passes, so that GPU time is not wasted on a broken guidance path.
23. As a RuiHeadStudio researcher, I want failures in model loading, condition formatting, latent encoding, and gradient construction to be diagnosable separately, so that fixes are localized.
24. As a RuiHeadStudio researcher, I want the SD1.5 ControlNet Baseline code path retained during the experiment, so that results can be compared and regressions can be isolated.
25. As a RuiHeadStudio researcher, I want a clear path to later ablations, so that true classifier-free guidance, higher resolution, and alternative control types can be evaluated after the first gradient path works.

## Implementation Decisions

- Create an isolated experiment environment for FLUX ControlNet Guidance. Do not upgrade or mutate the known-good SD1.5 environment.
- Keep the current 3D Avatar optimization pipeline. The change is a Guidance Backend replacement, not a rewrite of the training system.
- Preserve Multi-View Supervision. Each training step should continue sampling views across azimuth, elevation, and distance.
- Preserve the existing batch contract for rendered RGB, control conditions, prompt data, and camera metadata.
- Replace the SD1.5 ControlNet Baseline for this experiment with FLUX.1-dev plus FLUX.1-dev ControlNet Union Pro 2.0.
- Treat Pose Condition and Depth Condition as independent Control Conditions.
- Start with `control_image` semantics equivalent to independent pose and depth images.
- Start with independent conditioning scales near pose 0.9 and depth 0.8.
- Start with independent guidance end values near pose 0.65 and depth 0.8.
- Treat the Rendered RGB Sample as the latent being optimized. Do not use it as an init image, reference image, or preservation target.
- Use single conditional FLUX prediction with FLUX Guidance Scale in the first implementation.
- Defer true classifier-free guidance to a later ablation.
- Use FLUX flow-matching/rectified-flow style timestep and sigma handling rather than SD1.5 alpha-cumprod/DDIM assumptions.
- Build the FLUX guidance path as a deep module with a narrow interface: rendered RGB, control conditions, prompt data, and batch metadata in; guidance loss and optional diagnostics out.
- Keep geometry regularization, local scale-ratio constraints, surface constraints, sparsity, and temporal-window losses outside the FLUX guidance module.
- Make the FLUX Smoke Test the first executable milestone before training integration.
- Keep the SD1.5 baseline available for comparison and rollback during the branch.

## Testing Decisions

- Tests should validate external behavior and contracts rather than internal implementation details.
- The FLUX Smoke Test should validate that the Guidance Backend can load the model stack and accept independent Pose Condition and Depth Condition inputs.
- A condition-format test should validate that pose and depth are passed as independent Control Conditions, not merged into one image.
- A latent-shape test should validate that Rendered RGB Samples encode to the expected FLUX VAE latent shape at 512-pixel resolution.
- A timestep/sigma test should validate that sampled timesteps and sigmas are valid for the FLUX scheduler or transformer integration being used.
- A gradient test should validate that the SDS-like loss produces finite, nonzero gradients with respect to the Rendered RGB Sample.
- A training-step smoke test should validate that the existing training step can call the FLUX Guidance Backend without changing the geometry and temporal loss flow.
- A GPU smoke test should run with small batch size, 512 resolution, and few steps to detect out-of-memory failures before full training.
- Existing source-contract tests for local scale-ratio and temporal-window behavior should continue to pass, because those behaviors are intentionally preserved.
- Prior art in the codebase includes lightweight source-contract tests around guidance-adjacent training behavior and temporal losses; the FLUX tests can follow that style for early integration while adding one or more executable smoke tests for gradient correctness.

## Out of Scope

- Replacing the 3D Avatar representation.
- Removing FLAME binding.
- Removing Multi-View Supervision.
- Turning the workflow into single-view image-to-image or reference-image optimization.
- Rewriting local geometry losses, local scale-ratio losses, surface constraints, sparsity, or temporal-window losses.
- Making FLUX Guidance the default production baseline before smoke tests and at least one full run complete.
- True classifier-free guidance in the first implementation.
- Higher-than-512 smoke test resolution in the first milestone.
- Using canny, soft edge, or gray Control Conditions in the first milestone.
- Full benchmark comparison against all prior Thor, Saul Goodman, Iron Man, and other SD1.5 runs.

## Further Notes

- GitHub issues are disabled for this repository, so this PRD is stored as a local project document rather than published to an issue tracker with a `ready-for-agent` label.
- The first implementation should be treated as a probe branch, not a baseline replacement.
- The strongest early success signal is not visual quality. It is a correct, finite, nonzero FLUX SDS-like gradient back to the Rendered RGB Sample while pose/depth Control Conditions are active.
- Visual quality should be evaluated only after the gradient path, memory behavior, and training-step integration are stable.
