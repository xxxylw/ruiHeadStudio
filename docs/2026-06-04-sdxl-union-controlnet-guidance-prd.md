# PRD: SDXL Union ControlNet Guidance for RuiHeadStudio

## Problem Statement

RuiHeadStudio currently has a Guidance Backend built around the SD1.5 ControlNet Baseline. That baseline uses separate SD1.5 OpenPose and Depth ControlNets and can drive a FLAME-bound 3D Avatar through SDS, but it keeps image guidance tied to an older SD1.5 prior and a two-ControlNet runtime shape.

The next experiment should replace that SD1.5 Multi-ControlNet runtime path with SDXL Union ControlNet Guidance based on `xinsir/controlnet-union-sdxl-1.0`, while preserving the core RuiHeadStudio training shell: Multi-View Supervision, FLAME-derived Pose Condition and Depth Condition construction, Rendered RGB Sample optimization, and existing 3DGS geometry and temporal regularization.

## Solution

Build a new experimental SDXL Union ControlNet Guidance path. The experiment will replace the old SD1.5 guidance implementation in this branch rather than keeping a runtime switch, because the branch is dedicated to the SDXL Union probe.

The new Guidance Backend will consume:

- Rendered RGB Sample
- text prompt
- independent Pose Condition
- independent Depth Condition
- camera metadata already present in the multi-view batch, including azimuth, elevation, and distance

The first implementation will use:

- `stabilityai/stable-diffusion-xl-base-1.0`
- `madebyollin/sdxl-vae-fp16-fix`
- `xinsir/controlnet-union-sdxl-1.0`
- dedicated `stable-diffusion-xl-prompt-processor`
- guidance registration `controlnet-union-sdxl-guidance`
- Control Modes `["openpose", "depth"]`, internally mapped to `[0, 1]`
- 512-pixel smoke-test resolution
- batch size 1 for smoke tests
- standard two-way SDXL classifier-free guidance
- Euler scheduling
- constant SDS weighting for the first gradient probe

The implementation should prove the path in slices:

- validate the `ruiheadstudio-sdxl-union-controlnet` environment
- run ordinary SDXL Union 2D generation with FLAME pose and depth conditions
- validate SDXL prompt embeddings, including pooled embeddings
- validate finite nonzero SDS gradients to the Rendered RGB Sample
- connect SDXL Union guidance to the existing training interface
- run a few-step training smoke test before any full training attempt

## User Stories

1. As a RuiHeadStudio researcher, I want to replace the SD1.5 ControlNet Baseline with SDXL Union ControlNet Guidance, so that the 3D Avatar can benefit from a stronger SDXL image prior.
2. As a RuiHeadStudio researcher, I want to keep the training shell unchanged, so that the experiment isolates the Guidance Backend replacement.
3. As a RuiHeadStudio researcher, I want to keep FLAME-derived Pose Condition and Depth Condition construction unchanged, so that condition generation remains comparable with prior runs.
4. As a RuiHeadStudio researcher, I want Pose Condition and Depth Condition passed independently, so that pose and depth can retain separate strengths and schedules.
5. As a RuiHeadStudio researcher, I want Union Control Modes to be named in configuration, so that `openpose` and `depth` are not confused with opaque numeric ids.
6. As a RuiHeadStudio researcher, I want Union Control Modes mapped to xinsir/diffusers ids internally, so that the implementation satisfies the model contract.
7. As a RuiHeadStudio researcher, I want Multi-View Supervision preserved, so that the optimized avatar remains 3D-consistent rather than a single-view portrait.
8. As a RuiHeadStudio researcher, I want camera azimuth, elevation, and distance preserved in the batch contract, so that view-dependent prompting and diagnostics keep working.
9. As a RuiHeadStudio researcher, I want the Rendered RGB Sample treated as the optimization sample, so that SDS gradients update the 3D Avatar instead of preserving a reference image.
10. As a RuiHeadStudio researcher, I want a dedicated SDXL prompt processor, so that SDXL sequence and pooled embeddings are prepared at the correct boundary.
11. As a RuiHeadStudio researcher, I want the guidance module to fail fast on old SD1.5 prompt outputs, so that embedding contract mistakes are caught before model execution.
12. As a RuiHeadStudio researcher, I want the old SD1.5 guidance runtime path deleted in this branch, so that the probe branch stays focused on the replacement path.
13. As a RuiHeadStudio researcher, I want the new guidance registered as `controlnet-union-sdxl-guidance`, so that configuration clearly states the active Guidance Backend.
14. As a RuiHeadStudio researcher, I want the first 2D probe to use the official SDXL base, fp16-fix VAE, and xinsir Union ControlNet, so that model compatibility is validated before SDS work.
15. As a RuiHeadStudio researcher, I want 512-pixel smoke-test resolution first, so that memory and gradient correctness are validated before increasing resolution.
16. As a RuiHeadStudio researcher, I want ordinary 2D SDXL Union generation tested before SDS, so that Control Condition formatting and Control Mode mapping are proven independently.
17. As a RuiHeadStudio researcher, I want SDXL VAE latent encoding tested separately, so that tensor shape and dtype assumptions fail early.
18. As a RuiHeadStudio researcher, I want SDXL prompt pooled embeddings tested separately, so that UNet added-conditioning inputs are correct.
19. As a RuiHeadStudio researcher, I want standard two-way SDXL CFG in the first SDS implementation, so that training behavior matches SDXL pipeline semantics.
20. As a RuiHeadStudio researcher, I want NFSD, DSD, perp-neg, and null three-way guidance excluded from the first pass, so that the initial gradient path stays debuggable.
21. As a RuiHeadStudio researcher, I want Euler scheduling used consistently for 2D generation and SDS probes, so that scheduler assumptions are not split across stages.
22. As a RuiHeadStudio researcher, I want constant SDS weighting in the first gradient probe, so that scheduler weighting does not block validating the core gradient path.
23. As a RuiHeadStudio researcher, I want the SDS-like loss to produce finite nonzero RGB gradients, so that the Guidance Backend can actually drive 3DGS optimization.
24. As a RuiHeadStudio researcher, I want the SDXL Union Guidance Backend to expose the existing narrow training interface, so that geometry and temporal losses do not need to be rewritten.
25. As a RuiHeadStudio researcher, I want existing local geometry regularizers preserved, so that stronger SDXL guidance does not destabilize Gaussian scale or position behavior.
26. As a RuiHeadStudio researcher, I want existing temporal-window regularizers preserved, so that expression-driven avatar animation remains smooth.
27. As a RuiHeadStudio researcher, I want a dedicated environment named `ruiheadstudio-sdxl-union-controlnet`, so that this experiment is not confused with the FLUX probe environment.
28. As a RuiHeadStudio researcher, I want failures to write useful result notes, so that model access, OOM, dtype, shape, and gradient blockers are diagnosable later.
29. As a RuiHeadStudio researcher, I want a batch-size-1 training smoke test before full training, so that integration problems are caught before long GPU runs.
30. As a RuiHeadStudio researcher, I want visual quality evaluated only after model loading, 2D generation, SDS gradients, and smoke training pass, so that early validation focuses on correctness.
31. As a RuiHeadStudio researcher, I want a clear path to later ablations, so that 768/1024 resolution, ProMax, canny, softedge, normal, segment, and alternate SDS weighting can be evaluated after the first path works.

## Implementation Decisions

- Replace the SD1.5 Multi-ControlNet runtime path in this branch with SDXL Union ControlNet Guidance.
- Keep the current 3D Avatar optimization pipeline. The change is a Guidance Backend replacement, not a rewrite of the training system.
- Preserve Multi-View Supervision. Each training step should continue sampling views across azimuth, elevation, and distance.
- Preserve FLAME-derived Pose Condition and Depth Condition construction.
- Preserve the existing training interface shape: rendered RGB, control images, prompt output, and batch metadata in; `loss_sds` and diagnostics out.
- Delete the old SD1.5 guidance implementation file in this branch and create a new SDXL Union guidance module.
- Register the new guidance as `controlnet-union-sdxl-guidance`.
- Add a dedicated `stable-diffusion-xl-prompt-processor`.
- Add a dedicated SDXL prompt output object rather than overloading the existing SD1.5 prompt output with optional SDXL fields.
- The SDXL prompt output should provide positive and negative sequence embeddings plus positive and negative pooled embeddings.
- The SDXL prompt processor should preserve view-dependent prompting behavior.
- The SDXL Union guidance should fail fast if it receives the old SD1.5 prompt output contract.
- Use `stabilityai/stable-diffusion-xl-base-1.0` as the first base model.
- Use `madebyollin/sdxl-vae-fp16-fix` as the first VAE.
- Use `xinsir/controlnet-union-sdxl-1.0` as the first Union ControlNet.
- Use string Control Modes in configuration, starting with `["openpose", "depth"]`.
- Map Control Modes to xinsir/diffusers numeric ids internally, starting with `openpose -> 0` and `depth -> 1`.
- Save both string Control Modes and numeric Control Modes in probe reports.
- Start with Pose Condition and Depth Condition only.
- Start with condition scales near pose `1.0` and depth `0.8`.
- Start with independent control guidance schedules, including an early pose end around `0.65` and depth end around `0.8`.
- Treat 512-pixel guidance resolution as a smoke-test setting, not a final quality target.
- Use the SDXL VAE to encode the Rendered RGB Sample into the latent being optimized.
- Use standard two-way SDXL classifier-free guidance in the first SDS implementation.
- Do not include NFSD, DSD, perp-neg, or null three-way guidance in the first implementation.
- Use Euler scheduling for both ordinary 2D generation and the first SDS path.
- Use constant SDS weighting in the first implementation; sigma-based or alpha-based weighting can be a later ablation.
- Build a 2D probe script that uses the diffusers SDXL Union pipeline to validate model loading, VAE loading, Control Condition formatting, and Control Mode mapping.
- Build training guidance using explicit VAE, scheduler, Union ControlNet, and UNet calls rather than wrapping the full pipeline denoising loop.
- Keep local geometry losses, local scale-ratio constraints, surface constraints, sparsity, and temporal-window losses outside the SDXL Union guidance module.
- Use `ruiheadstudio-sdxl-union-controlnet` as the official environment name for documentation and scripts.

## Testing Decisions

- Tests and probes should validate external behavior and contracts rather than internal implementation details.
- Environment validation should check imports for torch, diffusers, transformers, `ControlNetUnionModel`, `StableDiffusionXLControlNetUnionPipeline`, `EulerAncestralDiscreteScheduler`, `AutoencoderKL`, and RuiHeadStudio FLAME condition utilities.
- The 2D generation probe should validate that the SDXL base, fp16-fix VAE, and xinsir Union ControlNet can load and consume independent Pose Condition and Depth Condition inputs.
- A Control Mode test should validate that configured names map to the expected numeric ids and reject unsupported names.
- A condition-format test should validate that pose and depth remain independent Control Conditions, not a merged image.
- A prompt-embedding test should validate positive and negative SDXL sequence embeddings and positive and negative pooled embeddings.
- A prompt-output contract test should validate that SDXL Union guidance rejects the old SD1.5 prompt output.
- A latent-shape test should validate that 512-pixel Rendered RGB Samples encode to the expected SDXL latent shape.
- A scheduler test should validate that Euler timestep sampling, noise addition, and model input scaling work for the selected scheduler.
- A gradient test should validate that the SDS-like loss produces finite, nonzero gradients with respect to the Rendered RGB Sample.
- A training-step smoke test should validate that the existing training shell can call SDXL Union guidance without changing geometry and temporal loss flow.
- A GPU smoke test should run with batch size 1, 512 guidance resolution, and few steps to detect OOM before full training.
- Existing source-contract tests for local scale-ratio and temporal-window behavior should continue to pass, because those behaviors are intentionally preserved.
- Prior art in the codebase includes lightweight source-contract tests around guidance-adjacent training behavior and temporal losses; SDXL Union tests can follow that style while adding executable probes for model and gradient correctness.

## Out of Scope

- Replacing the 3D Avatar representation.
- Removing FLAME binding.
- Removing Multi-View Supervision.
- Turning the workflow into single-view image-to-image or reference-image optimization.
- Rewriting local geometry losses, local scale-ratio losses, surface constraints, sparsity, or temporal-window losses.
- Making SDXL Union Guidance the default production baseline before smoke tests and at least one full run complete.
- Preserving the old SD1.5 guidance runtime path in this branch.
- Using FLUX for this branch.
- Using ProMax in the first implementation.
- Using canny, softedge, gray, normal, segment, or other Control Conditions in the first implementation.
- Using resolution higher than 512 for the first smoke-test milestone.
- Implementing NFSD, DSD, perp-neg, null three-way guidance, or alternative SDS weighting in the first implementation.
- Full benchmark comparison against prior Thor, Saul Goodman, Iron Man, Cristiano Ronaldo, or other SD1.5 runs.

## Further Notes

- GitHub issues are disabled for this repository, so this PRD is stored as a local project document rather than published to an issue tracker with a `ready-for-agent` label.
- The architectural decision is recorded in the SDXL Union ControlNet ADR.
- The first implementation should be treated as a probe branch, not a baseline replacement.
- The strongest early success signal is not visual quality. It is a correct, finite, nonzero SDXL Union SDS-like gradient back to the Rendered RGB Sample while pose and depth Control Conditions are active.
- Visual quality should be evaluated only after model loading, memory behavior, gradient correctness, and training-step integration are stable.
