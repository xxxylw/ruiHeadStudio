# SDXL Union ControlNet Probe

This directory tracks the execution slices for replacing the SD1.5 Multi-ControlNet Guidance Backend with SDXL Union ControlNet Guidance while preserving RuiHeadStudio's training shell, FLAME-derived Control Conditions, Multi-View Supervision, and 3DGS optimization flow.

## Decision Summary

- Use a new guidance registration: `controlnet-union-sdxl-guidance`.
- Delete the old SD1.5 `controlnet_guidance.py` runtime path in this branch, but use its SDS structure as reference.
- Use a dedicated `stable-diffusion-xl-prompt-processor` with SDXL sequence and pooled prompt embeddings.
- Use `stabilityai/stable-diffusion-xl-base-1.0`.
- Use `madebyollin/sdxl-vae-fp16-fix`.
- Use `xinsir/controlnet-union-sdxl-1.0`.
- Use Control Modes as names in config: `["openpose", "depth"]`; map them to xinsir/diffusers modes `[0, 1]` internally.
- First pass only uses Pose Condition and Depth Condition. Do not include canny, softedge, gray, normal, segment, or ProMax.
- Treat 512 as smoke-test resolution, not final quality resolution.
- Use standard two-way SDXL CFG with conservative guidance scale. Do not include NFSD, DSD, perp-neg, or null three-way guidance in the first pass.
- Use Euler scheduling and constant SDS weighting for the first pass.
- Official environment name: `ruiheadstudio-sdxl-union-controlnet`.

## Slices

1. `01_environment_validation` - validate imports and local runtime dependencies.
2. `02_sdxl_union_2d_generation` - run ordinary SDXL Union pipeline generation with FLAME pose/depth conditions.
3. `03_sdxl_prompt_embeddings` - validate SDXL prompt processor sequence and pooled embeddings, including view-dependent prompts.
4. `04_sdxl_union_sds_gradient` - validate finite nonzero SDS gradients with a rendered RGB sample.
5. `05_training_backend_integration` - connect SDXL Union guidance to the existing training interface.
6. `06_training_smoke_test` - run batch-size-1, 512-resolution, few-step training without OOM.

## Success Criteria

- The 2D generation slice loads the SDXL base model, fp16-fix VAE, and xinsir Union ControlNet; consumes independent FLAME Pose Condition and Depth Condition images; and saves inputs, output image, and a JSON report.
- The SDS gradient slice runs VAE encode, Euler noise addition, Union ControlNet residual prediction, SDXL UNet prediction, and SDS loss construction; `loss_sds` is finite and the RGB gradient is finite and nonzero.
- The training smoke test completes a short run with `batch_size=1` and logs `train/loss_sds` without OOM.

Failures should still produce a useful `RESULT.md` with GPU model, dtype, resolution, batch size, model names, command, and exact blocker.
