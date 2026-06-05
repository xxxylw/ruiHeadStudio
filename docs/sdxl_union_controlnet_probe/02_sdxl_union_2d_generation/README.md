# 02 SDXL Union 2D Generation

Goal: validate ordinary SDXL Union ControlNet generation before any SDS or 3D training integration.

The probe should:

- Generate 512-pixel FLAME Pose Condition and Depth Condition images.
- Load `xinsir/controlnet-union-sdxl-1.0` as `ControlNetUnionModel`.
- Load `madebyollin/sdxl-vae-fp16-fix` as the VAE.
- Load `stabilityai/stable-diffusion-xl-base-1.0` through `StableDiffusionXLControlNetUnionPipeline`.
- Pass `control_image=[pose, depth]`.
- Pass `control_mode=[0, 1]` from config names `["openpose", "depth"]`.
- Save pose image, depth image, generated image, and a JSON report.

The matching result directory is `outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/`.
