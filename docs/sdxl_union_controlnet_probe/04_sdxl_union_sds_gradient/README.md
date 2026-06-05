# 04 SDXL Union SDS Gradient

Goal: validate a finite nonzero SDS gradient with SDXL Union ControlNet before training integration.

The probe should run:

1. Rendered RGB Sample resize to 512.
2. SDXL VAE encode.
3. Euler timestep sampling and noise addition.
4. Union ControlNet residual prediction with Pose Condition and Depth Condition.
5. SDXL UNet noise prediction with standard two-way CFG.
6. Constant-weight SDS gradient construction.
7. `loss_sds` backward to the RGB sample.

The matching result directory is `outputs/sdxl_union_controlnet_probe/04_sdxl_union_sds_gradient/`.
