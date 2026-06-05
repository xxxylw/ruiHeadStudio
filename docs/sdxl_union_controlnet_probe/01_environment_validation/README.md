# 01 Environment Validation

Goal: validate the `ruiheadstudio-sdxl-union-controlnet` environment before downloading or loading full SDXL weights.

The validation script should import:

- `torch`
- `diffusers`
- `transformers`
- `diffusers.ControlNetUnionModel`
- `diffusers.StableDiffusionXLControlNetUnionPipeline`
- `diffusers.EulerAncestralDiscreteScheduler`
- `diffusers.AutoencoderKL`
- RuiHeadStudio FLAME condition utilities

The matching result directory is `outputs/sdxl_union_controlnet_probe/01_environment_validation/`.
