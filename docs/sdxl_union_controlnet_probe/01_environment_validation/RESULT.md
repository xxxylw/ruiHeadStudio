# 01 Environment Validation Result

Status: passed with the available `ruiheadstudio-flux-controlnet` environment used as the temporary SDXL Union validation environment.

## Passed

- `torch`: `2.1.2+cu118`
- `diffusers`: `0.34.0`
- `transformers`: `4.46.3`
- `diffusers.ControlNetUnionModel`
- `diffusers.StableDiffusionXLControlNetUnionPipeline`
- `diffusers.EulerAncestralDiscreteScheduler`
- `diffusers.AutoencoderKL`
- `threestudio.utils.head_v2.FlamePointswRandomExp`
- Hugging Face cache completeness for:
  - `stabilityai/stable-diffusion-xl-base-1.0`
  - `madebyollin/sdxl-vae-fp16-fix`
  - `xinsir/controlnet-union-sdxl-1.0`

## Notes

- The formal environment name remains `ruiheadstudio-sdxl-union-controlnet`, but it does not exist locally yet.
- An attempted clone from `ruiheadstudio-flux-controlnet` to `ruiheadstudio-sdxl-union-controlnet` ran for more than 15 minutes without completing and was stopped. A partial directory may exist and should be reviewed before retrying environment creation.
- Validation required GPU access outside the sandbox. Inside the sandbox, FLAME CUDA imports failed with `Unknown compute capability`.
- The temporary environment reports an xFormers binary mismatch: xFormers was built for PyTorch `2.1.2+cu121`, while the environment has PyTorch `2.1.2+cu118`. This disables xFormers memory-efficient attention but did not block imports.

## Artifact

- `outputs/sdxl_union_controlnet_probe/01_environment_validation/validate_flux_env_as_sdxl_union_gpu.json`
- `outputs/sdxl_union_controlnet_probe/01_environment_validation/check_cache_after_resume.json`
