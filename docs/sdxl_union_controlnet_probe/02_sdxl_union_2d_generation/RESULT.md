# 02 SDXL Union 2D Generation Result

Status: passed after completing the SDXL base model cache.

## Passed

- Generated 512-pixel RuiHeadStudio FLAME Pose Condition and Depth Condition images with the Thor prompt probe.
- Saved:
  - `outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/thor_probe_local_only/flame_pose.png`
  - `outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/thor_probe_local_only/flame_depth.png`
- Verified condition contract:
  - pose shape: `[1, 512, 512, 3]`
  - depth shape: `[1, 512, 512, 3]`
  - Control Modes: `["openpose", "depth"]`
  - Control Mode ids: `[0, 1]`
- Verified Hugging Face metadata/config access for:
  - `xinsir/controlnet-union-sdxl-1.0/config.json`
  - `madebyollin/sdxl-vae-fp16-fix/config.json`
  - `stabilityai/stable-diffusion-xl-base-1.0/model_index.json`
- Loaded the SDXL base, SDXL VAE, and xinsir Union ControlNet from local cache.
- Generated a 512-pixel SDXL Union output conditioned by the FLAME pose and depth maps:
  - `outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/thor_probe_after_cache/sdxl_union_pose_depth.png`

## Earlier Blocker

- Local-files-only model loading failed because the SDXL base UNet weights are not cached:
  - missing `unet/diffusion_pytorch_model.safetensors`
  - missing fallback `unet/diffusion_pytorch_model.bin`
- A network-enabled full 2D probe generated pose/depth, entered model loading/download, then made no visible progress for more than ten minutes and was stopped.
- A direct single-file download attempt for `stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors` also made no visible progress for about five minutes and was stopped.
- The cache was later completed with resumable Hugging Face downloads; `check_cache_after_resume.json` reports no missing files.

## Notes

- The initial implementation passed `variant="fp16"` to the SDXL pipeline. Local-files-only loading showed that the cached model snapshot did not contain matching fp16 variant files, so the implementation now relies on `torch_dtype=float16` without forcing a variant.
- The implementation relies on `torch_dtype=float16` without forcing a model variant, matching the available cache layout.

## Artifact

- `outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/thor_probe_local_only/probe_report.json`
- `outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/thor_probe_after_cache/probe_report.json`
