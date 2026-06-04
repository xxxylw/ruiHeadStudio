# 02 FLUX 2D Generation Result

Status: blocked on FLUX.1-dev gated Hugging Face access.

## Passed

- Generated RuiHeadStudio FLAME pose/depth conditions at 512 resolution.
- Saved:
  - `outputs/flux_controlnet_sdxl_probe/02_flux_2d_generation/flame_pose.png`
  - `outputs/flux_controlnet_sdxl_probe/02_flux_2d_generation/flame_depth.png`
- Verified condition contract:
  - pose shape: `[1, 512, 512, 3]`
  - depth shape: `[1, 512, 512, 3]`
  - control image order: `[pose, depth]`
  - control scales: `[0.9, 0.8]`
  - control guidance end: `[0.65, 0.8]`
  - `true_cfg_scale`: `1.0`
  - FLUX `guidance_scale`: `3.5`

## Failed

Ordinary FLUX ControlNet 2D generation could load/download the Shakker ControlNet repo, but failed when loading the base model:

`black-forest-labs/FLUX.1-dev` returned gated repo access error for `model_index.json`.

The current server Hugging Face credentials are not authorized for `black-forest-labs/FLUX.1-dev`.

Latest access gate recheck:

- Report: `outputs/flux_controlnet_sdxl_probe/02_flux_2d_generation/hf_access_report_latest.json`
- Result: still blocked by `black-forest-labs/FLUX.1-dev` gated access.
- Current HF identity: `qiruihuang`, token display name `dinov3-access`.

## Notes

- `HF_ENDPOINT=https://hf-mirror.com` failed metadata lookup with `huggingface_hub==0.36.0`.
- Overriding `HF_ENDPOINT=https://huggingface.co` fixed Shakker ControlNet metadata and allowed the ControlNet weights to download.
- The remaining blocker is model authorization, not mirror configuration.

## Next Step

Configure an authorized Hugging Face token for `black-forest-labs/FLUX.1-dev`, then rerun the generation probe.

Use `check_hf_flux_access.py` first to verify authorization without loading full weights.
