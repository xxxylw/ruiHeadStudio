# 02 FLUX 2D Generation

Goal: validate ordinary FLUX ControlNet generation with a fixed prompt and independent RuiHeadStudio FLAME Pose Condition and Depth Condition.

## Acceptance Criteria

- The probe generates `flame_pose.png` and `flame_depth.png` from `FlamePointswRandomExp.get_cond_pose_depth()`.
- The probe passes pose and depth as independent control images in the order `[pose, depth]`.
- The probe uses initial scales `[0.9, 0.8]` and guidance end `[0.65, 0.8]`.
- `true_cfg_scale` stays `1.0`; the first path uses FLUX `guidance_scale`, not true CFG.
- A successful generation writes `flux_controlnet_pose_depth.png` and `probe_report.json` to the matching result directory.

## Access Gate

Before rerunning the full generation probe after changing Hugging Face credentials, run:

```bash
HF_ENDPOINT=https://huggingface.co conda run -n ruiheadstudio-flux-controlnet python scripts/flux_controlnet_sdxl_probe/02_flux_2d_generation/check_hf_flux_access.py --output outputs/flux_controlnet_sdxl_probe/02_flux_2d_generation/hf_access_report.json
```

This checks `black-forest-labs/FLUX.1-dev/model_index.json` and `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/config.json` without loading model weights.

## Result Directory

`outputs/flux_controlnet_sdxl_probe/02_flux_2d_generation/`
