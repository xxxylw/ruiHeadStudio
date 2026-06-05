# 05 Training Backend Integration

Goal: connect `controlnet-union-sdxl-guidance` to the existing RuiHeadStudio training interface without changing the training shell, FLAME Control Condition construction, or 3DGS optimization losses.

Expected config changes:

- `system.guidance_type: "controlnet-union-sdxl-guidance"`
- `system.prompt_processor_type: "stable-diffusion-xl-prompt-processor"`
- SDXL base, VAE, and xinsir Union ControlNet model paths.
- `control_modes: ["openpose", "depth"]`
- `guidance_resolution: 512`

The matching result directory is `outputs/sdxl_union_controlnet_probe/05_training_backend_integration/`.
