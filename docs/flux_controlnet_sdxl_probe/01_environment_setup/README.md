# 01 Environment Setup

Goal: create an isolated conda environment for FLUX ControlNet Guidance without mutating the known-good `ruiheadstudio-bnbfix` environment.

## Acceptance Criteria

- A conda environment named `ruiheadstudio-flux-controlnet` exists.
- The environment keeps RuiHeadStudio's existing 3DGS runtime imports available.
- The environment can import FLUX-related diffusers pipeline classes.
- The environment validation script writes a machine-readable report to the matching result directory.

## Result Directory

`outputs/flux_controlnet_sdxl_probe/01_environment_setup/`
