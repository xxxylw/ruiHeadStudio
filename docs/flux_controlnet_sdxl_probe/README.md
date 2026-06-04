# FLUX ControlNet SDXL Probe Slices

This directory tracks the eight execution slices for replacing the SD1.5 ControlNet Baseline with FLUX ControlNet Guidance while preserving RuiHeadStudio Multi-View Supervision.

Each numbered slice has a matching ignored result directory under `outputs/flux_controlnet_sdxl_probe/`.

## Slices

1. `01_environment_setup` — isolated FLUX conda environment and import validation.
2. `02_flux_2d_generation` — fixed-prompt FLUX ControlNet 2D generation with independent Pose Condition and Depth Condition.
3. `03_flux_latent_prediction` — Rendered RGB Sample encode, timestep/sigma sample, and FLUX prediction probe.
4. `04_flux_sds_gradient` — SDS-like gradient probe with finite nonzero RGB gradients.
5. `05_training_backend_integration` — FLUX Guidance Backend connected to the existing training interface.
6. `06_training_smoke_test` — small batch, 512-pixel, few-step training smoke test.
7. `07_full_training_run` — full FLUX training run after smoke tests pass.
8. `08_ablation_backlog` — documented follow-up ablations after the first working path.

## Result Convention

Each slice result directory should contain:

- `commands.log` — commands run for the slice.
- `stdout.log` / focused logs — raw command output when useful.
- `RESULT.md` — summary, status, artifacts, and next dependency.
- generated images, videos, checkpoints, or probes as needed.
