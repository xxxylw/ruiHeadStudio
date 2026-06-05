# 06 Training Smoke Test

Goal: run a few training steps with SDXL Union ControlNet Guidance at `batch_size=1` and 512 guidance resolution.

This slice succeeds when a short run completes without OOM and logs finite `train/loss_sds`. Visual quality is not the success criterion for this slice.

The matching result directory is `outputs/sdxl_union_controlnet_probe/06_training_smoke_test/`.
