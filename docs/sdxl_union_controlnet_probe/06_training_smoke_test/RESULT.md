# 06 Training Smoke Test Result

Status: passed with a 3-step Thor smoke run.

## Attempted

Command:

```bash
MPLCONFIGDIR=/tmp HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 timeout 1200 conda run -n ruiheadstudio-flux-controlnet bash scripts/run_sdxl_union_thor_smoke.sh 0
```

The smoke script used the Thor prompt from `scripts/headstudio.sh`:

`a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation`

Smoke limits:

- `data.batch_size=1`
- `trainer.max_steps=3`
- `system.guidance.guidance_resolution=512`
- `system.guidance_type="controlnet-union-sdxl-guidance"`
- `system.prompt_processor_type="stable-diffusion-xl-prompt-processor"`

## Result

- The run loaded `controlnet-union-sdxl-guidance`.
- Lightning stopped normally at `max_steps=3`.
- `csv_logs/version_0/metrics.csv` contains finite `train/loss_sds` values for steps 0, 1, and 2.
- Test export completed after training.

## Artifacts

- `outputs/headstudio/a_DSLR_portrait_of_Thor_in_Marvel,_masterpiece,_Studio_Quality,_8k,_ultra-HD,_next_generation@20260604-213011/csv_logs/version_0/metrics.csv`
- `outputs/headstudio/a_DSLR_portrait_of_Thor_in_Marvel,_masterpiece,_Studio_Quality,_8k,_ultra-HD,_next_generation@20260604-213011/save/last.ply`
- `outputs/headstudio/a_DSLR_portrait_of_Thor_in_Marvel,_masterpiece,_Studio_Quality,_8k,_ultra-HD,_next_generation@20260604-213011/save/it3-test.mp4`
