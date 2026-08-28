#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT=$ROOT/outputs/text_gs_alignment_refine_alpha_retry5_20260828
BASE_PLY=$ROOT/outputs/text_gs_alignment_refine_20260828/refine_multicomponent/runs/refine_multicomponent/save/last.ply
LOG=$RUN_ROOT/refine_multicomponent.train.log

set +u
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
set -u
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"
mkdir -p "$RUN_ROOT/refine_multicomponent"

CUDA_VISIBLE_DEVICES=0 python3 launch.py \
  --config configs/headstudio_retry.yaml \
  --train \
  "exp_root_dir=$RUN_ROOT/refine_multicomponent" \
  "name=runs" \
  "tag=refine_multicomponent" \
  "use_timestamp=False" \
  "system.guidance.guidance_scale=25" \
  "trainer.max_steps=3000" \
  "data.batch_size=1" \
  "system.gaussian_init_ply=$BASE_PLY" \
  "system.gaussian_init_step=7000" \
  "system.clip_start_step=7000" \
  "system.max_grad=0.001" \
  "system.area_relax=True" \
  "system.loss.lambda_clip=0.006" \
  "system.clip_global_weight=0.20" \
  "system.clip_foreground_weight=0.55" \
  "system.clip_view_weight=0.25" > "$LOG" 2>&1

/home/huangqirui/miniconda3/envs/ruiheadstudio/bin/python \
  evaluation/run_evaluation.py \
  --batch-root "$RUN_ROOT/refine_multicomponent" \
  --output-dir "$RUN_ROOT/refine_multicomponent/eval/all_metrics" \
  --device cpu \
  --metrics all > "$RUN_ROOT/refine_multicomponent.eval.log" 2>&1
