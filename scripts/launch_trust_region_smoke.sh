#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_trust_region_20260830"
BASE_PLY="$ROOT/outputs/text_gs_alignment_refine_20260828/refine_multicomponent/runs/refine_multicomponent/save/last.ply"
TAG=trust_l003_anchor
OUT="$RUN_ROOT/$TAG"

[[ -f "$BASE_PLY" ]] || { echo "missing initialization PLY: $BASE_PLY" >&2; exit 1; }
mkdir -p "$OUT"
set +u
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
set -u
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 nohup python3 launch.py \
  --config configs/headstudio_retry.yaml \
  --train \
  "exp_root_dir=$OUT" \
  "name=runs" \
  "tag=$TAG" \
  "use_timestamp=False" \
  "system.guidance.guidance_scale=25" \
  "trainer.max_steps=3000" \
  "data.batch_size=1" \
  "system.gaussian_init_ply=$BASE_PLY" \
  "system.gaussian_init_step=7000" \
  "system.clip_start_step=7000" \
  "system.clip_decay_start_step=9000" \
  "system.clip_decay_end_step=10000" \
  "system.max_grad=0.001" \
  "system.area_relax=True" \
  "system.loss.lambda_clip=0.003" \
  "system.lambda_trust=0.02" \
  "system.clip_global_weight=0.20" \
  "system.clip_foreground_weight=0.55" \
  "system.clip_view_weight=0.25" \
  > "$OUT/train.log" 2>&1 < /dev/null &
echo $! > "$OUT/train.pid"
echo "launched tag=$TAG gpu=0 pid=$(cat "$OUT/train.pid")"
