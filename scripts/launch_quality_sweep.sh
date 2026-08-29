#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_refine_alpha_sweep_20260830"
BASE_PLY="$ROOT/outputs/text_gs_alignment_refine_20260828/refine_multicomponent/runs/refine_multicomponent/save/last.ply"

if [[ ! -f "$BASE_PLY" ]]; then
  echo "missing initialization PLY: $BASE_PLY" >&2
  exit 1
fi

source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

launch_variant() {
  local gpu="$1"
  local tag="$2"
  local lambda_clip="$3"
  local global_weight="$4"
  local foreground_weight="$5"
  local view_weight="$6"
  local out="$RUN_ROOT/$tag"

  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 launch.py \
    --config configs/headstudio_retry.yaml \
    --train \
    "exp_root_dir=$out" \
    "name=runs" \
    "tag=$tag" \
    "use_timestamp=False" \
    "system.guidance.guidance_scale=25" \
    "trainer.max_steps=3000" \
    "data.batch_size=1" \
    "system.gaussian_init_ply=$BASE_PLY" \
    "system.gaussian_init_step=7000" \
    "system.clip_start_step=7000" \
    "system.max_grad=0.001" \
    "system.area_relax=True" \
    "system.loss.lambda_clip=$lambda_clip" \
    "system.clip_global_weight=$global_weight" \
    "system.clip_foreground_weight=$foreground_weight" \
    "system.clip_view_weight=$view_weight" \
    > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
  echo "launched tag=$tag gpu=$gpu pid=$(cat "$out/train.pid")"
}

launch_variant 0 quality_l003 0.003 0.20 0.55 0.25
launch_variant 1 balanced_l004 0.004 0.30 0.45 0.25
launch_variant 2 global_l0035 0.0035 0.40 0.35 0.25
