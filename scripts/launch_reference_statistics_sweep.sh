#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_reference_statistics_sweep_20260830"
BASE_PLY="$ROOT/outputs/text_gs_alignment_content_quality_sweep_20260830/content_q0010/runs/content_q0010/save/last.ply"
PROMPT='a DSLR portrait of Elon Musk'
[[ -f "$BASE_PLY" ]] || { echo "missing initialization PLY: $BASE_PLY" >&2; exit 1; }
set +u
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
set -u
export HF_HUB_OFFLINE=1 DIFFUSERS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

launch_variant() {
  local gpu="$1" tag="$2" lambda_statistics="$3"
  local out="$RUN_ROOT/$tag"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 launch.py --config configs/headstudio_retry.yaml --train \
    "exp_root_dir=$out" "name=runs" "tag=$tag" "use_timestamp=False" \
    "system.prompt_processor.prompt=$PROMPT" "system.guidance.guidance_scale=25" \
    "trainer.max_steps=2000" "data.batch_size=1" "system.gaussian_init_ply=$BASE_PLY" \
    "system.gaussian_init_step=18000" "system.clip_start_step=18000" \
    "system.clip_decay_start_step=19000" "system.clip_decay_end_step=20000" \
    "system.quality_start_step=19000" "system.quality_ramp_end_step=20000" \
    "system.max_grad=0.0005" "system.area_relax=True" "system.loss.lambda_clip=0.0005" \
    "system.lambda_frequency_quality=0.0" "system.lambda_rendered_reference=0.0" \
    "system.lambda_reference_statistics=$lambda_statistics" \
    "system.clip_global_weight=0.50" "system.clip_foreground_weight=0.30" \
    "system.clip_view_weight=0.20" > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
  printf 'launched tag=%s gpu=%s pid=%s\n' "$tag" "$gpu" "$(cat "$out/train.pid")"
}

launch_variant 0 statistics_q001 0.001
launch_variant 1 statistics_q003 0.003
launch_variant 2 statistics_q006 0.006
