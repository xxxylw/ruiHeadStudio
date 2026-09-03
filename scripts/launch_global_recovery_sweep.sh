#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_global_recovery_sweep_20260830"
BASE_PLY="$ROOT/outputs/text_gs_alignment_frequency_quality_sweep_20260830/frequency_q0005/runs/frequency_q0005/save/last.ply"
PROMPT='a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation'
[[ -f "$BASE_PLY" ]] || { echo "missing initialization PLY: $BASE_PLY" >&2; exit 1; }
set +u
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
set -u
export HF_HUB_OFFLINE=1 DIFFUSERS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

launch_variant() {
  local gpu="$1" tag="$2" lambda_clip="$3" global="$4" foreground="$5" view="$6"
  local out="$RUN_ROOT/$tag"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 launch.py --config configs/headstudio_retry.yaml --train \
    "exp_root_dir=$out" "name=runs" "tag=$tag" "use_timestamp=False" \
    "system.prompt_processor.prompt=$PROMPT" "system.guidance.guidance_scale=25" \
    "trainer.max_steps=2000" "data.batch_size=1" "system.gaussian_init_ply=$BASE_PLY" \
    "system.gaussian_init_step=12000" "system.clip_start_step=12000" \
    "system.clip_decay_start_step=13000" "system.clip_decay_end_step=14000" \
    "system.quality_start_step=13000" "system.quality_ramp_end_step=14000" \
    "system.max_grad=0.0005" "system.area_relax=True" "system.loss.lambda_clip=$lambda_clip" \
    "system.lambda_frequency_quality=0.0005" "system.clip_global_weight=$global" \
    "system.clip_foreground_weight=$foreground" "system.clip_view_weight=$view" \
    > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
  printf 'launched tag=%s gpu=%s pid=%s\n' "$tag" "$gpu" "$(cat "$out/train.pid")"
}

launch_variant 0 global_l0005 0.0005 1.0 0.0 0.0
launch_variant 1 global_l0010 0.0010 1.0 0.0 0.0
launch_variant 2 mixed_l0010 0.0010 0.50 0.30 0.20
