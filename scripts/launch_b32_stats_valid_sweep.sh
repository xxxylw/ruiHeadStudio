#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_b32_content_valid_sweep_20260830"
BASE_ROOT="$ROOT/outputs/text_gs_alignment_content_quality_sweep_20260830"
PROMPT='a DSLR portrait of Elon Musk'
set +u
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
set -u
export HF_HUB_OFFLINE=1 DIFFUSERS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

launch_variant() {
  local gpu="$1" tag="$2" stats_weight="$3" base_tag="$4"
  local base_ply="$BASE_ROOT/$base_tag/runs/$base_tag/save/last.ply"
  local out="$RUN_ROOT/$tag"
  [[ -f "$base_ply" ]] || { echo "missing initialization PLY: $base_ply" >&2; exit 1; }
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 launch.py --config configs/headstudio_retry.yaml --train \
    "exp_root_dir=$out" "name=runs" "tag=$tag" "use_timestamp=False" \
    "system.prompt_processor.prompt=$PROMPT" "system.guidance.guidance_scale=25" \
    "trainer.max_steps=500" "trainer.precision=32-true" "data.batch_size=1" "system.gaussian_init_ply=$base_ply" \
    "system.gaussian_init_step=7000" "system.clip_start_step=7000" \
    "system.clip_decay_start_step=7400" "system.clip_decay_end_step=7500" \
    "system.quality_start_step=7000" "system.quality_ramp_end_step=7500" \
    "system.max_grad=0.0005" "system.area_relax=True" "system.loss.lambda_clip=0.0005" \
    "system.lambda_frequency_quality=0.0" "system.lambda_rendered_reference=0.0" \
    "system.lambda_reference_statistics=$stats_weight" \
    "system.clip_recovery_model_name=ViT-B/32" "system.clip_recovery_weight=0.15" \
    "system.clip_global_weight=1.0" "system.clip_foreground_weight=0.0" \
    "system.clip_view_weight=0.0" > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
  printf 'launched tag=%s gpu=%s pid=%s\n' "$tag" "$gpu" "$(cat "$out/train.pid")"
}

launch_variant 0 content_valid_q0002 0.0002 content_q0010
launch_variant 1 content_valid_q0005 0.0005 content_q0010
launch_variant 2 content_valid_q0010 0.0010 content_q0010
