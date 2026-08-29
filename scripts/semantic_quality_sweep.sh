#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_semantic_quality_sweep_20260830"
BASE_PLY="$ROOT/outputs/text_gs_alignment_refine_alpha_20260828/refine_semantic/runs/refine_semantic/save/last.ply"
PROMPT='a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation'

[[ -f "$BASE_PLY" ]] || { echo "missing initialization PLY: $BASE_PLY" >&2; exit 1; }
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
export HF_HUB_OFFLINE=1 DIFFUSERS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

launch_variant() {
  local gpu="$1" tag="$2" lambda_clip="$3"
  local out="$RUN_ROOT/$tag"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 launch.py \
    --config configs/headstudio_retry.yaml --train \
    "exp_root_dir=$out" "name=runs" "tag=$tag" "use_timestamp=False" \
    "system.prompt_processor.prompt=$PROMPT" \
    "system.guidance.guidance_scale=25" "trainer.max_steps=2000" \
    "data.batch_size=1" "system.gaussian_init_ply=$BASE_PLY" \
    "system.gaussian_init_step=10000" "system.clip_start_step=10000" \
    "system.clip_decay_start_step=11000" "system.clip_decay_end_step=12000" \
    "system.max_grad=0.0005" "system.area_relax=True" \
    "system.loss.lambda_clip=$lambda_clip" \
    "system.clip_global_weight=0.20" "system.clip_foreground_weight=0.55" \
    "system.clip_view_weight=0.25" \
    > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
  printf 'launched tag=%s gpu=%s pid=%s\n' "$tag" "$gpu" "$(cat "$out/train.pid")"
}

launch_variant 0 semantic_l0003 0.0003
launch_variant 1 semantic_l0006 0.0006
launch_variant 2 semantic_l0010 0.0010
