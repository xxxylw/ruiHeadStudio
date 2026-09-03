#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/text_gs_alignment_20260827}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-ruiheadstudio}"
PROMPT="a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation"

mkdir -p "$RUN_ROOT/logs"

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

run_variant() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local batch_root="$RUN_ROOT/$tag"
  local log_file="$batch_root/$tag.train.log"
  mkdir -p "$batch_root"
  {
    echo "tag=$tag"
    echo "gpu=$gpu"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_head=$(git rev-parse --short HEAD)"
    echo "branch=$(git branch --show-current)"
    echo "prompt=$PROMPT"
    echo "args=$*"
  } > "$batch_root/provenance.env"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" launch.py \
    --config configs/headstudio.yaml \
    --train \
    "exp_root_dir=$batch_root" \
    "name=runs" \
    "tag=$tag" \
    "use_timestamp=False" \
    "system.prompt_processor.prompt=$PROMPT" \
    "system.guidance.guidance_scale=25" \
    "trainer.max_steps=10000" \
    "system.max_grad=0.001" \
    "system.area_relax=True" \
    "$@" > "$log_file" 2>&1
  local exit_code="$?"
  set -e

  local status="failed"
  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
  fi
  {
    printf "index\ttag\tstatus\tstarted_at\tfinished_at\texit_code\ttrial_dir\tprompt\textra_args\n"
    printf "1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$tag" "$status" "" "$(date --iso-8601=seconds)" "$exit_code" "$batch_root/runs/$tag" "$PROMPT" "$*"
  } > "$batch_root/manifest.tsv"

  if [[ "$exit_code" -eq 0 ]]; then
    "$PYTHON_BIN" evaluation/run_evaluation.py \
      --batch-root "$batch_root" \
      --output-dir "$batch_root/eval/all_metrics" \
      --device cpu \
      --metrics all > "$batch_root/eval.log" 2>&1
  fi
  echo "$tag exit=$exit_code"
  return "$exit_code"
}

run_variant 0 global_clip_warm \
  "system.loss.lambda_clip=0.001" \
  "system.clip_start_step=8500" \
  "system.clip_global_weight=1.0" &
pid0=$!

run_variant 1 foreground_view_clip \
  "system.loss.lambda_clip=0.003" \
  "system.clip_start_step=7500" \
  "system.clip_foreground_weight=0.65" \
  "system.clip_view_weight=0.35" &
pid1=$!

run_variant 2 text_gs_multicomponent \
  "system.loss.lambda_clip=0.0025" \
  "system.clip_start_step=7500" \
  "system.clip_global_weight=0.45" \
  "system.clip_foreground_weight=0.35" \
  "system.clip_view_weight=0.20" &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1

mapfile -t summaries < <(find "$RUN_ROOT" -path "*/eval/all_metrics/summary.json" | sort)
if [[ "${#summaries[@]}" -gt 0 ]]; then
  "$PYTHON_BIN" scripts/summarize_alignment_dashboard.py \
    --output-dir "$RUN_ROOT/dashboard" \
    "${summaries[@]}"
fi

exit "$status"
