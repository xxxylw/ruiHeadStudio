#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_refine_alpha_sweep_20260830"
PROMPT='a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation'

cd "$ROOT"
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
set +u
conda activate ruiheadstudio
set -u

for tag in quality_l003 balanced_l004 global_l0035; do
  out="$RUN_ROOT/$tag"
  pid_file="$out/train.pid"
  [[ -f "$pid_file" ]] || { echo "missing PID file: $pid_file" >&2; exit 1; }
  pid=$(cat "$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  printf '1\t%s\tok\t\t\t0\t%s/runs/%s\t%s\t\n' "$tag" "$out" "$tag" "$PROMPT" > "$out/manifest.tsv"
  python3 evaluation/run_evaluation.py \
    --batch-root "$out" \
    --output-dir "$out/eval/all_metrics" \
    --device cpu \
    --metrics all > "$out/eval.log" 2>&1
  echo "evaluated tag=$tag"
done
