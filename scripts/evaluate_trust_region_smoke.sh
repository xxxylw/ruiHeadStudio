#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_trust_region_20260830"
TAG=trust_l003_anchor
OUT="$RUN_ROOT/$TAG"
PROMPT='a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation'

cd "$ROOT"
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
set +u
conda activate ruiheadstudio
set -u
pid=$(cat "$OUT/train.pid")
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
printf '1\t%s\tok\t\t\t0\t%s/runs/%s\t%s\t\n' "$TAG" "$OUT" "$TAG" "$PROMPT" > "$OUT/manifest.tsv"
python3 evaluation/run_evaluation.py \
  --batch-root "$OUT" \
  --output-dir "$OUT/eval/all_metrics" \
  --device cpu \
  --metrics all > "$OUT/eval.log" 2>&1
echo "evaluated tag=$TAG"
