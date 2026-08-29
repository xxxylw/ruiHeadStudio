#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangqirui/Projects/ruiHeadStudio
RUN_ROOT="$ROOT/outputs/text_gs_alignment_semantic_quality_sweep_20260830"
cd "$ROOT"
set +u
source /home/huangqirui/miniconda3/etc/profile.d/conda.sh
conda activate ruiheadstudio
set -u
export HF_HUB_OFFLINE=1 DIFFUSERS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for tag in semantic_l0003 semantic_l0006 semantic_l0010; do
  out="$RUN_ROOT/$tag"
  pid=$(cat "$out/train.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  status=failed
  [[ -f "$out/runs/$tag/save/last.ply" ]] && status=ok
  printf '1\t%s\t%s\t\t\t0\t%s/runs/%s\t%s\t\n' "$tag" "$status" "$out" "$tag" 'a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation' > "$out/manifest.tsv"
  if [[ "$status" == ok ]]; then
    python3 evaluation/run_evaluation.py --batch-root "$out" --output-dir "$out/eval/all_metrics" --device cpu --metrics all > "$out/eval.log" 2>&1
  fi
done
