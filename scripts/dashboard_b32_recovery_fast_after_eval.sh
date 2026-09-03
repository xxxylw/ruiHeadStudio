#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/huangqirui/Projects/ruiHeadStudio
cd "$ROOT"
for tag in recovery_fast_q015 recovery_fast_q030 recovery_fast_q045; do
  while [[ ! -f "outputs/text_gs_b32_recovery_fast_sweep_20260830/$tag/eval/all_metrics/summary.json" ]]; do
    sleep 60
  done
done
python3 scripts/summarize_alignment_dashboard.py \
  outputs/text_gs_b32_recovery_fast_sweep_20260830/recovery_fast_q015/eval/all_metrics/summary.json \
  outputs/text_gs_b32_recovery_fast_sweep_20260830/recovery_fast_q030/eval/all_metrics/summary.json \
  outputs/text_gs_b32_recovery_fast_sweep_20260830/recovery_fast_q045/eval/all_metrics/summary.json \
  --output-dir outputs/text_gs_b32_recovery_fast_sweep_20260830/dashboard
