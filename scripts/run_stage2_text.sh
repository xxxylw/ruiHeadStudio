#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/rui/.cache/huggingface}"
export CONDA_PREFIX="${CONDA_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$CONDA_PREFIX/bin/python" launch.py --config configs/headstudio_stage2_text.yaml --train "$@"
