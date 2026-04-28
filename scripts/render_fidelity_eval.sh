#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-$TRAIN_ENV_PREFIX}"
export CONDA_PREFIX="$TRAIN_ENV_PREFIX"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-118}"
export PATH="$TRAIN_ENV_PREFIX/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/lib:/usr/lib/wsl/lib"

run_in_clean_env() {
  env -i \
    HOME="${HOME}" \
    TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX}" \
    CONDA_PREFIX="${CONDA_PREFIX}" \
    PYTHONPATH="${PYTHONPATH}" \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    CUDA_HOME="${CUDA_HOME}" \
    BNB_CUDA_VERSION="${BNB_CUDA_VERSION}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PATH="${PATH}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    "$@"
}

FIDELITY_CKPT="${FIDELITY_CKPT:?Set FIDELITY_CKPT to the Stage2 checkpoint path}"
FIDELITY_TAG="${FIDELITY_TAG:-fidelity_eval}"
FIDELITY_PROMPT="${FIDELITY_PROMPT:-a realistic coherent character portrait, face and clothing together, stable identity, natural skin texture}"

run_in_clean_env "$TRAIN_ENV_PREFIX/bin/python" -c '
import runpy
import sys
import numpy as np

for name, value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
}.items():
    if not hasattr(np, name):
        setattr(np, name, value)

sys.argv = ["launch.py"] + sys.argv[1:]
runpy.run_path("launch.py", run_name="__main__")
' \
  --config configs/headstudio_stage2_text.yaml \
  --test \
  "system.weights=${FIDELITY_CKPT}" \
  "tag=${FIDELITY_TAG}" \
  "trainer.max_steps=1" \
  "system.prompt_processor.prompt=${FIDELITY_PROMPT}"
