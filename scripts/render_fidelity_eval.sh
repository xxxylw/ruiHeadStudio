#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"
FIDELITY_CKPT="${FIDELITY_CKPT:?Set FIDELITY_CKPT to the Stage2 checkpoint path}"
FIDELITY_TAG="${FIDELITY_TAG:-fidelity_eval}"
FIDELITY_PROMPT="${FIDELITY_PROMPT:-a realistic coherent character portrait, face and clothing together, stable identity, natural skin texture}"

conda run -n ruiheadstudio python launch.py \
  --config configs/headstudio_stage2_text.yaml \
  --test \
  "system.weights=${FIDELITY_CKPT}" \
  "tag=${FIDELITY_TAG}" \
  "trainer.max_steps=1" \
  "system.prompt_processor.prompt=${FIDELITY_PROMPT}"
