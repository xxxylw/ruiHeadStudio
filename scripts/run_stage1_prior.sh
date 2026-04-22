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

POSE_METADATA_JSON="${POSE_METADATA_JSON:-./collection/ruiheadstudio/flame_collections/curriculum/train_pose_metadata.json}"
if [[ "${FORCE_REBUILD_POSE_METADATA:-0}" == "1" || ! -f "$POSE_METADATA_JSON" ]]; then
  mkdir -p "$(dirname "$POSE_METADATA_JSON")"
  "$CONDA_PREFIX/bin/python" scripts/build_pose_difficulty_metadata.py \
    --input ./collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy \
    --input ./collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug \
    --input ./collection/ruiheadstudio/flame_collections/talkvid/per_clip \
    --output "$POSE_METADATA_JSON"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$CONDA_PREFIX/bin/python" launch.py --config configs/headstudio_stage1_prior.yaml --train \
  "data.pose_metadata_inputs=['${POSE_METADATA_JSON}']" \
  "$@"
