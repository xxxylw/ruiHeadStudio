#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/rui/.cache/huggingface}"
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
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HOME="${HF_HOME}" \
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

POSE_METADATA_JSON="${POSE_METADATA_JSON:-./collection/ruiheadstudio/flame_collections/curriculum/train_pose_metadata.json}"
REFERENCE_FIDELITY_ENABLED="${REFERENCE_FIDELITY_ENABLED:-false}"
REFERENCE_METADATA="${REFERENCE_METADATA:-}"
REFERENCE_LAMBDA_REF_PERSON="${REFERENCE_LAMBDA_REF_PERSON:-0.0}"
REFERENCE_LAMBDA_REF_FACE="${REFERENCE_LAMBDA_REF_FACE:-0.2}"
REFERENCE_LAMBDA_REF_TEMPORAL_FACE="${REFERENCE_LAMBDA_REF_TEMPORAL_FACE:-0.02}"
STAGE2_LAMBDA_SPARSITY="${STAGE2_LAMBDA_SPARSITY:-0.05}"
STAGE2_LAMBDA_OPAQUE="${STAGE2_LAMBDA_OPAQUE:-0.2}"
OPACITY_COVERAGE_ENABLED="${OPACITY_COVERAGE_ENABLED:-false}"
LAMBDA_OPACITY_COVERAGE="${LAMBDA_OPACITY_COVERAGE:-0.0}"
REAR_OPACITY_ENABLED="${REAR_OPACITY_ENABLED:-false}"
LAMBDA_REAR_OPACITY="${LAMBDA_REAR_OPACITY:-0.0}"
PRUNE_REGION_GUARD_ENABLED="${PRUNE_REGION_GUARD_ENABLED:-false}"
if [[ "${FORCE_REBUILD_POSE_METADATA:-0}" == "1" || ! -f "$POSE_METADATA_JSON" ]]; then
  mkdir -p "$(dirname "$POSE_METADATA_JSON")"
  run_in_clean_env "$TRAIN_ENV_PREFIX/bin/python" scripts/build_pose_difficulty_metadata.py \
    --input ./collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy \
    --input ./collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug \
    --input ./collection/ruiheadstudio/flame_collections/talkvid/per_clip \
    --output "$POSE_METADATA_JSON"
fi

WEIGHTS_ARG=()
if [[ -n "${STAGE1_CKPT:-}" ]]; then
  WEIGHTS_ARG=("system.weights=${STAGE1_CKPT}")
fi

run_in_clean_env "$TRAIN_ENV_PREFIX/bin/python" launch.py --config configs/headstudio_stage2_text.yaml --train \
  "data.pose_metadata_inputs=['${POSE_METADATA_JSON}']" \
  "system.reference_fidelity.enabled=${REFERENCE_FIDELITY_ENABLED}" \
  "system.reference_fidelity.metadata_path=${REFERENCE_METADATA}" \
  "system.loss.lambda_ref_person=${REFERENCE_LAMBDA_REF_PERSON}" \
  "system.loss.lambda_ref_face=${REFERENCE_LAMBDA_REF_FACE}" \
  "system.loss.lambda_ref_temporal_face=${REFERENCE_LAMBDA_REF_TEMPORAL_FACE}" \
  "system.loss.lambda_sparsity=${STAGE2_LAMBDA_SPARSITY}" \
  "system.loss.lambda_opaque=${STAGE2_LAMBDA_OPAQUE}" \
  "system.opacity_coverage.enabled=${OPACITY_COVERAGE_ENABLED}" \
  "system.rear_opacity.enabled=${REAR_OPACITY_ENABLED}" \
  "system.prune_region_guard.enabled=${PRUNE_REGION_GUARD_ENABLED}" \
  "system.loss.lambda_opacity_coverage=${LAMBDA_OPACITY_COVERAGE}" \
  "system.loss.lambda_rear_opacity=${LAMBDA_REAR_OPACITY}" \
  "${WEIGHTS_ARG[@]}" \
  "$@"
