#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-silver_haired_scientist_portrait}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
STAGE1_PROMPT="${STAGE1_PROMPT:-a neutral photorealistic human head portrait, realistic skin, natural face, studio lighting}"
STAGE2_PROMPT="${STAGE2_PROMPT:-a realistic studio portrait of a weathered middle aged man, short dark hair with subtle gray, natural skin tone, defined cheekbones, calm focused expression, clean realistic face, matte skin, soft even studio lighting, plain dark gray background, head and neck only, no clothing, no collar}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-4000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-10000}"
STAGE1_CKPT_INTERVAL="${STAGE1_CKPT_INTERVAL:-1000}"
STAGE1_VAL_INTERVAL="${STAGE1_VAL_INTERVAL:-1000}"
STAGE2_CKPT_INTERVAL="${STAGE2_CKPT_INTERVAL:-2000}"
STAGE2_VAL_INTERVAL="${STAGE2_VAL_INTERVAL:-2000}"
REFERENCE_FIDELITY_ENABLED="${REFERENCE_FIDELITY_ENABLED:-false}"
REFERENCE_METADATA="${REFERENCE_METADATA:-}"
REFERENCE_LAMBDA_REF_PERSON="${REFERENCE_LAMBDA_REF_PERSON:-0.05}"
REFERENCE_LAMBDA_REF_FACE="${REFERENCE_LAMBDA_REF_FACE:-0.2}"
REFERENCE_LAMBDA_REF_TEMPORAL_FACE="${REFERENCE_LAMBDA_REF_TEMPORAL_FACE:-0.02}"
OPACITY_COVERAGE_ENABLED="${OPACITY_COVERAGE_ENABLED:-false}"
LAMBDA_OPACITY_COVERAGE="${LAMBDA_OPACITY_COVERAGE:-0.0}"
REAR_OPACITY_ENABLED="${REAR_OPACITY_ENABLED:-false}"
LAMBDA_REAR_OPACITY="${LAMBDA_REAR_OPACITY:-0.0}"
PRUNE_REGION_GUARD_ENABLED="${PRUNE_REGION_GUARD_ENABLED:-false}"
OUTPUT_ROOT="outputs/${RUN_TAG}${RUN_TS}"
STAGE1_CKPT="${OUTPUT_ROOT}/headstudio-stage1-prior/ckpts/last.ckpt"

bash scripts/run_stage1_prior.sh \
  tag="${RUN_TAG}" \
  timestamp="${RUN_TS}" \
  trainer.max_steps="${STAGE1_MAX_STEPS}" \
  checkpoint.every_n_train_steps="${STAGE1_CKPT_INTERVAL}" \
  trainer.val_check_interval="${STAGE1_VAL_INTERVAL}" \
  "system.prompt_processor.prompt=${STAGE1_PROMPT}" \
  "$@"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Missing stage1 checkpoint: ${STAGE1_CKPT}" >&2
  exit 1
fi

export STAGE1_CKPT

REFERENCE_FIDELITY_ENABLED="${REFERENCE_FIDELITY_ENABLED}" \
REFERENCE_METADATA="${REFERENCE_METADATA}" \
REFERENCE_LAMBDA_REF_PERSON="${REFERENCE_LAMBDA_REF_PERSON}" \
REFERENCE_LAMBDA_REF_FACE="${REFERENCE_LAMBDA_REF_FACE}" \
REFERENCE_LAMBDA_REF_TEMPORAL_FACE="${REFERENCE_LAMBDA_REF_TEMPORAL_FACE}" \
OPACITY_COVERAGE_ENABLED="${OPACITY_COVERAGE_ENABLED}" \
LAMBDA_OPACITY_COVERAGE="${LAMBDA_OPACITY_COVERAGE}" \
REAR_OPACITY_ENABLED="${REAR_OPACITY_ENABLED}" \
LAMBDA_REAR_OPACITY="${LAMBDA_REAR_OPACITY}" \
PRUNE_REGION_GUARD_ENABLED="${PRUNE_REGION_GUARD_ENABLED}" \
bash scripts/run_stage2_text.sh \
  tag="${RUN_TAG}" \
  timestamp="${RUN_TS}" \
  trainer.max_steps="${STAGE2_MAX_STEPS}" \
  checkpoint.every_n_train_steps="${STAGE2_CKPT_INTERVAL}" \
  trainer.val_check_interval="${STAGE2_VAL_INTERVAL}" \
  "system.prompt_processor.prompt=${STAGE2_PROMPT}" \
  "$@"
