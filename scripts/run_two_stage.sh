#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-silver_haired_scientist_portrait}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
STAGE1_PROMPT="${STAGE1_PROMPT:-a neutral photorealistic human head portrait, realistic skin, natural face, studio lighting}"
STAGE2_PROMPT="${STAGE2_PROMPT:-a photorealistic DSLR portrait of a distinguished middle-aged scientist, silver hair, calm expression, realistic skin pores, subtle wrinkles, cinematic rim lighting, 85mm lens, shallow depth of field, studio backdrop}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-4000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-10000}"
STAGE1_CKPT_INTERVAL="${STAGE1_CKPT_INTERVAL:-1000}"
STAGE1_VAL_INTERVAL="${STAGE1_VAL_INTERVAL:-1000}"
STAGE2_CKPT_INTERVAL="${STAGE2_CKPT_INTERVAL:-2000}"
STAGE2_VAL_INTERVAL="${STAGE2_VAL_INTERVAL:-2000}"
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

bash scripts/run_stage2_text.sh \
  tag="${RUN_TAG}" \
  timestamp="${RUN_TS}" \
  trainer.max_steps="${STAGE2_MAX_STEPS}" \
  checkpoint.every_n_train_steps="${STAGE2_CKPT_INTERVAL}" \
  trainer.val_check_interval="${STAGE2_VAL_INTERVAL}" \
  "system.prompt_processor.prompt=${STAGE2_PROMPT}" \
  "$@"
