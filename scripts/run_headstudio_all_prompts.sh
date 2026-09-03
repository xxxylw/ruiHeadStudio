#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_NAME="$(basename "$0")"
DEFAULT_PYTHON="/home/rui/miniconda3/envs/ruiheadstudio-bnbfix/bin/python"
DEFAULT_NEGATIVE_PROMPT="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

MODE="run"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/outputs/headstudio_batch_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
SERVICE_NAME="${SERVICE_NAME:-ruiheadstudio-all-prompts-${RUN_ID}}"
START_INDEX="${START_INDEX:-1}"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [--run] [CUDA_DEVICE]
  ${SCRIPT_NAME} --systemd [CUDA_DEVICE]
  ${SCRIPT_NAME} --list

Modes:
  --run       Run all prompts serially in the current shell.
  --systemd   Start the serial run as a WSL user systemd background service.
  --list      Print the prompt list without starting training.

Environment overrides:
  RUN_ID       Timestamp-like run id. Default: current time.
  RUN_ROOT     Output root. Default: outputs/headstudio_batch_\${RUN_ID}
  PYTHON_BIN   Python executable. Default: ${DEFAULT_PYTHON}
  START_INDEX  First prompt index to run. Default: 1. Use with an existing RUN_ROOT to resume.

Examples:
  bash scripts/${SCRIPT_NAME} --systemd 0
  RUN_ROOT=/home/rui/of_work/code/ruiHeadStudio/outputs/my_batch bash scripts/${SCRIPT_NAME} --run 0
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--systemd" ]]; then
  MODE="systemd"
  shift
elif [[ "${1:-}" == "--list" ]]; then
  MODE="list"
  shift
elif [[ "${1:-}" == "--run" ]]; then
  MODE="run"
  shift
fi

if [[ "${1:-}" != "" ]]; then
  CUDA_DEVICE="$1"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${MODE}" != "list" ]]; then
  mkdir -p "${RUN_ROOT}" "${ROOT_DIR}/logs" "${MPLCONFIGDIR}"
fi

MASTER_LOG="${RUN_ROOT}/batch.log"
MANIFEST="${RUN_ROOT}/manifest.tsv"
STATUS_FILE="${RUN_ROOT}/status.env"

if [[ "${MODE}" == "systemd" ]]; then
  if ! command -v systemd-run >/dev/null 2>&1; then
    echo "systemd-run not found; run foreground mode instead:" >&2
    echo "  RUN_ROOT='${RUN_ROOT}' bash scripts/${SCRIPT_NAME} --run ${CUDA_DEVICE}" >&2
    exit 1
  fi

  if ! systemd-run --user --unit="${SERVICE_NAME}" --collect \
    --setenv=RUN_ID="${RUN_ID}" \
    --setenv=RUN_ROOT="${RUN_ROOT}" \
    --setenv=CUDA_DEVICE="${CUDA_DEVICE}" \
    --setenv=PYTHON_BIN="${PYTHON_BIN}" \
    --setenv=START_INDEX="${START_INDEX}" \
    --setenv=MPLCONFIGDIR="${MPLCONFIGDIR}" \
    --setenv=HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    --setenv=DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE}" \
    --setenv=TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    /bin/bash "${ROOT_DIR}/scripts/${SCRIPT_NAME}" --run "${CUDA_DEVICE}"; then
    echo "Failed to start ${SERVICE_NAME}.service" >&2
    exit 1
  fi

  cat > "${STATUS_FILE}" <<EOF
service=${SERVICE_NAME}.service
run_root=${RUN_ROOT}
cuda_device=${CUDA_DEVICE}
started_at=$(date --iso-8601=seconds)
EOF
  echo "Started ${SERVICE_NAME}.service"
  echo "Output root: ${RUN_ROOT}"
  echo "Follow log: tail -f '${MASTER_LOG}'"
  echo "Check service: systemctl --user status ${SERVICE_NAME}.service"
  exit 0
fi

if [[ "${MODE}" != "list" && ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${MODE}" != "list" ]]; then
  {
    echo "started_at=$(date --iso-8601=seconds)"
    echo "root_dir=${ROOT_DIR}"
    echo "run_root=${RUN_ROOT}"
    echo "cuda_device=${CUDA_DEVICE}"
    echo "python=${PYTHON_BIN}"
    echo "start_index=${START_INDEX}"
    echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || true)"
    echo "branch=$(git branch --show-current 2>/dev/null || true)"
  } | tee -a "${MASTER_LOG}"

  if [[ "${START_INDEX}" -le 1 || ! -s "${MANIFEST}" ]]; then
    printf "index\ttag\tstatus\tstarted_at\tfinished_at\texit_code\ttrial_dir\tprompt\textra_args\n" > "${MANIFEST}"
  fi
fi

COMMON_ARGS=(
  --config configs/headstudio.yaml
  --train
)

DATA_COMPAT_ARGS=()
if [[ -f "${ROOT_DIR}/logs/main_thor_numpy1_data/talkshow/project_converted_exp.npy" ]]; then
  DATA_COMPAT_ARGS=(
    "data.talkshow_train_path=${ROOT_DIR}/logs/main_thor_numpy1_data/talkshow/project_converted_exp.npy"
    "data.train_pose_inputs=[${ROOT_DIR}/logs/main_thor_numpy1_data/talkshow/project_converted_exp.npy,${ROOT_DIR}/logs/main_thor_numpy1_data/talkshow/synthetic_aug,${ROOT_DIR}/logs/main_thor_numpy1_data/talkvid/per_clip]"
  )
fi

slugify() {
  local value="$1"
  value="$(printf "%s" "${value}" | tr '[:upper:]' '[:lower:]')"
  value="$(printf "%s" "${value}" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
  printf "%s" "${value:0:80}"
}

run_one() {
  local idx="$1"
  local label="$2"
  local prompt="$3"
  shift 3

  if [[ "${idx}" -lt "${START_INDEX}" ]]; then
    return 0
  fi

  local tag
  tag="$(printf "%02d_%s" "${idx}" "$(slugify "${label}")")"
  local trial_dir="${RUN_ROOT}/runs/${tag}"
  local log_file="${RUN_ROOT}/${tag}.log"
  local started_at finished_at exit_code status extra_args
  started_at="$(date --iso-8601=seconds)"
  extra_args="$*"

  {
    echo
    echo "===== ${idx}: ${label} ====="
    echo "started_at=${started_at}"
    echo "tag=${tag}"
    echo "trial_dir=${trial_dir}"
    echo "prompt=${prompt}"
    echo "extra_args=${extra_args}"
  } | tee -a "${MASTER_LOG}" "${log_file}"

  "${PYTHON_BIN}" launch.py \
    "${COMMON_ARGS[@]}" \
    "exp_root_dir=${RUN_ROOT}" \
    "name=runs" \
    "tag=${tag}" \
    "use_timestamp=False" \
    "system.prompt_processor.prompt=${prompt}" \
    "$@" \
    "${DATA_COMPAT_ARGS[@]}" \
    2>&1 | tee -a "${MASTER_LOG}" "${log_file}"

  exit_code="${PIPESTATUS[0]}"
  finished_at="$(date --iso-8601=seconds)"
  if [[ "${exit_code}" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${idx}" "${tag}" "${status}" "${started_at}" "${finished_at}" "${exit_code}" \
    "${trial_dir}" "${prompt}" "${extra_args}" >> "${MANIFEST}"

  {
    echo "finished_at=${finished_at}"
    echo "exit_code=${exit_code}"
    echo "status=${status}"
  } | tee -a "${MASTER_LOG}" "${log_file}"

  return "${exit_code}"
}

list_one() {
  local idx="$1"
  local label="$2"
  local prompt="$3"
  shift 3

  printf "%02d\t%s\t%s\t%s\n" "${idx}" "$(slugify "${label}")" "${prompt}" "$*"
}

dispatch_one() {
  if [[ "${MODE}" == "list" ]]; then
    list_one "$@"
  else
    run_one "$@"
  fi
}

if [[ "${MODE}" == "list" ]]; then
  printf "index\ttag\tprompt\textra_args\n"
fi

overall_status=0
idx=1

dispatch_one "${idx}" "Joker in DC" "a DSLR portrait of Joker in DC, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Kratos in God of War" "a DSLR portrait of Kratos in God of War, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "I am Groot" "a head of I am Groot, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Vincent van Gogh nfsd" "a DSLR portrait of Vincent van Gogh, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.0008 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Batman" "a DSLR portrait of Batman, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True system.loss.lambda_scaling=100.0 || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Two-face in DC in Marvel" "a DSLR portrait of Two-face in DC in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Obama" "a DSLR portrait of Obama, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.guidance_scale=15 system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Elon Musk" "a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.guidance_scale=25 trainer.max_steps=10000 system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Large afro" "a head of a man with a large afro, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Alien" "a head of an alien, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Gandalf" "a DSLR portrait of Gandalf, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Geralt in The Witcher" "a DSLR portrait of Geralt in The Witcher, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True system.loss.lambda_scaling=100.0 || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Doctor Strange" "a DSLR portrait of Doctor Strange, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True system.loss.lambda_scaling=100.0 || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Hulk" "a head of Hulk, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Lionel Messi" "a DSLR portrait of Lionel Messi, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.guidance_scale=50 system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Caesar in Rise of the Planet of the Apes" "a head of Caesar in Rise of the Planet of the Apes, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Vincent van Gogh dsd" "a DSLR portrait of Vincent van Gogh, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_dsd=True system.max_grad=0.001 system.area_relax=True "system.prompt_processor.negative_prompt=${DEFAULT_NEGATIVE_PROMPT}" || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Salvador Dali" "a DSLR portrait of Salvador Dalí, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True "system.prompt_processor.negative_prompt=${DEFAULT_NEGATIVE_PROMPT}" || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Captain America" "a DSLR portrait of Captain America, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Spider Man" "a head of Spider Man, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Dwayne Johnson" "a DSLR portrait of Dwayne Johnson, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Terracotta Army" "a DSLR portrait of Terracotta Army, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Thanos in Marvel" "a head of Thanos in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 trainer.max_steps=10000 system.area_relax=True "system.prompt_processor.negative_prompt=${DEFAULT_NEGATIVE_PROMPT}" || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Thor in Marvel" "a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 trainer.max_steps=10000 system.area_relax=True "system.prompt_processor.negative_prompt=${DEFAULT_NEGATIVE_PROMPT}" || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Leo Tolstoy" "a DSLR portrait of Leo Tolstoy, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True "system.prompt_processor.negative_prompt=${DEFAULT_NEGATIVE_PROMPT}" || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Saul Goodman" "a DSLR portrait of Saul Goodman, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 trainer.max_steps=10000 system.area_relax=True || overall_status=1
idx=$((idx + 1))

dispatch_one "${idx}" "Iron Man" "a head of Iron Man, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True || overall_status=1

if [[ "${MODE}" == "list" ]]; then
  exit "${overall_status}"
fi

{
  echo
  echo "batch_finished_at=$(date --iso-8601=seconds)"
  echo "batch_exit_code=${overall_status}"
  echo "manifest=${MANIFEST}"
  echo "run_root=${RUN_ROOT}"
} | tee -a "${MASTER_LOG}"

exit "${overall_status}"
