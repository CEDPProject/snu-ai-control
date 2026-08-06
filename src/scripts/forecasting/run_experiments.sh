#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
MODEL_CFG="${MODEL_CFG:-${SRC_DIR}/configs/model_setting.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/saved_model}"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-${REPO_ROOT}/reports/forecasting}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-forecast_60min}"

MODELS="${MODELS:-DLinear iTransformer MICN PatchTST TimesNet TimeXer}"
ENVIRONMENTS="${ENVIRONMENTS:-room1 room2 livingroom}"
TARGET_KEYS="${TARGET_KEYS:-ventilation hvac}"

DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
MAX_RUNS="${MAX_RUNS:-0}"
EPOCHS="${EPOCHS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

interrupted=0
handle_interrupt() {
  interrupted=1
}
trap handle_interrupt INT TERM

read -r -a model_list <<< "${MODELS}"
read -r -a environment_list <<< "${ENVIRONMENTS}"
read -r -a target_key_list <<< "${TARGET_KEYS}"

target_name() {
  case "$1" in
    ventilation) printf '%s' '환기유효전력(W)' ;;
    hvac) printf '%s' '냉난방유효전력(W)' ;;
    *)
      printf 'Unknown target key: %s (use ventilation or hvac)\n' "$1" >&2
      return 2
      ;;
  esac
}

timestamp="$(date '+%Y%m%d_%H%M%S')"
log_dir="${REPORT_OUTPUT_DIR}/logs/${timestamp}"
mkdir -p "${log_dir}"
run_log="${log_dir}/run_status.tsv"
printf 'run\tmodel\tenvironment\ttarget_key\ttarget\tstatus\tlog\n' > "${run_log}"

total_runs=$((${#model_list[@]} * ${#environment_list[@]} * ${#target_key_list[@]}))
if (( MAX_RUNS > 0 && MAX_RUNS < total_runs )); then
  planned_runs="${MAX_RUNS}"
else
  planned_runs="${total_runs}"
fi

printf 'Forecast experiment matrix: %s run(s)\n' "${planned_runs}"
printf 'Models: %s\nEnvironments: %s\nTargets: %s\n' "${MODELS}" "${ENVIRONMENTS}" "${TARGET_KEYS}"
printf 'Input/forecast: 120/60 minute(s), split: 8:1:1, stride: 1\n'
printf 'Results: %s\nLogs: %s\n' "${RESULTS_DIR}" "${log_dir}"

run_number=0
success_count=0
failure_count=0
stop_requested=0

for model in "${model_list[@]}"; do
  for environment in "${environment_list[@]}"; do
    for target_key in "${target_key_list[@]}"; do
      if (( MAX_RUNS > 0 && run_number >= MAX_RUNS )); then
        stop_requested=1
        break
      fi

      target="$(target_name "${target_key}")"
      run_number=$((run_number + 1))
      log_file="${log_dir}/$(printf '%02d' "${run_number}")_${model}_${environment}_${target_key}.log"

      command=(
        "${PYTHON_BIN}" "${SRC_DIR}/main.py"
        --task forecast
        --model_name "${model}"
        --default_cfg "${DEFAULT_CFG}"
        --model_cfg "${MODEL_CFG}"
        --opts
        "DEFAULT.exp_name=${EXPERIMENT_NAME}"
        "DATASET.environment_name=${environment}"
        "DATASET.target_features=['${target}']"
        "DATASET.seq_len=120"
        "DATASET.pred_len=60"
        "DATASET.stride_len=1"
        "DATASET.split_rate=[0.8,0.1,0.1]"
        "RESULT.savedir=${RESULTS_DIR}"
      )
      if [[ -n "${EPOCHS}" ]]; then
        command+=("TRAIN.epoch=${EPOCHS}")
      fi

      printf '\n[%d/%d] %s / %s / %s\n' "${run_number}" "${planned_runs}" "${model}" "${environment}" "${target}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        status_row=(
          "${run_number}" "${model}" "${environment}" "${target_key}"
          "${target}" "dry-run" "${log_file}"
        )
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${status_row[@]}" >> "${run_log}"
        continue
      fi

      if "${command[@]}" 2>&1 | tee "${log_file}"; then
        success_count=$((success_count + 1))
        status="success"
      else
        failure_count=$((failure_count + 1))
        status="failed"
      fi
      status_row=(
        "${run_number}" "${model}" "${environment}" "${target_key}"
        "${target}" "${status}" "${log_file}"
      )
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${status_row[@]}" >> "${run_log}"

      if (( interrupted == 1 )); then
        printf '\nInterrupted: stopping the experiment matrix after run %d.\n' "${run_number}" >&2
        stop_requested=1
        break
      fi

      if [[ "${status}" == "failed" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
        stop_requested=1
        break
      fi

    done
    (( stop_requested == 1 )) && break
  done
  (( stop_requested == 1 )) && break
done

if (( interrupted == 1 )); then
  exit 130
fi

if [[ "${RUN_REPORT}" == "1" ]]; then
  report_command=(
    "${SCRIPT_DIR}/build_report.sh"

    --config "${DEFAULT_CFG}"
    --results-dir "${RESULTS_DIR}"
    --output-dir "${REPORT_OUTPUT_DIR}"
  )
  "${report_command[@]}"
fi

printf '\nCompleted: %d, failed: %d, status file: %s\n' "${success_count}" "${failure_count}" "${run_log}"
if (( failure_count > 0 )); then
  exit 1
fi
