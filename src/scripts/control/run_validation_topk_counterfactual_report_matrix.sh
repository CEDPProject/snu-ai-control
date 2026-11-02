#!/usr/bin/env bash
# Validation Top-K(7..50) sensitivity analysis for four completed Full-guard
# MPC traces.  This script never invokes main.py --task mpc: it reuses each
# frozen MPC action trace and its three completed counterfactual report-input
# prediction files.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
GPU_ID="${GPU_ID:-1}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
RUN_LABEL="${RUN_LABEL:-full_guard_component_report_matrix_20260908}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/reports/artifacts/${RUN_LABEL}}"
TOP_K_MIN="${TOP_K_MIN:-7}"
TOP_K_MAX="${TOP_K_MAX:-50}"

V3_REFERENCE="${V3_REFERENCE:-${REPO_ROOT}/saved_model/version3_260714_processed/${ENVIRONMENT}/mpc/action_constrained_v3}"
V4_REFERENCE="${V4_REFERENCE:-${REPO_ROOT}/saved_model/version4_260805_processed/${ENVIRONMENT}/mpc/action_constrained_v4}"
V5_REFERENCE="${V5_REFERENCE:-${REPO_ROOT}/saved_model/version5_260814_processed/${ENVIRONMENT}/mpc/action_constrained_v5}"
CONCAT_VERSIONS="version3_260714_processed,version4_260805_processed,version5_260814_processed"
CONCAT_LABEL="version3_260714_processed__version4_260805_processed__version5_260814_processed"
CONCAT_REFERENCE="${CONCAT_REFERENCE:-${REPO_ROOT}/saved_model/${CONCAT_LABEL}/${ENVIRONMENT}/mpc/${RUN_LABEL}_concat}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACT_DIR}/validation_topk_7_to_50}"

if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local label="$1"
  local versions="$2"
  local reference="$3"
  local output="${OUTPUT_ROOT}/${label}"
  local log_file="${OUTPUT_ROOT}/${label}.log"

  # Older v3/v4/v5 references predate the sidecar manifest.  The Python
  # runner safely discovers their ablation by its recorded reference path, so
  # only the two immutable MPC artifacts are required here.
  if [[ ! -f "${reference}/mpc_evaluation.json" || \
        ! -f "${reference}/mpc_counterfactual_total_power_report_predictions.csv" ]]; then
    printf '[%s] incomplete frozen MPC/report inputs: %s\n' "${label}" "${reference}" \
      | tee "${log_file}" >&2
    return 1
  fi
  printf '[%s] validation Top-K %s..%s start: %s\n' \
    "${label}" "${TOP_K_MIN}" "${TOP_K_MAX}" "$(date --iso-8601=seconds)" \
    | tee "${log_file}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validation_topk_counterfactual_report_matrix.py" \
    --default-cfg "${DEFAULT_CFG}" \
    --environment "${ENVIRONMENT}" \
    --data-versions "${versions}" \
    --reference-mpc-dir "${reference}" \
    --output-dir "${output}" \
    --top-k-min "${TOP_K_MIN}" \
    --top-k-max "${TOP_K_MAX}" \
    2>&1 | tee -a "${log_file}"
  printf '[%s] complete: %s\n' "${label}" "$(date --iso-8601=seconds)" \
    | tee -a "${log_file}"
}

printf 'Validation Top-K counterfactual report matrix\n'
printf '  K=%s..%s | environment=%s | CUDA_VISIBLE_DEVICES=%s\n' \
  "${TOP_K_MIN}" "${TOP_K_MAX}" "${ENVIRONMENT}" "${CUDA_VISIBLE_DEVICES:-<cpu>}"
printf '  Note: this is report aggregation/refitting on CPU; MILP is not re-solved.\n'

run_one version3_260714_processed version3_260714_processed "${V3_REFERENCE}"
run_one version4_260805_processed version4_260805_processed "${V4_REFERENCE}"
run_one version5_260814_processed version5_260814_processed "${V5_REFERENCE}"
run_one concat "${CONCAT_VERSIONS}" "${CONCAT_REFERENCE}"

printf 'Validation Top-K matrix complete: %s\n' "${OUTPUT_ROOT}"
