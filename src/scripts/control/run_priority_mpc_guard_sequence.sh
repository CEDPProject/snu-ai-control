#!/usr/bin/env bash
# Leakage-safe priority sequence for improving the counterfactual MPC report.
#
# Phase 1: factual report-model calibration, evaluated on recorded actions.
#          No MILP is solved here.
# Phase 2: Full-guard MPC with a first-action categorical trust region K=1/2
#          and a frozen-report action-effect limit of 1.95%.
#
# The Phase-2 report limit is fixed before test execution.  It uses only the
# current x_t, d_t and a report model selected/refit before test; it never sees
# test power targets.  Its total report-to-report MAPE is therefore a policy
# restriction audit, not proof of post-intervention real power accuracy.
#
# This script is intentionally sequential: one dataset/phase completes and
# writes its log before the next one starts.  Override DATASETS_CSV to narrow
# the queue during a diagnostic rerun.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
GPU_ID="${GPU_ID:-1}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
RUN_LABEL="${RUN_LABEL:-priority_mpc_guard_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/reports/artifacts/${RUN_LABEL}}"
DATASETS_CSV="${DATASETS_CSV:-version3_260714_processed,version4_260805_processed,version5_260814_processed,concat}"

if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
mkdir -p "${ARTIFACT_DIR}/logs"

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
if (( ${#DATASETS[@]} == 0 )); then
  echo 'DATASETS_CSV must contain at least one dataset token' >&2
  exit 2
fi

dataset_opts() {
  local dataset="$1"
  if [[ "${dataset}" == "concat" ]]; then
    printf '%s\n' \
      'DATASET.data_version=version3_260714_processed' \
      'DATASET.data_versions=[version3_260714_processed,version4_260805_processed,version5_260814_processed]'
  else
    printf '%s\n' \
      "DATASET.data_version=${dataset}" \
      'DATASET.data_versions=null'
  fi
}

run_phase1() {
  local dataset="$1"
  local log_file="${ARTIFACT_DIR}/logs/01_factual_report_calibration_${dataset}.log"
  local -a opts
  mapfile -t opts < <(dataset_opts "${dataset}")
  echo "[01][${dataset}] factual report calibration started $(date --iso-8601=seconds)" | tee "${log_file}"
  "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
    --task report_model_input_ablation \
    --default_cfg "${DEFAULT_CFG}" \
    --opts \
    "DATASET.environment_name=${ENVIRONMENT}" \
    "${opts[@]}" \
    "DEFAULT.exp_name=${RUN_LABEL}_01_factual_${dataset}" \
    'MPC.report_model_input_ablation.candidate_pool_expansion.enabled=false' \
    'MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.post_calibration.method=inner_train_component_affine' \
    2>&1 | tee -a "${log_file}"
  echo "[01][${dataset}] complete $(date --iso-8601=seconds)" | tee -a "${log_file}"
}

run_phase2() {
  local dataset="$1"
  local log_file="${ARTIFACT_DIR}/logs/02_full_guard_k1_k2_effect_gate_${dataset}.log"
  local -a opts
  mapfile -t opts < <(dataset_opts "${dataset}")
  echo "[02][${dataset}] Full Guard K=1,2 + 1.95% report-action gate started $(date --iso-8601=seconds)" | tee "${log_file}"
  "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
    --task action_distance_ablation \
    --default_cfg "${DEFAULT_CFG}" \
    --opts \
    "DATASET.environment_name=${ENVIRONMENT}" \
    "${opts[@]}" \
    "DEFAULT.exp_name=${RUN_LABEL}_02_guard_${dataset}" \
    'MPC.action_distance_ablation.distances=[1,2]' \
    'MPC.evaluation.report_action_effect_gate.enabled=true' \
    'MPC.evaluation.report_action_effect_gate.unavailable_model_policy=reject' \
    'MPC.evaluation.report_action_effect_gate.maximum_total_relative_change_percent=1.95' \
    'MPC.evaluation.report_action_effect_gate.maximum_component_relative_change_percent=null' \
    'MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.post_calibration.method=inner_train_component_affine' \
    'MPC.evaluation.max_windows=1' \
    2>&1 | tee -a "${log_file}"
  echo "[02][${dataset}] complete $(date --iso-8601=seconds)" | tee -a "${log_file}"
}

echo "Priority MPC guard sequence"
echo "  GPU visibility: ${CUDA_VISIBLE_DEVICES:-<cpu>}"
echo "  datasets: ${DATASETS_CSV}"
echo "  artifacts/logs: ${ARTIFACT_DIR}"

for dataset in "${DATASETS[@]}"; do
  run_phase1 "${dataset}"
done
for dataset in "${DATASETS[@]}"; do
  run_phase2 "${dataset}"
done

echo "All priority stages completed: ${ARTIFACT_DIR}"
