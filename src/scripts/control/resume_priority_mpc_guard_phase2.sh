#!/usr/bin/env bash
# Resume the interrupted priority MPC queue without repeating completed work.
#
# 1. Rebuild the three report-input summaries for the already-completed v3
#    K=1/K=2 action traces (no MILP solve).
# 2. Run the same Full-guard K=1/K=2 + frozen report-action-effect gate for
#    v4, v5, and concat in that order.  Each distance run performs its own
#    report-input sweep only after its MPC trace has been written.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
GPU_ID="${GPU_ID:-1}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
RUN_LABEL="${RUN_LABEL:-priority_mpc_guard_resume_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/reports/artifacts/${RUN_LABEL}}"
DATASETS_CSV="${DATASETS_CSV:-version4_260805_processed,version5_260814_processed,concat}"
V3_ROOT="${V3_ROOT:-${REPO_ROOT}/saved_model/version3_260714_processed/${ENVIRONMENT}/action_distance_ablation/priority_mpc_guard_20260909_141925_02_guard_version3_260714_processed_20260909_145051_431676}"

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

run_v3_saved_trace_sweep() {
  local distance="$1"
  local reference_dir="${V3_ROOT}/distance_k${distance}/mpc"
  local log_file="${ARTIFACT_DIR}/logs/00_v3_k${distance}_saved_trace_report_sweep.log"
  if [[ ! -f "${reference_dir}/mpc_evaluation.json" || ! -f "${reference_dir}/mpc_counterfactual_total_power_report_predictions.csv" ]]; then
    echo "[00][v3 K=${distance}] missing saved trace: ${reference_dir}" | tee "${log_file}"
    return 1
  fi
  echo "[00][v3 K=${distance}] report-input sweep (no MILP) started $(date --iso-8601=seconds)" | tee "${log_file}"
  "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
    --task counterfactual_report_input_ablation \
    --default_cfg "${DEFAULT_CFG}" \
    --opts \
    "DATASET.environment_name=${ENVIRONMENT}" \
    'DATASET.data_version=version3_260714_processed' \
    'DATASET.data_versions=null' \
    "DEFAULT.exp_name=${RUN_LABEL}_00_v3_k${distance}_report_sweep" \
    "MPC.counterfactual_report_input_ablation.reference_mpc_result_dir=${reference_dir}" \
    'MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]' \
    'MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.post_calibration.method=inner_train_component_affine' \
    2>&1 | tee -a "${log_file}"
  echo "[00][v3 K=${distance}] complete $(date --iso-8601=seconds)" | tee -a "${log_file}"
}

run_phase2_dataset() {
  local dataset="$1"
  local log_file="${ARTIFACT_DIR}/logs/02_full_guard_k1_k2_effect_gate_${dataset}.log"
  local -a opts
  mapfile -t opts < <(dataset_opts "${dataset}")
  echo "[02][${dataset}] Full Guard K=1,2 + report-action gate started $(date --iso-8601=seconds)" | tee "${log_file}"
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

echo "Priority MPC phase-2 resume"
echo "  GPU visibility: ${CUDA_VISIBLE_DEVICES:-<cpu>}"
echo "  datasets: ${DATASETS_CSV}"
echo "  logs: ${ARTIFACT_DIR}/logs"

run_v3_saved_trace_sweep 01
run_v3_saved_trace_sweep 02
for dataset in "${DATASETS[@]}"; do
  run_phase2_dataset "${dataset}"
done

echo "Priority MPC phase-2 resume completed: ${ARTIFACT_DIR}"
