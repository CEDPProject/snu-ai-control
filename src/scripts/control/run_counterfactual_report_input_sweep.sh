#!/usr/bin/env bash
# Reuse one completed MPC action trace and sweep only report-model inputs.
# This is intentionally not a second MPC policy experiment: MILP is not
# re-solved, so all three cases use exactly the same rows and MPC actions.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
DATA_VERSION="${DATA_VERSION:-version3_260714_processed}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
GPU_ID="${GPU_ID:-1}"
REFERENCE_MPC_DIR="${REFERENCE_MPC_DIR:?Set REFERENCE_MPC_DIR to a completed MPC result directory}"
EXP_NAME="${EXP_NAME:-counterfactual_report_input_sweep}"

if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" "${SRC_DIR}/main.py" \
  --task counterfactual_report_input_ablation \
  --default_cfg "${DEFAULT_CFG}" \
  --opts \
  "DATASET.data_version=${DATA_VERSION}" \
  "DATASET.data_versions=null" \
  "DATASET.environment_name=${ENVIRONMENT}" \
  "DEFAULT.exp_name=${EXP_NAME}" \
  "MPC.counterfactual_report_input_ablation.reference_mpc_result_dir=${REFERENCE_MPC_DIR}" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]"
