#!/usr/bin/env bash
# Full-guard MPC action trace를 기준으로 component-aware report model의
# 완전 비교 행렬을 실행한다.
#
# 축:
#   data      : v3, v4, v5, concat(v3+v4+v5)
#   input     : state+action+disturbance / state+action / action-only
#   output    : ventilation / HVAC / total (= component sum)
#   comparison: report(MPC) vs recorded actual,
#               report(MPC) vs report(recorded),
#               report(recorded) vs recorded actual (factual baseline)
#
# v3/v4/v5는 이미 완료된 Full-guard MPC first-action trace를 그대로 재사용한다.
# concat은 이전 trace가 완주하지 않았으므로 Full-guard MPC를 한 번 실행한 뒤,
# 그 trace에 대해 같은 3개 report input case를 자동으로 평가한다.
#
# 기본 실행은 GPU 1을 보이도록 CUDA_VISIBLE_DEVICES=1을 지정한다. 단 이 실험의
# SciPy MILP와 scikit-learn ExtraTrees report model은 CPU 연산이므로, GPU 1은
# neural evaluator가 선택될 때에만 실제 산술에 사용된다.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
GPU_ID="${GPU_ID:-1}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
RUN_LABEL="${RUN_LABEL:-full_guard_component_report_matrix_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/reports/artifacts/${RUN_LABEL}}"

V3_REFERENCE="${V3_REFERENCE:-${REPO_ROOT}/saved_model/version3_260714_processed/${ENVIRONMENT}/mpc/action_constrained_v3}"
V4_REFERENCE="${V4_REFERENCE:-${REPO_ROOT}/saved_model/version4_260805_processed/${ENVIRONMENT}/mpc/action_constrained_v4}"
V5_REFERENCE="${V5_REFERENCE:-${REPO_ROOT}/saved_model/version5_260814_processed/${ENVIRONMENT}/mpc/action_constrained_v5}"

if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
mkdir -p "${ARTIFACT_DIR}"

printf 'Full-guard component report matrix\n'
printf '  label=%s\n' "${RUN_LABEL}"
printf '  environment=%s | CUDA_VISIBLE_DEVICES=%s\n' "${ENVIRONMENT}" "${CUDA_VISIBLE_DEVICES:-<cpu>}"
printf '  artifacts=%s\n' "${ARTIFACT_DIR}"
printf '  v3/v4/v5: reuse completed MPC traces; concat: solve a new trace\n'

failures=0

run_reference_sweep() {
  local dataset="$1"
  local reference_dir="$2"
  local log_file="${ARTIFACT_DIR}/${dataset}_report_input_sweep.log"

  if [[ ! -f "${reference_dir}/mpc_evaluation.json" || \
        ! -f "${reference_dir}/mpc_counterfactual_total_power_report_predictions.csv" ]]; then
    printf '[%s] missing completed MPC reference: %s\n' "${dataset}" "${reference_dir}" \
      | tee "${log_file}"
    failures=$((failures + 1))
    return
  fi

  printf '[%s] report-input sweep start: %s\n' "${dataset}" "$(date --iso-8601=seconds)" \
    | tee "${log_file}"
  if env \
      GPU_ID="${GPU_ID}" \
      DATA_VERSION="${dataset}" \
      ENVIRONMENT="${ENVIRONMENT}" \
      REFERENCE_MPC_DIR="${reference_dir}" \
      EXP_NAME="${RUN_LABEL}_${dataset}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      DEFAULT_CFG="${DEFAULT_CFG}" \
      bash "${SCRIPT_DIR}/run_counterfactual_report_input_sweep.sh" \
      2>&1 | tee -a "${log_file}"; then
    printf '[%s] report-input sweep complete: %s\n' "${dataset}" "$(date --iso-8601=seconds)" \
      | tee -a "${log_file}"
  else
    printf '[%s] report-input sweep failed: %s\n' "${dataset}" "$(date --iso-8601=seconds)" \
      | tee -a "${log_file}"
    failures=$((failures + 1))
  fi
}

run_concat_mpc_and_sweep() {
  local versions='[version3_260714_processed,version4_260805_processed,version5_260814_processed]'
  local exp_name="${RUN_LABEL}_concat"
  local log_file="${ARTIFACT_DIR}/concat_full_guard_mpc_and_report_input_sweep.log"

  printf '[concat] Full-guard MPC start: %s\n' "$(date --iso-8601=seconds)" | tee "${log_file}"
  # The prior concat run used a 31 C upper temperature bound.  Retaining it
  # reproduces the only concat policy configuration that was feasible enough
  # to progress.  The generated evaluation JSON retains this exception.
  if "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
      --task mpc \
      --default_cfg "${DEFAULT_CFG}" \
      --opts \
      "DATASET.environment_name=${ENVIRONMENT}" \
      "DATASET.data_versions=${versions}" \
      "MPC.state_constraints.4.upper=31.0" \
      "MPC.evaluation.action_trust.enabled=true" \
      "MPC.objective.action_history.enabled=true" \
      "MPC.objective.action_history.penalty=2.0" \
      "MPC.objective.switch_penalty=0.5" \
      "MPC.component_kpi_gate.enabled=true" \
      "MPC.component_kpi_gate.mape_floor_w=5.0" \
      "MPC.vent_isolated_kpi.enabled=true" \
      "MPC.solver.mip_relative_gap=0.05" \
      "MPC.solver.time_limit_seconds=15" \
      "MPC.evaluation.max_windows=0" \
      "MPC.counterfactual_report_input_ablation.run_after_mpc=true" \
      "DEFAULT.exp_name=${exp_name}" \
      2>&1 | tee -a "${log_file}"; then
    printf '[concat] Full-guard MPC and report-input sweep complete: %s\n' \
      "$(date --iso-8601=seconds)" | tee -a "${log_file}"
  else
    printf '[concat] Full-guard MPC and report-input sweep failed: %s\n' \
      "$(date --iso-8601=seconds)" | tee -a "${log_file}"
    failures=$((failures + 1))
  fi
}

# These three runs reuse the immutable first-action trace, so their three
# input cases have identical rows, x_t, d_t, and z_t^MPC.
run_reference_sweep version3_260714_processed "${V3_REFERENCE}"
run_reference_sweep version4_260805_processed "${V4_REFERENCE}"
run_reference_sweep version5_260814_processed "${V5_REFERENCE}"

# This creates the missing concat trace and automatically runs the same
# counterfactual input ablation after the MILP evaluation is complete.
run_concat_mpc_and_sweep

# The four frozen traces now exist.  Reuse them to aggregate every validation
# Top-K prefix (7..50) for all report input bundles and comparisons.  This is
# deliberately after concat: it never re-solves MILP and cannot alter any
# action trace produced above.
if (( failures == 0 )); then
  env \
    GPU_ID="${GPU_ID}" \
    ENVIRONMENT="${ENVIRONMENT}" \
    RUN_LABEL="${RUN_LABEL}" \
    ARTIFACT_DIR="${ARTIFACT_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DEFAULT_CFG="${DEFAULT_CFG}" \
    bash "${SCRIPT_DIR}/run_validation_topk_counterfactual_report_matrix.sh"
fi

if (( failures > 0 )); then
  printf 'Matrix completed with %d failed stage(s). Inspect %s\n' "${failures}" "${ARTIFACT_DIR}" >&2
  exit 1
fi

printf 'Matrix completed successfully: %s\n' "${ARTIFACT_DIR}"
