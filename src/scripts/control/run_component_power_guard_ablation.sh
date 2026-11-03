#!/usr/bin/env bash
# Compare the new component-separated power models and first-action safeguards
# one configuration at a time.  Every run receives a unique exp_name, so no
# existing MPC result directory is overwritten.
#
# Default scope: v3 / livingroom.  Override only when deliberately running a
# different matrix, for example:
#   DATA_VERSION=version4_260805_processed ENVIRONMENT=room1 \
#     ./run_component_power_guard_ablation.sh
#
# The six MPC rows have a cumulative interpretation:
#   01 legacy evaluator/report, no new guard
#   02 component-separated evaluator only, no guard
#   03 component-separated evaluator and report model, no guard
#   04 + train-only global first-action support gate
#   05 + independent evaluator agreement/safety screen
#   06 + local-context gate (when calibratable) and transition cap
#
# A final report-model input ablation runs the three configured input bundles
# (state+action+disturbance, state+action, action-only) with the component
# two-stage report family.  It intentionally does not re-solve the MILP.
#
# Runs are strictly sequential: SciPy MILP and sklearn ExtraTrees are CPU/RAM
# bound, and concurrent runs can exhaust memory.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
DATA_VERSION="${DATA_VERSION:-version3_260714_processed}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
GPU_ID="${GPU_ID:-1}"
RUN_REPORT_INPUT_ABLATION="${RUN_REPORT_INPUT_ABLATION:-1}"

if [[ ! -f "${DEFAULT_CFG}" ]]; then
  printf 'MPC configuration not found: %s\n' "${DEFAULT_CFG}" >&2
  exit 2
fi
if [[ "${RUN_REPORT_INPUT_ABLATION}" != "0" && \
      "${RUN_REPORT_INPUT_ABLATION}" != "1" ]]; then
  printf 'RUN_REPORT_INPUT_ABLATION must be 0 or 1\n' >&2
  exit 2
fi

# The current six MPC runs use CPU-based affine/ExtraTrees/MILP components. Pin
# the visible GPU only so that an accidentally selected neural candidate does
# not contend with GPU 0; it does not accelerate the core MPC solve.
if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  COMPUTE_LABEL="CPU"
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  COMPUTE_LABEL="GPU ${GPU_ID} visible (core MPC remains CPU-bound)"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

timestamp="$(date '+%Y%m%d_%H%M%S')"
batch_label="component_power_guard_ablation_${timestamp}"
batch_dir="${REPO_ROOT}/reports/${batch_label}"
mkdir -p "${batch_dir}"
status_tsv="${batch_dir}/run_status.tsv"
manifest_md="${batch_dir}/experiment_manifest.md"

printf 'order\tname\ttask\tstatus\tstart\tend\tlog\tresult_hint\n' > "${status_tsv}"
printf '%s\n' \
  "# Component power and first-action guard ablation" \
  "" \
  "- Batch: \`${batch_label}\`" \
  "- Data: \`${DATA_VERSION}\` / \`${ENVIRONMENT}\`" \
  "- Execution: strictly sequential; ${COMPUTE_LABEL}" \
  "- Common horizon: the default 60-minute prediction / 10-minute control horizon." \
  "- Common evaluation: default chronological test windows, counterfactual dual-MAPE, and closed-loop backtest." \
  "- Important: every row preserves the same affine MILP and common state model. Only the listed evaluator/report family or gate settings differ." \
  "" \
  "| Order | Name | Delta from preceding row |" \
  "|---:|---|---|" \
  "| 01 | legacy_reference | ExtraTrees residual evaluator + direct total-power report; no new first-action gate or transition cap |" \
  "| 02 | component_evaluator_only | Component-separated evaluator; report model remains direct total-power; guards still off |" \
  "| 03 | component_report_model | Replaces the report model with component two-stage HVAC; guards still off |" \
  "| 04 | global_support_gate | Enables train-only global exact-action gate; local context and model screen off |" \
  "| 05 | model_screen | Adds validation-calibrated affine/evaluator power-agreement and evaluator safety screen |" \
  "| 06 | full_guard | Adds local context gate when enough validation contexts exist and caps changed categorical variables at 3 |" \
  "| 07 | report_input_ablation | No MILP; compares the three report-model input bundles with component two-stage HVAC family |" \
  > "${manifest_md}"

common_opts=(
  "DATASET.data_version=${DATA_VERSION}"
  "DATASET.data_versions=null"
  "DATASET.environment_name=${ENVIRONMENT}"
  "MPC.independent_evaluator.validation_model_families=[]"
)

ok=0
failed=0
run_mpc_case() {
  local order="$1"
  local name="$2"
  shift 2
  local run_name="${batch_label}_${order}_${name}"
  local log_file="${batch_dir}/${order}_${name}.log"
  local start_time
  local end_time
  local status
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '\n===== [%s/06] %s | start=%s =====\n' \
    "${order}" "${name}" "${start_time}"
  if "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
      --task mpc \
      --default_cfg "${DEFAULT_CFG}" \
      --opts \
      "${common_opts[@]}" \
      "DEFAULT.exp_name=${run_name}" \
      "$@" \
      > "${log_file}" 2>&1; then
    status="OK"
    ok=$((ok + 1))
  else
    status="FAIL"
    failed=$((failed + 1))
  fi
  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%s\t%s\tmpc\t%s\t%s\t%s\t%s\t%s\n' \
    "${order}" "${name}" "${status}" "${start_time}" "${end_time}" \
    "${log_file}" \
    "saved_model/${DATA_VERSION}/${ENVIRONMENT}/mpc/${run_name}" \
    >> "${status_tsv}"
  printf '===== [%s/06] %s -> %s | end=%s =====\n' \
    "${order}" "${name}" "${status}" "${end_time}"
}

# Keep the report family fixed inside each comparison row.  This makes the
# component-model effect visible instead of allowing a different family to
# silently win validation and confound the ablation.
run_mpc_case 01 legacy_reference \
  "MPC.independent_evaluator.model_family=extra_trees_residual_state" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[direct_total_extra_trees]" \
  "MPC.evaluation.action_trust.enabled=false" \
  "MPC.evaluation.action_model_screen.enabled=false" \
  "MPC.objective.maximum_changed_control_variables_per_step=null"

run_mpc_case 02 component_evaluator_only \
  "MPC.independent_evaluator.model_family=component_two_stage_hvac" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[direct_total_extra_trees]" \
  "MPC.evaluation.action_trust.enabled=false" \
  "MPC.evaluation.action_model_screen.enabled=false" \
  "MPC.objective.maximum_changed_control_variables_per_step=null"

run_mpc_case 03 component_report_model \
  "MPC.independent_evaluator.model_family=component_two_stage_hvac" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]" \
  "MPC.evaluation.action_trust.enabled=false" \
  "MPC.evaluation.action_model_screen.enabled=false" \
  "MPC.objective.maximum_changed_control_variables_per_step=null"

run_mpc_case 04 global_support_gate \
  "MPC.independent_evaluator.model_family=component_two_stage_hvac" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]" \
  "MPC.evaluation.action_trust.enabled=true" \
  "MPC.evaluation.action_trust.local_context_support.enabled=false" \
  "MPC.evaluation.action_model_screen.enabled=false" \
  "MPC.objective.maximum_changed_control_variables_per_step=null"

run_mpc_case 05 model_screen \
  "MPC.independent_evaluator.model_family=component_two_stage_hvac" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]" \
  "MPC.evaluation.action_trust.enabled=true" \
  "MPC.evaluation.action_trust.local_context_support.enabled=false" \
  "MPC.evaluation.action_model_screen.enabled=true" \
  "MPC.objective.maximum_changed_control_variables_per_step=null"

run_mpc_case 06 full_guard \
  "MPC.independent_evaluator.model_family=component_two_stage_hvac" \
  "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]" \
  "MPC.evaluation.action_trust.enabled=true" \
  "MPC.evaluation.action_trust.local_context_support.enabled=true" \
  "MPC.evaluation.action_trust.local_context_support.minimum_validation_contexts=30" \
  "MPC.evaluation.action_model_screen.enabled=true" \
  "MPC.objective.maximum_changed_control_variables_per_step=3"

if [[ "${RUN_REPORT_INPUT_ABLATION}" == "1" ]]; then
  log_file="${batch_dir}/07_report_input_ablation.log"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '\n===== [07/07] report_input_ablation | start=%s =====\n' "${start_time}"
  if "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
      --task report_model_input_ablation \
      --default_cfg "${DEFAULT_CFG}" \
      --opts \
      "${common_opts[@]}" \
      "DEFAULT.exp_name=${batch_label}_07_report_input_ablation" \
      "MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families=[component_hvac_two_stage_extra_trees]" \
      > "${log_file}" 2>&1; then
    status="OK"
    ok=$((ok + 1))
  else
    status="FAIL"
    failed=$((failed + 1))
  fi
  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '07\treport_input_ablation\treport_model_input_ablation\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${start_time}" "${end_time}" "${log_file}" \
    "saved_model/${DATA_VERSION}/${ENVIRONMENT}/report_model_input_ablation/${batch_label}_07_report_input_ablation_*" \
    >> "${status_tsv}"
  printf '===== [07/07] report_input_ablation -> %s | end=%s =====\n' \
    "${status}" "${end_time}"
fi

printf '\n[batch] complete: %d OK, %d FAIL. Status: %s\n' \
  "${ok}" "${failed}" "${status_tsv}"
exit "${failed}"
