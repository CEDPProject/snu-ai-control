#!/usr/bin/env bash
# 방안1(action 제약: action_trust + action_history + switch_penalty)
# + 방안2(component_kpi_gate: 환기/냉난방/총전력 MAPE·WAPE ≤target 판정)를
# 4개 데이터 구성 × livingroom 에 순차 적용한다.
#
# 각 run은 counterfactual 예측 직후 component_kpi_gate.{csv,json}을 산출하므로,
# "제어 풀고 → report(실제/MPC action) → 실제 비교"가 성분별로 1.95%를 넘는지
# run 폴더에서 바로 확인할 수 있다.
set -Euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
cd "${REPO_ROOT}"

ENVIRONMENT="${ENVIRONMENT:-livingroom}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"
CONCAT="version3_260714_processed,version4_260805_processed,version5_260814_processed"

# 데이터 구성 -> "data override | 실내온도 상한(median 기준)"
declare -A DATA_OPTS=(
  [v3]="DATASET.data_version=version3_260714_processed DATASET.data_versions=null MPC.state_constraints.4.upper=27.0"
  [v4]="DATASET.data_version=version4_260805_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [v5]="DATASET.data_version=version5_260814_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [concat]="DATASET.data_versions=[${CONCAT}] MPC.state_constraints.4.upper=31.0"
)
DATA_ORDER=(v3 v4 v5 concat)

# 방안1(제약) + 방안2(gate) 공통 옵션
COMMON_OPTS="MPC.evaluation.action_trust.enabled=true \
MPC.objective.action_history.enabled=true \
MPC.objective.action_history.penalty=2.0 \
MPC.objective.switch_penalty=0.5 \
MPC.component_kpi_gate.enabled=true \
MPC.component_kpi_gate.mape_floor_w=5.0 \
MPC.solver.mip_relative_gap=0.05 \
MPC.solver.time_limit_seconds=15"

TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="${REPO_ROOT}/reports/action_constrained_${TS}"
mkdir -p "${BATCH_LOG_DIR}"
STATUS="${BATCH_LOG_DIR}/run_status.tsv"
printf 'idx\tdata\tenv\tstatus\tstart\tend\tgate_summary\tlog\n' > "${STATUS}"

idx=0
for data_cfg in "${DATA_ORDER[@]}"; do
  idx=$((idx + 1))
  log="${BATCH_LOG_DIR}/${idx}_${data_cfg}_${ENVIRONMENT}.log"
  start="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '\n===== [%d/%d] %s/%s | start=%s =====\n' \
    "${idx}" "${#DATA_ORDER[@]}" "${data_cfg}" "${ENVIRONMENT}" "${start}"
  # shellcheck disable=SC2086
  if "${PYTHON_BIN}" "${SRC_DIR}/main.py" --task mpc --default_cfg "${DEFAULT_CFG}" --opts \
      DATASET.environment_name="${ENVIRONMENT}" \
      ${DATA_OPTS[$data_cfg]} \
      ${COMMON_OPTS} \
      MPC.evaluation.max_windows="${MAX_WINDOWS}" \
      DEFAULT.exp_name="action_constrained_${data_cfg}" \
      > "${log}" 2>&1; then
    status="OK"
  else
    status="FAIL"
  fi
  end="$(date '+%Y-%m-%d %H:%M:%S')"
  # gate 요약 추출 (있으면)
  gate=$(grep -oE "component_kpi_gate[^ ]*" "${log}" 2>/dev/null | head -1 || true)
  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${idx}" "${data_cfg}" "${ENVIRONMENT}" "${status}" "${start}" "${end}" "${gate}" "${log}" >> "${STATUS}"
  printf '===== [%d/%d] %s -> %s =====\n' "${idx}" "${#DATA_ORDER[@]}" "${data_cfg}" "${status}"
done
printf '\n[batch] done. status: %s\n' "${STATUS}"
