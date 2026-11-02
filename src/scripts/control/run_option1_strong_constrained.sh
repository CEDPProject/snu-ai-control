#!/usr/bin/env bash
# 옵션 1(강한 제약): MPC를 "학습 support가 충분한 관측 action" 안으로 강하게 밀어넣어,
# MPC가 고른 action이 과거 관측과 자주 매칭되게 한다. 목적은 MPC-선택 action의
# crossfit(실제-Y 매칭) 평가가 reportable해지고 KPI(≤1.95%)에 드는지 확인하는 것.
#
# 논리: report model은 반사실(취하지 않은 action)의 전력을 예측하려고 학습한 것이고,
#       그 신뢰도는 관측 action에서 검증(candidate-pool, 이미 통과)된다. 옵션 1은
#       MPC를 그 "검증된 support 영역" 안에 묶어, MPC 결정이 검증된 예측에 근거하게 한다.
#       제약을 강하게 줄수록 MPC 자유도(제어 의미)는 줄어드는 대가가 있다.
#
# 약한 제약(penalty 2.0/switch 0.5)에서는 crossfit이 v3/v4/v5 모두 None이었다.
# 여기서는 penalty 5.0/switch 2.0으로 강화한다. restrict_to_training_control_support는
# 기본 ON.
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
HISTORY_PENALTY="${HISTORY_PENALTY:-5.0}"
SWITCH_PENALTY="${SWITCH_PENALTY:-2.0}"
CONCAT="version3_260714_processed,version4_260805_processed,version5_260814_processed"

# support가 좋은 구성만: v4(단일 긴 데이터), concat(pooled)
declare -A DATA_OPTS=(
  [v4]="DATASET.data_version=version4_260805_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [concat]="DATASET.data_versions=[${CONCAT}] MPC.state_constraints.4.upper=31.0"
)
DATA_ORDER=(v4 concat)

COMMON_OPTS="MPC.evaluation.action_trust.enabled=true \
MPC.evaluation.restrict_to_training_control_support=true \
MPC.objective.action_history.enabled=true \
MPC.objective.action_history.penalty=${HISTORY_PENALTY} \
MPC.objective.switch_penalty=${SWITCH_PENALTY} \
MPC.component_kpi_gate.enabled=true \
MPC.component_kpi_gate.mape_floor_w=5.0 \
MPC.solver.mip_relative_gap=0.05 \
MPC.solver.time_limit_seconds=15"

TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="${REPO_ROOT}/reports/option1_strong_${TS}"
mkdir -p "${BATCH_LOG_DIR}"
STATUS="${BATCH_LOG_DIR}/run_status.tsv"
printf 'idx\tdata\tenv\tstatus\tstart\tend\tlog\n' > "${STATUS}"
printf '옵션1 강한 제약: history_penalty=%s switch_penalty=%s | data=%s\n' \
  "${HISTORY_PENALTY}" "${SWITCH_PENALTY}" "${DATA_ORDER[*]}"

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
      DEFAULT.exp_name="option1_strong_${data_cfg}" \
      > "${log}" 2>&1; then
    status="OK"
  else
    status="FAIL"
  fi
  end="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${idx}" "${data_cfg}" "${ENVIRONMENT}" "${status}" "${start}" "${end}" "${log}" >> "${STATUS}"
  printf '===== [%d/%d] %s -> %s =====\n' "${idx}" "${#DATA_ORDER[@]}" "${data_cfg}" "${status}"
done
printf '\n[batch] done. status: %s\n' "${STATUS}"
