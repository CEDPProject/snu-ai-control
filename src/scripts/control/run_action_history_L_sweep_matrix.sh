#!/usr/bin/env bash
# action-history lookback(L) 스윕 매트릭스 — 전 데이터 순차.
# 러너 기본 그리드(default_setting.yaml): L∈{0,1,2,5,10,30}, K(hard_distance)=7,
# history_group_mode=equipment, penalty=0.5. L=0은 factual baseline(개입 없음).
# 각 L>0은 전체 chronological test에 대해 Full-guard MPC(MILP)를 푼다 → 데이터당 수 시간.
# 빠른 신호를 위해 v4(가장 빠름)부터, concat(가장 느림)은 마지막.
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
CONCAT="version3_260714_processed,version4_260805_processed,version5_260814_processed"

# 데이터 구성 -> data override | 실내온도 상한(median 기준, action_constrained_matrix와 동일)
declare -A DATA_OPTS=(
  [v4]="DATASET.data_version=version4_260805_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [v5]="DATASET.data_version=version5_260814_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [v3]="DATASET.data_version=version3_260714_processed DATASET.data_versions=null MPC.state_constraints.4.upper=27.0"
  [concat]="DATASET.data_versions=[${CONCAT}] MPC.state_constraints.4.upper=31.0"
)
DATA_ORDER=(v4 v5 v3 concat)

TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="${REPO_ROOT}/reports/artifacts/action_history_L_sweep_${TS}"
mkdir -p "${BATCH_LOG_DIR}/logs"
STATUS="${BATCH_LOG_DIR}/run_status.tsv"
printf 'idx\tdata\tenv\tstatus\tstart\tend\tlog\n' > "${STATUS}"
printf 'action-history L 스윕: L=0/1/2/5/10/30, K=7 | data=%s\n' "${DATA_ORDER[*]}"

idx=0
for data_cfg in "${DATA_ORDER[@]}"; do
  idx=$((idx + 1))
  log="${BATCH_LOG_DIR}/logs/${idx}_${data_cfg}_${ENVIRONMENT}.log"
  start="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '\n===== [%d/%d] %s/%s | start=%s =====\n' \
    "${idx}" "${#DATA_ORDER[@]}" "${data_cfg}" "${ENVIRONMENT}" "${start}"
  # shellcheck disable=SC2086
  if "${PYTHON_BIN}" "${SRC_DIR}/main.py" --task action_history_ablation \
      --default_cfg "${DEFAULT_CFG}" --opts \
      DATASET.environment_name="${ENVIRONMENT}" \
      ${DATA_OPTS[$data_cfg]} \
      DEFAULT.exp_name="action_history_L_sweep_${data_cfg}" \
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
