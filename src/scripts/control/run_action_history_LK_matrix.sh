#!/usr/bin/env bash
# L×K 매트릭스 보완: K∈{1,2} 열 (K=7은 run_action_history_L_sweep_matrix.sh가 담당).
# L∈{0,1,2,5,10,30} × K∈{1,2} × 전 데이터. L=0은 factual(K 무관) → 데이터당 11 시나리오.
# 완료 후 K=7 결과와 합쳐 전체 L×K 그리드를 만든다.
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

declare -A DATA_OPTS=(
  [v4]="DATASET.data_version=version4_260805_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [v5]="DATASET.data_version=version5_260814_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [v3]="DATASET.data_version=version3_260714_processed DATASET.data_versions=null MPC.state_constraints.4.upper=27.0"
  [concat]="DATASET.data_versions=[${CONCAT}] MPC.state_constraints.4.upper=31.0"
)
DATA_ORDER=(v4 v5 v3 concat)

TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="${REPO_ROOT}/reports/artifacts/action_history_LK_k12_${TS}"
mkdir -p "${BATCH_LOG_DIR}/logs"
STATUS="${BATCH_LOG_DIR}/run_status.tsv"
printf 'idx\tdata\tenv\tstatus\tstart\tend\tlog\n' > "${STATUS}"
printf 'L×K 보완: L=0/1/2/5/10/30, K∈{1,2} | data=%s\n' "${DATA_ORDER[*]}"

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
      "MPC.action_history_ablation.hard_distances=[1,2]" \
      DEFAULT.exp_name="action_history_LK_k12_${data_cfg}" \
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
