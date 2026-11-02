#!/usr/bin/env bash
# 방안 1(제어 action 제약) + 방안 2(성분별 MAPE/WAPE gate)를 함께 켠 MPC 실행.
#
# 방안 1 — MPC first action을 "학습 support가 충분한 action" 안에서 고르고
#          (action_trust, 기본 ON), 최근 실제 적용 action에서 급격히 벗어나지
#          않도록 action_history 정규화와 switch_penalty를 강하게 준다. 이렇게 하면
#          case A(MPC action → report → 실제)가 case B(실제 action)에 가까워진다.
#          단, action_history/switch를 극단화해 MPC=recorded로 만들면 제어 의미가
#          사라지므로(순환논리), support 내 최적화 여지는 남기는 값을 기본으로 둔다.
# 방안 2 — component_kpi_gate(기본 ON)가 환기/냉난방/총전력을 설정 floor로
#          MAPE·WAPE 계산하고 ≤target pass/fail을 실제 action·MPC action 모두 판정해
#          run 폴더에 component_kpi_gate.{csv,json}으로 남긴다.
#
# 환경변수로 조정:
#   DATA_VERSION (기본 version3_260714_processed), ENVIRONMENT (기본 livingroom),
#   TEMP_UPPER (실내온도 상한, 기본 27), HISTORY_PENALTY (기본 2.0),
#   SWITCH_PENALTY (기본 0.5), GATE_FLOOR_W (기본 5.0), MAX_WINDOWS (기본 0=전체).
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
DATA_VERSION="${DATA_VERSION:-version3_260714_processed}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
TEMP_UPPER="${TEMP_UPPER:-27.0}"
HISTORY_PENALTY="${HISTORY_PENALTY:-2.0}"
SWITCH_PENALTY="${SWITCH_PENALTY:-0.5}"
GATE_FLOOR_W="${GATE_FLOOR_W:-5.0}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
cd "${REPO_ROOT}"

printf '방안1(action 제약)+방안2(component gate) MPC\n'
printf '  data=%s env=%s temp_upper=%s\n' "${DATA_VERSION}" "${ENVIRONMENT}" "${TEMP_UPPER}"
printf '  action_trust=ON, action_history.penalty=%s, switch_penalty=%s\n' \
  "${HISTORY_PENALTY}" "${SWITCH_PENALTY}"
printf '  component_kpi_gate=ON, floor=%sW, target=1.95%%\n' "${GATE_FLOOR_W}"

exec "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
  --task mpc \
  --default_cfg "${DEFAULT_CFG}" \
  --opts \
  DATASET.data_version="${DATA_VERSION}" \
  DATASET.data_versions=null \
  DATASET.environment_name="${ENVIRONMENT}" \
  MPC.state_constraints.4.upper="${TEMP_UPPER}" \
  MPC.evaluation.action_trust.enabled=true \
  MPC.objective.action_history.enabled=true \
  MPC.objective.action_history.penalty="${HISTORY_PENALTY}" \
  MPC.objective.switch_penalty="${SWITCH_PENALTY}" \
  MPC.component_kpi_gate.enabled=true \
  MPC.component_kpi_gate.mape_floor_w="${GATE_FLOOR_W}" \
  MPC.evaluation.max_windows="${MAX_WINDOWS}" \
  MPC.solver.mip_relative_gap=0.05 \
  MPC.solver.time_limit_seconds=15 \
  "$@"
