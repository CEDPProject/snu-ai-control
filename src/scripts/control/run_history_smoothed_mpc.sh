#!/usr/bin/env bash
# Full-guard MPC with a five-minute applied-action history regularizer.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
LOOKBACK_STEPS="${LOOKBACK_STEPS:-5}"
# K is a hard categorical-variable distance from the previous applied action.
# Keep 7 as the legacy unrestricted-history default; set HARD_DISTANCE=1 or 2
# for a deployable conservative policy.
HARD_DISTANCE="${HARD_DISTANCE:-7}"
GPU_ID="${GPU_ID:-cpu}"

if [[ ! -f "${DEFAULT_CFG}" ]]; then
  printf 'MPC configuration not found: %s\n' "${DEFAULT_CFG}" >&2
  exit 2
fi
if [[ ! "${LOOKBACK_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LOOKBACK_STEPS must be a positive integer, got: %s\n' "${LOOKBACK_STEPS}" >&2
  exit 2
fi
if [[ ! "${HARD_DISTANCE}" =~ ^[0-9]+$ ]] || (( HARD_DISTANCE > 7 )); then
  printf 'HARD_DISTANCE must be an integer in [0, 7], got: %s\n' "${HARD_DISTANCE}" >&2
  exit 2
fi
if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  # HiGHS MILP and ExtraTrees are CPU-bound. This only isolates optional
  # neural evaluator work when a GPU-backed family is explicitly selected.
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

printf 'Running Full guard MPC with applied-action history smoothing\n'
printf '  lookback: %s minute(s)\n' "${LOOKBACK_STEPS}"
printf '  first-action categorical hard distance K: %s\n' "${HARD_DISTANCE}"
printf '  history: recent applied MPC actions; five-minute rolling equipment budget\n'

# The values in default_setting.yaml define the penalty, recency weights,
# device-level groups, and rolling budget. Extra CLI options can override any
# of them, for example:
#   LOOKBACK_STEPS=10 HARD_DISTANCE=2 ./run_history_smoothed_mpc.sh \
#     MPC.objective.action_history.penalty=0.8
exec "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
  --task mpc \
  --default_cfg "${DEFAULT_CFG}" \
  --opts \
  "MPC.objective.maximum_changed_control_variables_per_step=${HARD_DISTANCE}" \
  "MPC.objective.action_history.enabled=true" \
  "MPC.objective.action_history.lookback_steps=${LOOKBACK_STEPS}" \
  "$@"
