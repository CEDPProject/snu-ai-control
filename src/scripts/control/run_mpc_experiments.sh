#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
GPU_ID="${GPU_ID:-0}"
RUN_FULL_MPC_AFTER_BENCHMARK="${RUN_FULL_MPC_AFTER_BENCHMARK:-1}"

if [[ ! -f "${DEFAULT_CFG}" ]]; then
  printf 'MPC configuration not found: %s\n' "${DEFAULT_CFG}" >&2
  exit 2
fi
if [[ "${RUN_FULL_MPC_AFTER_BENCHMARK}" != "0" && \
      "${RUN_FULL_MPC_AFTER_BENCHMARK}" != "1" ]]; then
  printf 'RUN_FULL_MPC_AFTER_BENCHMARK must be 0 or 1\n' >&2
  exit 2
fi

if [[ "${GPU_ID}" == "-1" || "${GPU_ID}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  DEVICE_OVERRIDE="cpu"
  COMPUTE_LABEL="CPU"
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  DEVICE_OVERRIDE="auto"
  COMPUTE_LABEL="GPU ${GPU_ID} (visible as cuda:0 inside the process)"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

printf 'Running MPC evaluator/action-scope experiment matrix\n'
printf '  config: %s\n' "${DEFAULT_CFG}"
printf '  neural compute: %s\n' "${COMPUTE_LABEL}"
printf '  full MPC after validation selection: %s\n' \
  "${RUN_FULL_MPC_AFTER_BENCHMARK}"
printf '  evaluation scopes: all test rows; candidate pools 7/9/10/12/13/17/41\n'
printf '  cumulative sweep: validation-selected Top-1 through Top-41\n'
printf '  artifacts: one unique run root with 01_all_data, 02_candidate_pool_experiments, 03_evaluators, 04_full_mpc/\n'
printf '  extra arguments are OmegaConf KEY=VALUE overrides\n'

exec "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
  --task evaluator_benchmark \
  --default_cfg "${DEFAULT_CFG}" \
  --opts \
  "MPC.independent_evaluator.neural_residual.device=${DEVICE_OVERRIDE}" \
  "MPC.independent_evaluator.forecasting.device=${DEVICE_OVERRIDE}" \
  "MPC.evaluator_benchmark.run_full_mpc_for_selected=${RUN_FULL_MPC_AFTER_BENCHMARK}" \
  "$@"
