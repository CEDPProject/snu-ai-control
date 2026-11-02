#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve every path from this file so the command works from any directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"

if [[ ! -f "${DEFAULT_CFG}" ]]; then
  printf 'MPC configuration not found: %s\n' "${DEFAULT_CFG}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# RESULT.savedir in the default configuration is relative. Running at the
# repository root keeps artifacts under <repo>/saved_model consistently.
cd "${REPO_ROOT}"

printf 'Running the basic MPC configuration\n'
printf '  config: %s\n' "${DEFAULT_CFG}"
printf '  results: %s/saved_model/<data_version>/<environment>/mpc/<exp_name>\n' \
  "${REPO_ROOT}"
printf '  logs: <results>/logs/mpc_<timestamp>_<pid>.log\n'
printf '  post-evaluation: Full guard MPC action trace + all 3 report-model input cases\n'

# No MPC option is overridden here: default_setting.yaml remains the single
# source of truth. Its default is 06 Full guard, and after MPC it reuses the
# saved action trace for all three report-model input cases. Extra CLI
# arguments may be appended, for example:
#   ./run_basic_mpc.sh --opts MPC.evaluation.max_windows=1
# To skip only the post-MPC input sweep during a smoke test:
#   ./run_basic_mpc.sh --opts MPC.counterfactual_report_input_ablation.run_after_mpc=false
exec "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
  --task mpc \
  --default_cfg "${DEFAULT_CFG}" \
  "$@"
