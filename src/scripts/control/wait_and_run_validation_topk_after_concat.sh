#!/usr/bin/env bash
# Wait for the current concat MPC run to finish both its action trace and the
# automatic three-input report ablation, then launch the Top-K matrix once.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

RUN_LABEL="${RUN_LABEL:-full_guard_component_report_matrix_20260908}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
CONCAT_LABEL="version3_260714_processed__version4_260805_processed__version5_260814_processed"
CONCAT_REFERENCE="${CONCAT_REFERENCE:-${REPO_ROOT}/saved_model/${CONCAT_LABEL}/${ENVIRONMENT}/mpc/${RUN_LABEL}_concat}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if (( POLL_SECONDS <= 0 || POLL_SECONDS > 60 )); then
  echo "POLL_SECONDS must be in 1..60" >&2
  exit 2
fi

while true; do
  manifest="${CONCAT_REFERENCE}/mpc_counterfactual_report_input_ablation_manifest.json"
  trace="${CONCAT_REFERENCE}/mpc_counterfactual_total_power_report_predictions.csv"
  if [[ -s "${manifest}" && -s "${trace}" ]]; then
    echo "[watcher] concat MPC and report-input ablation are complete: $(date --iso-8601=seconds)"
    break
  fi
  echo "[watcher] waiting for concat MPC/report-input ablation: $(date --iso-8601=seconds)"
  sleep "${POLL_SECONDS}"
done

exec bash "${SCRIPT_DIR}/run_validation_topk_counterfactual_report_matrix.sh"
