#!/usr/bin/env bash
# Full-guard applied-action-history constraint matrix. L=0 is the factual
# recorded-current-action baseline. Each L>0 expands across the configured K,
# history-group mode, and history-penalty values in default_setting.yaml.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"
DATA_VERSION="${DATA_VERSION:-version3_260714_processed}"
ENVIRONMENT="${ENVIRONMENT:-livingroom}"
GPU_ID="${GPU_ID:-1}"

if [[ "${GPU_ID}" == "cpu" || "${GPU_ID}" == "-1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  # MILP/affine/ExtraTrees remain CPU-bound; this isolates optional neural work.
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
  --task action_history_ablation \
  --default_cfg "${DEFAULT_CFG}" \
  --opts \
  "DATASET.data_version=${DATA_VERSION}" \
  "DATASET.data_versions=null" \
  "DATASET.environment_name=${ENVIRONMENT}" \
  "$@"
