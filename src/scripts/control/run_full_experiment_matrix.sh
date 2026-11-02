#!/usr/bin/env bash
# Sequentially run the comprehensive MPC evaluator_benchmark matrix over
# multiple data configurations x environments.
#
# Data configs (4): concat(v3+v4+v5), v3, v4, v5
# Environments (3): livingroom, room1, room2
# => 12 evaluator_benchmark runs, each producing the full
#    01_all_data / 02_candidate_pool_experiments / 03_evaluators / 04_full_mpc
#    tree under saved_model/<dataset_label>/<env>/mpc_experiments/<run>.
#
# Sequential on purpose: this host has 8 cores / ~13GB free RAM, so parallel
# runs on the 97k-row concat dataset would risk OOM.
set -Euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CFG="${DEFAULT_CFG:-${SRC_DIR}/configs/default_setting.yaml}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Pin to GPU1 by default (GPU0 is often shared with other tenants on this host).
# Override with GPU_ID=0 (or "" for CPU) if needed.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
cd "${REPO_ROOT}"

CONCAT="version3_260714_processed,version4_260805_processed,version5_260814_processed"
ENVIRONMENTS=(livingroom room1 room2)

# Per-dataset indoor-temperature upper bound (state_constraints index 4 = 실내온도).
# Set to each dataset's ~median indoor temp so violations stay SHALLOW: this is what
# keeps the MILP tractable. A fixed 26C is unachievable for the hot summer data (rooms
# run 31-40C), so it is violated deeply on every step -> the soft-constraint penalty
# dominates and flattens the objective -> branch-and-bound explodes (~15-38s/solve,
# 66% fallback). Matching the bound to each dataset (v3 May cool -> 27, v4/v5 hot -> 32,
# concat mixed -> 31) restores v3-like shallow violations -> ~0.13s/solve, 0 fallback
# (validated on v5/livingroom). state_constraints.4 is 실내온도(℃) in default_setting.yaml.

# data-config label -> OmegaConf override for that config (data version + temp bound)
declare -A DATA_OPTS=(
  [concat_v3v4v5]="DATASET.data_versions=[${CONCAT}] MPC.state_constraints.4.upper=31.0"
  [v3]="DATASET.data_version=version3_260714_processed DATASET.data_versions=null MPC.state_constraints.4.upper=27.0"
  [v4]="DATASET.data_version=version4_260805_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
  [v5]="DATASET.data_version=version5_260814_processed DATASET.data_versions=null MPC.state_constraints.4.upper=32.0"
)
# singles first (faster -> early results), concat last (largest ~4-8h).
DATA_ORDER=(v5 v3 v4 concat_v3v4v5)

# Relaxed MILP solver settings. The reprocessed/sparse-support data yields many
# near-tied allowed actions in a no-good-cut-restricted feasible region, so proving
# the default 1% optimality gap takes ~38s/solve (hitting the 60s limit) and would
# make the full matrix take weeks. Relaxing to a 5% gap + 15s limit collapses solves
# to ~2.7s (validated on v5/livingroom) with no meaningful quality loss: the affine
# power model's own WAPE (~9%) is far larger than the 5% optimality gap.
SOLVER_OPTS="MPC.solver.mip_relative_gap=0.05 MPC.solver.time_limit_seconds=15"

TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="${REPO_ROOT}/reports/mpc_rerun_${TS}"
mkdir -p "${BATCH_LOG_DIR}"
STATUS_TSV="${BATCH_LOG_DIR}/run_status.tsv"
printf 'idx\tdata_config\tenvironment\tstatus\tstart\tend\tlog\n' > "${STATUS_TSV}"

total=$(( ${#DATA_ORDER[@]} * ${#ENVIRONMENTS[@]} ))
idx=0
ok=0
fail=0
printf '[batch] %s runs total | log dir: %s\n' "${total}" "${BATCH_LOG_DIR}"

for data_cfg in "${DATA_ORDER[@]}"; do
  for env in "${ENVIRONMENTS[@]}"; do
    idx=$((idx + 1))
    log_file="${BATCH_LOG_DIR}/${idx}_${data_cfg}_${env}.log"
    start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '\n===== [%d/%d] data=%s env=%s | start=%s =====\n' \
      "${idx}" "${total}" "${data_cfg}" "${env}" "${start_ts}"
    # shellcheck disable=SC2086
    if "${PYTHON_BIN}" "${SRC_DIR}/main.py" \
        --task evaluator_benchmark \
        --default_cfg "${DEFAULT_CFG}" \
        --opts \
        DATASET.environment_name="${env}" \
        ${DATA_OPTS[$data_cfg]} \
        ${SOLVER_OPTS} \
        > "${log_file}" 2>&1; then
      status="OK"; ok=$((ok + 1))
    else
      status="FAIL"; fail=$((fail + 1))
    fi
    end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${idx}" "${data_cfg}" "${env}" "${status}" "${start_ts}" "${end_ts}" "${log_file}" \
      >> "${STATUS_TSV}"
    printf '===== [%d/%d] %s env=%s -> %s | end=%s =====\n' \
      "${idx}" "${total}" "${data_cfg}" "${env}" "${status}" "${end_ts}"
  done
done

printf '\n[batch] done: %d OK, %d FAIL (of %d). status: %s\n' \
  "${ok}" "${fail}" "${total}" "${STATUS_TSV}"
