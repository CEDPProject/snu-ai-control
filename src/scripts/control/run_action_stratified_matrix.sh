#!/usr/bin/env bash
# Run the supplementary action-stratified row evaluation for the same
# v3/v4/v5/concat x environment matrix as the strict B2 experiment.
#
# This does not replace the episode-grouped strict KPI. It gives every
# sufficiently observed exact joint action train/validation/test support using a
# deterministic per-action 8:1:1 row split, then saves independent results.
set -Eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
SRC_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd -- "$SRC_DIR/.." && pwd)"

PYTHON_BIN=python
DEFAULT_CFG="$SRC_DIR/configs/default_setting.yaml"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
# This runner trains only CPU total-power report models. Hide GPUs to avoid
# reserving a shared accelerator unnecessarily.
export CUDA_VISIBLE_DEVICES=""
cd "$REPO_ROOT"

CONCAT="version3_260714_processed,version4_260805_processed,version5_260814_processed"

TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG_DIR="$REPO_ROOT/reports/action_stratified_rerun_$TS"
mkdir -p "$BATCH_LOG_DIR"
STATUS_TSV="$BATCH_LOG_DIR/run_status.tsv"
printf 'idx\tdata_config\tenvironment\tstatus\tstart\tend\tlog\n' > "$STATUS_TSV"

total=12
idx=0
ok=0
fail=0
printf '[batch] %s action-stratified runs total | log dir: %s\n' \
  "$total" "$BATCH_LOG_DIR"

for data_cfg in v5 v3 v4 concat_v3v4v5; do
  case "$data_cfg" in
    v5)
      data_opts="DATASET.data_version=version5_260814_processed DATASET.data_versions=null"
      ;;
    v3)
      data_opts="DATASET.data_version=version3_260714_processed DATASET.data_versions=null"
      ;;
    v4)
      data_opts="DATASET.data_version=version4_260805_processed DATASET.data_versions=null"
      ;;
    concat_v3v4v5)
      data_opts="DATASET.data_versions=[$CONCAT]"
      ;;
    *)
      printf 'Unknown data configuration: %s\n' "$data_cfg" >&2
      exit 2
      ;;
  esac
  for env in livingroom room1 room2; do
    idx=$((idx + 1))
    log_file="$BATCH_LOG_DIR/$idx"_"$data_cfg"_"$env".log
    start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '\n===== [%d/%d] data=%s env=%s | start=%s =====\n' \
      "$idx" "$total" "$data_cfg" "$env" "$start_ts"
    # shellcheck disable=SC2086
    if "$PYTHON_BIN" "$SRC_DIR/main.py" \
        --task action_stratified_evaluation \
        --default_cfg "$DEFAULT_CFG" \
        --opts \
        DATASET.environment_name="$env" \
        $data_opts \
        > "$log_file" 2>&1; then
      status="OK"; ok=$((ok + 1))
    else
      status="FAIL"; fail=$((fail + 1))
    fi
    end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$idx" "$data_cfg" "$env" "$status" "$start_ts" "$end_ts" "$log_file" \
      >> "$STATUS_TSV"
    printf '===== [%d/%d] %s env=%s -> %s | end=%s =====\n' \
      "$idx" "$total" "$data_cfg" "$env" "$status" "$end_ts"
  done
done

printf '\n[batch] done: %d OK, %d FAIL (of %d). status: %s\n' \
  "$ok" "$fail" "$total" "$STATUS_TSV"
