#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python src/main.py --task control_benchmark --opts \
  DEFAULT.exp_name=six_control_methods \
  DATASET.datadir="$ROOT/../data/kweather" \
  DATASET.data_version=version4_260805_processed \
  "$@"
