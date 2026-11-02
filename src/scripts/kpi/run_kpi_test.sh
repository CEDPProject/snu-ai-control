#!/usr/bin/env bash
# 공인시험용 실행 스크립트 — 도메인 특화 시계열 예측 기술 성능 평가 (T-001)
#
# 시험 데이터(TL-2) 3개 공간(거실·방1·방2)마다 난수 시드 5개(40~44)로 모델을 학습하고,
# 시험 구간 전체를 예측해 공간별·시드별 MSE(표준화)와 공간별 평균·표준편차를 출력한다.
# 목표값과의 비교·적합 판정은 하지 않는다(시험기관이 판정).
#
# 사용 (어느 위치에서 실행해도 됨):
#   bash src/scripts/kpi/run_kpi_test.sh
#
# 바꿀 수 있는 값 (환경 변수):
#   SEEDS="40 41 42 43 44"          학습 난수 시드 목록
#   GPU=1                           사용할 GPU 번호
#   DATA_DIR=<시험 데이터 폴더>      기본: version4_260805_processed
#   OUT_DIR=kpi_release/test_run    결과 폴더 (이미 결과가 있으면 실행하지 않음)
# 예) 다른 시드로 시험:  SEEDS="1 2 3" OUT_DIR=kpi_release/test_run_seed123 bash src/scripts/kpi/run_kpi_test.sh
#
# 출력: <OUT_DIR>/console.txt (화면 출력), kpi_result.json, pred_vs_actual.png, learning_curve.png,
#       logs/ (학습 로그), models/seed<N>/<공간>/ (학습된 모델)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/../data/kweather/version4_260805_processed}"
OUT_DIR="${OUT_DIR:-kpi_release/test_run}"
SEEDS="${SEEDS:-40 41 42 43 44}"
GPU="${GPU:-1}"

cd "$REPO_DIR"
if [ -e "$OUT_DIR/kpi_result.json" ]; then
  echo "결과 폴더에 이전 결과가 있습니다: $OUT_DIR" >&2
  echo "폴더를 옮기거나 OUT_DIR을 다른 경로로 지정한 뒤 다시 실행하세요." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# SEEDS는 공백으로 나뉜 목록이라 따옴표 없이 넘긴다.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" python src/scripts/kpi/run_kpi.py \
  --data-dir "$DATA_DIR" \
  --envs livingroom room1 room2 \
  --out "$OUT_DIR" \
  --seeds $SEEDS 2>&1 | tee "$OUT_DIR/console.txt"
