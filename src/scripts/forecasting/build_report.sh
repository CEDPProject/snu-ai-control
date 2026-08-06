#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${SRC_DIR}/configs/default_setting.yaml"
RESULTS_DIR="${REPO_ROOT}/saved_model"
OUTPUT_DIR="${REPO_ROOT}/reports/forecasting"
MODELS=()
ENVIRONMENTS=()
TARGET_KEYS=()

usage() {
  usage_lines=(
    "Usage: build_report.sh [options]"
    "  --config PATH"
    "  --results-dir PATH"
    "  --output-dir PATH"
    "  --models MODEL..."
    "  --environments ENV..."
    "  --target-keys ventilation|hvac ..."
  )
  printf '%s\n' "${usage_lines[@]}"
}

while (( $# > 0 )); do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --models)
      shift
      while (( $# > 0 )) && [[ "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --environments)
      shift
      while (( $# > 0 )) && [[ "$1" != --* ]]; do
        ENVIRONMENTS+=("$1")
        shift
      done
      ;;
    --target-keys)
      shift
      while (( $# > 0 )) && [[ "$1" != --* ]]; do
        TARGET_KEYS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command=(
  "${PYTHON_BIN}" "${SRC_DIR}/reporting/forecasting_report.py"
  --config "${CONFIG}"
  --results-dir "${RESULTS_DIR}"
  --output-dir "${OUTPUT_DIR}"
)
(( ${#MODELS[@]} > 0 )) && command+=(--models "${MODELS[@]}")
(( ${#ENVIRONMENTS[@]} > 0 )) && command+=(--environments "${ENVIRONMENTS[@]}")
(( ${#TARGET_KEYS[@]} > 0 )) && command+=(--target-keys "${TARGET_KEYS[@]}")
"${command[@]}"

data_version="$("${PYTHON_BIN}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["DATASET"]["data_version"])' "${CONFIG}")"
report_dir="${OUTPUT_DIR}/${data_version}"
artifact_path="${report_dir}/artifact.json"
html_path="${report_dir}/forecasting_report.html"

default_builder=""
builder="${PORTABLE_REPORT_BUILDER:-${default_builder}}"
if command -v node >/dev/null 2>&1 && [[ -f "${builder}" ]]; then
  delivery_output="$(node "${builder}" --input "${artifact_path}" --output "${html_path}")"
  printf '%s\n' "${delivery_output}"
  if [[ "${delivery_output}" != *'"ok":true'* || ! -s "${html_path}" ]]; then
    printf 'Portable HTML delivery failed. Artifact: %s\n' "${artifact_path}" >&2
    exit 1
  fi
  printf 'Portable HTML: %s\n' "${html_path}"
else
  builder_message=(
    "Portable HTML builder not found; CSV, Markdown, and artifact.json were generated."
    "Set PORTABLE_REPORT_BUILDER to deliver_portable_artifact.mjs to regenerate HTML."
  )
  printf '%s\n' "${builder_message[@]}" >&2
fi
