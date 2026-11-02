#!/usr/bin/env bash
set -euo pipefail

# Report model 입력 묶음 ablation:
# 1) 현재 상태 + action + 현재 외란
# 2) 현재 상태 + action
# 3) action만
#
# MILP를 재실행하지 않고, 시간순 test의 기록 action에서 환기/냉난방 component
# MAPE를 비교한다. 결과는 saved_model/.../report_model_input_ablation/...에 저장된다.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

python src/main.py \
  --task report_model_input_ablation \
  --default_cfg src/configs/default_setting.yaml \
  --opts \
  "DEFAULT.exp_name=report_model_input_ablation"
