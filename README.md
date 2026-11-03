# SNU-AI-Control

건물 공조(환기·냉난방) 설비의 **전력 예측**과 **이산 제어 최적화(MPC)** 프레임워크.

- `forecast`: 측정 데이터로 미래 전력을 예측하는 딥러닝 프레임워크
- `mpc`: 실내 상태 제약 아래 이산 제어 조합의 전력 소비를 최소화하는 mixed-integer MPC
- 두 프레임워크는 같은 CSV를 쓸 수 있지만 모델·전처리·실행 루프는 공유하지 않음
  - `forecast`: 학습 구간 기준으로 스케일한 데이터 사용
  - `mpc`: 스케일하지 않은 물리 단위 데이터 사용
- 저장소에는 **데이터, 학습된 모델, 실험 결과가 포함되어 있지 않음** → [데이터 형식](#데이터-형식) 참고

## 목차

1. [설치](#설치)
2. [빠른 시작](#빠른-시작)
3. [데이터 형식](#데이터-형식)
4. [예측(forecast)](#예측forecast)
5. [MPC](#mpc)
6. [MPC 평가 프로토콜](#mpc-평가-프로토콜)
7. [제어 방법 6종 비교](#제어-방법-6종-비교)
8. [디렉터리 구조](#디렉터리-구조)

## 설치

```bash
git clone https://github.com/CEDPProject/snu-ai-control.git
cd snu-ai-control
pip install -r requirements.txt
```

- 모든 명령은 **저장소 루트**에서 실행
- MILP solver: SciPy 내장 HiGHS 사용 (상용 solver·`cvxpy` 불필요)
- 테스트: 합성 데이터만 사용하므로 데이터 없이 실행 가능

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## 빠른 시작

```bash
# 예측 모델 학습
python src/main.py --task forecast --model_name PatchTST

# MPC: affine 모델 학습 + 오프라인 폐루프 평가
python src/main.py --task mpc

# 설정 덮어쓰기: --opts 뒤에 KEY=VALUE (값은 YAML로 해석)
python src/main.py --task mpc --opts \
  DATASET.environment_name=room1 \
  MPC.evaluation.max_windows=10
```

| `--task` | 내용 |
|---|---|
| `forecast` | 딥러닝 전력 예측 학습·평가 (`--model_name` 필수) |
| `mpc` | MPC 모델 학습, 검증, 폐루프 정책 평가 |
| `evaluator_benchmark` | evaluator·action 범위 벤치마크 |
| `action_stratified_evaluation` | action 층화 행 단위 보조 평가 |
| `report_model_input_ablation` | 기록 action 기준 report model 입력 ablation |
| `counterfactual_report_input_ablation` | MPC action 기준 report model 입력 ablation |
| `action_distance_ablation`, `action_history_ablation` | action 거리·이력 regularizer ablation |
| `control_benchmark` | 제어 방법 6종 비교 |

- 기본 설정 파일: `src/configs/default_setting.yaml`(데이터·MPC·학습), `src/configs/model_setting.yaml`(예측 모델)
- 결과 경로: `saved_model/<data_version>/<environment>/<task 또는 model>/...`

## 데이터 형식

기본 설정은 "한 공간(방)에 설치된 환기·냉난방 장비의 1분 단위 운전 기록"을 가정함.
열 이름과 제어 변수를 설정 파일에서 바꾸면 비슷한 구조의 건물 HVAC 데이터에 그대로 적용 가능.

### 폴더 구조

```text
<datadir>/
└── <data_version>/              # 예: site_a_2025summer
    ├── <environment_name>.csv   # 공간 1개 = CSV 1개 (예: livingroom.csv)
    ├── room1.csv
    └── ...
```

| 설정 | 기본값 | 설명 |
|---|---|---|
| `DATASET.datadir` | `../../data/kweather/` | 데이터 루트. 절대경로 → 실행 위치 기준 → `src/` 기준 순으로 탐색 |
| `DATASET.data_version` | `version3_260714_processed` | `<datadir>` 아래 하위 폴더 |
| `DATASET.environment_name` | `livingroom` | CSV 파일 이름(확장자 제외) |
| `DATASET.data_versions` | `null` | 여러 수집 회차를 함께 쓸 때의 폴더 목록 |

- 데이터는 저장소 밖에 두고 실행 시 덮어쓰기 권장

  ```bash
  python src/main.py --task forecast --model_name DLinear --opts \
    DATASET.datadir=/path/to/data DATASET.data_version=site_a DATASET.environment_name=room1
  ```

- 여러 회차 사용(`data_versions`)
  - 각 회차를 별도 segment로 이어 붙임 (`merge_strategy: segmented_concat`)
  - segment 경계를 넘는 forecast window·MPC transition은 만들지 않음
  - 결과 경로: `saved_model/<version_a>__<version_b>/<environment>/...`

  ```bash
  python src/main.py --task mpc --opts \
    'DATASET.data_versions=[version3_260714_processed,version4_260805_processed]'
  ```

### 열 구성

한 행 = 한 시각의 관측값 + 그때 적용된 제어 상태. 열 역할은 모두 설정 파일에서 지정.

| 역할 | 설정 키 | 기본 예시 (단위) | 용도 |
|---|---|---|---|
| 시각 | (고정) `데이터 시간` | `2025-08-04 00:00:00` | 정렬·중복 제거·분할 |
| 상태 (state) | `MPC.state_constraints[].name` | 실내 온도(℃)·습도(%)·CO2(ppm)·PM10·PM2.5(㎍/㎥)·VOC(ppb) | 제약 대상, 예측 입력 |
| 외란 (disturbance) | `MPC.disturbance_variables` | 실외 온도·습도·PM10·PM2.5 | 상태 변화의 외부 요인 |
| 전력 (power) | `MPC.power_variables`, `DATASET.target_features` | 환기·냉난방 유효전력(W) | 최소화 대상, 예측 타깃 |
| 제어 (control) | `MPC.control_variables` | 전원 ON/OFF, 풍량 단계, 운전 모드, 설정온도 | MPC가 고르는 이산 action |

예시 (일부 열, 1분 간격):

```csv
데이터 시간,실내온도(℃),실내습도(%),실내이산화탄소(ppm),실외온도(℃),실외습도(%),환기전원,환기풍량,냉난방전원,냉난방운전모드,냉난방설정온도(℃),환기유효전력(W),냉난방유효전력(W)
2025-08-04 00:00:00,26.1,58.0,640,27.3,81.0,ON,2,ON,냉방,26,18.5,412.0
2025-08-04 00:01:00,26.0,58.2,642,27.3,81.0,ON,2,ON,냉방,26,18.4,409.7
```

### 데이터 요구사항

- **시각 열**
  - 이름 `데이터 시간` 고정 (변경 시 `src/data_provider/load_dataset.py`, `control_data.py` 수정)
  - `pandas.to_datetime`으로 읽을 수 있는 형식이면 됨
  - 같은 시각이 중복되면 마지막 행만 유지
- **샘플 간격**
  - MPC: **1분 간격** 필수 (`MPC.sample_interval_minutes: 1`)
  - 정확히 1분 차이인 연속 행만 transition으로 사용 → 결측 구간은 허용, 다른 간격은 미리 리샘플링
  - 예측: 행 단위 window(`seq_len`, `pred_len`)라 간격 검사 없음
- **제어 값**
  - `MPC.control_variables[].categories`에 적은 값만 허용 (범주 밖 값이면 오류, 빈 값은 `default`로 채움)
  - 숫자 범주는 문자열로 맞춰 비교 (예: `4.0` → `"4"`)
- **장비 논리 규칙** (`MPC.logical_rules`)
  - 예: "전원 OFF이면 풍량 0" → 같은 실제 명령의 중복 조합 제거
  - 기록 데이터가 `forbid` 규칙을 어기면 오류 → 장비에 맞게 수정하거나 `[]`로 비움
- **결측**
  - 예측: segment별 앞·뒤 값 채움 → 남은 수치는 0, 범주는 `unknown`
  - MPC: 수치 열 보간 후에도 결측이 남으면 오류
- **이상치**
  - 예측 로더가 `환기유효전력(W)`, `냉난방유효전력(W)` 1000 W 초과 행 제거 (`load_dataset.py`의 `outlier_thresholds`)
  - 그 밖의 타깃은 99.9 백분위수 기준 → 장비 용량에 맞게 조정
- **분할**
  - `start_date`/`end_date`로 기간 제한 → `split_date`가 있으면 날짜 기준, 없으면 `split_rate`(기본 `[0.8, 0.1, 0.1]`) 비율로 시간 순 분할
  - 기본 `start_date`는 예시 데이터 기준값 → 자신의 데이터에서는 `null` 또는 적절한 날짜로 변경

### 내 데이터에 적용하기

1. CSV를 `<datadir>/<data_version>/<공간>.csv`로 저장하고 시각 열 이름을 `데이터 시간`으로 변경
2. `default_setting.yaml`의 상태·외란·전력·제어 열과 제약 범위를 자신의 열 이름으로 변경
   - 최소 구성: 상태 1개, 전력 1개, 제어 변수 1개
   - 같은 형태의 합성 데이터로 전체 파이프라인을 돌리는 예: `tests/test_control_benchmark.py`

   ```yaml
   MPC:
     disturbance_variables: [outdoor_temp]
     power_variables: [power]
     control_variables:
       - {name: mode, categories: ['OFF', 'ON'], default: 'OFF'}
     logical_rules: []
     state_constraints:
       - {name: indoor_temp, lower: 20.0, upper: 26.0, penalty: 100.0}
   ```

3. 예측을 쓰는 경우 `DATASET.numerical_features`, `categorical_features`, `ord_categorical_feature`, `target_features`도 변경
4. 기본 열 이름이 아니면 이름에 의존하는 아래 코드도 확인
   - 냉난방 전력 성분 추정: `MPC.independent_evaluator.component_power.hvac_component_name`,
     `src/dynamics/component_power_evaluator.py`, `src/controllers/fixed_action_power_models.py`
     (`냉난방`, `hvac`, `cooling` 등 이름 토큰으로 판별)
   - 환기 전용 KPI: `src/controllers/vent_isolated_kpi.py` (`환기`로 시작하는 제어·전력 열)
   - 물리 residual 제어 모델: `src/models/control/neural.py` (장비 열 이름·범주 값)
   - ablation 결과 키: `src/runners/action_distance_ablation_runner.py`, `action_history_ablation_runner.py`
5. 원본 데이터 변환 예시: `src/scripts/preprocessing/prepare_*.py` (공간별 엑셀 → 위 CSV 형식)

## 예측(forecast)

```bash
python src/main.py --task forecast --model_name PatchTST
```

- **모델**: DLinear, iTransformer, MICN, PatchTST, TimesNet, TimeXer, FreTS, Koopa, TimeMixer, TSMixer
  - 구현: `src/models/prediction/`, 하이퍼파라미터: `src/configs/model_setting.yaml`
  - 기존 `models.MICN`, `models.create_model` 등의 import 경로는 호환용으로 유지
- **주요 설정** (`DATASET.*`)

  | 설정 | 기본값 | 설명 |
  |---|---|---|
  | `seq_len` / `pred_len` / `stride_len` | 120 / 60 / 1 | 입력 길이 / 예측 길이 / window 간격 (행 단위) |
  | `numerical_features` | 15개 열 | 수치 입력. **타깃 열도 반드시 포함** |
  | `categorical_features`, `ord_categorical_feature` | 제어 열 | 범주·순서형 입력 |
  | `cat_encode_method` | `label` | 범주 인코딩 (`label` 또는 `onehot`) |
  | `target_features` | 환기·냉난방 유효전력 | 예측 대상 |
  | `ignore_feature` | `null` | 입력에서 뺄 열 |
  | `scaler` | `standard` | 학습 구간 기준 스케일링 (`standard`, `minmax`) |

- **타깃 계약**
  - 마지막 n개 입력 채널을 타깃으로 가정하지 않음
  - Dataset이 입력·타깃 열 이름을 metadata로 기록하고, 채널 보존형 모델은 그 이름으로 출력 채널 선택
  - 타깃이 numerical input에 없으면 모델 생성 시 오류
- **일괄 실행**: `src/scripts/forecasting/run_experiments.sh` (환경 변수 `MODELS`, `ENVIRONMENTS`, `TARGET_KEYS`)
- **결과**: `saved_model/<data_version>/<environment>/<model>/<exp_name>/version<N>/`
  (`configs.yaml`, `best_model.pt`, `test_results.json`)

## MPC

### 문제 정의

- 1분 간격, prediction horizon 60 step, control horizon 10 step
- 7개 제어변수의 각 범주를 one-hot 이진변수로 표현 → solve 1회당 `10 × 20 = 200`개 이진변수
- solve 안에서는 10번째 제어를 horizon 끝까지 유지, 폐루프에서는 첫 1분 제어만 적용하고 매분 다시 최적화
- 학습하는 모델 2개

  ```text
  state(t+1) = A state(t) + B control(t) + E disturbance(t) + c
  power(t)   = F state(t) + G control(t) + H disturbance(t) + b
  ```

- **전력 모델**
  - 상태 동역학과 별도로 Ridge affine regression으로 학습
  - 테스트 R²가 기준보다 낮은 전력계는 `quality_warnings`에 기록, `fail_on_low_quality: true`이면 MPC 실행 중단
  - 전력 모델 MAPE: `|actual| > 0.1 W` 표본만 사용, 전력계별·전체 MAPE와 유효 표본 수·coverage 저장
  - 이 MAPE는 새 제어 명령을 실제 설비에 적용한 값이 아니므로 `affine_power_model_test_mape`로 구분
- **상태 제약** (`MPC.state_constraints`)
  - 상·하한은 실험용 초기값
  - 각 제약에 비음수 slack + 위반 비용 → 현재 상태가 이미 범위 밖이어도 MILP가 infeasible해지지 않음
- **장비 논리 규칙** (`MPC.logical_rules`)
  - 장비 OFF일 때 의미 없는 하위 변수는 대표값으로 고정 → 같은 명령을 여러 조합으로 표현하지 않음
- **외란 예측** (`disturbance_forecast`)
  - `persistence`(기본): 마지막 외란 관측값을 60분 유지. 배포 가능한 설정
  - 폐루프 평가에서 controller에는 이 예측만 주고, 독립 evaluator에는 실제 미래 외란을 적용 → 외란 예측 오차를 숨기지 않음
  - `perfect`: controller에 실제 미래 외란을 주는 상한선 실험
- **딥러닝 전력 모델로 교체 가능 여부** (MILP 내부 모델 기준)
  - affine·선형 piecewise: 현재 MILP에 그대로 포함 가능
  - ReLU 신경망: activation마다 big-M 이진변수 추가 필요 → 변수 수·계산량 급증, activation bound 필요
  - 임의 비선형 모델: 정확한 MILP가 아니므로 조합 열거·비선형 최적화·샘플 기반 제어 등 별도 optimizer 필요
  - 현재는 해석 가능성·물리 단위 검증·정확한 이산 최적화를 우선해 affine 모델을 기본으로 사용
  - neural·forecasting evaluator는 최적화 밖에서만 쓰이므로 Affine+MILP 구조와 무관

### 실행

```bash
# 기본 실행 (Full guard + report model 세 입력 평가)
./src/scripts/control/run_basic_mpc.sh

# 빠른 smoke test: 평가 window 수 제한, 후속 입력 ablation 생략
./src/scripts/control/run_basic_mpc.sh --opts \
  MPC.evaluation.max_windows=10 \
  MPC.counterfactual_report_input_ablation.run_after_mpc=false
```

### 주요 산출물

`saved_model/<data_version>/<environment>/mpc/<experiment>/`

| 파일 | 내용 |
|---|---|
| `affine_state_model.json`, `affine_power_model.json` | 최적화용 상태·전력 모델 계수와 열 이름 |
| `affine_model_metrics.json` | train/validation/test one-step 모델 지표 |
| `independent_evaluator_metadata.json` | 비선형 평가 plant의 학습 범위·설정 |
| `model_validation.json` | one-step·recursive 60분·lead-time별 모델 검증 |
| `mpc_evaluation.json` | 품질 gate, 폐루프 정책 KPI, window별 결과, 대표 지표 |
| `mpc_first_actions.csv` | 평가 window별 첫 제어 명령 |
| `mpc_closed_loop_steps.csv` | 매분 제어·solver·Affine/evaluator 전력 기록 |
| `chronological_action_split_support.json/.csv` | 시간 순 split별 joint action 표본·episode support |
| `fixed_optimal_action_split_support.csv` | 대표 평가 후보별 train/validation/test support |
| `fixed_optimal_action_test_predictions.csv` | 확정 action의 test 실제·예측 총전력, 행별 오차, 상태 제약 만족 여부 |
| `mpc_selected_action_crossfit_predictions.csv` | MPC 선택 조합의 out-of-fold 실제·예측 전력 (보조 지표) |
| `observed_combination_test_predictions.csv` | 이전 단일 조합 민감도 분석용 (기본 비활성) |

### action 이력 기반 완화 제어 (선택)

- 기본 MPC: 직전 action과의 one-step switching penalty만 사용
- `MPC.objective.action_history` 활성화 시 최근 L분간 **실제 적용 action** 전체와의 차이에 penalty 부여

  ```text
  J = J_MPC + λ_history × Σ(l=1..L) w_l × D(z_t, z_(t-l)^applied)
  ```

  - `D`: one-hot bit 수가 아니라 환기·냉난방 **장비 설정 묶음** 단위의 차이 (냉난방 OFF로 하위 설정이 함께 바뀌어도 1회 변경)
  - `rolling_change_budget`: 최근 5분 장비 단위 변화량 제한. 과거 이력만으로 이미 초과하면 큰 penalty slack으로 infeasible 회피
- 이력 구성
  - 폐루프: window 시작 전 관측 action으로 초기화 후, MPC가 실제 적용한 action만 추가 (미래 기록 action은 읽지 않음)
  - 행별 counterfactual 진단: 해당 행 이전의 기록 action만 사용하고 CSV에 출처 표시
- 실행

  ```bash
  ./src/scripts/control/run_history_smoothed_mpc.sh                    # 5분 이력, 변경 cap 7
  LOOKBACK_STEPS=10 ./src/scripts/control/run_history_smoothed_mpc.sh  # 10분 이력
  GPU_ID=1 ./src/scripts/control/run_action_history_ablation.sh        # L={0,1,2,5,10,30} 비교
  ```

  - `L=0`: 기록 action 고정 기준선, `L>0`: history-smoothed MPC
  - L 간 공정 비교를 위해 recency weight 합을 정규화하고 budget을 시간 길이에 비례
  - 결과: `saved_model/<data>/<environment>/action_history_ablation/<run>/`
    (`lookback_summary.csv`, `lookback_input_report_summary.csv`, `action_history_ablation_report.md`)
  - 감사용 기록: `mpc_closed_loop_steps.csv`의 `action_history_*` 열, `mpc_evaluation.json`의 `solver.action_history`
- `run_basic_mpc.sh`는 이전 실험 재현성을 위해 이 기능을 끈 상태로 유지

### 독립 evaluator

- 최적화용 Affine 모델과 분리된 평가 plant
- 선택지: `extra_trees_residual_state`(기본), `random_forest_residual_state`, `hist_gradient_boosting_residual_state`,
  `neural_residual`, `forecasting:<모델명>` (예: `forecasting:PatchTST`, 예측 모델 10종 사용 가능)
- `neural_residual`: Affine 계수를 재사용하지 않음. 별도 Ridge 선형 상태공간 모델(spectral radius ≤ 0.999로 안정화) + MLP 잔차 보정
- 시계열 evaluator 입력: 최근 `lookback_steps`의 상태·one-hot 제어·외란·**직전** 전력·시간 주기 feature (현재 전력은 입력에서 제외해 누출 방지)
- 정책별(recorded, hold-last, MPC) rollout session 분리 → 예측 이력이 섞이지 않음

```bash
python src/main.py --task mpc --opts MPC.independent_evaluator.model_family=neural_residual

# 후보 비교: train 80%로 학습, validation 10%로 지표 계산 (test는 선택에 사용 안 함)
python src/main.py --task mpc --opts \
  'MPC.independent_evaluator.validation_model_families=[random_forest_residual_state,neural_residual,forecasting:DLinear]'
```

- 정책 평가에 쓰는 family는 항상 `model_family`로 명시하며, 이 모델만 train+validation으로 재학습 후 test·정책 평가에 사용

## MPC 평가 프로토콜

실제 설비에 새 MPC 명령을 적용할 수 없으므로, 오프라인 평가를 여러 관점으로 나눠 보고함.

| 평가 | 질문 | 위치 (`mpc_evaluation.json`) |
|---|---|---|
| 대표 fixed-action MAPE | 고정한 최적 action의 총전력을 실제 Y로 얼마나 정확히 예측하는가 | `fixed_optimal_action_holdout_evaluation`, 최상위 `primary_metric` |
| counterfactual report MAPE | MPC가 바꾼 action이 예측 전력을 얼마나 바꾸는가 | `mpc_counterfactual_total_power_report_evaluation` |
| rolling cross-fit MAPE (보조) | MPC가 고른 조합을 과거 데이터만으로 검증하면 어떤가 | `mpc_selected_action_crossfit_evaluation` |
| 폐루프 정책 평가 | 같은 plant에서 MPC가 기록 제어보다 에너지·제약 면에서 나은가 | 정책 KPI, bootstrap |

### 기본 실행 구성 (Full guard)

- `run_basic_mpc.sh`의 기본값은 **Full guard**: 아래 네 가지를 동시에 적용
  - component two-stage HVAC evaluator
  - train 기준 global/local action support gate
  - validation 보정 model screen
  - step당 categorical 변수 변경 최대 3개
- MPC 완료 후, 저장된 first-action trace를 기준으로 report model의 세 입력 구성을 재학습·평가 (MILP 재실행 없음)
  1. 현재 상태 + action + 현재 외란
  2. 현재 상태 + action
  3. action만
- 세 경우는 같은 test 행·상태·외란·MPC action을 공유하고 report model 입력만 다름
- 결과: `mpc_counterfactual_report_input_ablation_manifest.json`(경로·action trace hash), `counterfactual_report_input_ablation_summary.csv`

### 데이터 분할

- controller(Affine): 시간 순 train 약 80%로 학습
- 기본 evaluator: test를 제외한 train+validation 90%로 학습, test 10%는 모델 검증·정책 백테스트에만 사용
- 명목 8:1:1 경계를 가장 가까운 episode 경계로 이동 → 같은 joint action 연속 운전이 두 split에 걸치지 않음 (행을 섞거나 복제하지 않음)
- `chronological_action_split_support`: 조합별 split 표본 수·완전 episode 수 기록
  - 조합 균형을 위한 random oversampling은 시간 누수·중복 평가 위험으로 사용하지 않음
  - 모든 split에서 30표본·episode 3개를 만족하는 조합이 없으면 `balanced_holdout_available=false`

### joint action 옵션 (`MPC.evaluation.action_options`)

| 옵션 | 기본값 | 내용 |
|---|---|---|
| `repeated_joint_actions.enabled` | `true` | train에서 30표본·독립 episode 3개를 확보한 exact joint action만 MPC 후보로 허용. 끄면 train에서 한 번이라도 관측된 조합 허용 |
| `category_consolidation.enabled` | `false` | `mappings`로 범주 통합(예: 설정온도 18·26 → 22)을 모든 split과 후보 공간에 동일 적용. mapping이 비거나 이름이 틀리면 실행 초기에 중단 |
| `episode_group_split.enabled` | `true` | split·cross-fit fold 경계를 episode 사이로 이동. 끄면 행 번호 경계 사용, 걸친 episode 수 기록 |
| `exclude_insufficient_episode_actions.enabled` | `true` | episode가 부족한 조합을 보조 cross-fit MAPE에서 제외 |

```bash
./src/scripts/control/run_basic_mpc.sh --opts \
  MPC.evaluation.action_options.repeated_joint_actions.enabled=false \
  MPC.evaluation.action_options.episode_group_split.enabled=false
```

```yaml
# 범주 통합은 YAML에 명시 권장 (source·representative 모두 control_variables에 있는 범주여야 함)
MPC:
  evaluation:
    action_options:
      category_consolidation:
        enabled: true
        mappings:
          냉난방설정온도(℃): {18: 22, 26: 22}
```

- 적용 설정·치환 건수: `evaluation_protocol.action_options`, `evaluation_protocol.control_preprocessing`
- 조합별 허용 여부·표본 수·제외 사유: `control_support.actions`

### 대표 지표: 실제 Y 기반 fixed-action MAPE

절차 (`fixed_optimal_action_holdout_evaluation`):

1. exact joint action의 연속 1분 운전을 하나의 episode로 정의하고, 전력·상태 Y 없이 action·시각·길이만으로 episode 전체를 train/validation/test 중 하나에 배정
2. 세 split 모두에 최소 support(train 5표본·1 episode, validation 3·1, test 5·1)가 있는 관측 action만 후보로 사용 (`top_k_actions=0`: 조건을 만족하는 전체)
3. train으로 학습해 고정한 evaluator로 모든 후보를 같은 validation context에 넣어 예측 상태 만족률·총전력 계산 → 만족률을 통과한 후보 중 예측 전력 최소 action 확정
4. 통과 후보가 없고 `allow_soft_constraint_fallback=true`이면 만족률 최고(동률 시 전력 최소) 후보 선택. MAPE는 계산하지만 정책 상태 제약 gate는 실패 처리
5. report model 후보(`action_median`, Huber residual, ExtraTrees residual, component-aware ExtraTrees)를 train으로 학습하고 확정 action의 validation MAPE로 하나 선택 (test Y 미사용)
6. 선택 모델을 train+validation으로 재학습·고정 후, 확정 action과 같은 test 기록 행에서 실제 총전력과 비교

```text
report(x_t, z_t, d_t) = [환기유효전력(W), 냉난방유효전력(W)]
total_power           = 환기유효전력(W) + 냉난방유효전력(W)
MAPE(%)               = 100 / n × Σ |actual − predicted| / |actual|
```

- report model은 두 성분을 동시에 직접 예측하고 총전력은 그 합. 총전력 MAPE가 대표값이며 성분별 MAPE/WAPE/MAE도 함께 기록
- report model은 MPC 목적함수·제약·action 선택에 연결되지 않음
- 분모가 `mape_min_actual_power_w` 이하인 행은 제외, exact-match 수·episode 수·유효 표본 수 함께 저장
- 후보가 없으면 실제 Y를 대체 생성하지 않고 `primary_metric.value=null`
- 대표값: `fixed_optimal_action_holdout_evaluation.primary_metric.value` (최상위 `primary_metric`에도 복사)
  - 기준값은 설정 `target_percent`, 달성 여부는 `target_achieved`와 policy gate `fixed_optimal_action_mape_target`
  - `reportable`, `selection`, `split`, `test_evaluation`과 함께 해석
- 모델 선택 근거: `fixed_action_power_model_comparison.json`/`.csv`
- 해석 시 주의
  - validation 표본이 적으면 목표를 달성해도 일반화 근거가 약함 → episode 수와 특정 episode 비중 확인
  - 희소 action에서 실제 Y MAPE를 내기 위한 평가 전용 분할. 미래 일반화·인과적 절감·실제 최적성을 증명하지 않음

**report model 입력 선택** (`fixed_optimal_action_holdout.total_power_report_model.input_features`)

```yaml
input_features:
  current_state: true       # x_t: 실내 상태 6개
  action: true              # z_t: 제어 7개의 one-hot 20개
  current_disturbance: true # d_t: 실외 외란 4개
```

- 최소 한 묶음 필요. `action: false`이면 기록 action과 MPC action의 예측이 같아져 제어 효과 해석 불가
- 실제 사용 feature: 결과 JSON의 `model_selection.model.input_features`
- 기록 action 기준 입력 ablation: `bash src/scripts/control/run_report_model_input_ablation.sh`
  - MILP 재실행 없음 → controller 비교가 아닌 report model 입력 비교
  - 결과: `saved_model/<data>/<environment>/report_model_input_ablation/<run>/`
  - 후보 풀 Top-K 민감도(`previous_7`, `expanded_9`~`expanded_17`, `exploratory_41`)를 함께 보고.
    validation이 고른 K만 대표값이고 나머지는 민감도 전용
  - `metric_targets`로 총전력·성분별 목표를 따로 설정, 판정도 환기·냉난방·총전력별로 기록

### counterfactual report MAPE

기록 test의 상태·외란을 유지한 채 행마다 MILP가 새 first action을 고르게 하고, 고정한 report model로 두 action의 전력을 비교.

```text
recorded x_t, d_t → MILP first action z_t^MPC
                  ├→ report(x_t, z_t^recorded, d_t) = ŷ_t^recorded
                  └→ report(x_t, z_t^MPC, d_t)      = ŷ_t^MPC

MPC vs recorded actual MAPE    = 100 / n × Σ |ŷ^MPC − y^recorded| / |y^recorded|
MPC vs recorded-report MAPE    = 100 / n × Σ |ŷ^MPC − ŷ^recorded| / |ŷ^recorded|
recorded_actual_difference     = y^recorded − ŷ^MPC
report_to_report_reduction     = ŷ^recorded − ŷ^MPC
```

- report model: 시간 순 train으로 학습, validation으로 선택, train+validation으로 재학습 후 고정
- 총전력과 환기·냉난방 성분별로 모두 계산
- 범위: 선택된 60분 test window의 모든 유효 transition (`row_stride_steps`, `maximum_evaluation_rows`로 제한 가능)
- 결과: `mpc_counterfactual_total_power_report_predictions.csv`(행별 `recorded_actual_*`, `recorded_action_report_predicted_*`, `mpc_action_report_predicted_*`), `..._evaluation.json`
- 해석 시 주의
  - 첫 번째 MAPE: 바뀐 action의 실제 전력이 없으므로 일반 forecast MAPE와 다름 (제어 효과 + 모델 오차 혼재)
  - 두 번째 MAPE·예상 절감: 실제 Y를 쓰지 않는 모델 기반 진단값 → 실제 절감률로 주장 불가
  - action 변경률, 유효 coverage, observed-action MAPE와 함께 보고

### 보조 지표: rolling cross-fit MAPE

- `mpc_selected_action_crossfit_evaluation`: 폐루프에서 MPC가 고른 action·context를 모아, 과거 데이터만으로 학습한 5개 expanding fold의 out-of-fold 실제 Y 중 같은 action·유사 context 행과 대응
- 조합별 총전력 MAPE를 MPC 선택 빈도 가중·동일 가중으로 함께 기록
- 시간 방향으로 더 엄격하지만 희소 action support·context coverage에 민감 → 대표 지표와 역할 구분
- `observed_control_combination_evaluation`: 이전 단일 조합 민감도 분석용, 기본 비활성

### 폐루프 정책 평가

60분 test window마다 다음을 60회 반복:

1. 현재 evaluator 상태와 persistence 외란 예측으로 60분 MILP 풀이
2. 첫 categorical 제어 조합만 적용
3. 실제 미래 외란 1 step과 독립 evaluator로 전력·다음 상태 생성
4. 다음 분에 새 상태로 재최적화

- recorded 제어·hold-last 정책도 같은 evaluator·초기 상태·외란으로 실행 → 같은 plant 조건에서 비교
- recorded 정책 rollout을 실제 계측과 비교해 evaluator MAPE·WAPE·60분 에너지 오차·상태 false-safe 기록
- MPC 궤적에서 Affine 전력 예측과 evaluator 전력의 MAPE·WAPE를 집계 → model exploitation 여부 확인
- `max_windows: 0`(기본) = 연속된 test window 전체, 양수는 smoke test용
- 전체 절감률 = `(Σbaseline − ΣMPC) / Σbaseline` (window 평균 아님), paired-window bootstrap 95% 신뢰구간 제공
- 훈련 구간에서 보지 못한 제어 조합은 MILP no-good cut으로 금지

**정책 채택 조건** (모두 만족해야 함)

- Affine controller와 독립 evaluator의 전력 품질 gate 통과
- recursive rollout 전력 WAPE·상태 false-safe 기준 통과
- window 20개 이상, bootstrap 절감률 하한 > 0%
- 모든 상태에서 제약 위반량·최대 위반이 baseline보다 악화되지 않음
- fallback, 1분 deadline miss, train-support 밖 제어 0건
- 대표 fixed-action MAPE가 exact-match·episode·유효 표본 기준 통과
- soft constraint fallback 없이 엄격 만족률을 통과한 대표 후보 존재

> 현재 결과는 오프라인 시뮬레이션이며 실제 설비 절감 근거가 아님.
> 최적 제어를 설비에 적용하기 전까지 `actual_closed_loop_mape.available=false`.

## 제어 방법 6종 비교

| ID | 모델·제어기 |
|---|---|
| `m1_affine_milp` | affine 모델 + MILP (기준선) |
| `m2_mode_history_milp` | 모드·이력 확장 모델 + MILP |
| `m3_physics_residual` | 물리 기반 구조 + 학습 잔차 모델 + categorical CEM |
| `m4_scenario_ensemble` | 시나리오 앙상블 + MILP |
| `m5_neural_cem` | 신경망 앙상블 + categorical CEM |
| `m6_decision_focused` | 제어 비용 proxy를 반영해 학습한 모델 + categorical CEM |

```bash
# 축소 예비 비교
bash src/scripts/control/run_six_control_methods.sh

# 원본 설정 전체 비교: 데이터 버전 × 공간 × 6방법
bash src/scripts/control/run_six_control_methods_full.sh
```

- 방법 선택: `CONTROL_BENCHMARK.selected_methods`
- 구현: `src/models/control/`, 비교 실행: `src/controllers/control_benchmark.py`, `src/controllers/full_control_backend.py`
- 전체 비교는 기존 MPC 파이프라인(60-step 폐루프, 가능한 전체 window, solver 60초·gap 1%, guard·gate·report 평가)을 그대로 재사용
- 공통 시간 분할과 독립 surrogate에서의 비교이며 실제 장비 절감 실증이 아님
- 장시간 실행의 상태·재시작 정보와 결과: `reports/artifacts/control_full_benchmark/`

## 디렉터리 구조

```text
src/
├── main.py                 # --task 분기
├── arguments.py            # argparse + OmegaConf 설정 조립
├── configs/                # default_setting.yaml, model_setting.yaml
├── data_provider/          # forecast 전처리, control_data.py(MPC 물리 단위 데이터 계약)
├── models/
│   ├── prediction/         # 예측 모델 10종 + factory
│   └── control/            # 제어용 상태·전력 모델 (affine, neural, ensemble, decision_focused)
├── dynamics/               # 최적화용 affine 모델, 독립 evaluator(ExtraTrees·RF·HGB·neural·forecasting)
├── controllers/            # MILP(mixed_integer_mpc.py), 평가·gate·파이프라인(pipeline.py)
├── runners/                # task별 실행 진입점
├── reporting/              # 결과 보고서 생성
├── layers/, losses/, optimizers/, utils/
├── scripts/
│   ├── control/            # MPC·ablation·6종 비교 실행 스크립트
│   ├── forecasting/        # 예측 일괄 실행
│   ├── kpi/                # 예측 성능 평가
│   └── preprocessing/      # 원본 데이터 → CSV 변환 예시
└── jupyter/                # 분석 노트북
tests/                      # unittest 기반 계약 테스트
```

실행 중 생성되며 git에서 제외되는 폴더:

- `saved_model/`: 학습 모델·설정·지표·MPC 평가 결과
- `reports/`: 일괄 실험 로그·비교 결과·보고서·그림
- `kpi_release/`: `src/scripts/kpi/` 평가 결과

## 라이선스

[MIT License](LICENSE)
