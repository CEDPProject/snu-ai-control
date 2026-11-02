# SNU-AI-Control

이 저장소에는 서로 독립적인 두 프레임워크가 공존합니다.

- `forecast`: 측정 데이터로 미래 전력을 예측하는 딥러닝 프레임워크
- `mpc`: 상태 제약 아래 이산 제어 조합의 전력 소비를 최소화하는 mixed-integer MPC

두 프레임워크는 같은 CSV를 사용할 수 있지만 모델, 전처리, 실행 루프를 공유하지
않습니다. 특히 MPC는 스케일되지 않은 물리 단위 데이터를 사용합니다.

## 설치

```bash
git clone https://github.com/CEDPProject/snu-ai-control.git
cd snu-ai-control
pip install -r requirements.txt
```

모든 명령은 저장소 루트에서 실행합니다. 테스트는 합성 데이터만 사용하므로 데이터 없이 돌릴 수 있습니다.

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

MPC는 SciPy에 포함된 HiGHS MILP solver를 사용하므로 별도의 상용 solver나
`cvxpy` 설치가 필요하지 않습니다.

## 데이터 형식

이 저장소에는 **데이터, 학습된 모델, 실험 결과가 포함되어 있지 않습니다.** 아래 형식에 맞춰 준비한 데이터를
사용하세요. 기본 설정은 "한 공간(방)에 설치된 환기·냉난방 장비의 분 단위 운전 기록" 형태를 가정하지만,
열 이름과 제어 변수를 설정 파일에서 바꾸면 비슷한 구조의 건물 HVAC 데이터에 그대로 적용할 수 있습니다.

### 1. 폴더 구조

```text
<datadir>/
└── <data_version>/              # 예: site_a_2025summer
    ├── <environment_name>.csv   # 공간 1개 = CSV 1개 (예: livingroom.csv)
    ├── room1.csv
    └── ...
```

| 설정 (`src/configs/default_setting.yaml`) | 기본값 | 설명 |
|---|---|---|
| `DATASET.datadir` | `../../data/kweather/` | 데이터 루트. 절대경로, 실행 위치 기준 상대경로, `src/` 기준 상대경로 순으로 찾음 |
| `DATASET.data_version` | `version3_260714_processed` | `<datadir>` 아래 하위 폴더 이름 |
| `DATASET.environment_name` | `livingroom` | 사용할 CSV 파일 이름(확장자 제외) |
| `DATASET.data_versions` | `null` | 여러 수집 회차를 함께 쓸 때 폴더 이름 목록. 각 회차는 별도 segment로 이어 붙이며 segment 경계를 넘는 window·transition은 만들지 않음 |

데이터는 저장소 밖에 두고 실행할 때 덮어쓰는 것을 권장합니다.

```bash
python src/main.py --task forecast --model_name DLinear --opts \
  DATASET.datadir=/path/to/data DATASET.data_version=site_a DATASET.environment_name=room1
```

### 2. CSV 열 구성

한 행은 한 시각의 관측값과 그때 적용된 제어 상태입니다. 열은 역할에 따라 다섯 종류로 나뉘며,
각 역할에 어떤 열을 쓸지는 모두 설정 파일에서 지정합니다.

| 역할 | 설정 키 | 기본 예시 (단위) | 사용처 |
|---|---|---|---|
| 시각 | (고정) `데이터 시간` | `2025-08-04 00:00:00` | 정렬·중복 제거·분할 |
| 상태 (state) | `MPC.state_constraints[].name` | 실내 온도(℃), 습도(%), CO2(ppm), PM10·PM2.5(㎍/㎥), VOC(ppb) | 제약 대상, 예측 입력 |
| 외란 (disturbance) | `MPC.disturbance_variables` | 실외 온도·습도·PM10·PM2.5 | 상태 변화의 외부 요인 |
| 전력 (power) | `MPC.power_variables`, `DATASET.target_features` | 환기·냉난방 유효전력(W) | 최소화 대상, 예측 타깃 |
| 제어 (control) | `MPC.control_variables` | 전원 ON/OFF, 풍량 단계, 운전 모드, 설정온도 | MPC가 고르는 이산 action |

예측(forecast)에서는 열을 자료형 기준으로 다시 지정합니다.

- `DATASET.numerical_features`: 수치 입력. **타깃 열도 반드시 포함**해야 합니다.
- `DATASET.categorical_features`, `DATASET.ord_categorical_feature`: 문자열·순서형 범주 입력.
  `cat_encode_method`(`label` 또는 `onehot`)로 인코딩합니다.
- `DATASET.target_features`: 예측할 열.
- `DATASET.ignore_feature`: 입력에서 뺄 열.

예시 (일부 열, 1분 간격):

```csv
데이터 시간,실내온도(℃),실내습도(%),실내이산화탄소(ppm),실외온도(℃),실외습도(%),환기전원,환기풍량,냉난방전원,냉난방운전모드,냉난방설정온도(℃),환기유효전력(W),냉난방유효전력(W)
2025-08-04 00:00:00,26.1,58.0,640,27.3,81.0,ON,2,ON,냉방,26,18.5,412.0
2025-08-04 00:01:00,26.0,58.2,642,27.3,81.0,ON,2,ON,냉방,26,18.4,409.7
```

### 3. 데이터 요구사항

- **시각 열 이름**은 `데이터 시간`으로 고정되어 있습니다. 다른 이름이면 CSV에서 바꾸거나
  `src/data_provider/load_dataset.py`, `src/data_provider/control_data.py`를 수정하세요.
  `pandas.to_datetime`으로 읽을 수 있는 형식이면 되며, 같은 시각이 여러 번 나오면 마지막 행만 남깁니다.
- **샘플 간격**: MPC는 **1분 간격**을 요구합니다(`MPC.sample_interval_minutes: 1`). 정확히 1분 차이인 연속 행만
  transition으로 사용하므로 결측 구간이 있어도 되지만, 다른 간격의 데이터는 미리 1분으로 리샘플링해야 합니다.
  예측은 행 단위 window(`seq_len`, `pred_len`)를 쓰므로 간격을 검사하지 않습니다.
- **제어 열의 값**은 `MPC.control_variables[].categories`에 적은 값 중 하나여야 합니다. 범주에 없는 값이 있으면
  오류가 나고, 빈 값은 `default`로 채웁니다. 숫자 범주는 문자열로 맞춰 비교합니다(예: `4.0` → `"4"`).
- **장비 논리 규칙**(`MPC.logical_rules`): 예를 들어 "전원 OFF이면 풍량 0" 같은 규칙으로 중복 조합을 없앱니다.
  기록 데이터가 `forbid` 규칙을 어기면 오류가 나므로, 자신의 장비에 맞게 규칙을 고치거나 비워 두세요(`[]`).
- **단위**: MPC는 스케일하지 않은 물리 단위를 그대로 사용합니다. 예측은 학습 구간으로 맞춘 scaler
  (`DATASET.scaler: standard|minmax`)를 씁니다.
- **결측**: 예측은 segment별로 앞·뒤 값으로 채운 뒤 수치는 0, 범주는 `unknown`으로 채웁니다.
  MPC는 수치 열을 보간한 뒤에도 남는 결측이 있으면 오류를 냅니다.
- **이상치**: 예측 로더는 `환기유효전력(W)`, `냉난방유효전력(W)`이 1000 W를 넘는 행을 제거합니다
  (`load_dataset.py`의 `outlier_thresholds`). 다른 타깃은 99.9 백분위수를 기준으로 합니다. 장비 용량에 맞게 고치세요.
- **분할**: `start_date`/`end_date`로 기간을 자르고, `split_date`가 있으면 날짜로, 없으면 `split_rate`
  (기본 `[0.8, 0.1, 0.1]`) 비율로 시간 순 분할합니다. 기본 `start_date`는 예시 데이터에 맞춘 값이므로
  자신의 데이터에서는 `null`이나 적절한 날짜로 바꾸세요.

### 4. 내 데이터에 적용하기

1. CSV를 `<datadir>/<data_version>/<공간>.csv`로 두고 시각 열 이름을 `데이터 시간`으로 맞춥니다.
2. `default_setting.yaml`에서 상태·외란·전력·제어 열과 제약 범위를 자신의 열 이름으로 바꿉니다.
   최소 구성은 상태 1개, 전력 1개, 제어 변수 1개입니다. 예를 들어 제어 변수가 운전 모드 하나뿐이라면:

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

   같은 형태의 합성 데이터로 전체 파이프라인을 돌리는 예가 `tests/test_control_benchmark.py`에 있습니다.
3. 예측을 쓴다면 `DATASET.numerical_features`, `categorical_features`, `ord_categorical_feature`,
   `target_features`도 같은 열 이름으로 바꿉니다.
4. 열 이름에 의존하는 부분이 남아 있으니, 기본 열 이름이 아니라면 아래도 확인하세요.
   - 냉난방 전력 성분 추정: `independent_evaluator.component_power.hvac_component_name` 설정,
     `src/dynamics/component_power_evaluator.py`, `src/controllers/fixed_action_power_models.py`
     (`냉난방`, `hvac`, `cooling` 등의 이름 토큰으로 판별)
   - 환기 전용 KPI: `src/controllers/vent_isolated_kpi.py` (`환기`로 시작하는 제어·전력 열)
   - 물리 residual 제어 모델: `src/models/control/neural.py` (장비 열 이름과 범주 값)
   - ablation runner의 결과 키: `src/runners/action_distance_ablation_runner.py`,
     `src/runners/action_history_ablation_runner.py`
5. `src/scripts/preprocessing/prepare_*.py`는 원본 수집 데이터(공간별 엑셀)를 위 CSV 형식으로 바꾸는 예시입니다.
   자신의 원본 형식에 맞게 참고해서 작성하세요.

### 5. 로컬 산출물

다음 폴더는 실행 중에 만들어지며 `.gitignore`로 저장소에서 제외됩니다.

- `saved_model/`: 학습한 모델, 설정, 지표, MPC 평가 결과
- `reports/`: 일괄 실험 로그, 비교 결과, 보고서·그림
- `kpi_release/`: `src/scripts/kpi/` 성능 평가 스크립트의 결과

## 모델 폴더 분리와 제어 6종 비교

- 전력 예측 모델 구현: `src/models/prediction/`
- 제어용 상태·전력 모델 구현: `src/models/control/`
- 기존 `models.MICN`, `models.create_model` 등의 import는 호환 shim으로 유지합니다.

기존 forecast/MPC 실행을 유지하면서 6종의 **축소 예비 비교**를 실행하려면:

```bash
bash src/scripts/control/run_six_control_methods.sh
```

**원본 설정 전체 비교**(v3/v4/v5/concat × 3개 공간 × 6개 방법, 72개 조합)는:

```bash
bash src/scripts/control/run_six_control_methods_full.sh
```

원본 60-step 폐루프·가능한 전체 창·solver 60초/1%·guard/gate/report 평가를 재사용한다.
장시간 실행의 상태·재시작 정보와 산출물은 `reports/artifacts/control_full_benchmark/` 아래에
저장된다(로컬 산출물이며 저장소에는 포함하지 않는다).

`CONTROL_BENCHMARK.selected_methods`로 기준선·모드/이력·물리 residual·scenario
ensemble·비선형 categorical CEM·decision-focused 중 선택합니다.
공통 시간 분할과 독립 surrogate에서 평가하며 실제 장비 절감 실증은 아닙니다.
방법별 구현은 `src/models/control/`, 비교 실행은 `src/controllers/control_benchmark.py`와
`src/controllers/full_control_backend.py`에 있습니다.

## 실행

저장소 루트에서 실행합니다.

예측 모델 학습:

```bash
python src/main.py --task forecast --model_name PatchTST
```

MPC affine 모델 학습 및 오프라인 평가:

```bash
python src/main.py --task mpc
```

다른 환경이나 평가 창 개수는 명시적으로 덮어쓸 수 있습니다.

```bash
python src/main.py --task mpc --opts \
  DATASET.environment_name=room1 \
  MPC.evaluation.max_windows=10
```

결과는 기본적으로
`saved_model/<data_version>/<environment>/mpc/<experiment>/`에 저장됩니다.
여러 데이터 버전을 함께 쓰는 경우에는
`saved_model/<version_a>__<version_b>/<environment>/...`처럼 병합 label 아래에
저장됩니다.

여러 회차 데이터를 하나의 실험에 사용할 때는 `DATASET.data_versions`를 명시합니다.
각 version은 하나의 segment로 취급되며, forecast sample은 segment 경계를 넘지
않습니다. MPC transition도 같은 segment 안의 1분 간격만 유효합니다.

```bash
python src/main.py --task forecast --model_name PatchTST --opts \
  'DATASET.data_versions=[version3_260714_processed,version4_260805_processed]'

python src/main.py --task mpc --opts \
  'DATASET.data_versions=[version3_260714_processed,version4_260805_processed]'
```

- `affine_state_model.json`: 최적화용 상태 동역학 계수와 열 이름
- `affine_power_model.json`: 최적화용 독립 affine 전력 모델
- `affine_model_metrics.json`: train/validation/test one-step 모델 지표
- `independent_evaluator_metadata.json`: nonlinear 평가 plant의 학습 범위와 설정
- `model_validation.json`: one-step·recursive 60분·lead-time별 모델 검증
- `mpc_evaluation.json`: 품질 gate, 폐루프 정책 KPI, window별 결과
- `chronological_action_split_support.json`: 시간순 split별 joint action 표본·완전
  episode support와 평가 가능 여부
- `chronological_action_split_support.csv`: 조합별 train/validation/test support 표
- `mpc_first_actions.csv`: 각 평가 window의 첫 제어 명령
- `mpc_closed_loop_steps.csv`: 매분 제어·solver·Affine/evaluator 전력 기록
- `fixed_optimal_action_split_support.csv`: 공식 평가 후보별 episode-disjoint
  train/validation/test 표본·episode support
- `fixed_optimal_action_test_predictions.csv`: 고정된 최종 action의 test 실제·예측
  총 공조전력, 행별 절대백분율오차와 상태 제약 만족 여부
- `mpc_selected_action_crossfit_predictions.csv`: MPC 선택 조합과 context가
  일치한 과거 행의 out-of-fold 실제·예측 전력과 행별 오차(보조 지표)
- `observed_combination_test_predictions.csv`: 이전 chronological 단일 조합
  민감도 분석용 실제·예측 전력(기본 비활성)

## MPC 정의

기본 설정은 1분 간격, prediction horizon 60 step, control horizon 10 step입니다.
7개 제어변수의 각 카테고리를 one-hot 이진변수로 모델링하므로 한 solve에는
`10 × 20 = 200`개의 이진변수가 있습니다. 각 solve 내부에서는 control horizon
이후 10번째 제어를 prediction horizon 끝까지 유지하지만, 폐루프 평가에서는 첫
1분 제어만 evaluator에 적용하고 다음 상태에서 즉시 60분 MPC를 다시 풉니다.

학습되는 식은 다음 두 개입니다.

```text
state(t+1) = A state(t) + B control(t) + E disturbance(t) + c
power(t)   = F state(t) + G control(t) + H disturbance(t) + b
```

전력 모델은 상태 동역학과 별도로 Ridge affine regression으로 학습됩니다.
테스트 R²가 설정 기준보다 낮은 전력 계측기는 결과의 quality_warnings에 기록됩니다.
fail_on_low_quality를 true로 두면 이런 모델로 MPC를 실행하지 않고 즉시 중단합니다.
전력모델 MAPE는 각 train/test 구간의 실측 전력과 affine 예측을 비교한 값이며,
기본적으로 `|actual| > 0.1 W`인 표본만 사용하며 전력계별 MAPE, 전체 MAPE,
유효 표본 수와 coverage를 함께 저장합니다. 이 값은 새 MPC 제어 명령을 실제 설비에
적용한 closed-loop 오차가 아니므로 `mpc_evaluation.json`에서도
`affine_power_model_test_mape`라는 이름으로 구분합니다.
상태 상·하한은 `src/configs/default_setting.yaml`의 실험용 초기값이며 각 제약에
nonnegative slack과 위반 비용을 둡니다. 기준값이 과도하거나 현재 상태가 이미
범위를 벗어나도 MILP가 불필요하게 infeasible해지지 않습니다.

환기/HVAC 전원과 하위 설정 간 물리적 관계는 `logical_rules`에 선언합니다.
모든 제어변수는 최적화 대상이지만 장비가 OFF일 때 의미가 없는 하위 변수는
대표값으로 고정하여 동일한 실제 명령을 여러 조합으로 표현하지 않습니다.

`disturbance_forecast: persistence`는 마지막 외란 관측값을 60분 동안 유지하는
배포 가능한 기본값입니다. 폐루프 평가에서도 controller에는 이 예측만 제공하고,
독립 evaluator에는 실제 미래 외란을 한 step씩 적용하여 외란 예측 오차를 숨기지
않습니다. `perfect`는 controller에 실제 미래 외란을 주는 상한선 실험입니다.

### 최근 적용 action 이력 기반 완화 제어

기본 MPC는 직전 action과의 one-step switching penalty만 사용한다. 최근 설정 이력
전체에서 급격히 벗어나는 action을 억제해야 할 때는
`MPC.objective.action_history`를 활성화한다. `lookback_steps: 5`는 현재 first
action을 1, 2, 3, 4, 5분 전의 **실제 적용 action**과 모두 비교한다는 뜻이다.

```text
J = J_MPC + λ_history × Σ(l=1..L) w_l × D(z_t, z_(t-l)^applied)
```

`D`는 원시 one-hot bit 수가 아니라 환기·냉난방 장비 설정 묶음의 차이를 센다.
따라서 냉난방 OFF canonicalization으로 풍량·운전모드·온도가 함께 대표값으로
바뀌어도 하나의 냉난방 장비 설정 변경으로 해석한다. 추가로 최근 5분의 장비 단위
변화량을 `rolling_change_budget`으로 제한하며, 과거 이력만으로 budget이 이미
초과된 경우에는 큰 penalty의 slack을 남겨 MILP infeasible을 피한다.

폐루프에서는 window 시작 전의 관측 action으로 history를 bootstrap한 뒤, 첫 step
부터는 이전에 MPC가 실제 적용한 action만 history에 추가한다. 미래의 기록 action은
읽지 않는다. 반면 행별 counterfactual 진단은 독립 행 평가이므로 해당 행 이전의
기록 action을 bootstrap으로만 사용하며 CSV에 그 source를 명시한다.

다음 스크립트는 categorical 변경 cap을 7(제어변수 전체)로 열고, 기본 5분 history
regularizer를 활성화한다.

```bash
./src/scripts/control/run_history_smoothed_mpc.sh

# 10분 history 민감도 실행
LOOKBACK_STEPS=10 ./src/scripts/control/run_history_smoothed_mpc.sh
```

모든 비교값 `L={0,1,2,5,10,30}`을 같은 Full guard 조건으로 순차 실행하려면
아래 runner를 사용한다. `L=0`은 현재 기록 action을 고정한 factual 기준선이고,
`L>0`은 바로 이전 action부터 L분 이력까지를 함께 보는 history-smoothed MPC다.
각 L의 MPC action trace에는 세 report-model 입력 bundle이 모두 적용된다.

```bash
GPU_ID=1 ./src/scripts/control/run_action_history_ablation.sh
```

산출물은 `saved_model/<data>/<environment>/action_history_ablation/<run>/`에
분리된다. `lookback_summary.csv`는 L별 MPC trace 요약,
`lookback_input_report_summary.csv`는 L×3 입력 bundle별 환기·냉난방·총전력
action-effect MAPE, `action_history_ablation_report.md`는 사람이 읽는 요약이다.
L>0 간에는 recency weight 합을 같은 값으로 정규화하고 rolling change budget을
시간 길이에 비례시켜, 단순히 L이 큰 실험에 더 큰 regularization이 걸리지 않게 한다.

`mpc_closed_loop_steps.csv`의 `action_history_*` 열과
`mpc_evaluation.json`의 `solver.action_history`에서 lag별 거리, 가중 거리, 5분
누적 변화량, budget overflow를 감사할 수 있다. 기본 `run_basic_mpc.sh`는 이전
실험 재현성을 위해 이 기능을 비활성화한 상태로 유지한다.

## MPC 평가 프로토콜

### 기본 실행: Full guard와 세 입력 report-model 결과를 함께 저장

`./src/scripts/control/run_basic_mpc.sh`의 기본값은 06 **Full guard**다. 즉
component two-stage HVAC evaluator, train-only global/local action support gate,
validation 보정 model screen, 한 step당 최대 세 categorical 변수 변경 제한을
동시에 적용한다. 이 기본값을 바꾸지 않는 한 별도 `--opts`가 없어도 적용된다.

MPC가 완료되면 저장된 first-action trace를 reference로 하여 report model의 세 입력
구성을 순차적으로 모두 재학습·평가한다.

1. `현재 상태 + action + 현재 외란`
2. `현재 상태 + action`
3. `action만`

이 post-evaluation은 MILP를 다시 풀지 않는다. 세 case는 동일한 test row,
state, disturbance, Full-guard MPC action을 공유하고 report model 입력만 다르다.
따라서 `report(x, z_MPC, d)`와 `report(x, z_recorded, d)`의 action-effect MAPE를
입력 구성별로 공정하게 비교할 수 있다. 산출물 경로와 action-trace hash는 원 MPC
결과의 `mpc_counterfactual_report_input_ablation_manifest.json`에 기록되고, 세 case의
상세 수치는 연결된 `counterfactual_report_input_ablation_summary.csv`에 저장된다.

smoke test에서만 아래처럼 후속 sweep을 끌 수 있다.

```bash
./src/scripts/control/run_basic_mpc.sh --opts \
  MPC.counterfactual_report_input_ablation.run_after_mpc=false
```

평가 모델은 최적화에 사용한 Affine 모델과 분리합니다. controller는 chronological
train 약 80%에서 학습한 Affine 상태·전력 모델을 사용합니다. 기본값에서는 명목
8:1:1 경계를 동일 joint-action의 연속 운전 episode가 두 split에 걸리지 않도록 가장
가까운 episode 경계로 이동하며, 행을 섞거나 복제하지 않습니다. 기본 evaluator는 기존과
동일하게 test를 제외한 train+validation 90%에서 상태 변화량과 전력을 학습하는
ExtraTrees이며, `model_family`를 바꾼 경우에만 다른 evaluator를 학습합니다. test
10%는 controller/evaluator 모델 검증과 정책 백테스트에만 사용합니다.

`chronological_action_split_support`는 각 조합이 train/validation/test에 몇 번, 몇 개의
완전한 독립 episode로 존재하는지를 기록합니다. 모든 조합을 억지로 같은 수로 만드는
random oversampling은 시간 누수와 중복 평가를 만들 수 있어 사용하지 않습니다. 각
split에서 기본 30표본·완전 episode 3개를 모두 만족하는 조합이 없으면
`balanced_holdout_available=false`로 남기고 조합 균형 holdout 성능을 주장하지 않습니다.

### Joint action 데이터 옵션

아래 네 옵션은 `MPC.evaluation.action_options`에서 서로 독립적으로 제어합니다.

1. `repeated_joint_actions.enabled`(기본 `true`): train에서 기본 30표본과 독립
   episode 3개를 모두 확보한 exact joint action만 MPC 후보로 허용합니다. 끄면
   `restrict_to_training_control_support=true`인 한 train에서 한 번이라도 관측된
   조합은 후보가 될 수 있습니다.
2. `category_consolidation.enabled`(기본 `false`): 설정한 `mappings`를 train,
   validation, test의 기록 제어와 MPC 후보 공간에 동일하게 적용합니다. 켰는데
   mapping이 비어 있거나 변수·카테고리 이름이 잘못되면 실행 초기에 중단합니다.
3. `episode_group_split.enabled`(기본 `true`): chronological 순서는 유지하면서
   8:1:1 split 및 expanding cross-fit fold 경계를 완전한 episode 사이로 옮깁니다.
   끄면 명목 행 번호 경계를 그대로 사용하며 결과에 crossing episode 수를 기록합니다.
4. `exclude_insufficient_episode_actions.enabled`(기본 `true`): 독립 episode가
   기준보다 적은 조합을 보조 rolling cross-fit MAPE에서 제외합니다. 끄더라도 표본
   수, context 거리, 유효 전력 분모, coverage 기준은 그대로 유지됩니다. 공식
   fixed-action 평가는 별도 `fixed_optimal_action_holdout` support 설정을 사용합니다.

Boolean 옵션은 실행 시 바로 덮어쓸 수 있습니다.

```bash
./src/scripts/control/run_basic_mpc.sh --opts \
  MPC.evaluation.action_options.repeated_joint_actions.enabled=false \
  MPC.evaluation.action_options.episode_group_split.enabled=false \
  MPC.evaluation.action_options.exclude_insufficient_episode_actions.enabled=false
```

카테고리 통합은 대표값 선택이 데이터 의미에 영향을 주므로 YAML에 명시하는 방식을
권장합니다. source와 representative 모두 `MPC.control_variables`에 선언된 기존
카테고리여야 합니다.

```yaml
MPC:
  evaluation:
    action_options:
      category_consolidation:
        enabled: true
        mappings:
          냉난방설정온도(℃):
            18: 22
            26: 22
```

실행 결과에는 적용 설정과 통합 치환 건수가
`mpc_evaluation.json.evaluation_protocol.action_options` 및
`evaluation_protocol.control_preprocessing`에 저장됩니다. 조합별 허용 여부,
train 표본·episode 수, 제외 사유는 `control_support.actions`에서 확인할 수 있습니다.

사용 가능한 evaluator는 `extra_trees_residual_state`(기본),
`random_forest_residual_state`, `hist_gradient_boosting_residual_state`,
`neural_residual`, `forecasting_sequence`입니다. forecasting selector에는
`forecasting:PatchTST`처럼 `model_setting.yaml`에 등록된 모델명을 붙입니다. DLinear,
iTransformer, MICN, PatchTST, TimesNet, TimeXer, FreTS, Koopa, TimeMixer, TSMixer를
같은 adapter로 사용할 수 있습니다.

### 실제 Y 기반 공식 오프라인 MAPE

실제 설비에 새로운 MPC 명령을 적용할 수 없으므로, 공식 대표 지표는
`fixed_optimal_action_holdout_evaluation`에서 이미지의 평가 절차와 같은 형태로
계산합니다.

1. exact joint categorical action의 연속 1분 운전을 하나의 episode로 정의합니다.
   전력·상태 Y를 보지 않고 action, timestamp, episode 길이만 사용하여 episode 전체를
   평가 전용 train/validation/test 중 하나에 배정합니다. 따라서 같은 episode나 행이
   둘 이상의 split에 들어가지 않습니다.
2. train·validation·test에 각각 최소 support를 확보할 수 있는 관측 action만 후보로
   둡니다. 기본값은 support를 만족하는 모든 action(`top_k_actions=0`)이며, 후보별
   train 5표본·1개 episode, validation 3표본·1개 episode, test 5표본·1개 episode가
   필요합니다.
3. 선택용 evaluator는 grouped train 행으로만 학습하고 고정합니다. 모든 후보를
   동일한 validation 상태·외란 context에 순전파하여 예측 joint-state 만족률과
   예측 총 공조전력을 계산합니다. 설정한 만족률을 통과한 후보 중 예측 전력이 가장
   작은 하나를 최종 action으로 확정합니다.
4. 엄격 만족률을 통과한 후보가 없고 `allow_soft_constraint_fallback=true`이면,
   만족률이 가장 높은 후보를 먼저 고르고 동률일 때 예측 전력이 가장 작은 후보를
   선택합니다. 이 경우 MAPE 계산은 계속하지만 별도의 policy acceptance 상태 제약
   gate는 실패하므로 안전한 최적 제어로 주장할 수 없습니다.
5. 공식 총전력 report model 후보(`action_median`, Huber residual, ExtraTrees
   residual, component-aware ExtraTrees)는 grouped train으로만 학습하고, 확정 action의
   grouped validation MAPE로 하나를 선택합니다. family 선택에 test Y는 사용하지
   않습니다.
6. 선택 family를 grouped train+validation으로 재학습하고 파라미터를 고정합니다.
   확정 action과 정확히 같은 test 기록 행만 순전파하여 실제 총 공조전력 Y와 예측값의
   표준 MAPE를 직접 평균합니다. 이 report model은 MPC/MILP 목적함수·제약·action
   선택에 연결되지 않으며, 기존 component-sum evaluator 결과는 비교 진단으로 남깁니다.

```text
report(x_t, z_t, d_t) = [환기유효전력(W), 냉난방유효전력(W)]
total_power = 환기유효전력(W) + 냉난방유효전력(W)
MAPE(%) = 100 / n × sum(|actual - predicted| / |actual|)
```

`total_power_report_model`은 두 성분을 **동시에 직접 예측**하고, 총전력은 두
성분 예측값의 합으로 계산한다. 따라서 모든 report-model 산출물은 총전력 MAPE를
대표값으로 유지하면서도 `환기유효전력(W)`, `냉난방유효전력(W)` 각각의 MAPE/WAPE/MAE,
유효 MAPE 표본 수와 행별 실제·예측값을 함께 기록한다.

입력 묶음은 `MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model
.input_features`에서 독립적으로 선택한다.

```yaml
input_features:
  current_state: true       # x_t: 실내 상태 6개
  action: true              # z_t: 7개 categorical control의 one-hot 20개
  current_disturbance: true # d_t: 실외 외란 4개
```

기본값은 세 묶음 모두 사용한다. 최소 한 묶음은 켜야 하며, counterfactual MPC
비교에서 `action: false`이면 기록 action과 MPC action의 report 예측이 동일해지므로
제어 효과를 해석할 수 없다. 입력 선택·차원·실제 feature 이름은 결과 JSON의
`model_selection.model.input_features`에 남는다.

세 입력 조합의 component MAPE를 비교하려면 아래 task를 실행한다.

```bash
bash src/scripts/control/run_report_model_input_ablation.sh
```

기본 case는 `현재 상태+action+현재 외란`, `현재 상태+action`, `action만`이다.
각 case는 같은 시간순 train/validation/test split에서 family를 validation으로 선택하고,
기록 test action의 실제 환기·냉난방 전력과 비교한다. MILP는 재실행하지 않으므로 이는
controller 성능 비교가 아니라 **report model 입력 ablation**이다. 결과는
`saved_model/<data>/<environment>/report_model_input_ablation/<run>/` 아래 case별
`evaluation.json`, `test_predictions.csv`, 그리고 통합
`report_model_input_ablation_summary.csv`에 저장된다.

기본 설정에서는 같은 세 입력 case마다 하나의 공통 outcome-blind episode-group
holdout에서 validation 순위로 후보를 누적해 `previous_7`, `expanded_9`,
`expanded_10`, `expanded_12`, `expanded_13`, `expanded_17`, `exploratory_41`의
Top-K prefix를 함께 보고한다. 이는 기존 슬라이드와 동일하게 **총전력** `Pooled
MAPE`, coverage, worst action, action pass, 판정을 보고하고, 같은 고정 report
model의 환기·냉난방 component Pooled MAPE를 함께 기록한다. report-model family와
candidate 순위는 validation만으로 고정되고 test 전력은 선택에 쓰이지 않는다. 단,
validation이 선택한 K 하나만 공식 selected-K이고 나머지는 sensitivity-only다.
`previous_7`은 과거 run 재현이 아니라 이번 input-ablation의 validation-ranked
Top-7이라는 표시명이다. 결과는
`candidate_pool_expansion_summary.csv`, `candidate_pool_expansion_report.md`,
`expanded_candidate_pool/<입력 case>/`에 저장된다. 총전력 판정과
component MAPE를 같은 KPI로 혼동하지 않는다. `metric_targets`에서 총전력,
기본 component, 전력별 component MAPE 목표를 각각 설정할 수 있고, 표에는
`환기 판정`, `냉난방 판정`, `총전력 판정`을 독립적으로 기록한다.

공식 대표값은 하나로 고정한 action의 **총 공조유효전력 직접 행 평균 MAPE**입니다.
분모가 `mape_min_actual_power_w` 이하인 행은 제외하며, exact-match 수, 독립 episode
수, 유효 MAPE 표본 수를 함께 저장합니다. 설정 support를 만족하지 않거나 fallback을
명시적으로 끈 상태에서 선택 가능한 후보가 없으면 실제 Y를 대체 생성하지 않고
`primary_metric.value=null`로 남깁니다.

대표값 경로는
`mpc_evaluation.json.fixed_optimal_action_holdout_evaluation.primary_metric.value`
이며 같은 섹션의 `reportable`, `selection`, `split`, `test_evaluation`과 함께 읽어야
합니다. 편의를 위해 같은 객체가 JSON 최상위 `primary_metric`에도 복사됩니다.
기본 목표는 `primary_metric.target_percent=1.950`이고 달성 여부는
`primary_metric.target_achieved` 및 policy gate
`fixed_optimal_action_mape_target`에 기록됩니다.

모델 선택 근거는 `fixed_action_power_model_comparison.json`과 CSV에 저장됩니다.
행별 공식 예측, 기존 component-sum 예측, APE는
`fixed_optimal_action_test_predictions.csv`에서 함께 감사할 수 있습니다. validation
표본이 적으면 목표를 달성해도 일반화 증거가 약하므로 validation/test episode 수와
특정 episode가 차지하는 비중을 반드시 함께 확인해야 합니다.

이 grouped split은 희소한 반복 action에서 공식 수식의 실제-Y MAPE를 산출하기 위한
평가 전용 분할입니다. 미래 시점 일반화 성능을 주장하지 않으며, action을 실제 장비에
무작위 배정한 실험도 아니므로 선택 action의 인과적 절감 효과나 실제 최적성을 증명하지
않습니다. 미래 일반화와 정책 거동은 아래 chronological 평가를 별도로 확인해야 합니다.

### MPC 선택 action 기반 counterfactual report MAPE

`mpc_counterfactual_total_power_report`는 기록 test의 현재 state·외란을 유지한 채,
각 행마다 MILP가 새 first action을 선택하도록 한 평가다. report model은 시간순
train으로 family를 학습하고 validation으로 하나를 선택한 뒤 train+validation으로
재학습하여 test 전에 고정한다.

```text
recorded x_t, d_t → MILP first action z_t^MPC
                  ├→ frozen report model(x_t, z_t^recorded, d_t) = ŷ_t^recorded
                  └→ frozen report model(x_t, z_t^MPC, d_t)      = ŷ_t^MPC
```

두 MAPE를 함께 저장한다.

```text
MPC action vs recorded actual MAPE(%)
= 100 / n × sum(|ŷ_t^MPC - y_t^recorded| / |y_t^recorded|)

MPC action vs recorded-action report MAPE(%)
= 100 / n × sum(|ŷ_t^MPC - ŷ_t^recorded| / |ŷ_t^recorded|)
```

두 번째 값은 state·외란·report model을 고정하고 action만 바꾼 모델 기반 상대 차이다.
따라서 실제-Y forecast accuracy가 아니라 MPC action-effect 진단값이다. 두 baseline에
대한 예상 절감량도 함께 저장한다. 이 두 MAPE와 예상 절감량은 총전력뿐 아니라
환기유효전력·냉난방유효전력 각각에 대해서도 산출한다.

```text
recorded_actual_difference = y^recorded - ŷ^MPC
report_to_report_predicted_reduction = ŷ^recorded - ŷ^MPC
```

기본값은 일반 MPC policy와 동일하게 선택된 60분 test window의 모든 유효 transition을
사용한다. `row_stride_steps`, `maximum_evaluation_rows`로 실행량을 제한할 수 있다.
행별 기록은 `mpc_counterfactual_total_power_report_predictions.csv`, 요약은
`mpc_counterfactual_total_power_report_evaluation.json` 및
`mpc_evaluation.json.mpc_counterfactual_total_power_report_evaluation`에 저장된다.
행별 CSV에는 `recorded_actual_<전력명>`,
`recorded_action_report_predicted_<전력명>`,
`mpc_action_report_predicted_<전력명>` 열이 포함된다.

첫 번째 MAPE는 변경 action의 실제 전력이 없으므로 일반적인 동일-treatment 실제-Y
forecast MAPE와 다르며, 예상 제어 효과와 report model 오차가 함께 들어간다. 두 번째
MAPE·예상 절감률은 실제 Y를 쓰지 않으므로 실제 장비 절감률로 주장할 수 없다. action
변경률, 두 MAPE의 유효 coverage, report model의 별도 observed-action MAPE를 함께
보고해야 한다.

### 보조 MPC-selected-action rolling cross-fit MAPE

`mpc_selected_action_crossfit_evaluation`은 삭제하지 않고 보조 진단으로 유지합니다.
모든 폐루프 window에서 MPC가 선택한 action·context를 수집하고, 과거 데이터만으로
학습한 5개 expanding fold evaluator의 out-of-fold 실제 Y 중 exact action과 유사
context를 만족하는 행을 대응시킵니다. 조합별 총전력 MAPE를 MPC action 빈도로
가중한 값과 조합 동일 가중값을 함께 기록합니다. 이는 시간 방향 검증에는 더 엄격하지만
희소 action support와 context coverage에 민감하므로 공식 fixed-action MAPE와 역할이
다릅니다.

기존 `observed_control_combination_evaluation`은 validation에서 하나의 관측 조합을
선정하여 chronological test의 동일 조합으로 확인하는 이전 민감도 분석용으로만
유지하며 기본값은 비활성입니다.

각 60분 test window에서는 다음 순서를 60회 반복합니다.

1. 현재 evaluator 상태와 persistence 외란 예측으로 60분 MILP를 푼다.
2. 첫 번째 categorical 제어 조합만 적용한다.
3. 실제 미래 외란 한 step과 독립 evaluator로 전력·다음 상태를 생성한다.
4. 다음 분에 새로운 상태로 다시 최적화한다.

recorded control과 hold-last 정책도 같은 evaluator·초기 상태·외란에서 실행하므로
counterfactual 비교의 plant 조건이 동일합니다. recorded 정책의 evaluator rollout은
실제 계측 궤적과 다시 비교하여 evaluator MAPE, WAPE, 60분 에너지 오차와 상태
false-safe를 함께 기록합니다. MPC가 실제 선택한 궤적에서는 Affine 전력 예측과
독립 evaluator 전력의 MAPE·WAPE도 window별 및 전체로 집계하여 model exploitation
가능성을 확인합니다.

시계열 evaluator 입력은 최근 `lookback_steps`의 상태, one-hot 제어, 외란, 직전
전력과 시간 주기 feature입니다. 현재 시점 전력을 target 입력으로 사용하지 않고
이전 1분 전력을 사용해 누출을 막습니다. recorded, hold-last, MPC는 각각 독립
rollout session을 만들기 때문에 한 정책의 예측 이력이 다른 정책에 섞이지 않습니다.

`neural_residual`은 MPC의 Affine 계수를 재사용하지 않습니다. 별도 Ridge 선형
상태공간 모델을 학습하고 spectral radius를 기본 0.999 이하로 안정화한 뒤, 상태와
전력의 선형 예측 오차를 MLP가 보정합니다.

활성 evaluator는 다음처럼 변경합니다.

```bash
python src/main.py --task mpc --opts \
  MPC.independent_evaluator.model_family=neural_residual

python src/main.py --task mpc --opts \
  'MPC.independent_evaluator.model_family=forecasting:PatchTST'
```

`validation_model_families`에 후보를 넣으면 활성 family와 후보들을 train 80%로만
학습하고 validation 10%에서 one-step, recursive rollout, lead-time, power
MAPE/WAPE, 상태 false-safe 지표를 계산합니다. test는 후보 선택에 사용하지 않습니다.
정책 평가에 사용할 family는 항상 `model_family`로 명시하며, 이 모델만
train+validation으로 다시 학습한 뒤 test 및 정책 평가에 사용합니다.

```bash
python src/main.py --task mpc --opts \
  'MPC.independent_evaluator.validation_model_families=[random_forest_residual_state,neural_residual,forecasting:DLinear]'
```

기본 `max_windows: 0`은 연속된 test window 전체를 의미합니다. 빠른 smoke test만
양수로 제한합니다. 전체 에너지 절감률은 window 절감률 평균이 아니라
`(sum(baseline) - sum(MPC)) / sum(baseline)`으로 계산하고 paired-window bootstrap
95% 신뢰구간을 제공합니다.

offline simulated policy를 채택하려면 다음 조건을 모두 만족해야 합니다.

- Affine controller와 독립 evaluator의 전력 품질 gate 통과
- recursive rollout power WAPE와 상태 false-safe 기준 통과
- 최소 20개 window 및 bootstrap 절감률 하한이 0% 초과
- 모든 상태에서 제약 위반량과 최대 위반이 baseline보다 악화되지 않음
- fallback, 1분 deadline miss, train-support OOD 제어가 모두 0건
- 공식 fixed-action 실제-Y MAPE가 exact-match·episode·유효 표본 기준을 통과
- soft state-constraint fallback 없이 엄격 만족률을 통과한 공식 후보가 존재

훈련 구간에서 보지 못한 논리적으로 유효한 전체 제어 조합은 MILP no-good cut으로
직접 금지합니다. 현재 offline 결과는 실제 설비 절감 증거가 아니며 최적 제어를
설비에 적용하기 전까지 `actual_closed_loop_mape.available=false`로 기록됩니다.

## 딥러닝 전력 모델로 교체할 수 있는가?

이 절은 **MPC 최적화 내부의 전력 모델**에 대한 설명입니다. 위에서 선택한 neural
또는 forecasting evaluator는 최적화 바깥에서만 실행되므로 Affine+MILP 구조를
변경하지 않습니다.

가능하지만 모델 형태에 따라 최적화 문제도 바뀝니다.

- affine 또는 선형 piecewise 모델은 현재 MILP에 직접 포함할 수 있습니다.
- ReLU 신경망은 activation마다 big-M 이진변수를 추가하면 MILP로 바꿀 수 있지만
  변수 수와 계산량이 크게 증가하며 유효한 activation bound가 필요합니다.
- 임의의 비선형 모델을 그대로 사용하면 정확한 MILP가 아니므로 조합 열거,
  비선형 최적화, 샘플 기반 제어 같은 별도 optimizer가 필요합니다.

현재 구현은 해석 가능성, 물리 단위 검증, 정확한 이산 최적화를 우선하여 affine
모델을 기본 계약으로 둡니다.

## 구조

```text
src/
├── main.py                       # task dispatcher
├── runners/
│   ├── forecast_runner.py        # 딥러닝 예측 실행
│   └── mpc_runner.py             # MPC 실행
├── controllers/
│   ├── mixed_integer_mpc.py      # SciPy/HiGHS MILP
│   ├── action_crossfit.py        # MPC 선택 조합 rolling cross-fit 실제-Y MAPE
│   ├── fixed_action_evaluation.py # 공식 단일 조합 grouped holdout 실제-Y MAPE
│   ├── fixed_action_power_models.py # 평가전용 component-aware total-power 모델 선택
│   ├── evaluation.py             # recursive/closed-loop KPI와 bootstrap
│   └── pipeline.py               # 학습·검증·정책 평가·artifact 저장
├── dynamics/
│   ├── affine.py                 # 최적화용 상태/전력 모델
│   ├── evaluator_base.py         # stateless/sequence evaluator 공통 계약
│   ├── evaluator_factory.py      # 설정 기반 evaluator 생성
│   ├── nonlinear.py              # 기본 ExtraTrees 평가 plant
│   ├── sklearn_evaluators.py     # RF/HistGradientBoosting 평가 plant
│   └── neural_evaluators.py      # linear+MLP residual 및 forecasting adapter
├── data_provider/
│   ├── control_data.py           # MPC 물리 단위 데이터 계약
│   └── ...                       # forecast 전처리
├── models/                       # forecast 모델; sequence evaluator에서도 재사용
└── configs/
    ├── default_setting.yaml
    └── model_setting.yaml
```

## 예측 프레임워크의 타깃 계약

예측 모델은 더 이상 마지막 n개 입력 채널이 타깃이라고 가정하지 않습니다.
Dataset이 실제 categorical/numerical 입력 열과 타깃 열 이름을 metadata에
기록하고 채널 보존형 모델은 그 이름으로 출력 채널을 선택합니다. 타깃은 반드시
numerical input에도 포함되어야 하며 아니면 모델 생성 시 명확한 오류가 납니다.
