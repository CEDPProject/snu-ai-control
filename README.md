# SNU-AI-Control

이 저장소에는 서로 독립적인 두 프레임워크가 공존합니다.

- `forecast`: 측정 데이터로 미래 전력을 예측하는 딥러닝 프레임워크
- `mpc`: 상태 제약 아래 이산 제어 조합의 전력 소비를 최소화하는 mixed-integer MPC

두 프레임워크는 같은 CSV를 사용할 수 있지만 모델, 전처리, 실행 루프를 공유하지
않습니다. 특히 MPC는 스케일되지 않은 물리 단위 데이터를 사용합니다.

## 설치

```bash
pip install -r requirements.txt
```

MPC는 SciPy에 포함된 HiGHS MILP solver를 사용하므로 별도의 상용 solver나
`cvxpy` 설치가 필요하지 않습니다.

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

## MPC 평가 프로토콜

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
   두고 전체 관측 빈도 Top-K를 남깁니다. 기본값은 Top-8, 후보별 train 20표본·3개
   episode, validation 3표본·3개 episode, test 30표본·3개 episode입니다.
3. 선택용 evaluator는 grouped train 행으로만 학습하고 고정합니다. 모든 후보를
   동일한 validation 상태·외란 context에 순전파하여 예측 joint-state 만족률과
   예측 총 공조전력을 계산합니다. 설정한 만족률을 통과한 후보 중 예측 전력이 가장
   작은 하나를 최종 action으로 확정합니다.
4. 엄격 만족률을 통과한 후보가 없고 `allow_soft_constraint_fallback=true`이면,
   만족률이 가장 높은 후보를 먼저 고르고 동률일 때 예측 전력이 가장 작은 후보를
   선택합니다. 이 경우 MAPE 계산은 계속하지만 별도의 policy acceptance 상태 제약
   gate는 실패하므로 안전한 최적 제어로 주장할 수 없습니다.
5. 공식 총전력 report model 후보(`action_median`, Huber residual, ExtraTrees
   residual, direct-total ExtraTrees)는 grouped train으로만 학습하고, 확정 action의
   grouped validation MAPE로 하나를 선택합니다. family 선택에 test Y는 사용하지
   않습니다.
6. 선택 family를 grouped train+validation으로 재학습하고 파라미터를 고정합니다.
   확정 action과 정확히 같은 test 기록 행만 순전파하여 실제 총 공조전력 Y와 예측값의
   표준 MAPE를 직접 평균합니다. 이 report model은 MPC/MILP 목적함수·제약·action
   선택에 연결되지 않으며, 기존 component-sum evaluator 결과는 비교 진단으로 남깁니다.

```text
total_power = 환기유효전력(W) + 냉난방유효전력(W)
MAPE(%) = 100 / n × sum(|actual - predicted| / |actual|)
```

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
│   ├── fixed_action_power_models.py # 평가전용 direct total-power 모델 선택
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
