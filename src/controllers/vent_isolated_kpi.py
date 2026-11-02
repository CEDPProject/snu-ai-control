"""환기-격리(component-isolated) report model KPI — additive, default-safe.

환기 전력(``환기유효전력(W)``)은 환기 제어(전원/풍량/운전모드)의 거의 결정적
함수이며 냉난방 입력과 사실상 독립이다. 그런데 joint report model은 환기 head가
냉난방 입력에 오염되어(동일 환기 제어인데도 냉난방 변경만으로 환기 예측이 수 W
흔들림) 전체 test에서 환기 MAPE가 크게 뜬다.

이 모듈은 **냉난방 입력을 배제한 환기-전용 예측기**를 별도로 적합해, MPC가 냉난방을
어떻게 바꾸든 불변인 환기 전력 예측의 정확도를 KPI로 산출한다. 기존 파이프라인·평가
경로(joint report model, candidate-pool, counterfactual)는 전혀 바꾸지 않는다 —
counterfactual 산출물(프레임)을 입력으로 받아 추가 산출물만 낸다.

주장 한계(claim boundary):
  이 KPI는 '환기 전력이 환기 제어만으로 얼마나 잘 예측되는가'만 측정한다. MPC가
  환기 sub-action을 관측된 다른 값으로 바꾼 행은 실제 전력이 존재하지 않는
  반사실이므로, **실제-대조 정확도는 '환기 sub-action 유지' 행에서만** 계산하고
  그 커버리지를 함께 공개한다. 예측기가 학습에서 본 적 없는 환기 제어로 외삽한
  비율(seen_support_percent)도 함께 보고한다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from controllers.component_kpi_gate import (
    component_mape_percent,
    component_wape_percent,
)

CLAIM_BOUNDARY = 'vent_isolated_report_predicts_vent_power_from_vent_controls_only'


def _norm(value: Any) -> str:
    """제어 라벨을 프레임/디코더 양쪽에서 동일하게 정규화한다.

    풍량처럼 숫자형 라벨은 ``2``/``2.0``/``'2'`` 이 모두 ``'2'`` 로 모이도록,
    정수값이면 정수 문자열로 통일한다. 그 외에는 따옴표·공백을 제거한 문자열.
    """
    text = str(value).strip().strip("'\"")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(number) and float(number).is_integer():
        return str(int(number))
    return repr(number)


def resolve_vent_control_names(data) -> list[str]:
    """인코더의 제어 변수 중 이름이 '환기'로 시작하는 것들(전원/풍량/운전모드)."""
    names = [
        str(variable.name)
        for variable in data.encoder.variables
        if str(variable.name).startswith('환기')
    ]
    return names


def resolve_vent_target_name(data) -> str:
    """전력 성분 중 '환기'가 들어간 이름(예: 환기유효전력(W))."""
    for name in data.power_names:
        if '환기' in str(name):
            return str(name)
    raise ValueError('환기 전력 성분(power_names에 "환기" 포함)을 찾을 수 없습니다')


class VentIsolatedPredictor:
    """환기 제어 조합 → 환기 전력. 학습 그룹평균 lookup(냉난방 무관, 결정적).

    환기 전력이 환기 제어의 결정적 함수라는 실증(그룹내 std ≈ 2W)에 근거한다.
    비선형 트리 회귀와 수치가 동일하므로, 재현성·해석성을 위해 그룹평균을 쓴다.
    """

    def __init__(self, table: Mapping[tuple, float], global_mean: float):
        self._table = dict(table)
        self._global_mean = float(global_mean)

    @property
    def seen_keys(self) -> set:
        return set(self._table.keys())

    def predict_one(self, key: tuple) -> float:
        return float(self._table.get(key, self._global_mean))

    def predict(self, keys: Sequence[tuple]) -> np.ndarray:
        return np.asarray([self.predict_one(k) for k in keys], dtype=float)


def fit_vent_isolated_predictor(
    data,
    refit_rows: Sequence[int],
    *,
    vent_control_names: Sequence[str],
    target_name: str,
) -> VentIsolatedPredictor:
    """train(+val) 행에서 환기 제어 조합별 환기 전력 평균으로 예측기를 적합한다."""
    rows = np.asarray(data.valid_transition_rows(refit_rows), dtype=int)
    if rows.size == 0:
        raise ValueError('환기-격리 예측기 학습에 사용할 유효 전이 행이 없습니다')
    target_index = list(data.power_names).index(str(target_name))
    keys: list[tuple] = []
    values: list[float] = []
    for row in rows:
        decoded = data.encoder.decode_action(data.controls[row])
        key = tuple(_norm(decoded[name]) for name in vent_control_names)
        keys.append(key)
        values.append(float(data.powers[row, target_index]))
    values_array = np.asarray(values, dtype=float)
    frame = pd.DataFrame({'key': keys, 'y': values_array})
    table = frame.groupby('key')['y'].mean().to_dict()
    return VentIsolatedPredictor(table, float(values_array.mean()))


def _frame_vent_keys(
    frame: pd.DataFrame, prefix: str, vent_control_names: Sequence[str]
) -> list[tuple]:
    columns = [f'{prefix}{name}' for name in vent_control_names]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f'counterfactual 프레임에 컬럼 없음: {missing}')
    keys = []
    for _, record in frame[columns].iterrows():
        keys.append(tuple(_norm(record[column]) for column in columns))
    return keys


def vent_isolated_kpi_from_counterfactual(
    data,
    frame: pd.DataFrame,
    refit_rows: Sequence[int],
    *,
    vent_control_names: Sequence[str] | None = None,
    target_name: str | None = None,
    floor_w: float = 5.0,
    target_percent: float = 1.950,
) -> list[dict[str, Any]]:
    """환기-격리 예측기로 recorded/MPC action 환기 전력 KPI 행들을 만든다.

    - recorded_action: 실제 환기 전력 vs 예측(recorded 환기 제어). 전체 test 행에서
      유효(실제값이 recorded action 것이므로). 외삽 비율도 함께 공개.
    - mpc_action_vent_retained: MPC가 환기 sub-action을 유지한 행에서만 실제 대조.
      (환기를 바꾼 행은 반사실 → 실제 없음. 커버리지로 공개.)
    """
    if vent_control_names is None:
        vent_control_names = resolve_vent_control_names(data)
    if target_name is None:
        target_name = resolve_vent_target_name(data)
    vent_control_names = list(vent_control_names)

    frame = frame.reset_index(drop=True)
    total_rows = int(len(frame))
    actual_column = f'recorded_actual_{target_name}'
    if actual_column not in frame.columns:
        raise KeyError(f'counterfactual 프레임에 실제값 컬럼 없음: {actual_column}')
    actual = frame[actual_column].to_numpy(dtype=float)

    predictor = fit_vent_isolated_predictor(
        data,
        refit_rows,
        vent_control_names=vent_control_names,
        target_name=target_name,
    )
    recorded_keys = _frame_vent_keys(frame, 'recorded_control_', vent_control_names)
    mpc_keys = _frame_vent_keys(frame, 'mpc_control_', vent_control_names)
    recorded_pred = predictor.predict(recorded_keys)
    mpc_pred = predictor.predict(mpc_keys)

    seen = predictor.seen_keys
    vent_retained = np.asarray(
        [rk == mk for rk, mk in zip(recorded_keys, mpc_keys)], dtype=bool
    )
    recorded_seen = np.asarray([k in seen for k in recorded_keys], dtype=bool)
    mpc_seen = np.asarray([k in seen for k in mpc_keys], dtype=bool)

    inputs_used = ','.join(vent_control_names)
    rows: list[dict[str, Any]] = []

    def _row(action_source, mask, predicted, coverage_denominator, seen_mask):
        mask = np.asarray(mask, dtype=bool)
        sub_actual = actual[mask]
        sub_pred = np.asarray(predicted)[mask]
        mape, valid = component_mape_percent(sub_actual, sub_pred, floor_w)
        wape = component_wape_percent(sub_actual, sub_pred)
        coverage = (
            100.0 * float(mask.sum()) / float(coverage_denominator)
            if coverage_denominator
            else 0.0
        )
        seen_percent = (
            100.0 * float(np.asarray(seen_mask)[mask].sum()) / float(mask.sum())
            if mask.sum()
            else None
        )
        return {
            'action_source': action_source,
            'target': str(target_name),
            'model': 'vent_isolated_group_mean',
            'inputs_used': inputs_used,
            'sample_count': int(mask.sum()),
            'coverage_percent': coverage,
            'seen_support_percent': seen_percent,
            'mape_valid_samples': int(valid),
            'floor_w': float(floor_w),
            'target_percent': float(target_percent),
            'mape_percent': mape,
            'wape_percent': wape,
            'mape_pass': bool(mape is not None and mape <= target_percent),
            'wape_pass': bool(wape is not None and wape <= target_percent),
        }

    # recorded action: 실제값이 recorded action에 대응하므로 전체 행에서 유효.
    rows.append(
        _row(
            'recorded_action',
            np.ones(total_rows, dtype=bool),
            recorded_pred,
            total_rows,
            recorded_seen,
        )
    )
    # MPC action: 환기 sub-action 유지 행에서만 실제 대조 가능.
    rows.append(
        _row(
            'mpc_action_vent_retained',
            vent_retained,
            mpc_pred,
            total_rows,
            mpc_seen,
        )
    )
    return rows


def summarize_vent_isolated(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """KPI 행들의 통과 여부 요약(어떤 행이라도 ≤target이면 pass)."""
    any_pass = any(bool(row.get('mape_pass') or row.get('wape_pass')) for row in rows)
    best = None
    for row in rows:
        for metric in ('mape_percent', 'wape_percent'):
            value = row.get(metric)
            if value is None:
                continue
            if best is None or value < best['value']:
                best = {
                    'action_source': row['action_source'],
                    'metric': metric,
                    'value': float(value),
                    'coverage_percent': row.get('coverage_percent'),
                }
    return {
        'claim_boundary': CLAIM_BOUNDARY,
        'any_metric_pass': bool(any_pass),
        'best': best,
    }
