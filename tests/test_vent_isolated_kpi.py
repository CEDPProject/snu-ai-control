"""환기-격리 KPI 모듈 계약 테스트."""

import unittest

import numpy as np
import pandas as pd

from controllers.vent_isolated_kpi import (
    _norm,
    fit_vent_isolated_predictor,
    resolve_vent_control_names,
    resolve_vent_target_name,
    summarize_vent_isolated,
    vent_isolated_kpi_from_counterfactual,
)

# 두 개의 관측된 환기 액션(결정적 전력) + 한 개의 미관측 액션
_ACTIONS = {
    0: ({'환기전원': 'ON', '환기풍량': '2', '환기운전모드': '환기'}, 40.0),
    1: ({'환기전원': 'ON', '환기풍량': '4', '환기운전모드': '바이패스'}, 60.0),
    2: ({'환기전원': 'OFF', '환기풍량': '0', '환기운전모드': '꺼짐'}, 24.0),
}


class _FakeVar:
    def __init__(self, name):
        self.name = name


class _FakeEncoder:
    variables = [
        _FakeVar('환기전원'),
        _FakeVar('환기풍량'),
        _FakeVar('환기운전모드'),
        _FakeVar('냉난방전원'),
    ]

    def decode_action(self, vector):
        code = int(vector[0])
        decoded = dict(_ACTIONS[code][0])
        decoded['냉난방전원'] = 'ON' if vector[1] else 'OFF'
        return decoded


class _FakeData:
    """환기 전력이 환기 액션의 결정적 함수인 합성 데이터."""

    def __init__(self, codes, hvac_flags, vent_power_noise=0.0):
        self.encoder = _FakeEncoder()
        self.power_names = ('환기유효전력(W)', '냉난방유효전력(W)')
        self.controls = np.column_stack([codes, hvac_flags]).astype(float)
        vent = np.array([_ACTIONS[int(c)][1] for c in codes], dtype=float)
        vent = vent + vent_power_noise
        hvac = np.array(hvac_flags, dtype=float) * 100.0
        self.powers = np.column_stack([vent, hvac])
        self.states = np.zeros((len(codes), 1))

    def valid_transition_rows(self, rows):
        rows = np.asarray(list(rows), dtype=int)
        return np.unique(rows)


def _frame_row(rec_code, mpc_code, actual):
    rec = _ACTIONS[rec_code][0]
    mpc = _ACTIONS[mpc_code][0]
    return {
        'recorded_control_환기전원': rec['환기전원'],
        'recorded_control_환기풍량': int(float(rec['환기풍량'])),
        'recorded_control_환기운전모드': rec['환기운전모드'],
        'mpc_control_환기전원': mpc['환기전원'],
        'mpc_control_환기풍량': int(float(mpc['환기풍량'])),
        'mpc_control_환기운전모드': mpc['환기운전모드'],
        'recorded_actual_환기유효전력(W)': actual,
    }


class NormalizeTest(unittest.TestCase):
    def test_numeric_variants_collapse(self):
        self.assertEqual(_norm(2), '2')
        self.assertEqual(_norm(2.0), '2')
        self.assertEqual(_norm('2'), '2')
        self.assertEqual(_norm("'2'"), '2')

    def test_string_labels_preserved(self):
        self.assertEqual(_norm('ON'), 'ON')
        self.assertEqual(_norm("'환기'"), '환기')


class ResolveTest(unittest.TestCase):
    def test_resolve_vent_controls_and_target(self):
        data = _FakeData([0, 1], [0, 1])
        self.assertEqual(
            resolve_vent_control_names(data),
            ['환기전원', '환기풍량', '환기운전모드'],
        )
        self.assertEqual(resolve_vent_target_name(data), '환기유효전력(W)')


class PredictorTest(unittest.TestCase):
    def test_group_mean_lookup_matches_deterministic_power(self):
        codes = [0, 0, 1, 1]
        data = _FakeData(codes, [0, 1, 0, 1])
        predictor = fit_vent_isolated_predictor(
            data,
            np.arange(4),
            vent_control_names=['환기전원', '환기풍량', '환기운전모드'],
            target_name='환기유효전력(W)',
        )
        self.assertAlmostEqual(
            predictor.predict_one(('ON', '2', '환기')), 40.0
        )
        self.assertAlmostEqual(
            predictor.predict_one(('ON', '4', '바이패스')), 60.0
        )
        # 미관측 키는 전역 평균으로 폴백
        self.assertAlmostEqual(
            predictor.predict_one(('OFF', '0', '꺼짐')), 50.0
        )


class KpiTest(unittest.TestCase):
    def _data(self):
        codes = [0, 1, 0, 1, 0, 1]
        return _FakeData(codes, [0, 1, 0, 1, 0, 1])

    def test_recorded_and_mpc_retained_pass(self):
        data = self._data()
        # 4행: MPC가 환기 유지(냉난방만 변경). 2행: MPC가 환기 변경(반사실).
        frame = pd.DataFrame([
            _frame_row(0, 0, 40.0),
            _frame_row(1, 1, 60.0),
            _frame_row(0, 0, 40.0),
            _frame_row(1, 1, 60.0),
            _frame_row(0, 1, 40.0),  # 환기 변경 → retained 아님
            _frame_row(1, 0, 60.0),  # 환기 변경 → retained 아님
        ])
        rows = vent_isolated_kpi_from_counterfactual(
            data, frame, np.arange(6), floor_w=5.0, target_percent=1.95
        )
        by_source = {r['action_source']: r for r in rows}
        rec = by_source['recorded_action']
        mpc = by_source['mpc_action_vent_retained']

        self.assertEqual(rec['sample_count'], 6)
        self.assertAlmostEqual(rec['coverage_percent'], 100.0)
        self.assertEqual(rec['mape_percent'], 0.0)  # 결정적 → 완벽 예측
        self.assertTrue(rec['mape_pass'])

        self.assertEqual(mpc['sample_count'], 4)  # 환기 유지 행만
        self.assertAlmostEqual(mpc['coverage_percent'], 100.0 * 4 / 6)
        self.assertEqual(mpc['mape_percent'], 0.0)
        self.assertTrue(mpc['mape_pass'])

    def test_seen_support_percent_reports_extrapolation(self):
        # 학습엔 code 0,1만. 프레임 recorded는 모두 관측된 것.
        data = _FakeData([0, 1, 0, 1], [0, 1, 0, 1])
        frame = pd.DataFrame([_frame_row(0, 0, 40.0), _frame_row(1, 1, 60.0)])
        rows = vent_isolated_kpi_from_counterfactual(
            data, frame, np.arange(4), floor_w=5.0, target_percent=1.95
        )
        rec = {r['action_source']: r for r in rows}['recorded_action']
        self.assertAlmostEqual(rec['seen_support_percent'], 100.0)

    def test_summary_flags_pass(self):
        data = self._data()
        frame = pd.DataFrame([_frame_row(0, 0, 40.0), _frame_row(1, 1, 60.0)])
        rows = vent_isolated_kpi_from_counterfactual(
            data, frame, np.arange(6), floor_w=5.0, target_percent=1.95
        )
        summary = summarize_vent_isolated(rows)
        self.assertTrue(summary['any_metric_pass'])
        self.assertEqual(
            summary['claim_boundary'],
            'vent_isolated_report_predicts_vent_power_from_vent_controls_only',
        )
        self.assertIsNotNone(summary['best'])


if __name__ == '__main__':
    unittest.main()
