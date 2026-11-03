import unittest

import numpy as np
import pandas as pd

from controllers.component_kpi_gate import (
    component_kpi_gate_from_counterfactual,
    component_mape_percent,
    component_wape_percent,
    summarize_gate,
)


class ComponentMetricTest(unittest.TestCase):
    def test_mape_floor_excludes_near_zero(self):
        actual = np.array([10.0, 20.0, 0.05, 4.0])
        predicted = np.array([11.0, 18.0, 0.5, 4.0])
        # floor 0.1: 유효 3개(10,20,4) → mean(0.1,0.1,0)=6.6667%
        value, valid = component_mape_percent(actual, predicted, 0.1)
        self.assertEqual(valid, 3)
        self.assertAlmostEqual(value, 6.66666667, places=5)
        # floor 5: 유효 2개(10,20) → 10%
        value, valid = component_mape_percent(actual, predicted, 5.0)
        self.assertEqual(valid, 2)
        self.assertAlmostEqual(value, 10.0, places=5)

    def test_mape_none_when_no_valid_samples(self):
        actual = np.array([0.0, 0.05])
        predicted = np.array([1.0, 1.0])
        value, valid = component_mape_percent(actual, predicted, 0.1)
        self.assertIsNone(value)
        self.assertEqual(valid, 0)

    def test_wape_is_robust_to_near_zero(self):
        actual = np.array([10.0, 20.0, 0.05, 4.0])
        predicted = np.array([11.0, 18.0, 0.5, 4.0])
        # 100*(1+2+0.45+0)/(10+20+0.05+4)=10.1322%
        self.assertAlmostEqual(component_wape_percent(actual, predicted), 10.13216, places=4)

    def test_wape_none_when_denominator_zero(self):
        self.assertIsNone(component_wape_percent(np.zeros(3), np.ones(3)))


class ComponentGateTest(unittest.TestCase):
    def _frame(self):
        # 환기: 안정(≈40W), 정확 예측 → pass. 냉난방: near-zero, 부정확 → fail.
        n = 50
        vent_actual = np.full(n, 40.0)
        vent_pred = vent_actual * 1.005  # 0.5% 오차 → target 통과
        hvac_actual = np.array([0.2, 30.0] * (n // 2))
        hvac_pred = hvac_actual + 5.0
        return pd.DataFrame(
            {
                "recorded_actual_환기유효전력(W)": vent_actual,
                "recorded_action_report_predicted_환기유효전력(W)": vent_pred,
                "mpc_action_report_predicted_환기유효전력(W)": vent_pred,
                "recorded_actual_냉난방유효전력(W)": hvac_actual,
                "recorded_action_report_predicted_냉난방유효전력(W)": hvac_pred,
                "mpc_action_report_predicted_냉난방유효전력(W)": hvac_pred,
                "recorded_total_hvac_power_w": vent_actual + hvac_actual,
                "recorded_action_report_predicted_total_hvac_power_w": vent_pred + hvac_pred,
                "mpc_action_report_predicted_total_hvac_power_w": vent_pred + hvac_pred,
            }
        )

    def test_gate_judges_both_action_sources_and_all_targets(self):
        names = ["환기유효전력(W)", "냉난방유효전력(W)"]
        rows = component_kpi_gate_from_counterfactual(
            self._frame(), names, floor_w=5.0, target_percent=1.95
        )
        # 2 sources x (2 components + total) = 6 rows
        self.assertEqual(len(rows), 6)
        sources = {r["action_source"] for r in rows}
        self.assertEqual(sources, {"recorded_action", "mpc_action"})
        vent = next(
            r for r in rows if r["target"] == "환기유효전력(W)" and r["action_source"] == "recorded_action"
        )
        self.assertTrue(vent["mape_pass"])  # 0.5% ≤ target
        hvac = next(
            r for r in rows if r["target"] == "냉난방유효전력(W)" and r["action_source"] == "recorded_action"
        )
        self.assertFalse(hvac["mape_pass"])  # 부정확 → 실패

    def test_empty_frame_returns_no_rows(self):
        self.assertEqual(
            component_kpi_gate_from_counterfactual(pd.DataFrame(), ["환기유효전력(W)"], floor_w=5.0, target_percent=1.95),
            [],
        )

    def test_summary_reports_passing_targets(self):
        names = ["환기유효전력(W)", "냉난방유효전력(W)"]
        rows = component_kpi_gate_from_counterfactual(
            self._frame(), names, floor_w=5.0, target_percent=1.95
        )
        summary = summarize_gate(rows)
        self.assertTrue(summary["any_target_or_metric_passes_target"])
        self.assertTrue(any(p["target"] == "환기유효전력(W)" for p in summary["passing"]))


if __name__ == "__main__":
    unittest.main()
