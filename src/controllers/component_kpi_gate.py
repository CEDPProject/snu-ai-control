"""성분별·총전력 전력예측을 설정 가능한 floor로 MAPE/WAPE 계산하고 pass/fail을 판정한다.

방안 2(지표 조정)의 마무리 모듈.

배경: 냉난방유효전력은 실제값의 절반 이상이 near-zero(≤1W)라 MAPE 분모가 폭발한다.
따라서 (a) MAPE의 near-zero 분모를 floor로 제외하거나 (b) WAPE(Σ|오차|/Σ|실제|)를
쓰는 것이 간헐부하에 대한 올바른 평가다. 기존 코드는 WAPE와 floor 기반 MAPE를
'계산·저장'만 했고 공식 pass/fail 판정으로는 쓰지 않았다. 이 모듈은 그 판정을 추가한다.

원칙:
- MAPE/WAPE 정의는 기존 프레임워크와 동일(단순·투명한 수식).
- 실제(recorded) action 기반 예측과 MPC action 기반 예측을 **둘 다** 판정한다.
- floor와 target(≤1.95%)은 설정으로 조정 가능하다.
- 이 값은 report model의 전력예측 정확도이며, MPC 절감/제어 성능이 아니다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

TOTAL_TARGET_LABEL = "총전력"


def component_mape_percent(
    actual: np.ndarray, predicted: np.ndarray, floor_w: float
) -> tuple[float | None, int]:
    """MAPE(%) = 100/N · Σ_{|actual|>floor} |actual-pred|/|actual|. (값, 유효표본수)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > float(floor_w)
    valid = int(mask.sum())
    if valid == 0:
        return None, 0
    value = float(
        100.0 * np.mean(np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask]))
    )
    return value, valid


def component_wape_percent(
    actual: np.ndarray, predicted: np.ndarray
) -> float | None:
    """WAPE(%) = 100 · Σ|actual-pred| / Σ|actual|. near-zero에 강건한 간헐부하 지표."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum(np.abs(actual)))
    if denominator <= 1e-12:
        return None
    return float(100.0 * np.sum(np.abs(actual - predicted)) / denominator)


def _target_row(
    *,
    action_source: str,
    target: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    floor_w: float,
    target_percent: float,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[finite]
    predicted = predicted[finite]
    mape, valid = component_mape_percent(actual, predicted, floor_w)
    wape = component_wape_percent(actual, predicted)
    return {
        "action_source": action_source,
        "target": target,
        "sample_count": int(len(actual)),
        "mape_valid_samples": valid,
        "floor_w": float(floor_w),
        "target_percent": float(target_percent),
        "mape_percent": mape,
        "wape_percent": wape,
        "mape_pass": bool(mape is not None and mape <= target_percent),
        "wape_pass": bool(wape is not None and wape <= target_percent),
        # MAPE 또는 WAPE 중 하나라도 통과하면 pass(둘 다 정당한 지표이므로).
        "any_metric_pass": bool(
            (mape is not None and mape <= target_percent)
            or (wape is not None and wape <= target_percent)
        ),
    }


def component_kpi_gate_from_counterfactual(
    frame: pd.DataFrame | None,
    power_names: Sequence[str],
    *,
    floor_w: float,
    target_percent: float,
) -> list[dict[str, Any]]:
    """counterfactual 예측 DataFrame에서 성분별·총전력 pass/fail 행 목록을 만든다.

    frame 컬럼 규약(mpc_counterfactual_total_power_report_predictions.csv):
      recorded_actual_<component>
      recorded_action_report_predicted_<component>   (실제 action → report)
      mpc_action_report_predicted_<component>        (MPC action → report)
      recorded_total_hvac_power_w
      recorded_action_report_predicted_total_hvac_power_w
      mpc_action_report_predicted_total_hvac_power_w
    """
    if frame is None or len(frame) == 0:
        return []
    names = [str(name) for name in power_names]
    sources = [
        ("recorded_action", "recorded_action_report_predicted_"),
        ("mpc_action", "mpc_action_report_predicted_"),
    ]
    rows: list[dict[str, Any]] = []
    for action_source, prefix in sources:
        # 성분별
        for name in names:
            actual_col = f"recorded_actual_{name}"
            pred_col = f"{prefix}{name}"
            if actual_col not in frame.columns or pred_col not in frame.columns:
                continue
            rows.append(
                _target_row(
                    action_source=action_source,
                    target=name,
                    actual=pd.to_numeric(frame[actual_col], errors="coerce"),
                    predicted=pd.to_numeric(frame[pred_col], errors="coerce"),
                    floor_w=floor_w,
                    target_percent=target_percent,
                )
            )
        # 총전력
        total_actual_col = "recorded_total_hvac_power_w"
        total_pred_col = f"{prefix}total_hvac_power_w"
        if total_actual_col in frame.columns and total_pred_col in frame.columns:
            rows.append(
                _target_row(
                    action_source=action_source,
                    target=TOTAL_TARGET_LABEL,
                    actual=pd.to_numeric(frame[total_actual_col], errors="coerce"),
                    predicted=pd.to_numeric(frame[total_pred_col], errors="coerce"),
                    floor_w=floor_w,
                    target_percent=target_percent,
                )
            )
    return rows


def summarize_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """gate 행에서 '어느 target/지표라도 통과했는지'를 요약한다."""
    rows = list(rows)
    any_pass = any(bool(r.get("any_metric_pass")) for r in rows)
    passed = [
        {
            "action_source": r["action_source"],
            "target": r["target"],
            "metric": ("mape" if r.get("mape_pass") else "wape"),
            "value_percent": (
                r.get("mape_percent") if r.get("mape_pass") else r.get("wape_percent")
            ),
        }
        for r in rows
        if r.get("any_metric_pass")
    ]
    return {
        "any_target_or_metric_passes_target": any_pass,
        "passing": passed,
        "claim_boundary": "report_model_power_prediction_accuracy_not_control_savings",
    }
