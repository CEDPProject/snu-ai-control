"""strict/provisional MPC KPI를 coverage·신뢰등급과 함께 요약한다.

이 모듈의 목적은 support가 부족해 공식 strict KPI가 null이 되는 run도 "실패"로
끝내지 않고, 완화(relaxed) 조건에서의 값·coverage·신뢰등급을 남겨 해석 가능하게
만드는 것이다.

원칙 (반드시 유지):
- strict 공식 KPI의 의미와 계산은 그대로 둔다. 이 모듈은 요약만 한다.
- provisional(완화) 값은 절대 공식값으로 승격하지 않는다. 항상
  ``claim_boundary="diagnostic_only_not_official_kpi"``를 붙인다.
- 모든 값에는 coverage(실제 평가된 test 행 비율)와 confidence_tier를 병기한다.

순수 함수 모듈이라 파이프라인 의존성이 없다. runner가 이미 계산한 결과(dict/rows)만
받아 요약 dict를 돌려준다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

CLAIM_BOUNDARY = "diagnostic_only_not_official_kpi"
DEFAULT_TARGET_MAPE_PERCENT = 1.950
STRICT_SOURCE = "full_mpc_selected_evaluator.official_mape"


def _to_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    result = _to_float(value)
    return int(result) if result is not None else None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _confidence_tier(
    *,
    reportable: bool,
    target_achieved: bool,
    coverage_percent: float | None,
    sample_count: int | None,
    tiers: Mapping[str, Any],
) -> str:
    """coverage와 sample 수 기준으로 신뢰등급을 매긴다.

    높음/중간은 target 통과 + coverage·sample 충족, 낮음은 target 통과이나 규모가
    작음, 진단용은 reportable이나 target 실패, 없음은 값 자체가 없음.
    """
    if not reportable:
        return "none"
    if not target_achieved:
        return "diagnostic"
    coverage = coverage_percent if coverage_percent is not None else 0.0
    samples = sample_count if sample_count is not None else 0
    high = dict(tiers.get("high") or {})
    medium = dict(tiers.get("medium") or {})
    if coverage >= float(high.get("minimum_coverage_percent", 40.0)) and samples >= int(
        high.get("minimum_sample_count", 5000)
    ):
        return "high"
    if coverage >= float(medium.get("minimum_coverage_percent", 20.0)) and samples >= int(
        medium.get("minimum_sample_count", 1000)
    ):
        return "medium"
    return "low"


def _select_provisional(
    rows: Sequence[Mapping[str, Any]],
    include_scenarios: Sequence[str],
    selection_rule: str,
    fallback_rule: str,
) -> dict[str, Any] | None:
    """완화 시나리오 행에서 provisional 대표값을 고른다.

    1) target 통과한 시나리오가 있으면 selection_rule로 선택
       (target_pass_highest_coverage: coverage 최대).
    2) 없으면 fallback_rule로 선택(lowest_mape_diagnostic: MAPE 최저, 진단용).
    """
    allowed = {str(name) for name in include_scenarios} if include_scenarios else None
    candidates: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("scenario"))
        if allowed is not None and name not in allowed:
            continue
        if not _to_bool(row.get("reportable")):
            continue
        mape = _to_float(row.get("pooled_test_mape_percent"))
        if mape is None:
            continue
        candidates.append(
            {
                "source": name,
                "mape_percent": mape,
                "coverage_percent": _to_float(row.get("test_exact_match_coverage_percent")),
                "evaluated_sample_count": _to_int(row.get("test_exact_match_count")),
                "selected_candidate_count": _to_int(row.get("selected_candidate_count")),
                "target_achieved": _to_bool(row.get("target_achieved")),
            }
        )
    if not candidates:
        return None

    passed = [c for c in candidates if c["target_achieved"]]
    if passed:
        if selection_rule == "target_pass_highest_coverage":
            chosen = max(passed, key=lambda c: (c["coverage_percent"] or 0.0))
        else:  # 안전한 기본값: MAPE 최저
            chosen = min(passed, key=lambda c: c["mape_percent"])
        chosen["applied_rule"] = selection_rule
        return chosen

    # target 통과가 없으면 진단용 fallback
    if fallback_rule == "lowest_mape_diagnostic":
        chosen = min(candidates, key=lambda c: c["mape_percent"])
    else:
        chosen = max(candidates, key=lambda c: (c["coverage_percent"] or 0.0))
    chosen["applied_rule"] = fallback_rule
    return chosen


def build_kpi_coverage_summary(
    *,
    full_mpc_result: Mapping[str, Any] | None,
    candidate_summary_rows: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """strict/provisional/coverage/confidence 요약 dict를 만든다.

    Parameters
    ----------
    full_mpc_result:
        ``_run_full_mpc_for_selected`` 반환 dict. strict 값(``official_mape``)과
        strict coverage(``strict_coverage_percent`` 등)를 읽는다. None이면 strict는
        평가 불가로 처리한다.
    candidate_summary_rows:
        ``_candidate_pool_experiments``가 만든 시나리오 요약 행. provisional 선택에
        쓴다.
    config:
        ``MPC.provisional_kpi_summary`` 설정(dict). None이거나 enabled=false이면
        None을 돌려준다.
    """
    cfg = dict(config or {})
    if not cfg or not bool(cfg.get("enabled", False)):
        return None

    target = float(cfg.get("target_mape_percent", DEFAULT_TARGET_MAPE_PERCENT))
    provisional_cfg = dict(cfg.get("provisional") or {})
    include_scenarios = list(provisional_cfg.get("include_scenarios") or [])
    selection_rule = str(provisional_cfg.get("selection_rule", "target_pass_highest_coverage"))
    fallback_rule = str(provisional_cfg.get("fallback_rule", "lowest_mape_diagnostic"))
    tiers = dict(cfg.get("confidence_tiers") or {})

    # ---- strict ----
    official = dict((full_mpc_result or {}).get("official_mape") or {})
    strict_reportable = _to_bool(official.get("reportable"))
    strict_target = _to_bool(official.get("target_achieved"))
    strict_coverage = _to_float((full_mpc_result or {}).get("strict_coverage_percent"))
    strict_samples = _to_int((full_mpc_result or {}).get("strict_evaluated_sample_count"))
    strict = {
        "source": STRICT_SOURCE,
        "mape_percent": _to_float(official.get("value")),
        "reportable": strict_reportable,
        "target_achieved": strict_target,
        "selected_candidate_count": _to_int(
            (full_mpc_result or {}).get("strict_selected_candidate_count")
        ),
        "evaluated_sample_count": strict_samples,
        "coverage_percent": strict_coverage,
        "confidence_tier": _confidence_tier(
            reportable=strict_reportable,
            target_achieved=strict_target,
            coverage_percent=strict_coverage,
            sample_count=strict_samples,
            tiers=tiers,
        ),
    }

    # ---- provisional ----
    chosen = _select_provisional(
        candidate_summary_rows or [], include_scenarios, selection_rule, fallback_rule
    )
    if chosen is None:
        provisional: dict[str, Any] = {
            "source": None,
            "mape_percent": None,
            "reportable": False,
            "target_achieved": False,
            "coverage_percent": None,
            "evaluated_sample_count": None,
            "confidence_tier": "none",
            "claim_boundary": CLAIM_BOUNDARY,
        }
    else:
        provisional = {
            "source": f"candidate_pool_experiments.{chosen['source']}",
            "mape_percent": chosen["mape_percent"],
            "reportable": True,
            "target_achieved": chosen["target_achieved"],
            "selected_candidate_count": chosen.get("selected_candidate_count"),
            "coverage_percent": chosen["coverage_percent"],
            "evaluated_sample_count": chosen["evaluated_sample_count"],
            "confidence_tier": _confidence_tier(
                reportable=True,
                target_achieved=chosen["target_achieved"],
                coverage_percent=chosen["coverage_percent"],
                sample_count=chosen["evaluated_sample_count"],
                tiers=tiers,
            ),
            "selection_rule": chosen.get("applied_rule"),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    strict_ok = strict_reportable and strict_target
    interpretation = {
        "primary_claim": (
            "strict official KPI passes"
            if strict_ok
            else "strict official KPI is null or below target"
        ),
        "coverage_warning": (
            f"strict coverage is {strict_coverage:.1f}% — only exact-match candidate-pool "
            "test rows were evaluated"
            if strict_coverage is not None
            else "strict KPI has no coverage; read the provisional value with its coverage"
        ),
        "provisional_claim_boundary": (
            "provisional values are diagnostic/supporting metrics, never the official KPI"
        ),
    }

    return {
        "target_mape_percent": target,
        "strict": strict,
        "provisional": provisional,
        "interpretation": interpretation,
    }


def kpi_coverage_summary_rows(summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """요약 dict를 CSV 저장용 평면 행(strict/provisional 각 1행)으로 변환한다."""
    if not summary:
        return []
    rows: list[dict[str, Any]] = []
    for metric_type in ("strict", "provisional"):
        metric = dict(summary.get(metric_type) or {})
        rows.append(
            {
                "metric_type": metric_type,
                "source": metric.get("source"),
                "mape_percent": metric.get("mape_percent"),
                "reportable": metric.get("reportable"),
                "target_achieved": metric.get("target_achieved"),
                "selected_candidate_count": metric.get("selected_candidate_count"),
                "evaluated_sample_count": metric.get("evaluated_sample_count"),
                "coverage_percent": metric.get("coverage_percent"),
                "confidence_tier": metric.get("confidence_tier"),
                "claim_boundary": metric.get("claim_boundary"),
            }
        )
    return rows
