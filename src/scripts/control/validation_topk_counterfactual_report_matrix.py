#!/usr/bin/env python
"""Validation-Top-K view of an already completed MPC/report-model run.

The MPC policy is *not* re-solved here.  Its first-action trace and the three
counterfactual report-model prediction files are immutable inputs.  This
script only does the following:

1. construct an outcome-blind, exact-action-stratified row split;
2. fit one canonical ``state + action + disturbance`` report model on its
   train partition and rank exact joint actions by their validation
   total-power MAPE; and
3. for every validation prefix K, filter the saved MPC trace by the recorded
   action and aggregate all three report-input bundles and all three MAPE
   comparisons.

Thus the test power target never decides which action enters a Top-K prefix,
and a change in K does not change the controller, its chosen MPC action, or a
report-model prediction.  It only changes which *recorded-action test rows*
are included in the disclosed evaluation cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from controllers.candidate_pool_evaluation import _fit_model  # noqa: E402
from controllers.evaluation import power_prediction_metrics  # noqa: E402
from controllers.action_stratified_holdout import (  # noqa: E402
    build_action_stratified_row_holdout,
)
from data_provider.control_data import load_control_data  # noqa: E402
from data_provider.dataset_versions import dataset_label_from_cfg  # noqa: E402


CASE_ORDER = (
    "state_action_disturbance",
    "state_action",
    "action_only",
)
COMPARISONS = (
    (
        "report_mpc_vs_recorded_actual",
        "report(MPC) ↔ 실제",
        "recorded_actual",
        "mpc_action_report_predicted",
    ),
    (
        "report_mpc_vs_report_recorded",
        "report(MPC) ↔ report(recorded)",
        "recorded_action_report_predicted",
        "mpc_action_report_predicted",
    ),
    (
        "report_recorded_vs_recorded_actual",
        "report(recorded) ↔ 실제",
        "recorded_actual",
        "recorded_action_report_predicted",
    ),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-ranked Top-K aggregation for frozen MPC actions."
    )
    parser.add_argument("--default-cfg", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument(
        "--data-versions",
        required=True,
        help="Comma-separated dataset versions; use one version for a single dataset.",
    )
    parser.add_argument("--reference-mpc-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-min", type=int, default=7)
    parser.add_argument("--top-k-max", type=int, default=50)
    return parser.parse_args()


def _config(args: argparse.Namespace):
    cfg = OmegaConf.load(args.default_cfg)
    versions = [item.strip() for item in args.data_versions.split(",") if item.strip()]
    if not versions:
        raise ValueError("--data-versions must name at least one version")
    if args.top_k_min <= 0 or args.top_k_max < args.top_k_min:
        raise ValueError("Top-K range must satisfy 0 < minimum <= maximum")
    OmegaConf.update(cfg, "DATASET.environment_name", args.environment)
    OmegaConf.update(cfg, "DATASET.data_versions", versions)
    # ``data_versions`` is authoritative.  Keeping a single legacy data_version
    # as well would make the provenance unclear in generated artifacts.
    OmegaConf.update(cfg, "DATASET.data_version", versions[0])
    return cfg, versions


def _find_ablation_dir(reference_dir: Path, result_root: Path, dataset_label: str, environment: str) -> Path:
    """Find the unique completed input-ablation tied to this exact MPC run."""
    manifest = reference_dir / "mpc_counterfactual_report_input_ablation_manifest.json"
    if manifest.is_file():
        value = _load_json(manifest).get(
            "counterfactual_report_input_ablation_result_directory"
        )
        if value:
            candidate = Path(str(value)).expanduser().resolve()
            if (candidate / "counterfactual_report_input_ablation.json").is_file():
                return candidate

    root = result_root / dataset_label / environment / "counterfactual_report_input_ablation"
    matches = []
    for summary_path in root.glob("*/counterfactual_report_input_ablation.json"):
        summary = _load_json(summary_path)
        reported_reference = (
            summary.get("reference_mpc_run") or {}
        ).get("result_directory")
        if reported_reference and Path(str(reported_reference)).resolve() == reference_dir:
            matches.append(summary_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            "Unable to identify one report-input ablation for reference MPC run "
            f"{reference_dir}; matches={matches}"
        )
    return matches[0]


def _action_key(values: np.ndarray) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(values, dtype=float)).astype(int).tolist())


def _source_row_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_row_index" not in frame.columns:
        raise ValueError("Prediction file lacks source_row_index")
    result = frame.copy()
    result["source_row_index"] = pd.to_numeric(
        result["source_row_index"], errors="raise"
    ).astype(int)
    if result["source_row_index"].duplicated().any():
        raise ValueError("Prediction file contains duplicated source_row_index")
    return result.sort_values("source_row_index").reset_index(drop=True)


def _load_case_frames(ablation_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    summary_path = ablation_dir / "counterfactual_report_input_ablation.json"
    summary = _load_json(summary_path)
    available = summary.get("case_results") or {}
    missing = [name for name in CASE_ORDER if name not in available]
    if missing:
        raise ValueError(
            "Input-ablation result does not contain required cases: " + ", ".join(missing)
        )
    frames: dict[str, pd.DataFrame] = {}
    common_rows: np.ndarray | None = None
    for name in CASE_ORDER:
        path = ablation_dir / "cases" / name / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved predictions for {name}: {path}")
        frame = _source_row_frame(pd.read_csv(path, encoding="utf-8-sig"))
        rows = frame["source_row_index"].to_numpy(dtype=int)
        if common_rows is None:
            common_rows = rows
        elif not np.array_equal(common_rows, rows):
            raise ValueError("Input-ablation cases do not have identical frozen MPC rows")
        frames[name] = frame
    return summary, frames


def _candidate_ranking(
    cfg,
    data,
    report_family: str,
    reference_evaluation: dict[str, Any],
    trace_recorded_controls: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank support-qualified exact actions using validation only.

    The action-stratified split puts each repeated exact action into train,
    validation, and test rows with a fixed seed.  It is outcome-blind but not
    episode-disjoint and not a future-time generalization test.  It is used
    here solely because the requested Top-K range (7..50) needs an action
    comparison cohort that contains the same combinations in every partition.
    A final control-only overlap gate requires an action to occur at least five
    times in the already fixed MPC trace; it never inspects trace power values.
    """
    del reference_evaluation  # trace provenance is checked by the caller.
    holdout = build_action_stratified_row_holdout(
        data,
        cfg.MPC.evaluation.fixed_optimal_action_holdout,
        split_rate=cfg.DATASET.split_rate,
        random_state=int(cfg.DEFAULT.seed),
    )
    candidates = holdout.metadata.get("candidates") or []
    if not candidates:
        raise ValueError("No exact joint action meets action-stratified support")
    minimum_trace_rows = int(
        cfg.MPC.evaluation.fixed_optimal_action_holdout.minimum_test_exact_matches
    )
    trace_keys = [_action_key(row) for row in trace_recorded_controls]
    trace_counts: dict[tuple[int, ...], int] = {}
    for key in trace_keys:
        trace_counts[key] = trace_counts.get(key, 0) + 1
    candidates = [
        candidate
        for candidate in candidates
        if trace_counts.get(tuple(candidate["encoded_control"]), 0) >= minimum_trace_rows
    ]
    if not candidates:
        raise ValueError(
            "No action has both action-stratified support and the predeclared "
            "MPC-trace control overlap minimum"
        )
    report_cfg = OmegaConf.to_container(
        cfg.MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model,
        resolve=True,
    )
    report_cfg["input_features"] = {
        "current_state": True,
        "action": True,
        "current_disturbance": True,
    }
    report_model = _fit_model(
        data,
        holdout.train_rows,
        report_family,
        report_cfg,
        int(cfg.DEFAULT.seed),
    )
    threshold = float(cfg.MPC.affine_models.mape_min_actual_power_w)
    ranking: list[dict[str, Any]] = []
    controls = np.rint(np.asarray(data.controls, dtype=float)).astype(int)
    validation_rows = data.valid_transition_rows(holdout.validation_rows)
    for candidate in candidates:
        encoded = np.asarray(candidate["encoded_control"], dtype=int)
        rows = validation_rows[np.all(controls[validation_rows] == encoded, axis=1)]
        states, candidate_controls, disturbances, _, powers = data.model_arrays_for_rows(rows)
        predicted = report_model.predict_components(states, candidate_controls, disturbances)
        actual = np.asarray(powers, dtype=float)
        metric = power_prediction_metrics(
            actual.sum(axis=1, keepdims=True),
            predicted.sum(axis=1, keepdims=True),
            ("total_hvac_power_w",),
            threshold,
        )
        mape = metric["mape_percent"]
        valid_samples = int(metric["mape_valid_samples_per_output"][0])
        if mape is None:
            continue
        ranking.append(
            {
                "encoded_control": encoded.tolist(),
                "action": candidate["action"],
                "validation_total_power_mape_percent": float(mape),
                "validation_total_power_valid_mape_samples": valid_samples,
                "validation_exact_rows": int(len(rows)),
                "frozen_trace_recorded_action_count": trace_counts[
                    tuple(encoded.tolist())
                ],
                "support": candidate["support"],
            }
        )
    ranking.sort(
        key=lambda item: (
            item["validation_total_power_mape_percent"],
            -item["validation_exact_rows"],
            tuple(item["encoded_control"]),
        )
    )
    for rank, item in enumerate(ranking, start=1):
        item["validation_rank"] = rank
    return ranking, {
        "protocol": {
            "name": "outcome_blind_action_stratified_validation_ranked_top_k",
            "candidate_membership": (
                "exact action with configured train/validation/test row support under "
                "the fixed-seed action-stratified allocation"
            ),
            "ranking": "action-stratified train-only report model validation total-power MAPE",
            "selection_uses_test_power_targets": False,
            "episode_overlap_between_splits_allowed": True,
            "temporal_generalization_claim": False,
        },
        "split_transition_counts": holdout.metadata["split_transition_counts"],
        "candidate_count_after_control_only_trace_overlap_gate": len(candidates),
        "candidate_count_with_validation_mape": len(ranking),
        "minimum_training_samples_per_action": int(
            cfg.MPC.evaluation.fixed_optimal_action_holdout.minimum_training_samples_per_candidate
        ),
        "minimum_validation_mape_samples_per_action": int(
            cfg.MPC.evaluation.fixed_optimal_action_holdout.minimum_validation_samples_per_candidate
        ),
        "minimum_test_support_rows_per_action": int(
            cfg.MPC.evaluation.fixed_optimal_action_holdout.minimum_test_exact_matches
        ),
        "validation_ranking_model_family": report_family,
        "validation_ranking_model_input": "state + action + disturbance",
        "validation_ranking_model_fit_rows": int(len(holdout.train_rows)),
        "validation_ranking_rows": int(len(holdout.validation_rows)),
        "test_targets_used_for_action_selection": False,
        "trace_control_overlap_uses_power_or_state_targets": False,
    }


def _metric_rows(
    *,
    data_label: str,
    k_requested: int,
    ranking: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    encoded_recorded_controls: np.ndarray,
    power_names: tuple[str, ...],
    threshold: float,
) -> list[dict[str, Any]]:
    available = len(ranking)
    if k_requested > available:
        status = "unavailable_insufficient_validation_actions"
        selected_keys: set[tuple[int, ...]] = set()
    else:
        status = "available"
        selected_keys = {
            tuple(item["encoded_control"])
            for item in ranking[:k_requested]
        }
    mask = np.asarray(
        [_action_key(row) in selected_keys for row in encoded_recorded_controls],
        dtype=bool,
    )
    selected_rows = int(mask.sum())
    total_rows = int(len(mask))
    coverage = float(100.0 * selected_rows / total_rows) if total_rows else None
    action_hash = hashlib.sha256(
        json.dumps(sorted(selected_keys), separators=(",", ":")).encode("utf-8")
    ).hexdigest() if selected_keys else None
    rows: list[dict[str, Any]] = []
    for case_name in CASE_ORDER:
        frame = frames[case_name]
        for comparison_key, comparison_label, actual_prefix, predicted_prefix in COMPARISONS:
            for component_index, output_name in enumerate(power_names):
                actual = frame.loc[mask, f"{actual_prefix}_{output_name}"].to_numpy(
                    dtype=float
                )[:, None]
                predicted = frame.loc[
                    mask, f"{predicted_prefix}_{output_name}"
                ].to_numpy(dtype=float)[:, None]
                metric = (
                    power_prediction_metrics(actual, predicted, (output_name,), threshold)
                    if selected_rows
                    else None
                )
                rows.append(
                    {
                        "dataset": data_label,
                        "top_k_requested": k_requested,
                        "top_k_effective": k_requested if status == "available" else None,
                        "status": status,
                        "available_validation_actions": available,
                        "selected_action_sha256": action_hash,
                        "selected_trace_rows": selected_rows,
                        "trace_rows": total_rows,
                        "coverage_percent": coverage,
                        "report_model_input": case_name,
                        "comparison": comparison_key,
                        "comparison_label": comparison_label,
                        "output": output_name,
                        "mape_percent": None if metric is None else metric["mape_percent"],
                        "mape_valid_samples": None if metric is None else int(metric["mape_valid_samples_per_output"][0]),
                        "mape_denominator_coverage_percent": None if metric is None else float(metric["mape_coverage_percent_per_output"][0]),
                    }
                )
            actual_components = np.column_stack([
                frame.loc[mask, f"{actual_prefix}_{name}"].to_numpy(dtype=float)
                for name in power_names
            ])
            predicted_components = np.column_stack([
                frame.loc[mask, f"{predicted_prefix}_{name}"].to_numpy(dtype=float)
                for name in power_names
            ])
            metric = (
                power_prediction_metrics(
                    actual_components.sum(axis=1, keepdims=True),
                    predicted_components.sum(axis=1, keepdims=True),
                    ("total_hvac_power_w",),
                    threshold,
                )
                if selected_rows
                else None
            )
            rows.append(
                {
                    "dataset": data_label,
                    "top_k_requested": k_requested,
                    "top_k_effective": k_requested if status == "available" else None,
                    "status": status,
                    "available_validation_actions": available,
                    "selected_action_sha256": action_hash,
                    "selected_trace_rows": selected_rows,
                    "trace_rows": total_rows,
                    "coverage_percent": coverage,
                    "report_model_input": case_name,
                    "comparison": comparison_key,
                    "comparison_label": comparison_label,
                    "output": "total_hvac_power_w",
                    "mape_percent": None if metric is None else metric["mape_percent"],
                    "mape_valid_samples": None if metric is None else int(metric["mape_valid_samples_per_output"][0]),
                    "mape_denominator_coverage_percent": None if metric is None else float(metric["mape_coverage_percent_per_output"][0]),
                }
            )
    return rows


def _ranking_frame(ranking: Iterable[dict[str, Any]], data) -> pd.DataFrame:
    rows = []
    for item in ranking:
        flat = {
            "validation_rank": item["validation_rank"],
            "validation_total_power_mape_percent": item["validation_total_power_mape_percent"],
            "validation_total_power_valid_mape_samples": item[
                "validation_total_power_valid_mape_samples"
            ],
            "validation_exact_rows": item["validation_exact_rows"],
            "frozen_trace_recorded_action_count": item[
                "frozen_trace_recorded_action_count"
            ],
            "encoded_control": json.dumps(item["encoded_control"], ensure_ascii=False),
        }
        flat.update({f"control_{key}": value for key, value in item["action"].items()})
        flat.update(item["support"])
        rows.append(flat)
    return pd.DataFrame(rows)


def _write_markdown(path: Path, metadata: dict[str, Any], results: pd.DataFrame) -> None:
    lines = [
        "# Validation Top-K counterfactual report matrix",
        "",
        "- **고정된 것:** 원 MPC first-action trace, 각 입력 case의 report-model 예측값, test 행의 상태·외란.",
        "- **변하는 것:** validation MAPE로 순위를 매긴 recorded exact joint action의 누적 prefix K만 바뀝니다.",
        "- **선정 규칙:** canonical `S+A+D` report model을 fixed-seed action-stratified train에서 학습하고 validation MAPE로 action을 순위화합니다. action은 train 5행·validation 3행·test 5행 이상이어야 하며, 고정된 MPC trace 안에서도 기록 action 5행 이상이어야 합니다. 이 overlap gate는 control label만 읽고 test 전력값은 action·K·model 선택에 사용하지 않습니다.",
        "- **해석 경계:** 같은 action의 episode가 train/validation/test에 나뉠 수 있는 supplementary row-stratified protocol입니다. 따라서 coverage 확대/입력 비교용 sensitivity이며, episode-independent strict KPI를 대체하지 않습니다.",
        "- **행 필터:** 저장된 MPC trace 중 **기록된 현재 action**이 Top-K prefix에 속하는 행만 세 비교에 공통 적용합니다. 따라서 coverage는 policy trace에서 실제로 채점된 행의 비율입니다.",
        "- **비교:** `report(MPC) ↔ 실제`, `report(MPC) ↔ report(recorded)`, `report(recorded) ↔ 실제`. MAPE 분모는 각각 비교의 왼쪽 항이며, |분모| ≤ 0.1 W 행은 해당 출력 MAPE에서 제외됩니다.",
        "- `K > available_validation_actions`는 조용히 다른 K로 대체하지 않고 `unavailable`로 기록합니다.",
        "",
        "## 실행 메타데이터",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 총전력 MAPE 요약",
        "",
        "전체(환기/냉난방/총전력)와 세 비교의 long-form 값은 `validation_topk_component_report_matrix.csv`에 있습니다.",
        "",
        "| K | 입력 | 비교 | Coverage | 총전력 MAPE | 상태 |",
        "|---:|---|---|---:|---:|---|",
    ]
    total = results[results["output"] == "total_hvac_power_w"]
    for _, row in total.iterrows():
        value = "—" if pd.isna(row["mape_percent"]) else f"{float(row['mape_percent']):.4f}%"
        coverage = "—" if pd.isna(row["coverage_percent"]) else f"{float(row['coverage_percent']):.2f}%"
        lines.append(
            f"| {int(row['top_k_requested'])} | {row['report_model_input']} | "
            f"{row['comparison_label']} | {coverage} | {value} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cfg, versions = _config(args)
    reference_dir = Path(args.reference_mpc_dir).expanduser().resolve()
    required = (
        reference_dir / "mpc_evaluation.json",
        reference_dir / "mpc_counterfactual_total_power_report_predictions.csv",
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("MPC reference is incomplete: " + ", ".join(missing))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_label = dataset_label_from_cfg(cfg)
    ablation_dir = _find_ablation_dir(
        reference_dir,
        Path(cfg.RESULT.savedir).expanduser().resolve(),
        data_label,
        args.environment,
    )
    ablation_summary, frames = _load_case_frames(ablation_dir)
    report_family = str(
        ablation_summary["case_results"]["state_action_disturbance"]["model_selection"][
            "selected_family"
        ]
    )
    data = load_control_data(cfg)
    frame0 = frames[CASE_ORDER[0]]
    source_rows = frame0["source_row_index"].to_numpy(dtype=int)
    if np.any(source_rows < 0) or np.any(source_rows >= len(data.controls)):
        raise ValueError("Saved MPC trace has source rows outside loaded data")
    recorded_controls = np.asarray(data.controls[source_rows], dtype=float)
    reference_evaluation = _load_json(reference_dir / "mpc_evaluation.json")
    ranking, split_metadata = _candidate_ranking(
        cfg,
        data,
        report_family,
        reference_evaluation,
        recorded_controls,
    )
    all_rows: list[dict[str, Any]] = []
    for k in range(args.top_k_min, args.top_k_max + 1):
        all_rows.extend(
            _metric_rows(
                data_label=data_label,
                k_requested=k,
                ranking=ranking,
                frames=frames,
                encoded_recorded_controls=recorded_controls,
                power_names=tuple(data.power_names),
                threshold=float(cfg.MPC.affine_models.mape_min_actual_power_w),
            )
        )
    result_frame = pd.DataFrame(all_rows)
    metadata = {
        "framework": "validation_topk_counterfactual_report_matrix",
        "dataset_label": data_label,
        "data_versions": versions,
        "environment": args.environment,
        "reference_mpc_result_directory": str(reference_dir),
        "counterfactual_report_input_ablation_directory": str(ablation_dir),
        "top_k_requested_range": [args.top_k_min, args.top_k_max],
        "input_cases": list(CASE_ORDER),
        "comparisons": [item[0] for item in COMPARISONS],
        "power_outputs": [*data.power_names, "total_hvac_power_w"],
        "mape_floor_w": float(cfg.MPC.affine_models.mape_min_actual_power_w),
        "ranking": split_metadata,
        "important_interpretation": (
            "This is a validation-selected action-cohort sensitivity analysis of an "
            "already completed MPC policy. It does not re-run MILP and does not "
            "claim that Top-K filtering improves the policy itself."
        ),
    }
    _write_csv(output_dir / "validation_topk_component_report_matrix.csv", result_frame)
    _write_csv(output_dir / "validation_topk_action_ranking.csv", _ranking_frame(ranking, data))
    _write_json(output_dir / "validation_topk_component_report_matrix.json", metadata)
    _write_markdown(output_dir / "validation_topk_component_report_matrix.md", metadata, result_frame)
    print(
        "Validation Top-K aggregation complete | "
        f"dataset={data_label} | available_actions={len(ranking)} | "
        f"rows={len(result_frame)} | output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
