#!/usr/bin/env python3
"""Build a source-backed Kweather forecasting experiment report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


DEFAULT_MODELS = ["DLinear", "iTransformer", "MICN", "PatchTST", "TimesNet", "TimeXer"]
DEFAULT_ENVIRONMENTS = ["room1", "room2", "livingroom"]
ENVIRONMENT_LABELS = {"room1": "방 1", "room2": "방 2", "livingroom": "거실"}
TARGETS = {
    "ventilation": {"column": "환기유효전력(W)", "label": "환기유효전력"},
    "hvac": {"column": "냉난방유효전력(W)", "label": "냉난방유효전력"},
}
METRIC_FIELDS = [
    "prediction_accuracy", "prediction_accuracy*100", "mse", "mae", "rmse",
    "mape", "mspe", "wape", "smape", "rse", "nmae",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir.parent
    repo_root = src_dir.parent
    parser = argparse.ArgumentParser(
        description="Report Kweather data quality, splits, and forecast results."
    )
    parser.add_argument("--config", type=Path, default=src_dir / "configs" / "default_setting.yaml")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "saved_model")
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "reports" / "forecasting",
        help="Root output directory; a data-version subdirectory is added.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--environments", nargs="+", default=DEFAULT_ENVIRONMENTS)
    parser.add_argument(
        "--target-keys", nargs="+", choices=tuple(TARGETS), default=list(TARGETS)
    )
    parser.add_argument(
        "--outlier-threshold", type=float, default=1000.0,
        help="Exclude rows whose selected target exceeds this value.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def resolve_data_dir(datadir: str, data_version: str, src_dir: Path) -> Path:
    configured = Path(datadir).expanduser()
    candidates = [configured, Path.cwd() / configured, src_dir / configured]
    checked = []
    for candidate in candidates:
        version_dir = candidate.resolve() / data_version
        checked.append(str(version_dir))
        if version_dir.is_dir():
            return version_dir
    raise FileNotFoundError("Data version directory not found. Searched: " + ", ".join(checked))


def safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else round(number, 6)


def json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: json_value(value) for key, value in row.items()} for row in rows]


def timestamp_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def window_count(rows: int, seq_len: int, pred_len: int, stride: int) -> int:
    max_start = rows - seq_len - pred_len
    return 0 if max_start < 0 else (max_start // stride) + 1


def profile_data(
    data_dir: Path,
    environments: list[str],
    target_keys: list[str],
    split_rate: list[float],
    start_date: Any,
    end_date: Any,
    seq_len: int,
    pred_len: int,
    stride: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quality_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for environment in environments:
        csv_path = data_dir / f"{environment}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing environment CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "데이터 시간" not in frame:
            raise KeyError(f"'데이터 시간' column is missing from {csv_path}")

        timestamps = pd.to_datetime(frame["데이터 시간"], errors="coerce")
        valid_timestamps = timestamps.dropna()
        diffs = valid_timestamps.diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)]
        irregular = int((diffs != pd.Timedelta(minutes=1)).sum())
        max_gap = (
            round(float(positive_diffs.max().total_seconds() / 60), 3)
            if not positive_diffs.empty else None
        )
        invalid_timestamps = int(timestamps.isna().sum())
        duplicate_timestamps = int(timestamps.duplicated(keep=False).sum())
        is_monotonic = bool(valid_timestamps.is_monotonic_increasing)
        missing_by_column = frame.isna().sum()
        missing_cells = int(missing_by_column.sum())
        columns_with_missing = int((missing_by_column > 0).sum())
        max_missing_count = int(missing_by_column.max())
        max_missing_rate = round(max_missing_count / len(frame), 6) if len(frame) else None
        missing_cell_rate = round(missing_cells / frame.size, 6) if frame.size else None
        high_missing_columns = ", ".join(
            str(column) for column in missing_by_column.index
            if len(frame) and missing_by_column[column] / len(frame) >= 0.1
        )

        for target_key in target_keys:
            target = TARGETS[target_key]
            column = target["column"]
            if column not in frame:
                raise KeyError(f"'{column}' column is missing from {csv_path}")
            numeric_target = pd.to_numeric(frame[column], errors="coerce")
            outlier_mask = numeric_target > threshold
            missing_target = int(numeric_target.isna().sum())
            zero_count = int((numeric_target == 0).sum())
            valid_count = int(numeric_target.notna().sum())

            filtered_timestamps = timestamps.loc[~outlier_mask]
            filtered_target = numeric_target.loc[~outlier_mask]
            start_dt = pd.to_datetime(start_date) if start_date is not None else filtered_timestamps.min()
            end_dt = pd.to_datetime(end_date) if end_date is not None else filtered_timestamps.max()
            date_mask = (
                filtered_timestamps.notna()
                & (filtered_timestamps >= start_dt)
                & (filtered_timestamps <= end_dt)
            )
            filtered_timestamps = filtered_timestamps.loc[date_mask]
            filtered_target = filtered_target.loc[date_mask]
            n_samples = len(filtered_target)
            train_size = int(n_samples * split_rate[0])
            val_size = int(n_samples * split_rate[1])
            boundaries = {
                "Train": (0, train_size),
                "Val": (train_size, train_size + val_size),
                "Test": (train_size + val_size, n_samples),
            }
            for split_name, (begin, finish) in boundaries.items():
                split_time = filtered_timestamps.iloc[begin:finish]
                count = finish - begin
                split_rows.append({
                    "environment": environment,
                    "environment_label": ENVIRONMENT_LABELS.get(environment, environment),
                    "target_key": target_key,
                    "target": column,
                    "target_label": target["label"],
                    "split": split_name,
                    "cohort": f"{ENVIRONMENT_LABELS.get(environment, environment)} · {target['label']} · {split_name}",
                    "rows": count,
                    "forecast_windows": window_count(count, seq_len, pred_len, stride),
                    "start": timestamp_text(split_time.iloc[0]) if count else "—",
                    "end": timestamp_text(split_time.iloc[-1]) if count else "—",
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "stride": stride,
                })

            descriptive = filtered_target.describe(percentiles=[0.25, 0.5, 0.75])
            distribution_rows.append({
                "environment": environment,
                "environment_label": ENVIRONMENT_LABELS.get(environment, environment),
                "target_key": target_key,
                "target": column,
                "target_label": target["label"],
                "cohort": f"{ENVIRONMENT_LABELS.get(environment, environment)} · {target['label']}",
                "count": int(descriptive.get("count", 0)),
                "mean_w": safe_number(descriptive.get("mean")),
                "std_w": safe_number(descriptive.get("std")),
                "min_w": safe_number(descriptive.get("min")),
                "q1_w": safe_number(descriptive.get("25%")),
                "median_w": safe_number(descriptive.get("50%")),
                "q3_w": safe_number(descriptive.get("75%")),
                "max_w": safe_number(descriptive.get("max")),
                "zero_rate": round(float((filtered_target == 0).mean()), 6) if n_samples else None,
                "rows_after_filter": n_samples,
            })
            quality_rows.append({
                "environment": environment,
                "environment_label": ENVIRONMENT_LABELS.get(environment, environment),
                "target_key": target_key,
                "target": column,
                "target_label": target["label"],
                "raw_rows": len(frame),
                "raw_columns": len(frame.columns),
                "raw_start": timestamp_text(valid_timestamps.min()),
                "raw_end": timestamp_text(valid_timestamps.max()),
                "rows_after_filter": n_samples,
                "missing_target": missing_target,
                "missing_target_rate": round(missing_target / len(frame), 6) if len(frame) else None,
                "total_missing_cells": missing_cells,
                "missing_cell_rate": missing_cell_rate,
                "columns_with_missing": columns_with_missing,
                "max_column_missing_count": max_missing_count,
                "max_column_missing_rate": max_missing_rate,
                "columns_over_10pct_missing": high_missing_columns,
                "outliers_over_threshold": int(outlier_mask.sum()),
                "outlier_threshold_w": threshold,
                "zero_count": zero_count,
                "zero_rate": round(zero_count / valid_count, 6) if valid_count else None,
                "invalid_timestamps": invalid_timestamps,
                "duplicate_timestamp_rows": duplicate_timestamps,
                "irregular_intervals": irregular,
                "max_positive_gap_minutes": max_gap,
                "timestamp_monotonic": is_monotonic,
            })
    return quality_rows, split_rows, distribution_rows


def version_number(path: Path) -> int:
    return int(path.name[7:]) if path.name.startswith("version") and path.name[7:].isdigit() else -1


def scan_results(
    results_dir: Path,
    data_version: str,
    models: list[str],
    environments: list[str],
    target_keys: list[str],
    seq_len: int,
    pred_len: int,
    expected_epochs: int,
) -> tuple[list[dict[str, Any]], int]:
    desired_targets = {TARGETS[key]["column"]: key for key in target_keys}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    ignored = 0

    if results_dir.is_dir():
        for metrics_path in results_dir.rglob("test_results.json"):
            config_path = metrics_path.with_name("configs.yaml")
            if not config_path.is_file():
                ignored += 1
                continue
            try:
                config = load_yaml(config_path)
                dataset_cfg = config.get("DATASET", {})
                model_cfg = config.get("MODEL", {})
                if dataset_cfg.get("data_version") != data_version:
                    continue
                environment = str(dataset_cfg.get("environment_name", ""))
                model = str(model_cfg.get("modelname", ""))
                targets = dataset_cfg.get("target_features") or []
                if isinstance(targets, str):
                    targets = [targets]
                if (
                    environment not in environments
                    or model not in models
                    or len(targets) != 1
                    or targets[0] not in desired_targets
                    or int(dataset_cfg.get("seq_len", -1)) != seq_len
                    or int(dataset_cfg.get("pred_len", -1)) != pred_len
                ):
                    continue
                target_key = desired_targets[targets[0]]
                with metrics_path.open("r", encoding="utf-8") as stream:
                    metrics = json.load(stream)
                candidate = {
                    "mtime": metrics_path.stat().st_mtime,
                    "version": version_number(metrics_path.parent),
                    "metrics_path": metrics_path,
                    "metrics": metrics,
                    "train_epochs": int(config.get("TRAIN", {}).get("epoch", 0)),
                    "experiment": config.get("DEFAULT", {}).get("exp_name"),
                }
                candidate["is_full"] = candidate["train_epochs"] >= expected_epochs
                key = (environment, target_key, model)
                previous = selected.get(key)
                candidate_rank = (
                    candidate["is_full"], candidate["mtime"], candidate["version"]
                )
                previous_rank = (
                    (previous["is_full"], previous["mtime"], previous["version"])
                    if previous else (False, -1, -1)
                )
                if candidate_rank > previous_rank:
                    selected[key] = candidate
            except (OSError, ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError):
                ignored += 1

    rows: list[dict[str, Any]] = []
    for environment in environments:
        for target_key in target_keys:
            target = TARGETS[target_key]
            for model in models:
                candidate = selected.get((environment, target_key, model))
                row: dict[str, Any] = {
                    "environment": environment,
                    "environment_label": ENVIRONMENT_LABELS.get(environment, environment),
                    "target_key": target_key,
                    "target": target["column"],
                    "target_label": target["label"],
                    "model": model,
                    "status": (
                        "완료" if candidate and candidate["is_full"]
                        else "스모크" if candidate
                        else "미실행"
                    ),
                    "train_epochs": candidate.get("train_epochs") if candidate else None,
                    "expected_epochs": expected_epochs,
                    "experiment": candidate.get("experiment") if candidate else None,
                    "version": (
                        f"version{candidate['version']}"
                        if candidate and candidate["version"] >= 0 else None
                    ),
                    "result_path": str(candidate["metrics_path"]) if candidate else None,
                }
                for field in METRIC_FIELDS:
                    row[field] = safe_number(candidate["metrics"].get(field)) if candidate else None
                rows.append(row)
    return rows, ignored


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def metric_text(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def build_markdown(
    data_version: str,
    generated_at: str,
    quality_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    models: list[str],
    environments: list[str],
    target_keys: list[str],
    seq_len: int,
    pred_len: int,
    stride: int,
    split_rate: list[float],
    threshold: float,
    ignored_results: int,
) -> str:
    completed = sum(row["status"] == "완료" for row in result_rows)
    smoke = sum(row["status"] == "스모크" for row in result_rows)
    planned = len(result_rows)
    raw_counts = [row["raw_rows"] for row in quality_rows]
    starts = [row["raw_start"] for row in quality_rows if row["raw_start"] != "—"]
    ends = [row["raw_end"] for row in quality_rows if row["raw_end"] != "—"]
    outliers = sum(row["outliers_over_threshold"] for row in quality_rows)
    missing_targets = sum(row["missing_target"] for row in quality_rows)
    max_column_missing_rate = max(row["max_column_missing_rate"] for row in quality_rows)
    high_missing_sets = sorted({
        row["columns_over_10pct_missing"] for row in quality_rows if row["columns_over_10pct_missing"]
    })
    lines = [
        "# 케이웨더 전력량 시계열 예측 실험 보고서", "",
        f"- 생성 시각: {generated_at}",
        f"- 데이터 버전: {data_version}", "",
        "## 기술 요약", "",
        "현재 제공된 CSV에는 발전량 컬럼이 없으므로, 이 태스크는 첨부 페이지와 동일한 "
        "**전력 소비량 예측**으로 정의했다. 환기유효전력과 냉난방유효전력을 서로 독립된 "
        "단일 타깃으로 학습·평가한다.", "",
        f"- 실험 행렬: {len(models)}개 모델 × {len(environments)}개 공간 × "
        f"{len(target_keys)}개 타깃 = **{planned}회**",
        f"- 입력/예측: 최근 **{seq_len}분** → 미래 **{pred_len}분**, stride {stride}",
        f"- 분할: Train/Val/Test = {split_rate[0]:.0%}/{split_rate[1]:.0%}/"
        f"{split_rate[2]:.0%}, 시간순 분할",
        f"- 정식 완료 결과: **{completed}/{planned}회**, 스모크 결과: **{smoke}회**", "",
        "## 핵심 데이터 점검", "",
        f"- 공간별 원시 행 수 범위: **{min(raw_counts):,}~{max(raw_counts):,}행**",
        f"- 전체 수집 범위: **{min(starts)} ~ {max(ends)}**",
        f"- 타깃 기준 {threshold:,.0f}W 초과 이상치 합계: **{outliers:,}건** "
        "(공간×타깃 단위 집계)",
        f"- 타깃 결측 합계: **{missing_targets:,}건** (공간×타깃 단위 집계)",
        f"- 최대 단일 컬럼 결측률: **{max_column_missing_rate:.2%}**",
        f"- 10% 이상 결측 컬럼: **{' / '.join(high_missing_sets) or '없음'}**", "",
        "원시 행 수와 학습 샘플 수는 다르다. 한 샘플이 입력 120행과 정답 60행을 "
        "필요로 하므로, 아래 표의 예측 윈도우가 실제 Dataset 길이다.", "",
        "## 데이터 분할", "",
        markdown_table(
            ["공간", "타깃", "분할", "행 수", "예측 윈도우", "시작", "종료"],
            [[
                row["environment_label"], row["target_label"], row["split"],
                f"{row['rows']:,}", f"{row['forecast_windows']:,}",
                row["start"], row["end"],
            ] for row in split_rows],
        ), "",
        "## 모델 평가표", "",
        "전력량 정확도는 1 - MAE / mean(y_true)이며, MSE·MAE·RMSE는 역정규화된 "
        "W 단위 예측값으로 계산된다. 냉난방처럼 0이 많은 타깃은 평균이 작아 정확도가 "
        "음수가 되거나 과도하게 변할 수 있으므로 MAE/RMSE를 함께 판단해야 한다.", "",
    ]
    for target_key in target_keys:
        for environment in environments:
            subset = [
                row for row in result_rows
                if row["target_key"] == target_key and row["environment"] == environment
            ]
            lines.extend([
                f"### {ENVIRONMENT_LABELS.get(environment, environment)} · "
                f"{TARGETS[target_key]['label']}(W)", "",
                markdown_table(
                    ["모델", "전력량 정확도", "MSE", "MAE", "RMSE", "Epoch", "상태"],
                    [[
                        row["model"], metric_text(row["prediction_accuracy"]),
                        metric_text(row["mse"]), metric_text(row["mae"]),
                        metric_text(row["rmse"]), row["train_epochs"], row["status"],
                    ] for row in subset],
                ), "",
            ])
    lines.extend([
        "## 방법론", "",
        "1. 각 공간 CSV를 시간 순서 그대로 읽고 타임스탬프·결측·중복·1분 간격을 점검한다.",
        f"2. 선택된 타깃이 {threshold:,.0f}W를 초과하는 행을 제거한다.",
        "3. 수치·범주·순서형 결측은 시간순 forward-fill 후 필요한 선두 구간만 back-fill한다.",
        "4. 설정 날짜 범위를 적용한 뒤 8:1:1로 연속 분할한다. 임의 셔플은 사용하지 않는다.",
        "5. Train 통계로 정규화하고 테스트 예측을 역정규화해 W 단위 지표를 계산한다.",
        "6. 동일 조합은 정식 epoch 결과를 우선하고, 없으면 최신 스모크 결과를 별도 표시한다.", "",
        "## 한계와 견고성", "",
        "- 원시 데이터에는 발전량 컬럼이 없다. 실제 발전량 예측이 목표라면 발전량·일사량·"
        "기상예보 컬럼과 별도의 타깃 정의가 필요하다.",
        "- 단순 8:1:1 한 번의 홀드아웃 결과는 계절 일반화를 보장하지 않는다. "
        "운영 적용 전 rolling-origin backtest가 필요하다.",
        "- 전력량 정확도는 평균 실제값이 0에 가까울 때 불안정하고 [0, 1] 범위에 제한되지 않는다. "
        "모델 선정은 MAE/RMSE를 주지표로 하고 정확도는 보조지표로 사용한다.",
        "- 1분 간격이 아닌 구간, 중복 타임스탬프, 결측치는 품질 표에서 확인해야 한다.",
        "- 결측 대치는 연속 시계열 전체에 적용되므로 시작부 back-fill이 필요한 새 데이터셋은 "
        "분할별 대치 또는 train-fitted imputer로 재검증해야 한다.",
        f"- 형식이 맞지 않아 무시한 결과 파일: **{ignored_results}개**", "",
        "## 다음 단계", "",
        "1. run_experiments.sh로 36개 실험을 실행한다.",
        "2. 자동 재생성된 표에서 타깃별 MAE/RMSE 최우수 모델을 선택한다.",
        "3. persistence/seasonal-naive 기준선을 추가하고 rolling-origin 검증을 수행한다.",
        "4. MPC 입력으로 연결할 때 60분 구간별 오차와 불확실성을 함께 전달한다.", "",
        "## 추가로 결정할 질문", "",
        "- 여기서 말한 ‘발전량’이 실제 발전 설비 출력인지, 현재 CSV의 소비전력인지?",
        "- 운영 모델 선정의 1차 기준을 MAE, RMSE, 전력량 정확도 중 무엇으로 둘지?",
        "- 60개 시점 평균 외에 10/30/60분 lead-time별 성능도 표준 지표로 둘지?", "",
    ])
    return "\n".join(lines)


def make_source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    query_text: str,
    generated_at: str,
    tables: list[str],
    filters: list[str],
    definitions: dict[str, str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "python/pandas",
            "sql": query_text,
            "description": description,
            "executed_at": generated_at,
            "tables_used": tables,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def build_artifact(
    data_version: str,
    generated_at: str,
    quality_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    models: list[str],
    environments: list[str],
    target_keys: list[str],
    seq_len: int,
    pred_len: int,
    stride: int,
    split_rate: list[float],
    threshold: float,
) -> dict[str, Any]:
    completed = sum(row["status"] == "완료" for row in result_rows)
    smoke = sum(row["status"] == "스모크" for row in result_rows)
    full_epoch_requirement = result_rows[0]["expected_epochs"] if result_rows else 0
    planned = len(result_rows)
    unique_raw_rows = {row["environment"]: row["raw_rows"] for row in quality_rows}
    max_latest = max(
        pd.to_datetime(row["raw_end"]) for row in quality_rows if row["raw_end"] != "—"
    )
    age_days = max(0, (pd.Timestamp.now() - max_latest).days)
    max_column_missing_rate = max(row["max_column_missing_rate"] for row in quality_rows)
    headline_rows = [{
        "environment_count": len(environments),
        "model_count": len(models),
        "target_count": len(target_keys),
        "planned_runs": planned,
        "completed_runs": completed,
        "smoke_runs": smoke,
        "full_epoch_requirement": full_epoch_requirement,
        "seq_len_minutes": seq_len,
        "forecast_minutes": pred_len,
        "latest_data": timestamp_text(max_latest),
        "data_age_days": age_days,
        "rows_min": min(unique_raw_rows.values()),
        "rows_max": max(unique_raw_rows.values()),
    }]
    sources = [
        make_source(
            "dataset_csv", f"Kweather {data_version} CSV",
            f"data/kweather/{data_version}/environments.csv",
            "Profiles source CSVs and reproduces target-specific filtering and chronological splits.",
            (
                f"SELECT * FROM read_csv_auto('data/kweather/{data_version}/*.csv', "
                "union_by_name=true, filename=true);"
            ),
            generated_at, [f"{environment}.csv" for environment in environments],
            [
                f"target <= {threshold:g} W", f"split={split_rate}",
                f"seq_len={seq_len}", f"pred_len={pred_len}", f"stride={stride}",
            ],
            {
                "forecast_windows": "max(0, floor((rows - seq_len - pred_len) / stride) + 1)",
                "outlier": f"selected target > {threshold:g} W",
                "zero_rate": "count(target == 0) / count(non-null target)",
            },
        ),
        make_source(
            "forecast_results", "Forecast test results and saved configs",
            f"saved_model/{data_version}/results.json",
            "Prefers full-epoch results matching the experiment definition and labels shorter runs as smoke tests.",
            (
                f"SELECT * FROM read_json_auto('saved_model/{data_version}/**/"
                "test_results.json', filename=true);"
            ),
            generated_at, ["test_results.json", "configs.yaml"],
            [
                f"data_version={data_version}", f"models={models}", "one target per run",
                f"seq_len={seq_len}", f"pred_len={pred_len}",
                f"full_result_epoch>={full_epoch_requirement}",
            ],
            {
                "prediction_accuracy": "1 - MAE / mean(y_true)",
                "MSE": "mean((prediction - actual)^2)",
                "MAE": "mean(abs(prediction - actual))",
                "RMSE": "sqrt(MSE)",
                "WAPE": "sum(abs(prediction - actual)) / sum(abs(actual))",
                "SMAPE": "mean(2 * abs(prediction - actual) / (abs(prediction) + abs(actual)))",
            },
        ),
    ]
    cards = [
        {
            "id": "environment_count", "description": "Environment CSV count.",
            "dataset": "headline", "sourceId": "dataset_csv",
            "metrics": [{"label": "분석 공간", "field": "environment_count", "format": "number"}],
        },
        {
            "id": "planned_runs", "description": "Planned experiment combinations.",
            "dataset": "headline", "sourceId": "dataset_csv",
            "metrics": [{"label": "계획 실험", "field": "planned_runs", "format": "number"}],
        },
        {
            "id": "completed_runs", "description": "Matching completed test results.",
            "dataset": "headline", "sourceId": "forecast_results",
            "metrics": [{"label": "완료 실험", "field": "completed_runs", "format": "number"}],
        },
        {
            "id": "forecast_minutes", "description": "Future one-minute forecast steps.",
            "dataset": "headline", "sourceId": "dataset_csv",
            "metrics": [{"label": "예측 구간(분)", "field": "forecast_minutes", "format": "number"}],
        },
    ]
    charts = [{
        "id": "mean_power_by_space",
        "title": "공간·타깃별 평균 유효전력",
        "subtitle": "0W 구간을 포함한 필터 후 평균이며 zero rate와 함께 해석한다.",
        "headerMarkdown": (
            "**질문:** 공간과 장비 타깃 사이의 전력 규모가 얼마나 다른가?  "
            "**선정 이유:** 동일 오차도 타깃 규모에 따라 의미가 달라 모델 표 해석의 기준이 된다."
        ),
        "type": "bar",
        "dataset": "target_distribution",
        "sourceId": "dataset_csv",
        "encodings": {
            "x": {"field": "cohort", "type": "nominal", "label": "공간 · 타깃"},
            "y": {
                "field": "mean_w", "type": "quantitative",
                "label": "평균 유효전력(W)", "format": "number",
            },
            "color": {"field": "target_label", "type": "nominal", "label": "타깃"},
            "tooltip": [
                {"field": "median_w", "type": "quantitative", "label": "중앙값(W)", "format": "number"},
                {"field": "max_w", "type": "quantitative", "label": "최대값(W)", "format": "number"},
                {"field": "zero_rate", "type": "quantitative", "label": "0W 비율", "format": "percent"},
            ],
        },
        "yAxisTitle": "평균 유효전력(W)",
        "valueFormat": "number",
        "layout": "full",
    }]
    tables = [
        {
            "id": "split_table", "title": "시간순 분할과 실제 예측 윈도우",
            "subtitle": "원시 행 수와 120+60분 윈도우 구성 후 샘플 수를 구분한다.",
            "dataset": "split_summary", "sourceId": "dataset_csv",
            "defaultSort": {"field": "environment_label", "direction": "asc"},
            "columns": [
                {"field": "environment_label", "label": "공간", "type": "text"},
                {"field": "target_label", "label": "타깃", "type": "text"},
                {"field": "split", "label": "분할", "type": "text"},
                {"field": "rows", "label": "행 수", "format": "number"},
                {"field": "forecast_windows", "label": "예측 윈도우", "format": "number"},
                {"field": "start", "label": "시작", "type": "text"},
                {"field": "end", "label": "종료", "type": "text"},
            ],
        },
        {
            "id": "quality_table", "title": "타깃별 데이터 품질",
            "subtitle": "결측·이상치·0W·중복 및 비정상 시간 간격을 함께 점검한다.",
            "dataset": "data_quality", "sourceId": "dataset_csv",
            "defaultSort": {"field": "environment_label", "direction": "asc"},
            "columns": [
                {"field": "environment_label", "label": "공간", "type": "text"},
                {"field": "target_label", "label": "타깃", "type": "text"},
                {"field": "raw_rows", "label": "원시 행", "format": "number"},
                {"field": "missing_cell_rate", "label": "전체 결측률", "format": "percent"},
                {"field": "max_column_missing_rate", "label": "최대 컬럼 결측률", "format": "percent"},
                {"field": "columns_over_10pct_missing", "label": "10%+ 결측 컬럼", "type": "text"},
                {"field": "missing_target", "label": "타깃 결측", "format": "number"},
                {"field": "outliers_over_threshold", "label": ">1000W", "format": "number"},
                {"field": "zero_rate", "label": "0W 비율", "format": "percent"},
                {"field": "duplicate_timestamp_rows", "label": "중복 시간 행", "format": "number"},
                {"field": "irregular_intervals", "label": "비정상 간격", "format": "number"},
                {"field": "max_positive_gap_minutes", "label": "최대 간격(분)", "format": "number"},
            ],
        },
        {
            "id": "model_table", "title": "모델별 테스트 성능",
            "subtitle": f"정식 완료는 {full_epoch_requirement} epoch 이상이며, 없으면 최신 스모크를 별도 표시한다.",
            "dataset": "model_results", "sourceId": "forecast_results",
            "defaultSort": {"field": "environment_label", "direction": "asc"},
            "columns": [
                {"field": "environment_label", "label": "공간", "type": "text"},
                {"field": "target_label", "label": "타깃", "type": "text"},
                {"field": "model", "label": "모델", "type": "text"},
                {"field": "train_epochs", "label": "Epoch", "format": "number"},
                {"field": "prediction_accuracy", "label": "전력량 정확도", "format": "number"},
                {"field": "mse", "label": "MSE", "format": "number"},
                {"field": "mae", "label": "MAE", "format": "number"},
                {"field": "rmse", "label": "RMSE", "format": "number"},
                {"field": "wape", "label": "WAPE(%)", "format": "number"},
                {"field": "smape", "label": "SMAPE(%)", "format": "number"},
                {"field": "status", "label": "상태", "type": "text"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# 케이웨더 전력량 시계열 예측 모델 기반 실험"},
        {
            "id": "technical_summary", "type": "markdown", "sourceId": "dataset_csv",
            "body": (
                "## 기술 요약\n\n현재 CSV에는 발전량 컬럼이 없어 **환기유효전력과 "
                "냉난방유효전력 소비량**을 각각 예측한다. 6개 모델, 3개 공간, 2개 타깃의 "
                "36개 단일 타깃 실험을 120분 입력 → 60분 예측으로 정의했다."
            ),
        },
        {
            "id": "metrics", "type": "metric-strip",
            "cardIds": ["environment_count", "planned_runs", "completed_runs", "forecast_minutes"],
        },
        {
            "id": "finding", "type": "markdown", "sourceId": "dataset_csv",
            "body": (
                "## 핵심 관찰\n\n아래 평균 전력 비교는 타깃별 규모 차이를 보여준다. "
                "냉난방은 0W 비중에 따라 평균과 1 - MAE/평균 실제값 지표가 크게 변하므로 "
                "모델 선정에서는 MAE와 RMSE를 함께 봐야 한다."
            ),
        },
        {"id": "power_chart", "type": "chart", "chartId": "mean_power_by_space", "layout": "full"},
        {
            "id": "scope", "type": "markdown", "sourceId": "dataset_csv",
            "body": (
                "## 범위·데이터·지표\n\n시간 순서를 유지한 8:1:1 분할을 사용한다. "
                f"선택 타깃이 {threshold:g}W를 초과하는 행은 제거하며, 각 분할 내부에서 "
                "120+60분 연속 윈도우를 stride 1로 구성한다."
            ),
        },
        {"id": "split", "type": "table", "tableId": "split_table", "layout": "full"},
        {
            "id": "quality_intro", "type": "markdown", "sourceId": "dataset_csv",
            "body": (
                "## 데이터 품질\n\n원시 행 수만으로 학습 가능성을 판단할 수 없다. "
                f"타깃 결측과 0W뿐 아니라 시간 공백을 확인한다. 최대 단일 컬럼 결측률은 **{max_column_missing_rate:.2%}**이다."
            ),
        },
        {"id": "quality", "type": "table", "tableId": "quality_table", "layout": "full"},
        {
            "id": "results_intro", "type": "markdown", "sourceId": "forecast_results",
            "body": (
                "## 모델 비교\n\n첨부 페이지와 동일하게 전력량 정확도와 MSE를 포함하되, "
                "정확도 분모가 작은 경우를 대비해 MAE·RMSE를 같이 보고한다. "
                f"현재 정식 완료는 {completed}/{planned}회, 스모크는 {smoke}회이다. "
                "스모크는 정식 완료에 포함하지 않고, 미실행 행에는 수치를 채우지 않는다."
            ),
        },
        {"id": "results", "type": "table", "tableId": "model_table", "layout": "full"},
        {
            "id": "methodology", "type": "markdown", "sourceId": "dataset_csv",
            "body": (
                "## 방법론\n\n1. 원본 CSV와 설정을 읽어 스키마·시간축·결측·이상치를 점검한다.\n"
                "2. 타깃 이상치를 제거하고 결측을 forward-fill 후 필요한 선두 구간만 back-fill한다.\n"
                "3. 시간순 8:1:1로 분할하고 Train 통계로 정규화한다.\n"
                "4. 한 실행에 한 타깃만 학습하고 테스트 예측을 역정규화한다.\n"
                "5. 같은 조합 결과가 여러 개면 현재 정의와 일치하는 가장 최근 결과를 선택한다."
            ),
        },
        {
            "id": "limitations", "type": "markdown", "sourceId": "dataset_csv",
            "body": (
                "## 한계와 견고성\n\n- 실제 발전량 예측에는 발전량·일사량·기상예보가 필요하다.\n"
                "- 단일 8:1:1 홀드아웃은 계절 일반화를 보장하지 않아 rolling-origin 검증이 필요하다.\n"
                "- 전력량 정확도는 평균 실제값이 0에 가까우면 음수 또는 큰 값이 될 수 있다.\n"
                f"- 최대 단일 컬럼 결측률은 {max_column_missing_rate:.2%}이므로 대치 민감도를 검증해야 한다.\n"
                "- 시작부 back-fill이 필요한 새 데이터는 분할별 대치 또는 train-fitted imputer로 재검증해야 한다.\n"
                f"- 최신 데이터 시점은 {timestamp_text(max_latest)}이며 생성일 기준 약 {age_days}일 전이다."
            ),
        },
        {
            "id": "next_steps", "type": "markdown",
            "body": (
                "## 다음 단계\n\n1. 36개 실험을 완료한다.\n"
                "2. 타깃별 MAE/RMSE 최우수 모델을 선택한다.\n"
                "3. persistence 기준선과 rolling-origin backtest를 추가한다.\n"
                "4. MPC 연결 전 lead-time별 오차와 예측 불확실성을 산출한다."
            ),
        },
        {
            "id": "questions", "type": "markdown",
            "body": (
                "## 추가로 결정할 질문\n\n- ‘발전량’이 실제 발전 설비 출력인지 현재 소비전력인지?\n"
                "- 운영 모델 선정의 주지표를 MAE, RMSE, 전력량 정확도 중 무엇으로 둘지?\n"
                "- 10/30/60분 lead-time별 성능을 표준 보고 항목으로 둘지?"
            ),
        },
    ]
    snapshot: dict[str, Any] = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready" if completed == planned else "partial",
        "datasets": {
            "headline": to_records(headline_rows),
            "data_quality": to_records(quality_rows),
            "split_summary": to_records(split_rows),
            "target_distribution": to_records(distribution_rows),
            "model_results": to_records(result_rows),
        },
    }
    if completed != planned:
        snapshot["accessIssues"] = [{
            "id": "forecast_results_incomplete",
            "dataset": "model_results",
            "message": (
                f"현재 실험 정의와 일치하는 테스트 결과가 {completed}/{planned}개입니다. "
                "미완료 조합은 모델 성능 표에 '미실행'으로 표시됩니다."
            ),
        }]
    return {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report",
            "title": "케이웨더 전력량 시계열 예측 모델 기반 실험",
            "description": "Data profile, chronological split, experiment matrix, and evaluation.",
            "generatedAt": generated_at,
            "cards": cards, "charts": charts, "tables": tables,
            "sources": sources, "blocks": blocks,
        },
        "snapshot": snapshot,
        "sources": sources,
        "package_info": {
            "root": ".", "manifestPath": "artifact.json", "snapshotPath": "artifact.json"
        },
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    dataset_cfg = config["DATASET"]
    src_dir = Path(__file__).resolve().parents[1]
    data_version = str(dataset_cfg["data_version"])
    data_dir = resolve_data_dir(str(dataset_cfg["datadir"]), data_version, src_dir)
    split_rate = [float(value) for value in dataset_cfg["split_rate"]]
    if len(split_rate) != 3 or not math.isclose(sum(split_rate), 1.0, abs_tol=1e-8):
        raise ValueError(f"split_rate must have three values summing to 1: {split_rate}")
    seq_len = int(dataset_cfg["seq_len"])
    pred_len = int(dataset_cfg["pred_len"])
    stride = int(dataset_cfg["stride_len"])
    expected_epochs = int(config["TRAIN"]["epoch"])
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    quality_rows, split_rows, distribution_rows = profile_data(
        data_dir, args.environments, args.target_keys, split_rate,
        dataset_cfg.get("start_date"), dataset_cfg.get("end_date"),
        seq_len, pred_len, stride, args.outlier_threshold,
    )
    result_rows, ignored_results = scan_results(
        args.results_dir.resolve(), data_version, args.models, args.environments,
        args.target_keys, seq_len, pred_len, expected_epochs,
    )
    output_dir = args.output_dir.resolve() / data_version
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "data_quality.csv", quality_rows)
    write_csv(output_dir / "split_summary.csv", split_rows)
    write_csv(output_dir / "target_distribution.csv", distribution_rows)
    write_csv(output_dir / "model_results.csv", result_rows)

    markdown = build_markdown(
        data_version, generated_at, quality_rows, split_rows, result_rows,
        args.models, args.environments, args.target_keys, seq_len, pred_len,
        stride, split_rate, args.outlier_threshold, ignored_results,
    )
    (output_dir / "forecasting_report.md").write_text(markdown, encoding="utf-8")
    artifact = build_artifact(
        data_version, generated_at, quality_rows, split_rows, distribution_rows,
        result_rows, args.models, args.environments, args.target_keys,
        seq_len, pred_len, stride, split_rate, args.outlier_threshold,
    )
    with (output_dir / "artifact.json").open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)

    completed = sum(row["status"] == "완료" for row in result_rows)
    print(f"Data directory: {data_dir}")
    print(f"Report directory: {output_dir}")
    print(f"Completed results: {completed}/{len(result_rows)}")
    print(f"Artifact: {output_dir / 'artifact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
