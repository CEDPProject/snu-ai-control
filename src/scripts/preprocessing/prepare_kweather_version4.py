#!/usr/bin/env python
"""Prepare Kweather version4 Excel files for forecast and MPC pipelines."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "데이터 시간",
    "실내미세먼지(㎍/㎥)",
    "실내초미세먼지(㎍/㎥)",
    "실내이산화탄소(ppm)",
    "실내휘발성유기화합물(ppb)",
    "실내온도(℃)",
    "실내습도(%)",
    "실외미세먼지(㎍/㎥)",
    "실외초미세먼지(㎍/㎥)",
    "실외온도(℃)",
    "실외습도(%)",
    "환기전원",
    "환기풍량",
    "환기운전모드",
    "환기누적전력량(Wh)",
    "환기유효전력(W)",
    "냉난방전원",
    "냉난방운전모드",
    "냉난방풍량",
    "냉난방설정온도(℃)",
    "냉난방누적전력량(Wh)",
    "냉난방유효전력(W)",
    "냉난방유효전력(W)_분차분",
]

ROOM_FOLDERS = {
    "livingroom": "거실",
    "room1": "방1",
    "room2": "방2",
}

NUMERIC_COLUMNS = [
    "실내미세먼지(㎍/㎥)",
    "실내초미세먼지(㎍/㎥)",
    "실내이산화탄소(ppm)",
    "실내휘발성유기화합물(ppb)",
    "실내온도(℃)",
    "실내습도(%)",
    "실외미세먼지(㎍/㎥)",
    "실외초미세먼지(㎍/㎥)",
    "실외온도(℃)",
    "실외습도(%)",
    "환기풍량",
    "환기누적전력량(Wh)",
    "환기유효전력(W)",
    "냉난방설정온도(℃)",
    "냉난방누적전력량(Wh)",
    "냉난방유효전력(W)",
    "냉난방유효전력(W)_분차분",
]

CONTROL_COLUMNS = [
    "환기전원",
    "환기풍량",
    "환기운전모드",
    "냉난방전원",
    "냉난방운전모드",
    "냉난방풍량",
    "냉난방설정온도(℃)",
]

STRING_CONTROL_COLUMNS = [
    "환기전원",
    "환기운전모드",
    "냉난방전원",
    "냉난방운전모드",
    "냉난방풍량",
]

DEFAULT_CONTROLS = {
    "환기전원": "OFF",
    "환기풍량": 0.0,
    "환기운전모드": "꺼짐",
    "냉난방전원": "OFF",
    "냉난방운전모드": "냉방",
    "냉난방풍량": "약풍",
    "냉난방설정온도(℃)": 22.0,
}

ALLOWED_CATEGORIES = {
    "환기전원": {"OFF", "ON"},
    "환기풍량": {0.0, 2.0, 4.0, 6.0},
    "환기운전모드": {"꺼짐", "공기청정", "바이패스", "환기"},
    "냉난방전원": {"OFF", "ON"},
    "냉난방운전모드": {"냉방", "난방"},
    "냉난방풍량": {"약풍", "중풍", "강풍"},
    "냉난방설정온도(℃)": {18.0, 22.0, 26.0},
}


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat(sep=" ")
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _normalize_text(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if text.upper() in {"ON", "OFF"}:
        return text.upper()
    return unicodedata.normalize("NFC", text)


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _find_room_dir(raw_root: Path, korean_name: str) -> Path:
    wanted = _nfc(korean_name)
    for path in raw_root.iterdir():
        if path.is_dir() and _nfc(path.name) == wanted:
            return path
    raise FileNotFoundError(f"Room directory not found: {raw_root / korean_name}")


def _find_processed_excel(room_dir: Path) -> Path:
    candidates = []
    for path in room_dir.glob("*.xlsx"):
        name = _nfc(path.name)
        if "가공" in name and "미가공" not in name:
            candidates.append(path)
    if len(candidates) != 1:
        found = [path.name for path in candidates]
        raise FileNotFoundError(
            f"Expected exactly one processed Excel in {room_dir}, found {found}"
        )
    return candidates[0]


def _value_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for column in columns:
        values = frame[column].astype("string").fillna("<NA>")
        result[column] = {str(k): int(v) for k, v in values.value_counts().items()}
    return result


def _invalid_categories(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    invalid: dict[str, dict[str, int]] = {}
    for column, allowed in ALLOWED_CATEGORIES.items():
        series = frame[column]
        if column in {"환기풍량", "냉난방설정온도(℃)"}:
            values = pd.to_numeric(series, errors="coerce")
            mask = values.notna() & ~values.isin(allowed)
            counts = values[mask].value_counts()
        else:
            values = series.map(_normalize_text)
            mask = values.notna() & ~values.isin(allowed)
            counts = values[mask].value_counts()
        if len(counts):
            invalid[column] = {str(k): int(v) for k, v in counts.items()}
    return invalid


def _time_profile(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values("데이터 시간")
    timestamps = ordered["데이터 시간"]
    deltas = timestamps.diff().dropna()
    non_one_minute = deltas[deltas != pd.Timedelta(minutes=1)]
    return {
        "start": timestamps.min(),
        "end": timestamps.max(),
        "one_minute_transition_count": int((deltas == pd.Timedelta(minutes=1)).sum()),
        "transition_count": int(len(deltas)),
        "gap_count": int(len(non_one_minute)),
        "gap_durations": {
            str(k): int(v) for k, v in non_one_minute.value_counts().head(10).items()
        },
    }


def _max_missing_run(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    best = current = 0
    for value in missing:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _fill_controls(frame: pd.DataFrame) -> dict[str, int]:
    before = frame[CONTROL_COLUMNS].isna().sum()

    for column in STRING_CONTROL_COLUMNS:
        frame[column] = frame[column].map(_normalize_text)

    for column in CONTROL_COLUMNS:
        frame[column] = frame[column].ffill().bfill()
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(DEFAULT_CONTROLS[column])

    for column in ["환기풍량", "냉난방설정온도(℃)"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].ffill().bfill().fillna(DEFAULT_CONTROLS[column])

    filled = (before - frame[CONTROL_COLUMNS].isna().sum()).clip(lower=0)
    return {column: int(count) for column, count in filled.items() if int(count) > 0}


def _canonicalize_off_controls(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}

    ventilation_off = frame["환기전원"].map(_normalize_text).eq("OFF")
    before = frame.loc[ventilation_off, ["환기풍량", "환기운전모드"]].copy()
    frame.loc[ventilation_off, "환기풍량"] = 0.0
    frame.loc[ventilation_off, "환기운전모드"] = "꺼짐"
    after = frame.loc[ventilation_off, ["환기풍량", "환기운전모드"]]
    counts["ventilation_off_rows"] = int(ventilation_off.sum())
    counts["ventilation_off_field_changes"] = int((before != after).sum().sum())

    hvac_off = frame["냉난방전원"].map(_normalize_text).eq("OFF")
    before = frame.loc[
        hvac_off, ["냉난방운전모드", "냉난방풍량", "냉난방설정온도(℃)"]
    ].copy()
    frame.loc[hvac_off, "냉난방운전모드"] = "냉방"
    frame.loc[hvac_off, "냉난방풍량"] = "약풍"
    frame.loc[hvac_off, "냉난방설정온도(℃)"] = 22.0
    after = frame.loc[
        hvac_off, ["냉난방운전모드", "냉난방풍량", "냉난방설정온도(℃)"]
    ]
    counts["hvac_off_rows"] = int(hvac_off.sum())
    counts["hvac_off_field_changes"] = int((before != after).sum().sum())

    return counts


def _fill_numeric(frame: pd.DataFrame) -> dict[str, Any]:
    before = frame[NUMERIC_COLUMNS].isna().sum()
    max_missing_run_before = {
        column: _max_missing_run(frame[column])
        for column in NUMERIC_COLUMNS
        if int(before[column]) > 0
    }

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[NUMERIC_COLUMNS] = (
        frame[NUMERIC_COLUMNS].interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    if frame[NUMERIC_COLUMNS].isna().any().any():
        missing = frame[NUMERIC_COLUMNS].columns[
            frame[NUMERIC_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(f"Unresolved numeric missing values: {missing}")

    filled = (before - frame[NUMERIC_COLUMNS].isna().sum()).clip(lower=0)
    return {
        "filled_count_by_column": {
            column: int(count) for column, count in filled.items() if int(count) > 0
        },
        "max_missing_run_before": max_missing_run_before,
    }


def _prepare_room(excel_path: Path, output_file: Path) -> dict[str, Any]:
    frame = pd.read_excel(excel_path)
    raw_columns = list(frame.columns)
    columns_match = raw_columns == EXPECTED_COLUMNS
    if not columns_match:
        missing = sorted(set(EXPECTED_COLUMNS) - set(raw_columns))
        extra = sorted(set(raw_columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            f"Unexpected schema in {excel_path}: missing={missing}, extra={extra}"
        )

    rows_before = len(frame)
    nulls_before = frame.isna().sum()
    invalid_before = _invalid_categories(frame)
    category_counts_before = _value_counts(frame, CONTROL_COLUMNS)

    frame["데이터 시간"] = pd.to_datetime(frame["데이터 시간"], errors="raise")
    frame = frame.sort_values("데이터 시간").drop_duplicates("데이터 시간", keep="last")
    duplicates_removed = rows_before - len(frame)
    frame = frame.reset_index(drop=True)

    control_fill_counts = _fill_controls(frame)
    canonicalized_counts = _canonicalize_off_controls(frame)
    numeric_fill_report = _fill_numeric(frame)

    frame = frame[EXPECTED_COLUMNS]
    nulls_after = frame.isna().sum()
    invalid_after = _invalid_categories(frame)
    if any(int(v) for v in nulls_after.values):
        unresolved = {
            column: int(count) for column, count in nulls_after.items() if int(count) > 0
        }
        raise ValueError(f"Unresolved missing values remain in {output_file}: {unresolved}")
    if invalid_after:
        raise ValueError(f"Unsupported control categories remain in {output_file}: {invalid_after}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")

    time_info = _time_profile(frame)
    return {
        "source_file": str(excel_path),
        "output_file": str(output_file),
        "columns_match_expected": columns_match,
        "rows_before": int(rows_before),
        "rows_after": int(len(frame)),
        "duplicate_timestamps_removed": int(duplicates_removed),
        "nulls_before": {
            column: int(count) for column, count in nulls_before.items() if int(count) > 0
        },
        "nulls_after": {
            column: int(count) for column, count in nulls_after.items() if int(count) > 0
        },
        "control_fill_counts": control_fill_counts,
        "numeric_fill_report": numeric_fill_report,
        "canonicalized_counts": canonicalized_counts,
        "invalid_categories_before": invalid_before,
        "invalid_categories_after": invalid_after,
        "control_category_counts_before": category_counts_before,
        "control_category_counts_after": _value_counts(frame, CONTROL_COLUMNS),
        "time_profile": time_info,
        "power_summary_after": {
            column: {
                "min": float(frame[column].min()),
                "p50": float(frame[column].quantile(0.50)),
                "p95": float(frame[column].quantile(0.95)),
                "p99": float(frame[column].quantile(0.99)),
                "max": float(frame[column].max()),
                "negative_count": int((frame[column] < 0).sum()),
                "over_1000w_count": int((frame[column] > 1000).sum()),
            }
            for column in ["환기유효전력(W)", "냉난방유효전력(W)"]
        },
    }


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Kweather version4_260805 전처리 리포트",
        "",
        "## 요약",
        "",
        "- 원본 Excel은 보존하고 `version4_260805_processed`에 프레임워크 입력 CSV를 생성했다.",
        "- 기존 `version3_260714_processed`와 같은 23개 컬럼 순서를 유지했다.",
        "- 제어 카테고리는 현재 MPC 설정의 허용값과 맞도록 OFF 상태를 대표 제어값으로 정규화했다.",
        "- 숫자 결측은 시간순 선형 보간 후 앞/뒤 채움으로 해소했다.",
        "",
        "## 기준 데이터 형태",
        "",
        "기존 프레임워크는 `version3_260714_processed/{livingroom,room1,room2}.csv` 형태를 기준으로 동작한다. 따라서 version4도 아래 조건을 맞추도록 변환했다.",
        "",
        "- 파일 단위: 공간별 CSV 3개(`livingroom.csv`, `room1.csv`, `room2.csv`).",
        "- 시간 단위: `데이터 시간` 기준 시간순 1분 로그. 일부 2~3분 gap은 보존한다.",
        "- 컬럼 단위: 기존 processed 데이터와 동일한 23개 컬럼 및 동일한 컬럼 순서.",
        "- Forecast 입력: 스케일링 전 원본 CSV를 dataloader가 읽고 내부에서 encoding/scaling한다.",
        "- MPC 입력: 같은 CSV를 스케일 없이 물리 단위로 읽고, 제어 변수는 설정 파일의 카테고리와 정확히 일치해야 한다.",
        "",
        "## 원본 선택 기준",
        "",
        "`version4_260805`에는 공간별로 `가공.xlsx`와 `미가공.xlsx`가 함께 있다. 이번 변환은 `가공.xlsx`를 기준으로 했다.",
        "",
        "- `가공.xlsx`: 기존 processed 데이터와 같은 23개 컬럼을 이미 포함한다.",
        "- `미가공.xlsx`: `냉난방유효전력(W)_분차분` 컬럼이 없고, 냉난방 유효전력 계산 전 상태에 가깝다.",
        "- 따라서 모델 학습과 MPC 평가 계약을 맞추려면 `가공.xlsx`가 더 직접적인 입력이다.",
        "",
        "## 산출 파일",
        "",
        "| environment | rows | start | end | output |",
        "|---|---:|---|---|---|",
    ]

    for env, item in report["environments"].items():
        time_info = item["time_profile"]
        lines.append(
            f"| {env} | {item['rows_after']:,} | {time_info['start']} | "
            f"{time_info['end']} | `{item['output_file']}` |"
        )

    lines.extend(
        [
            "",
            "## 주요 수정 사항",
            "",
            "| environment | control fills | numeric fills | OFF canonical field changes | invalid categories before -> after | 1-min transitions / total | gaps |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for env, item in report["environments"].items():
        control_fills = sum(item["control_fill_counts"].values())
        numeric_fills = sum(item["numeric_fill_report"]["filled_count_by_column"].values())
        canonical_changes = (
            item["canonicalized_counts"]["ventilation_off_field_changes"]
            + item["canonicalized_counts"]["hvac_off_field_changes"]
        )
        invalid_before = item["invalid_categories_before"] or {}
        invalid_after = item["invalid_categories_after"] or {}
        time_info = item["time_profile"]
        lines.append(
            f"| {env} | {control_fills:,} | {numeric_fills:,} | "
            f"{canonical_changes:,} | `{invalid_before}` -> `{invalid_after}` | "
            f"{time_info['one_minute_transition_count']:,} / {time_info['transition_count']:,} | "
            f"{time_info['gap_count']:,} |"
        )

    lines.extend(
        [
            "",
            "## 수정 정책",
            "",
            "1. 컬럼 검증",
            "",
            "   기존 processed CSV의 23개 컬럼과 정확히 일치하는지 확인했다. 컬럼이 빠지거나 추가되면 변환을 중단하도록 했다.",
            "",
            "2. 시간 정렬과 중복 제거",
            "",
            "   `데이터 시간`을 datetime으로 파싱하고 시간순으로 정렬했다. 동일 timestamp가 있으면 마지막 값을 남기는 정책을 넣었지만, 이번 version4에서는 중복 timestamp가 없었다.",
            "",
            "3. 제어값 결측 보정",
            "",
            "   `환기전원`, `환기풍량`, `환기운전모드`, `냉난방전원`, `냉난방운전모드`, `냉난방풍량`, `냉난방설정온도(℃)`의 결측은 앞뒤 관측값으로 채웠다. 짧은 장비 로그 누락은 실제 제어 변경보다 수집 누락일 가능성이 높기 때문이다. 그래도 남는 값이 있으면 각 장비의 기본 OFF 대표값으로 채우도록 했다.",
            "",
            "4. OFF 상태 canonicalization",
            "",
            "   MPC의 logical rule과 같은 의미가 되도록 OFF 상태의 하위 제어값을 대표값으로 고정했다.",
            "",
            "   - `환기전원=OFF`이면 `환기풍량=0`, `환기운전모드=꺼짐`.",
            "   - `냉난방전원=OFF`이면 `냉난방운전모드=냉방`, `냉난방풍량=약풍`, `냉난방설정온도(℃)=22`.",
            "",
            "5. 숫자 결측 보정",
            "",
            "   상태, 외란, 누적전력, 유효전력 컬럼은 숫자로 강제 변환한 뒤 시간순 선형 보간을 적용하고 양끝은 앞/뒤 채움으로 마무리했다. 결과 CSV에는 결측이 남지 않는다.",
            "",
            "6. 지원되지 않는 카테고리 처리",
            "",
            "   `냉난방설정온도(℃)=25`가 소수 행 존재했지만 모두 `냉난방전원=OFF` 상태였다. OFF 상태에서는 설정온도가 실제 제어 의미를 갖지 않으므로, 현재 MPC 대표값인 `22`로 정규화했다.",
            "",
            "7. 유지한 것",
            "",
            "   1분보다 긴 timestamp gap은 행을 삭제하거나 새 행을 삽입하지 않고 그대로 보존했다. MPC의 transition 생성 로직은 1분 간격 transition만 사용하므로, 원본 시간 구조를 숨기지 않는 편이 더 안전하다.",
            "",
            "## 세부 수정량",
            "",
            "### 제어 결측 보정",
            "",
            "| environment | filled controls |",
            "|---|---|",
        ]
    )
    for env, item in report["environments"].items():
        lines.append(f"| {env} | `{item['control_fill_counts']}` |")

    lines.extend(
        [
            "",
            "### 숫자 결측 보정",
            "",
            "| environment | filled numeric columns | max missing run before fill |",
            "|---|---|---|",
        ]
    )
    for env, item in report["environments"].items():
        fill_counts = item["numeric_fill_report"]["filled_count_by_column"]
        max_runs = item["numeric_fill_report"]["max_missing_run_before"]
        lines.append(f"| {env} | `{fill_counts}` | `{max_runs}` |")

    lines.extend(
        [
            "",
            "### 전력값 검증",
            "",
            "| environment | ventilation power W min/p50/p95/p99/max | hvac power W min/p50/p95/p99/max | negative | >1000W |",
            "|---|---|---|---:|---:|",
        ]
    )
    for env, item in report["environments"].items():
        vent = item["power_summary_after"]["환기유효전력(W)"]
        hvac = item["power_summary_after"]["냉난방유효전력(W)"]
        negative = vent["negative_count"] + hvac["negative_count"]
        over_1000 = vent["over_1000w_count"] + hvac["over_1000w_count"]
        lines.append(
            f"| {env} | {vent['min']:.3g}/{vent['p50']:.3g}/{vent['p95']:.3g}/{vent['p99']:.3g}/{vent['max']:.3g} | "
            f"{hvac['min']:.3g}/{hvac['p50']:.3g}/{hvac['p95']:.3g}/{hvac['p99']:.3g}/{hvac['max']:.3g} | "
            f"{negative:,} | {over_1000:,} |"
        )

    lines.extend(
        [
            "",
            "## 데이터 품질 메모",
            "",
            "- 1분 간격이 아닌 transition은 제거하지 않았다. MPC와 forecasting dataloader가 시간순 데이터를 사용하되, MPC transition 생성 시 1분 간격만 유효 transition으로 사용한다.",
            "- 실외 환경 변수가 가장 큰 보간 대상이므로 외부 기상 원천 데이터가 추가되면 다시 대체하는 것이 좋다.",
            "- `냉난방설정온도(℃)=25`는 모두 `냉난방전원=OFF` 행에서만 발견되어, MPC 논리 규칙과 동일하게 OFF 대표값 `22`로 정규화했다.",
            "- 전력 컬럼의 음수·1,000W 초과 이상치 수는 위 표를 참고한다.",
            "- 기본 설정 파일의 `DATASET.data_version`은 바꾸지 않았다. 기존 실험 재현성을 위해 version3 기본값은 유지하고, version4를 쓸 때 `DATASET.data_version=version4_260805_processed`를 명시한다.",
            "",
            "## 상세 JSON",
            "",
            f"- `{report['json_report_path']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    src_dir = _repo_src_dir()
    default_kweather_root = (src_dir / "../../data/kweather").resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_kweather_root / "version4_260805",
        help="Directory containing the version4 room Excel folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_kweather_root / "version4_260805_processed",
        help="Directory where processed CSV files will be written.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=src_dir.parent / "reports/docs",
        help="Directory where the Markdown preprocessing report will be written.",
    )
    parser.add_argument(
        "--json-report-dir",
        type=Path,
        default=src_dir.parent / "reports/artifacts/preprocessing",
        help="Directory where the structured JSON preprocessing report will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    report_dir = args.report_dir.resolve()
    json_report_dir = args.json_report_dir.resolve()

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw version4 directory not found: {raw_root}")

    report: dict[str, Any] = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "environments": {},
    }

    for env, korean_folder in ROOM_FOLDERS.items():
        room_dir = _find_room_dir(raw_root, korean_folder)
        excel_path = _find_processed_excel(room_dir)
        output_file = output_root / f"{env}.csv"
        report["environments"][env] = _prepare_room(excel_path, output_file)

    json_report_path = json_report_dir / "kweather_version4_260805_preprocessing_report.json"
    report["json_report_path"] = str(json_report_path)
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    markdown_report_path = report_dir / "kweather_version4_260805_preprocessing_report.md"
    _write_markdown_report(report, markdown_report_path)

    print(f"Processed CSVs written to: {output_root}")
    print(f"JSON report written to: {json_report_path}")
    print(f"Markdown report written to: {markdown_report_path}")


if __name__ == "__main__":
    main()
