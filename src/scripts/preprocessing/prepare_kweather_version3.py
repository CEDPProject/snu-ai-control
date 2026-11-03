#!/usr/bin/env python
"""Reprocess Kweather version3 Excel files under the shared processed contract.

version3_260714_processed는 초기 파이프라인으로 만들어져 결측 보간 / 제어 결측
채움 / OFF canonicalization 이 적용되지 않았고, 허용 밖 카테고리(냉난방설정온도=25)와
다수의 결측이 남아 있었다. 이 스크립트는 version4/version5와 **동일한 전처리 계약**을
version3 원본에 적용해 세 버전의 processed 계약을 통일한다.

- 원본: 공간별(거실·방1·방2) 가공 Excel 파일
  (기존 `version3_260714_processed`와 행 수가 일치하는 원천)
- 변환 로직은 :mod:`prepare_kweather_version4`를 그대로 재사용한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import prepare_kweather_version4 as v4
from prepare_kweather_version4 import (
    ROOM_FOLDERS,
    _find_processed_excel,
    _find_room_dir,
    _json_default,
    _prepare_room,
)

VERSION_LABEL = "version3_260714"
BASELINE_LABEL = "version4_260805_processed / version5_260814_processed"


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_markdown_report(report: dict[str, Any], path: Path, version_label: str) -> None:
    lines = [
        f"# Kweather {version_label} 재처리 리포트",
        "",
        "## 요약",
        "",
        f"- 기존 `{version_label}_processed`는 초기 파이프라인 산출물로, 결측 보간 / 제어 결측 채움 / OFF canonicalization 이 적용되지 않았고 허용 밖 카테고리(냉난방설정온도=25)와 다수의 결측이 남아 있었다.",
        f"- 이 재처리는 version4/version5와 동일한 전처리 계약(`prepare_kweather_version4.py`)을 `{version_label}` 원본에 적용해 세 버전의 processed 계약을 통일한다.",
        "- 원본은 공간별(거실·방1·방2) 가공 Excel 파일이며, 기존 processed와 행 수가 일치한다.",
        "- 제어 카테고리는 현재 MPC 설정 허용값에 맞도록 OFF 상태를 대표 제어값으로 정규화했다.",
        "- 숫자 결측은 시간순 선형 보간 후 앞/뒤 채움으로 해소했다(결과 CSV 결측 0).",
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
            "version4/version5와 동일한 정책을 적용했다. 세부 구현은 `prepare_kweather_version4.py`를 참고한다.",
            "",
            "1. 컬럼 검증: 기존 processed CSV의 23개 컬럼과 정확히 일치하는지 확인하고, 다르면 변환을 중단한다.",
            "2. 시간 정렬과 중복 제거: `데이터 시간`을 datetime으로 파싱해 오름차순 정렬하고, 동일 timestamp가 있으면 마지막 값을 남긴다.",
            "3. 제어값 결측 보정: 제어 컬럼 결측을 forward/backward fill로 채우고, 그래도 남으면 각 장비의 기본 OFF 대표값으로 채운다.",
            "4. OFF 상태 canonicalization: `환기전원=OFF`이면 `환기풍량=0`, `환기운전모드=꺼짐`. `냉난방전원=OFF`이면 `냉난방운전모드=냉방`, `냉난방풍량=약풍`, `냉난방설정온도(℃)=22`.",
            "5. 숫자 결측 보정: 숫자 컬럼을 강제 변환한 뒤 시간순 선형 보간 + 앞/뒤 채움으로 마무리한다. 결과 CSV에는 결측이 남지 않는다.",
            "6. 지원되지 않는 카테고리 처리: 허용값 밖 제어 카테고리(예: `냉난방설정온도(℃)=25`)가 OFF 상태에서 나타나면 OFF canonicalization으로 대표값(22)으로 정규화된다.",
            "7. 유지한 것: 1분보다 긴 timestamp gap은 행을 삭제하거나 삽입하지 않고 그대로 보존한다.",
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
            "- 실외 환경 변수가 가장 큰 보간 대상이다. 외부 기상 원천 데이터가 추가되면 다시 대체하는 것이 좋다.",
            "- 허용값 밖 제어 카테고리(`냉난방설정온도=25`)는 모두 냉난방 OFF 행에서만 나타났고, OFF canonicalization으로 대표값 22로 정규화됐다.",
            "- 이 재처리로 기존 v3 processed 대비 결측이 채워지고 OFF 하위 제어가 대표값으로 고정되므로, v3 기반 기존 학습/실험 산출물은 재실행이 필요할 수 있다.",
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
        default=default_kweather_root / "data_arxiv" / VERSION_LABEL,
        help="Directory containing the version3 room Excel folders (거실/방1/방2).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_kweather_root / f"{VERSION_LABEL}_processed",
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
        raise FileNotFoundError(f"Raw {VERSION_LABEL} directory not found: {raw_root}")

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

    json_report_path = json_report_dir / f"kweather_{VERSION_LABEL}_reprocessing_report.json"
    report["json_report_path"] = str(json_report_path)
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    markdown_report_path = report_dir / f"kweather_{VERSION_LABEL}_reprocessing_report.md"
    _write_markdown_report(report, markdown_report_path, VERSION_LABEL)

    print(f"Processed CSVs written to: {output_root}")
    print(f"JSON report written to: {json_report_path}")
    print(f"Markdown report written to: {markdown_report_path}")


if __name__ == "__main__":
    main()
