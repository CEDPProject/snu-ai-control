"""Recorded-action-anchored Full-guard MPC distance sweep.

For every declared categorical-variable distance K, this runner builds a fresh
Full-guard MPC action trace.  Its first action can differ from the *current
recorded* action by at most K control variables.  The three report-model input
bundles are then evaluated on that immutable trace without re-solving MILP.
K=0 is an offline factual replay baseline; K>0 is a model-based diagnostic.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import re
from typing import Any

from omegaconf import OmegaConf
import pandas as pd

from controllers.pipeline import run_mpc_pipeline
from data_provider.dataset_versions import dataset_label_from_cfg, versions_from_cfg
from runners.counterfactual_report_input_ablation_runner import (
    run_counterfactual_report_input_ablation,
)
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC.action_distance_ablation')


def _plain(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def _safe_name(value: str) -> str:
    result = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._')
    if not result:
        raise ValueError('Experiment name must contain a filesystem-safe token')
    return result


def _result_directory(cfg) -> Path:
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run_name = _safe_name(f'{cfg.DEFAULT.exp_name}_{timestamp}_{os.getpid()}')
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'action_distance_ablation'
        / run_name
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    log_path = result_dir / 'logs' / 'action_distance_ablation.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _distances(cfg) -> list[int]:
    specification = _plain(cfg.MPC.action_distance_ablation)
    raw = specification.get('distances')
    if not isinstance(raw, list) or not raw:
        raise TypeError('MPC.action_distance_ablation.distances must be a nonempty list')
    maximum = len(cfg.MPC.control_variables)
    distances = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError('Every action-distance value must be an integer')
        if not 0 <= value <= maximum:
            raise ValueError(
                f'Action distance must be in [0, {maximum}], got {value}'
            )
        distances.append(int(value))
    if distances != sorted(distances) or len(distances) != len(set(distances)):
        raise ValueError('Action distances must be unique and ascending')
    return distances


def _scenario_cfg(cfg, distance: int):
    """Fix every changed element to the approved Full-guard experiment."""
    specification = _plain(cfg.MPC.action_distance_ablation)
    families = specification.get('report_model_candidate_families') or []
    if not isinstance(families, list) or not families:
        raise TypeError('report_model_candidate_families must be a nonempty list')
    if specification.get('first_action_reference') != 'recorded_current_action':
        raise ValueError(
            'action_distance_ablation requires first_action_reference='
            'recorded_current_action'
        )
    if specification.get('evaluation_row_scope') != 'all_chronological_test':
        raise ValueError(
            'action_distance_ablation requires evaluation_row_scope='
            'all_chronological_test'
        )
    closed_loop_windows = specification.get('closed_loop_max_windows')
    if isinstance(closed_loop_windows, bool) or not isinstance(
        closed_loop_windows, int
    ) or closed_loop_windows <= 0:
        raise TypeError('closed_loop_max_windows must be a positive integer')

    return OmegaConf.merge(
        cfg,
        {
            'DEFAULT': {'exp_name': f'{cfg.DEFAULT.exp_name}_distance_k{distance}'},
            'MPC': {
                'independent_evaluator': {'model_family': 'component_two_stage_hvac'},
                'objective': {
                    'maximum_changed_control_variables_per_step': int(distance),
                },
                'evaluation': {
                    # The action-distance table evaluates the all-test
                    # first-action trace. One closed-loop window is retained
                    # only as a compact quality artifact per K.
                    'max_windows': int(closed_loop_windows),
                    'action_trust': {
                        'enabled': True,
                        'local_context_support': {
                            'enabled': True,
                            'minimum_validation_contexts': 30,
                        },
                    },
                    'action_model_screen': {'enabled': True},
                    'fixed_optimal_action_holdout': {
                        'total_power_report_model': {
                            'candidate_families': list(families),
                        },
                    },
                    'mpc_counterfactual_total_power_report': {
                        'enabled': True,
                        'first_action_reference': 'recorded_current_action',
                        'evaluation_row_scope': 'all_chronological_test',
                        'row_stride_steps': 1,
                        'maximum_evaluation_rows': 0,
                    },
                },
                # The runner invokes the three-case sweep explicitly after the
                # current K action trace is written.
                'counterfactual_report_input_ablation': {'run_after_mpc': False},
            },
        },
    )


def _distance_summary_row(distance: int, mpc_dir: Path, summary) -> dict[str, Any]:
    evaluation = summary['mpc_counterfactual_total_power_report_evaluation']
    test = evaluation['test_evaluation']
    return {
        'distance_k': int(distance),
        'status': 'OK',
        'mpc_result_directory': str(mpc_dir),
        'first_action_reference': evaluation['protocol']['first_action_reference'],
        'evaluated_transition_count': test['evaluated_transition_count'],
        'mean_changed_control_variables_from_reference': test[
            'changed_control_variables_from_reference'
        ]['mean'],
        'max_changed_control_variables_from_reference': test[
            'changed_control_variables_from_reference'
        ]['maximum'],
        'mpc_action_vs_recorded_actual_total_power_mape_percent': (
            evaluation['mape_cases']['mpc_action_vs_recorded_actual']['value']
        ),
        'report_to_report_total_power_mape_percent': evaluation['mape_cases'][
            'mpc_action_vs_recorded_action_report'
        ]['value'],
        'solver_fallback_count': test['solver_fallback_count'],
    }


def _input_rows(distance: int, sweep_dir: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(
        sweep_dir / 'counterfactual_report_input_ablation_summary.csv',
        encoding='utf-8-sig',
    )
    rows = []
    for source in frame.to_dict('records'):
        rows.append({
            'distance_k': int(distance),
            'case': str(source['case']),
            'description': str(source['description']),
            'total_report_to_report_mape_percent': source[
                'report_to_report_total_power_mape_percent'
            ],
            'vent_report_to_report_mape_percent': source[
                '환기유효전력(W)_report_to_report_mape_percent'
            ],
            'hvac_report_to_report_mape_percent': source[
                '냉난방유효전력(W)_report_to_report_mape_percent'
            ],
            'total_mpc_action_report_vs_recorded_actual_mape_percent': source[
                'mpc_action_report_vs_actual_total_power_mape_percent'
            ],
            'vent_mpc_action_report_vs_recorded_actual_mape_percent': source[
                '환기유효전력(W)_mpc_action_report_vs_actual_mape_percent'
            ],
            'hvac_mpc_action_report_vs_recorded_actual_mape_percent': source[
                '냉난방유효전력(W)_mpc_action_report_vs_actual_mape_percent'
            ],
            'counterfactual_report_input_ablation_directory': str(sweep_dir),
        })
    return rows


def _write_markdown(path: Path, distance_rows, input_rows) -> None:
    lines = [
        '# Full guard action-distance sweep',
        '',
        '- K=0은 현재 기록 action을 그대로 재현하는 factual baseline이다.',
        '- K>0은 기록 action에 가까운 MPC action의 model-based diagnostic이다.',
        '- 입력 bundle의 report-to-report MAPE는 같은 MPC trace에서 action만 바꾼 효과다.',
        '',
        '## 거리별 MPC trace',
        '',
        '| K | 평가 행 | 평균 변경 변수 수 | 최대 변경 변수 수 | MPC↔기록 실제 총전력 MAPE | report↔report 총전력 MAPE | fallback |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in distance_rows:
        if row['status'] != 'OK':
            lines.append(f"| {row['distance_k']} | — | — | — | 실패 | 실패 | — |")
            continue
        lines.append(
            '| {distance_k} | {evaluated_transition_count} | '
            '{mean_changed_control_variables_from_reference:.4f} | '
            '{max_changed_control_variables_from_reference} | '
            '{mpc_action_vs_recorded_actual_total_power_mape_percent:.4f}% | '
            '{report_to_report_total_power_mape_percent:.4f}% | '
            '{solver_fallback_count} |'.format(**row)
        )
    lines.extend([
        '',
        '## 입력 bundle별 action-effect',
        '',
        '| K | Report model 입력 | 환기 MAPE | 냉난방 MAPE | 총전력 MAPE |',
        '|---:|---|---:|---:|---:|',
    ])
    for row in input_rows:
        lines.append(
            '| {distance_k} | {description} | '
            '{vent_report_to_report_mape_percent:.4f}% | '
            '{hvac_report_to_report_mape_percent:.4f}% | '
            '{total_report_to_report_mape_percent:.4f}% |'.format(**row)
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_action_distance_ablation(cfg):
    """Run every K and emit a compact 8 × 3 summary plus full artifacts."""
    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(cfg, result_dir)
    distances = _distances(cfg)
    LOGGER.info('Action-distance ablation started: log=%s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d | data=%s | versions=%s | '
        'environment=%s | distances=%s',
        platform.python_version(), platform.platform(), os.getpid(),
        dataset_label_from_cfg(cfg), list(versions_from_cfg(cfg)),
        cfg.DATASET.environment_name, distances,
    )
    _write_json(result_dir / 'experiment_manifest.json', {
        'framework': 'recorded_action_anchored_full_guard_distance_ablation',
        'distances': distances,
        'first_action_reference': 'recorded_current_action',
        'evaluation_row_scope': 'all_chronological_test',
        'report_model_input_cases': [
            _plain(case) for case in cfg.MPC.report_model_input_ablation.cases
        ],
        'claim_boundary': (
            'K=0 is factual replay. K>0 is a recorded-action-anchored, '
            'model-based diagnostic rather than measured post-intervention power.'
        ),
    })

    distance_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_cases = {
        str(case.name) for case in cfg.MPC.report_model_input_ablation.cases
    }
    for position, distance in enumerate(distances, start=1):
        scenario_dir = result_dir / f'distance_k{distance:02d}'
        scenario_dir.mkdir(parents=True, exist_ok=True)
        mpc_dir = scenario_dir / 'mpc'
        LOGGER.info(
            'Distance %d/%d started: K=%d | artifacts=%s',
            position, len(distances), distance, scenario_dir,
        )
        try:
            scenario_cfg = _scenario_cfg(cfg, distance)
            mpc_summary = run_mpc_pipeline(scenario_cfg, mpc_dir)
            distance_row = _distance_summary_row(distance, mpc_dir, mpc_summary)
            sweep_cfg = OmegaConf.merge(
                scenario_cfg,
                {'MPC': {'counterfactual_report_input_ablation': {
                    'reference_mpc_result_dir': str(mpc_dir),
                }}},
            )
            sweep_summary = run_counterfactual_report_input_ablation(sweep_cfg)
            sweep_dir = Path(sweep_summary['result_directory'])
            rows = _input_rows(distance, sweep_dir)
            actual_cases = {row['case'] for row in rows}
            if actual_cases != expected_cases:
                raise RuntimeError(
                    'Expected exactly the configured three report-input cases; '
                    f'expected={sorted(expected_cases)}, actual={sorted(actual_cases)}'
                )
            distance_row['counterfactual_report_input_ablation_directory'] = str(
                sweep_dir
            )
            distance_rows.append(distance_row)
            input_rows.extend(rows)
            _write_json(scenario_dir / 'scenario_manifest.json', {
                'distance_k': int(distance),
                'mpc_result_directory': str(mpc_dir),
                'counterfactual_report_input_ablation_directory': str(sweep_dir),
                'first_action_reference': 'recorded_current_action',
                'evaluation_row_scope': 'all_chronological_test',
            })
            LOGGER.info(
                'Distance K=%d complete: action-effect total MAPE=%s | '
                'mean changed variables=%s | report cases=%d',
                distance,
                distance_row['report_to_report_total_power_mape_percent'],
                distance_row['mean_changed_control_variables_from_reference'],
                len(rows),
            )
        except Exception as error:
            LOGGER.exception('Distance K=%d failed', distance)
            failure = f'K={distance}: {type(error).__name__}: {error}'
            failures.append(failure)
            distance_rows.append({
                'distance_k': int(distance),
                'status': 'FAIL',
                'mpc_result_directory': str(mpc_dir),
                'error': failure,
            })
        _write_csv(result_dir / 'distance_summary.csv', distance_rows)
        _write_csv(result_dir / 'distance_input_report_summary.csv', input_rows)
        _write_markdown(
            result_dir / 'action_distance_ablation_report.md',
            distance_rows,
            input_rows,
        )

    summary = {
        'framework': 'recorded_action_anchored_full_guard_distance_ablation',
        'result_directory': str(result_dir),
        'distances': distances,
        'distance_rows': distance_rows,
        'input_rows': input_rows,
        'failures': failures,
        'status': 'OK' if not failures else 'PARTIAL_FAILURE',
    }
    _write_json(result_dir / 'action_distance_ablation.json', summary)
    if failures:
        raise RuntimeError(
            'Action-distance ablation finished with failures; artifacts were '
            f'preserved at {result_dir}: ' + ' | '.join(failures)
        )
    LOGGER.info('Action-distance ablation complete: result_dir=%s', result_dir)
    return summary
