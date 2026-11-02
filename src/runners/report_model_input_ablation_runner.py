"""Chronological report-model input ablation without re-solving MPC.

This task measures report-model prediction fidelity on recorded chronological
test actions.  It intentionally does not run MILP: changing report-model
inputs must not change the controller policy, and repeatedly solving the same
controller would only duplicate expensive work.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import re
from typing import Dict

from omegaconf import OmegaConf
import pandas as pd

from controllers.action_crossfit import align_boundaries_to_control_episodes
from controllers.candidate_pool_evaluation import (
    evaluate_fixed_action_candidate_pool_top_k_sweep,
)
from controllers.fixed_action_evaluation import (
    build_fixed_action_episode_holdout,
)
from controllers.fixed_action_power_models import ReportModelInputSelection
from controllers.total_power_evaluation import (
    evaluate_all_chronological_test_total_power,
)
from data_provider.control_data import load_control_data
from data_provider.dataset_versions import dataset_label_from_cfg, versions_from_cfg
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC.report_input_ablation')


def _plain(config) -> Dict[str, object]:
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def _safe_directory_name(value: str) -> str:
    name = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._')
    if not name:
        raise ValueError('Report-model input ablation case name is empty')
    return name


def _result_directory(cfg) -> Path:
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run_name = f'{cfg.DEFAULT.exp_name}_{timestamp}_{os.getpid()}'
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'report_model_input_ablation'
        / run_name
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    log_path = result_dir / 'logs' / 'report_model_input_ablation.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _chronological_boundaries(cfg, data) -> tuple[int, int]:
    nominal_train_end, nominal_test_start = data.split_indices(cfg.DATASET.split_rate)
    if not bool(cfg.MPC.evaluation.action_options.episode_group_split.enabled):
        return int(nominal_train_end), int(nominal_test_start)
    try:
        train_end, test_start = align_boundaries_to_control_episodes(
            data,
            [nominal_train_end, nominal_test_start],
        )
    except ValueError as error:
        LOGGER.warning(
            'Episode boundary alignment failed; nominal boundaries used: %s',
            error,
        )
        train_end, test_start = nominal_train_end, nominal_test_start
    return int(train_end), int(test_start)


def _nonnegative_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a nonnegative number')
    result = float(value)
    if result < 0.0:
        raise ValueError(f'{name} must be nonnegative')
    return result


def _metric_targets(expansion_config, power_names, default_total_target):
    """Resolve independently configurable total and component MAPE gates."""
    raw = _plain(expansion_config.get('metric_targets') or {})
    total_target = _nonnegative_float(
        raw.get('total_power_mape_percent', default_total_target),
        'total_power_mape_percent',
    )
    default_component_target = _nonnegative_float(
        raw.get('default_component_mape_percent', total_target),
        'default_component_mape_percent',
    )
    by_power = _plain(raw.get('by_power_output') or {})
    unknown_outputs = sorted(set(by_power) - set(power_names))
    if unknown_outputs:
        raise ValueError(
            'Unknown component MAPE target output(s): '
            + ', '.join(str(value) for value in unknown_outputs)
        )
    component_targets = {
        name: _nonnegative_float(
            by_power.get(name, default_component_target),
            f'by_power_output.{name}',
        )
        for name in power_names
    }
    minimum_component_samples = raw.get('minimum_component_mape_samples', 1)
    if (
        isinstance(minimum_component_samples, bool)
        or not isinstance(minimum_component_samples, int)
        or minimum_component_samples <= 0
    ):
        raise ValueError('minimum_component_mape_samples must be a positive integer')
    return {
        'total_power_mape_percent': total_target,
        'component_mape_percent_by_output': component_targets,
        'minimum_component_mape_samples': int(minimum_component_samples),
    }


def _metric_verdict(value, target, valid_samples, minimum_samples, reportable):
    """Classify one MAPE without treating a missing value as a failed score."""
    available = (
        value is not None
        and valid_samples is not None
        and int(valid_samples) >= int(minimum_samples)
        and bool(reportable)
    )
    if not available:
        return {'reportable': False, 'target_achieved': False, 'verdict': '미보고'}
    achieved = float(value) <= float(target) + 1e-12
    return {
        'reportable': True,
        'target_achieved': bool(achieved),
        'verdict': '통과' if achieved else '실패',
    }


def _power_metric_assessment(
    *,
    total_value,
    total_valid_samples,
    total_reportable,
    component_metrics,
    metric_targets,
):
    """Create separate total/ventilation/HVAC MAPE assessments."""
    total = {
        'mape_percent': total_value,
        'target_percent': metric_targets['total_power_mape_percent'],
        'valid_mape_samples': total_valid_samples,
        **_metric_verdict(
            total_value,
            metric_targets['total_power_mape_percent'],
            total_valid_samples,
            1,
            total_reportable,
        ),
    }
    components = {}
    for name, target in metric_targets['component_mape_percent_by_output'].items():
        value = (component_metrics.get('mape_percent_by_output') or {}).get(name)
        valid_samples = (
            component_metrics.get('mape_valid_samples_by_output') or {}
        ).get(name)
        coverage = (
            component_metrics.get('mape_coverage_percent_by_output') or {}
        ).get(name)
        components[name] = {
            'mape_percent': value,
            'target_percent': target,
            'valid_mape_samples': valid_samples,
            'mape_coverage_percent': coverage,
            **_metric_verdict(
                value,
                target,
                valid_samples,
                metric_targets['minimum_component_mape_samples'],
                value is not None,
            ),
        }
    return {'total_power': total, 'components': components}


def _case_summary(case, evaluation, power_names, metric_targets) -> Dict[str, object]:
    test_evaluation = evaluation['test_evaluation']
    components = test_evaluation['component_power_prediction']
    inputs = evaluation['model_selection']['model']['input_features']
    assessment = _power_metric_assessment(
        total_value=evaluation['primary_metric']['value'],
        total_valid_samples=test_evaluation['valid_primary_mape_sample_count'],
        total_reportable=evaluation['reportable'],
        component_metrics=components,
        metric_targets=metric_targets,
    )
    evaluation['metric_assessment'] = assessment
    row = {
        'case': case['name'],
        'description': case.get('description'),
        'enabled_input_groups': ' + '.join(inputs['enabled_groups']),
        'input_feature_dimension': inputs['feature_dimension'],
        'selected_model_family': evaluation['model_selection']['selected_family'],
        'total_power_mape_percent': evaluation['primary_metric']['value'],
        'total_power_mape_diagnostic_percent': (
            evaluation['primary_metric']['diagnostic_value']
        ),
        'reportable': evaluation['reportable'],
        # ``evaluate_all_chronological_test_total_power`` keeps split sizes in
        # its top-level ``split`` section.  The test-evaluation section holds
        # only prediction metrics.  Reading the count here from ``split``
        # avoids coupling the ablation summary to an unrelated metric schema.
        'test_transition_count': evaluation['split']['test_transition_count'],
        'valid_total_power_mape_coverage_percent': (
            test_evaluation['valid_mape_coverage_percent']
        ),
        'total_power_target_percent': assessment['total_power']['target_percent'],
        'total_power_verdict': assessment['total_power']['verdict'],
    }
    for name in power_names:
        row[f'{name}_mape_percent'] = components['mape_percent_by_output'].get(name)
        row[f'{name}_wape_percent'] = components['wape_percent_by_output'].get(name)
        row[f'{name}_mae_w'] = components['mae_w_by_output'].get(name)
        row[f'{name}_valid_mape_samples'] = (
            components['mape_valid_samples_by_output'].get(name)
        )
        row[f'{name}_mape_coverage_percent'] = (
            components['mape_coverage_percent_by_output'].get(name)
        )
        row[f'{name}_target_percent'] = assessment['components'][name][
            'target_percent'
        ]
        row[f'{name}_verdict'] = assessment['components'][name]['verdict']
    return row


def _top_k_prefixes(expansion_config):
    """Validate image-style labels and their cumulative candidate counts."""
    prefixes = expansion_config.get('top_k_prefixes') or []
    if OmegaConf.is_list(prefixes):
        prefixes = OmegaConf.to_container(prefixes, resolve=True)
    if not isinstance(prefixes, list) or not prefixes:
        raise TypeError(
            'MPC.report_model_input_ablation.candidate_pool_expansion.'
            'top_k_prefixes must be a nonempty list'
        )
    normalized = []
    labels = []
    counts = []
    for raw in prefixes:
        prefix = _plain(raw)
        label = str(prefix.get('label', '')).strip()
        count = prefix.get('k')
        if (
            not label
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(
                'Each candidate-pool expansion prefix needs nonempty label '
                'and positive integer k'
            )
        labels.append(label)
        counts.append(count)
        normalized.append({'label': label, 'k': count})
    if len(labels) != len(set(labels)) or len(counts) != len(set(counts)):
        raise ValueError('Candidate-pool expansion labels and k values must be unique')
    if counts != sorted(counts):
        raise ValueError('Candidate-pool expansion k values must be ascending')
    return normalized


def _top_k_expansion_summary_row(
    case,
    prefix,
    result,
    curve_row,
    power_names,
    metric_targets,
):
    """Flatten a single cumulative Top-K point for the slide-style table."""
    requested_k = int(prefix['k'])
    metadata = result.get('total_power_model_selection') or {}
    if curve_row is None:
        row = {
            'case': case['name'],
            'case_description': case.get('description'),
            'scenario': prefix['label'],
            'requested_actions': requested_k,
            'actions': None,
            'available_in_sweep': False,
            'available_actions': result.get('sweep_candidate_count'),
            'coverage_percent': None,
            'test_rows': None,
            'test_episodes': None,
            'total_power_pooled_mape_percent': None,
            'total_power_worst_action_mape_percent': None,
            'total_power_action_pass_rate_percent': None,
            'total_power_target_percent': metric_targets[
                'total_power_mape_percent'
            ],
            'total_power_reportable': False,
            'total_power_target_achieved': False,
            'total_power_verdict': '미가용',
            'test_interpretation': 'unavailable_prefix',
            'selected_model_family': metadata.get('selected_family'),
            'selected_candidate_ids': '',
        }
        for name in power_names:
            row[f'{name}_pooled_mape_percent'] = None
            row[f'{name}_pooled_wape_percent'] = None
            row[f'{name}_valid_mape_samples'] = None
            row[f'{name}_mape_coverage_percent'] = None
            row[f'{name}_target_percent'] = metric_targets[
                'component_mape_percent_by_output'
            ][name]
            row[f'{name}_verdict'] = '미가용'
        return row

    component_metrics = curve_row['test_component_power_prediction']
    assessment = _power_metric_assessment(
        total_value=curve_row['test_pooled_mape_percent'],
        total_valid_samples=curve_row['test_valid_mape_sample_count'],
        total_reportable=bool(curve_row['test_reportable']),
        component_metrics=component_metrics,
        metric_targets=metric_targets,
    )
    curve_row['metric_assessment'] = assessment
    row = {
        'case': case['name'],
        'case_description': case.get('description'),
        'scenario': prefix['label'],
        'requested_actions': requested_k,
        'actions': int(curve_row['k']),
        'available_in_sweep': True,
        'available_actions': result['sweep_candidate_count'],
        'coverage_percent': curve_row['test_exact_match_coverage_percent'],
        'test_rows': curve_row['test_exact_match_count'],
        'test_episodes': curve_row['test_exact_match_episode_count'],
        'total_power_pooled_mape_percent': curve_row['test_pooled_mape_percent'],
        'total_power_worst_action_mape_percent': curve_row[
            'test_worst_action_mape_percent'
        ],
        'total_power_action_pass_rate_percent': curve_row[
            'test_action_target_pass_rate_percent'
        ],
        'total_power_target_percent': assessment['total_power']['target_percent'],
        'total_power_reportable': assessment['total_power']['reportable'],
        'total_power_target_achieved': assessment['total_power'][
            'target_achieved'
        ],
        'total_power_verdict': assessment['total_power']['verdict'],
        'test_interpretation': curve_row['test_interpretation'],
        'selected_model_family': metadata.get('selected_family'),
        'selected_candidate_ids': ','.join(curve_row['selected_candidate_ids']),
    }
    for name in power_names:
        row[f'{name}_pooled_mape_percent'] = (
            component_metrics.get('mape_percent_by_output') or {}
        ).get(name)
        row[f'{name}_pooled_wape_percent'] = (
            component_metrics.get('wape_percent_by_output') or {}
        ).get(name)
        row[f'{name}_valid_mape_samples'] = (
            component_metrics.get('mape_valid_samples_by_output') or {}
        ).get(name)
        row[f'{name}_mape_coverage_percent'] = (
            component_metrics.get('mape_coverage_percent_by_output') or {}
        ).get(name)
        row[f'{name}_target_percent'] = assessment['components'][name][
            'target_percent'
        ]
        row[f'{name}_verdict'] = assessment['components'][name]['verdict']
    return row


def _run_candidate_pool_expansion_for_case(
    cfg,
    data,
    *,
    case,
    report_model_config,
    holdout,
    expansion_config,
    artifact_root: Path,
    random_state: int,
    metric_targets,
):
    """Evaluate one frozen report model over cumulative Top-K action prefixes."""
    prefixes = _top_k_prefixes(expansion_config)
    pool_config = OmegaConf.merge(
        cfg.MPC.evaluation.fixed_action_candidate_pool,
        expansion_config.get('candidate_pool_overrides') or {},
        {
            'minimum_k': 1,
            'maximum_k': max(prefix['k'] for prefix in prefixes),
            'step': 1,
            'k_selection_rule': 'largest_k_within_validation_target',
            'fallback_rule': 'minimum_validation_mape',
            'minimum_pooled_validation_mape_samples': int(
                expansion_config['minimum_pooled_validation_mape_samples']
            ),
        },
    )
    case_root = artifact_root / _safe_directory_name(case['name'])
    LOGGER.info(
        'Expanded Top-K pool started: case=%s | available_actions=%d | '
        'requested_k=%s',
        case['name'],
        int(holdout.metadata['candidate_count']),
        [prefix['k'] for prefix in prefixes],
    )
    result, audit_rows = evaluate_fixed_action_candidate_pool_top_k_sweep(
        data,
        holdout,
        config=pool_config,
        total_power_report_config=report_model_config,
        mape_min_actual_power_w=float(cfg.MPC.affine_models.mape_min_actual_power_w),
        random_state=int(random_state),
        report_model_refit_scope=str(
            OmegaConf.select(
                cfg,
                'MPC.evaluation.report_model_refit_scope',
                default='train_validation',
            )
        ),
    )
    result['input_ablation_expansion'] = {
        'method': 'validation_ranked_cumulative_top_k',
        'purpose': (
            'Report image-style K prefixes using one report-model family '
            'selected on validation and frozen before test.'
        ),
        'top_k_prefixes': prefixes,
        'holdout_overrides': _plain(
            expansion_config.get('holdout_overrides') or {}
        ),
        'candidate_pool_overrides': _plain(
            expansion_config.get('candidate_pool_overrides') or {}
        ),
        'effective_holdout_protocol': holdout.metadata['protocol'],
        'report_model_input_ablation_case': dict(case),
        'test_targets_used_for_model_or_k_selection': False,
    }
    case_root.mkdir(parents=True, exist_ok=True)
    result['artifact_directory'] = str(case_root)
    curve_by_k = {int(row['k']): row for row in result.get('curve') or []}
    summary_rows = []
    for prefix in prefixes:
        row = _top_k_expansion_summary_row(
            case,
            prefix,
            result,
            curve_by_k.get(int(prefix['k'])),
            data.power_names,
            metric_targets,
        )
        row['artifact_directory'] = str(case_root)
        summary_rows.append(row)
    _write_json(case_root / 'evaluation.json', result)
    _write_json(
        case_root / 'expansion_config.json',
        result['input_ablation_expansion'],
    )
    _write_csv(case_root / 'top_k_curve.csv', result.get('curve') or [])
    _write_csv(
        case_root / 'candidate_order.csv',
        pd.json_normalize(result.get('candidate_order') or []).to_dict('records'),
    )
    _write_csv(case_root / 'all_candidate_test_predictions.csv', audit_rows)
    LOGGER.info(
        'Expanded Top-K pool complete: case=%s | available=%s | '
        'model=%s | official_k=%s',
        case['name'],
        result.get('sweep_candidate_count'),
        (result.get('total_power_model_selection') or {}).get('selected_family'),
        (result.get('official_selection') or {}).get('selected_k'),
    )
    return result, summary_rows


def _format_percent(value, decimals=4):
    if value is None:
        return '—'
    return f'{float(value):.{decimals}f}%'


def _write_expansion_markdown(path: Path, rows, power_names) -> None:
    """Create slide-ready per-input tables alongside machine-readable CSV."""
    lines = [
        '# Report model 입력별 expanded candidate-pool 결과',
        '',
        '- 모든 표는 같은 outcome-blind episode-group holdout 규칙을 사용한다.',
        '- 환기유효전력 MAPE, 냉난방유효전력 MAPE, 총전력 pooled MAPE를 각각의 '
        '목표값으로 별도 판정한다. worst action과 action pass는 총전력 기준이다.',
        '- 후보 순위·report-model family·공식 K는 validation만으로 선택하며, test '
        '전력은 선택 이후에만 평가한다. `official_selected_k` 외 prefix는 '
        'sensitivity-only다.',
        '',
    ]
    case_names = []
    for row in rows:
        if row['case'] not in case_names:
            case_names.append(row['case'])
    for case_name in case_names:
        case_rows = [row for row in rows if row['case'] == case_name]
        description = case_rows[0].get('case_description') or case_name
        lines.extend([
            f'## {case_name} — {description}',
            '',
            '| 시나리오 | Actions | Coverage | '
            f'{power_names[0]} MAPE | 환기 판정 | {power_names[1]} MAPE | '
            '냉난방 판정 | 총전력 Pooled MAPE | 총전력 판정 | Worst action | '
            'Action pass | 구분 |',
            '|---|---:|---:|---:|---|---:|---|---:|---|---:|---:|---|',
        ])
        for row in case_rows:
            lines.append(
                '| {scenario} | {actions} | {coverage} | {first} | '
                '{first_verdict} | {second} | {second_verdict} | {total} | '
                '{total_verdict} | {worst} | {pass_rate} | {interpretation} |'.format(
                    scenario=row['scenario'],
                    actions=(
                        row['actions'] if row['actions'] is not None else '—'
                    ),
                    coverage=_format_percent(row['coverage_percent'], 2),
                    first=_format_percent(
                        row[f'{power_names[0]}_pooled_mape_percent']
                    ),
                    first_verdict=row[f'{power_names[0]}_verdict'],
                    second=_format_percent(
                        row[f'{power_names[1]}_pooled_mape_percent']
                    ),
                    second_verdict=row[f'{power_names[1]}_verdict'],
                    total=_format_percent(row['total_power_pooled_mape_percent']),
                    total_verdict=row['total_power_verdict'],
                    worst=_format_percent(
                        row['total_power_worst_action_mape_percent']
                    ),
                    pass_rate=_format_percent(
                        row['total_power_action_pass_rate_percent'],
                        2,
                    ),
                    interpretation=row['test_interpretation'],
                )
            )
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def refresh_report_model_input_ablation_assessment(cfg, result_directory):
    """Backfill independent MAPE verdicts into a completed ablation run.

    This reads only stored predictions/metrics.  It never fits a model or
    solves MILP, so a reporting-rule update does not alter an experiment's
    learned model, selected actions, or measured MAPE values.
    """
    result_dir = Path(result_directory).resolve()
    result_path = result_dir / 'report_model_input_ablation.json'
    if not result_path.is_file():
        raise FileNotFoundError(f'No ablation result file: {result_path}')
    with result_path.open(encoding='utf-8') as stream:
        result = json.load(stream)
    power_names = tuple(result['data']['power_names'])
    ablation_config = _plain(cfg.MPC.report_model_input_ablation)
    expansion_config = _plain(
        ablation_config.get('candidate_pool_expansion') or {}
    )
    metric_targets = _metric_targets(
        expansion_config,
        power_names,
        default_total_target=float(
            cfg.MPC.evaluation.fixed_action_candidate_pool.target_mape_percent
        ),
    )

    chronological_rows = []
    for entry in result.get('cases') or []:
        case = entry['case']
        evaluation = entry['evaluation']
        row = _case_summary(case, evaluation, power_names, metric_targets)
        row['artifact_directory'] = entry.get('artifact_directory')
        chronological_rows.append(row)
        artifact_directory = entry.get('artifact_directory')
        if artifact_directory:
            _write_json(
                Path(artifact_directory) / 'evaluation.json',
                evaluation,
            )

    expansion = result.get('expanded_candidate_pool') or {}
    prefixes = expansion.get('top_k_prefixes') or _top_k_prefixes(
        expansion_config
    )
    expansion_rows = []
    for entry in expansion.get('cases') or []:
        case = entry['case']
        top_k_sweep = entry.get('top_k_sweep') or {}
        curve_by_k = {
            int(row['k']): row for row in top_k_sweep.get('curve') or []
        }
        for prefix in prefixes:
            row = _top_k_expansion_summary_row(
                case,
                prefix,
                top_k_sweep,
                curve_by_k.get(int(prefix['k'])),
                power_names,
                metric_targets,
            )
            row['artifact_directory'] = top_k_sweep.get('artifact_directory')
            expansion_rows.append(row)
        artifact_directory = top_k_sweep.get('artifact_directory')
        if artifact_directory:
            artifact_dir = Path(artifact_directory)
            _write_json(artifact_dir / 'evaluation.json', top_k_sweep)
            _write_csv(
                artifact_dir / 'top_k_curve.csv',
                top_k_sweep.get('curve') or [],
            )

    result['summary_rows'] = chronological_rows
    result.setdefault('expanded_candidate_pool', {})['summary_rows'] = (
        expansion_rows
    )
    result['expanded_candidate_pool']['metric_targets'] = metric_targets
    _write_json(result_path, result)
    _write_csv(
        result_dir / 'report_model_input_ablation_summary.csv',
        chronological_rows,
    )
    _write_csv(
        result_dir / 'candidate_pool_expansion_summary.csv',
        expansion_rows,
    )
    _write_expansion_markdown(
        result_dir / 'candidate_pool_expansion_report.md',
        expansion_rows,
        power_names,
    )
    return {
        'result_directory': str(result_dir),
        'metric_targets': metric_targets,
        'chronological_case_count': len(chronological_rows),
        'expanded_prefix_row_count': len(expansion_rows),
    }


def run_report_model_input_ablation(cfg):
    """Run the configured observed-action report-model input ablation."""
    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(cfg, result_dir)
    LOGGER.info('Report-model input ablation started: log=%s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d | data=%s | versions=%s',
        platform.python_version(),
        platform.platform(),
        os.getpid(),
        dataset_label_from_cfg(cfg),
        list(versions_from_cfg(cfg)),
    )
    LOGGER.info(
        'Compute: report model uses scikit-learn on CPU; MILP is not run in '
        'this input-only ablation.'
    )

    data = load_control_data(cfg)
    train_end, test_start = _chronological_boundaries(cfg, data)
    train_rows = range(0, train_end - 1)
    validation_rows = range(train_end, test_start - 1)
    test_rows = range(test_start, len(data.states) - 1)
    if min(len(train_rows), len(validation_rows), len(test_rows)) <= 0:
        raise ValueError('Chronological report-model ablation split is empty')

    ablation_config = _plain(cfg.MPC.report_model_input_ablation)
    raw_cases = ablation_config.get('cases') or []
    if not raw_cases:
        raise ValueError('MPC.report_model_input_ablation.cases must be nonempty')
    cases = [_plain(case) for case in raw_cases]
    case_names = [str(case.get('name', '')) for case in cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError('Report-model input ablation case names must be unique')

    base_report_config = _plain(
        cfg.MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model
    )
    expansion_config = _plain(
        ablation_config.get('candidate_pool_expansion') or {}
    )
    metric_targets = _metric_targets(
        expansion_config,
        data.power_names,
        default_total_target=float(
            cfg.MPC.evaluation.fixed_action_candidate_pool.target_mape_percent
        ),
    )
    expansion_enabled = bool(expansion_config.get('enabled', False))
    expansion_root = result_dir / 'expanded_candidate_pool'
    expansion_holdout = None
    expansion_prefixes = []
    if expansion_enabled:
        expansion_prefixes = _top_k_prefixes(expansion_config)
        holdout_config = OmegaConf.merge(
            cfg.MPC.evaluation.fixed_optimal_action_holdout,
            expansion_config.get('holdout_overrides') or {},
        )
        expansion_holdout = build_fixed_action_episode_holdout(
            data,
            holdout_config,
            split_rate=cfg.DATASET.split_rate,
        )
        LOGGER.info(
            'Expanded Top-K reporting enabled: cases=%d | available_actions=%d '
            '| requested_k=%s',
            len(cases),
            int(expansion_holdout.metadata['candidate_count']),
            [prefix['k'] for prefix in expansion_prefixes],
        )
    results = []
    summary_rows = []
    expansion_results = []
    expansion_summary_rows = []
    case_root = result_dir / 'cases'
    for index, case in enumerate(cases):
        name = str(case.get('name', ''))
        description = str(case.get('description', ''))
        input_features = case.get('input_features')
        if not isinstance(input_features, dict):
            raise TypeError(
                f'Report-model input ablation case {name!r} needs '
                'an input_features mapping'
            )
        input_selection = ReportModelInputSelection.from_config(input_features)
        case_config = {
            **base_report_config,
            'input_features': _plain(input_features),
        }
        LOGGER.info(
            'Case %d/%d started: %s | inputs=%s',
            index + 1,
            len(cases),
            name,
            input_selection.to_metadata(
                state_names=data.state_names,
                control_feature_names=data.encoder.feature_names,
                disturbance_names=data.disturbance_names,
            )['enabled_groups'],
        )
        evaluation, audit_rows = evaluate_all_chronological_test_total_power(
            data,
            training_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            report_model_config=case_config,
            minimum_actual_power_w=float(
                cfg.MPC.affine_models.mape_min_actual_power_w
            ),
            target_mape_percent=float(case_config['target_mape_percent']),
            minimum_validation_samples=int(
                case_config['minimum_validation_mape_samples']
            ),
            minimum_test_samples=int(
                cfg.MPC.all_test_data_evaluation.minimum_test_mape_samples
            ),
            random_state=int(cfg.DEFAULT.seed) + index * 100,
            report_model_refit_scope=str(
                OmegaConf.select(
                    cfg,
                    'MPC.evaluation.report_model_refit_scope',
                    default='train_validation',
                )
            ),
        )
        evaluation['ablation_case'] = {
            'name': name,
            'description': description,
            'input_features': _plain(input_features),
            'purpose': (
                'Observed-action chronological component-power prediction '
                'ablation. No MILP action is selected or re-solved.'
            ),
        }
        case_dir = case_root / _safe_directory_name(name)
        summary = _case_summary(
            evaluation['ablation_case'],
            evaluation,
            data.power_names,
            metric_targets,
        )
        _write_json(case_dir / 'evaluation.json', evaluation)
        _write_json(case_dir / 'input_config.json', evaluation['ablation_case'])
        _write_csv(case_dir / 'test_predictions.csv', audit_rows)
        summary['artifact_directory'] = str(case_dir)
        summary_rows.append(summary)
        results.append({
            'case': evaluation['ablation_case'],
            'artifact_directory': str(case_dir),
            'evaluation': evaluation,
        })
        LOGGER.info(
            'Case %s complete: total_mape=%s | component_mape=%s',
            name,
            summary['total_power_mape_diagnostic_percent'],
            {
                power_name: summary[f'{power_name}_mape_percent']
                for power_name in data.power_names
            },
        )
        if expansion_enabled:
            expanded_result, rows = _run_candidate_pool_expansion_for_case(
                cfg,
                data,
                case=evaluation['ablation_case'],
                report_model_config=case_config,
                holdout=expansion_holdout,
                expansion_config=expansion_config,
                artifact_root=expansion_root,
                random_state=int(cfg.DEFAULT.seed) + index * 10_000,
                metric_targets=metric_targets,
            )
            expansion_results.append({
                'case': evaluation['ablation_case'],
                'top_k_sweep': expanded_result,
            })
            expansion_summary_rows.extend(rows)

    result = {
        'framework': 'report_model_input_ablation',
        'purpose': (
            'Compare report-model input groups on observed actions. This does '
            'not evaluate changed MPC actions or re-run MILP.'
        ),
        'data': {
            'dataset_label': dataset_label_from_cfg(cfg),
            'source_versions': list(data.source_versions),
            'environment_name': str(cfg.DATASET.environment_name),
            'power_names': list(data.power_names),
        },
        'split': {
            'train_end_exclusive': train_end,
            'validation_start': train_end,
            'test_start': test_start,
            'train_transition_count': len(train_rows),
            'validation_transition_count': len(validation_rows),
            'test_transition_count': len(test_rows),
            'episode_boundary_alignment_enabled': bool(
                cfg.MPC.evaluation.action_options.episode_group_split.enabled
            ),
        },
        'cases': results,
        'summary_rows': summary_rows,
        'expanded_candidate_pool': {
            'enabled': expansion_enabled,
            'purpose': (
                'For each report-model input case, report declared cumulative '
                'Top-K action prefixes under one outcome-blind episode-group '
                'holdout. Total-power KPI is primary; component MAPE is '
                'secondary.'
            ),
            'top_k_prefixes': expansion_prefixes if expansion_enabled else [],
            'metric_targets': metric_targets,
            'cases': expansion_results,
            'summary_rows': expansion_summary_rows,
        },
    }
    _write_json(result_dir / 'report_model_input_ablation.json', result)
    _write_csv(result_dir / 'report_model_input_ablation_summary.csv', summary_rows)
    if expansion_enabled:
        _write_csv(
            result_dir / 'candidate_pool_expansion_summary.csv',
            expansion_summary_rows,
        )
        _write_expansion_markdown(
            result_dir / 'candidate_pool_expansion_report.md',
            expansion_summary_rows,
            data.power_names,
        )
    LOGGER.info('Report-model input ablation complete: result_dir=%s', result_dir)
    return result
