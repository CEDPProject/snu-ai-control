"""Counterfactual report-model input ablation with a frozen MPC policy.

The three report-model input bundles are compared on *identical* saved MPC
first actions.  Re-solving MILP for each bundle would be wasteful and could
introduce solver noise even though the report model is evaluation-only.  This
runner instead reuses the audited action CSV from one completed MPC result,
fits each report model using only that run's chronological train/validation
rows, and evaluates both report-to-report and actual-baseline metrics.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from controllers.evaluation import power_prediction_metrics
from controllers.fixed_action_power_models import ReportModelInputSelection
from controllers.total_power_evaluation import (
    fit_frozen_chronological_total_power_model,
)
from data_provider.control_data import load_control_data
from data_provider.dataset_versions import dataset_label_from_cfg, versions_from_cfg
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC.counterfactual_report_input_ablation')


def _plain(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def _safe_name(value: str) -> str:
    name = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._')
    if not name:
        raise ValueError('Ablation case name must contain a safe file name token')
    return name


def _result_directory(cfg) -> Path:
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run_name = f'{cfg.DEFAULT.exp_name}_{stamp}_{os.getpid()}'
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'counterfactual_report_input_ablation'
        / run_name
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    log_path = result_dir / 'logs' / 'counterfactual_report_input_ablation.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _reference_paths(value: object) -> tuple[Path, Path, Path]:
    if value is None or not str(value).strip():
        raise ValueError(
            'MPC.counterfactual_report_input_ablation.'
            'reference_mpc_result_dir must name a completed MPC result directory'
        )
    result_dir = Path(str(value)).expanduser().resolve()
    evaluation_path = result_dir / 'mpc_evaluation.json'
    prediction_path = (
        result_dir / 'mpc_counterfactual_total_power_report_predictions.csv'
    )
    for path in (evaluation_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(
                'Counterfactual report-input ablation requires reference artifact: '
                f'{path}'
            )
    return result_dir, evaluation_path, prediction_path


def _load_reference(value: object) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    result_dir, evaluation_path, prediction_path = _reference_paths(value)
    with evaluation_path.open(encoding='utf-8') as stream:
        evaluation = json.load(stream)
    frame = pd.read_csv(prediction_path, encoding='utf-8-sig')
    if frame.empty:
        raise ValueError('Reference MPC counterfactual prediction CSV is empty')
    required = {'source_row_index', 'timestamp'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            'Reference counterfactual prediction CSV is missing columns: '
            + ', '.join(missing)
        )
    return result_dir, evaluation, frame


def _reference_split(evaluation: Mapping[str, Any], data) -> tuple[np.ndarray, np.ndarray]:
    split = (evaluation.get('model_validation') or {}).get('data_split') or {}
    train_end = split.get('train_end_exclusive')
    test_start = split.get('test_start')
    if train_end is None or test_start is None:
        raise ValueError(
            'Reference mpc_evaluation.json does not retain chronological split '
            'boundaries required for leakage-safe report-model refitting'
        )
    train_end = int(train_end)
    test_start = int(test_start)
    if not 1 < train_end < test_start < len(data.states):
        raise ValueError(
            'Reference chronological split boundaries are incompatible with the '
            'currently loaded data'
        )
    return (
        np.arange(0, train_end - 1, dtype=int),
        np.arange(train_end, test_start - 1, dtype=int),
    )


def _reference_actions(data, frame: pd.DataFrame, prefix: str) -> np.ndarray:
    values: dict[str, pd.Series] = {}
    for variable in data.encoder.variables:
        column = f'{prefix}_control_{variable.name}'
        if column not in frame.columns:
            raise ValueError(
                f'Reference prediction CSV lacks required control column {column!r}'
            )
        values[variable.name] = frame[column]
    return data.encoder.transform(pd.DataFrame(values, index=frame.index))


def _reference_rows(data, frame: pd.DataFrame) -> np.ndarray:
    numeric = pd.to_numeric(frame['source_row_index'], errors='raise').to_numpy()
    if not np.allclose(numeric, np.rint(numeric)):
        raise ValueError('Reference source_row_index values must be integral')
    rows = np.rint(numeric).astype(int)
    if len(np.unique(rows)) != len(rows):
        raise ValueError(
            'Reference counterfactual rows must be unique; duplicate rows would '
            'change their intended policy-window weighting'
        )
    valid = data.valid_transition_rows(rows)
    if len(valid) != len(rows) or not np.array_equal(valid, np.sort(rows)):
        raise ValueError(
            'Reference rows are not valid one-minute transitions in the loaded data'
        )
    return rows


def _metric_pair(actual, predicted, names, threshold: float) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    component = power_prediction_metrics(actual, predicted, names, threshold)
    total = power_prediction_metrics(
        actual.sum(axis=1, keepdims=True),
        predicted.sum(axis=1, keepdims=True),
        ('total_hvac_power_w',),
        threshold,
    )
    return {'components': component, 'total_power': total}


def _hash_reference_actions(rows: np.ndarray, controls: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    digest.update(np.rint(np.asarray(controls, dtype=float)).astype(np.int8).tobytes())
    return digest.hexdigest()


def _report_input_feature_metadata(model) -> dict[str, Any]:
    """Read report-input provenance without assuming an unwrapped model class.

    Post-calibrated models deliberately wrap the selected family while keeping
    prediction semantics identical.  Reporting must therefore rely on the
    stable metadata contract first and only use the legacy attribute when it
    is explicitly available.
    """
    metadata = getattr(model, 'input_feature_metadata', None)
    if isinstance(metadata, dict):
        return metadata
    to_metadata = getattr(model, 'to_metadata', None)
    model_metadata = to_metadata() if callable(to_metadata) else {}
    metadata = model_metadata.get('input_features')
    if not isinstance(metadata, dict):
        raise TypeError(
            'Counterfactual report model must expose input_feature_metadata '
            'or include input_features in to_metadata()'
        )
    return metadata


def _summary_row(case, selection, metrics, power_names) -> dict[str, Any]:
    report_to_report = metrics['report_to_report']
    observed = metrics['recorded_action_report_vs_recorded_actual']
    counterfactual = metrics['mpc_action_report_vs_recorded_actual']
    input_metadata = _report_input_feature_metadata(selection.model)
    row = {
        'case': case['name'],
        'description': case.get('description'),
        'enabled_input_groups': ' + '.join(
            input_metadata['enabled_groups']
        ),
        'input_feature_dimension': input_metadata['feature_dimension'],
        'selected_model_family': selection.selected_family,
        'report_to_report_total_power_mape_percent': report_to_report[
            'total_power'
        ]['mape_percent'],
        'recorded_action_report_vs_actual_total_power_mape_percent': observed[
            'total_power'
        ]['mape_percent'],
        'mpc_action_report_vs_actual_total_power_mape_percent': counterfactual[
            'total_power'
        ]['mape_percent'],
    }
    for name in power_names:
        row[f'{name}_report_to_report_mape_percent'] = report_to_report[
            'components'
        ]['mape_percent_by_output'][name]
        row[f'{name}_recorded_action_report_vs_actual_mape_percent'] = observed[
            'components'
        ]['mape_percent_by_output'][name]
        row[f'{name}_mpc_action_report_vs_actual_mape_percent'] = counterfactual[
            'components'
        ]['mape_percent_by_output'][name]
    return row


def run_counterfactual_report_input_ablation(cfg):
    """Refit report input variants against one immutable MPC-action trace."""
    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(cfg, result_dir)

    sweep_cfg = _plain(cfg.MPC.counterfactual_report_input_ablation)
    reference_dir, reference_evaluation, reference_frame = _load_reference(
        sweep_cfg.get('reference_mpc_result_dir')
    )
    data = load_control_data(cfg)
    training_rows, validation_rows = _reference_split(reference_evaluation, data)
    rows = _reference_rows(data, reference_frame)
    recorded_controls = _reference_actions(data, reference_frame, 'recorded')
    mpc_controls = _reference_actions(data, reference_frame, 'mpc')
    states = np.asarray(data.states[rows], dtype=float)
    disturbances = np.asarray(data.disturbances[rows], dtype=float)
    actual_components = np.asarray(data.powers[rows], dtype=float)

    report_config = _plain(
        cfg.MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model
    )
    counterfactual_cfg = _plain(
        cfg.MPC.evaluation.mpc_counterfactual_total_power_report
    )
    threshold = float(cfg.MPC.affine_models.mape_min_actual_power_w)
    cases = _plain(cfg.MPC.report_model_input_ablation).get('cases') or []
    if not isinstance(cases, list) or not cases:
        raise ValueError('MPC.report_model_input_ablation.cases must be nonempty')

    LOGGER.info('Counterfactual report-input ablation started')
    LOGGER.info('Log file: %s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d',
        platform.python_version(), platform.platform(), os.getpid(),
    )
    LOGGER.info(
        'Reference MPC run: %s | rows=%d | data=%s | versions=%s | environment=%s',
        reference_dir, len(rows), dataset_label_from_cfg(cfg),
        list(versions_from_cfg(cfg)), cfg.DATASET.environment_name,
    )

    expected_action_hash = _hash_reference_actions(rows, mpc_controls)
    case_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for position, raw_case in enumerate(cases):
        case = _plain(raw_case)
        name = str(case.get('name', '')).strip()
        if not name:
            raise ValueError('Every report-model input ablation case needs a name')
        input_features = case.get('input_features')
        if not isinstance(input_features, dict):
            raise TypeError(f'Case {name!r} must define input_features as a mapping')
        # Validate early so a malformed case cannot silently use a different bundle.
        ReportModelInputSelection.from_config(input_features)
        case_config = dict(report_config)
        case_config['input_features'] = dict(input_features)
        LOGGER.info(
            'Case %d/%d: %s | input_groups=%s', position + 1, len(cases),
            name,
            '+'.join(ReportModelInputSelection.from_config(input_features).to_metadata(
                state_names=data.state_names,
                control_feature_names=data.encoder.feature_names,
                disturbance_names=data.disturbance_names,
            )['enabled_groups']),
        )
        selection = fit_frozen_chronological_total_power_model(
            data,
            training_rows=training_rows,
            validation_rows=validation_rows,
            report_model_config=case_config,
            minimum_actual_power_w=threshold,
            target_mape_percent=float(counterfactual_cfg['target_mape_percent']),
            minimum_validation_samples=int(
                counterfactual_cfg['minimum_validation_mape_samples']
            ),
            # Match the observed-action input-ablation seed schedule.  When a
            # distance-0 trace is identical to recorded actions and uses the
            # same chronological cohort, this makes the factual baseline
            # directly reproducible rather than merely approximately similar.
            random_state=int(cfg.DEFAULT.seed) + position * 100,
        )
        recorded_prediction = selection.model.predict_components(
            states, recorded_controls, disturbances
        )
        mpc_prediction = selection.model.predict_components(
            states, mpc_controls, disturbances
        )
        metrics = {
            # The reported action is the MAPE denominator here.  This is an
            # action-effect diagnostic, never a sensor-grounded forecast error.
            'report_to_report': _metric_pair(
                recorded_prediction, mpc_prediction, data.power_names, threshold
            ),
            'recorded_action_report_vs_recorded_actual': _metric_pair(
                actual_components, recorded_prediction, data.power_names, threshold
            ),
            'mpc_action_report_vs_recorded_actual': _metric_pair(
                actual_components, mpc_prediction, data.power_names, threshold
            ),
        }
        reduction_w = recorded_prediction.sum(axis=1) - mpc_prediction.sum(axis=1)
        recorded_total = recorded_prediction.sum(axis=1)
        valid_reduction = np.abs(recorded_total) > threshold
        action_hash = _hash_reference_actions(rows, mpc_controls)
        if action_hash != expected_action_hash:
            raise RuntimeError('Reference MPC action trace changed during the sweep')
        case_result = {
            'ablation_case': {
                'name': name,
                'description': case.get('description'),
                'input_features': dict(input_features),
            },
            'reference_mpc_run': {
                'result_directory': str(reference_dir),
                'counterfactual_transition_count': int(len(rows)),
                'source_row_indices_are_unique': True,
                'mpc_action_trace_sha256': expected_action_hash,
                'meaning': (
                    'The saved MPC first action is reused exactly. The MILP is not '
                    're-solved; only the report model input bundle changes.'
                ),
            },
            'protocol': {
                'report_model_training': (
                    'chronological train-only family selection on validation, then '
                    'train+validation refit before the fixed reference test rows'
                ),
                'report_to_report_definition': (
                    'mean(abs(report(x, mpc_action, d) - report(x, recorded_action, d)) '
                    '/ abs(report(x, recorded_action, d))) * 100'
                ),
                'minimum_power_for_mape_w': threshold,
                'no_milp_resolve': True,
            },
            'model_selection': selection.to_metadata(),
            'metrics': metrics,
            'estimated_report_to_report_energy_reduction': {
                'mean_reduction_w': float(np.mean(reduction_w)),
                'mean_reduction_percent': (
                    None
                    if not np.any(valid_reduction)
                    else float(np.mean(
                        reduction_w[valid_reduction]
                        / recorded_total[valid_reduction]
                    ) * 100.0)
                ),
            },
        }
        case_dir = result_dir / 'cases' / _safe_name(name)
        _write_json(case_dir / 'evaluation.json', case_result)
        prediction_frame = reference_frame.loc[:, [
            column for column in reference_frame.columns
            if column == 'source_row_index' or column == 'timestamp'
            or column.startswith('recorded_control_')
            or column.startswith('mpc_control_')
        ]].copy()
        for index, output_name in enumerate(data.power_names):
            prediction_frame[f'recorded_actual_{output_name}'] = actual_components[:, index]
            prediction_frame[
                f'recorded_action_report_predicted_{output_name}'
            ] = recorded_prediction[:, index]
            prediction_frame[f'mpc_action_report_predicted_{output_name}'] = (
                mpc_prediction[:, index]
            )
        prediction_frame['recorded_action_report_predicted_total_hvac_power_w'] = (
            recorded_total
        )
        prediction_frame['mpc_action_report_predicted_total_hvac_power_w'] = (
            mpc_prediction.sum(axis=1)
        )
        _write_csv(case_dir / 'predictions.csv', prediction_frame)
        case_results[name] = case_result
        summary_rows.append(_summary_row(case, selection, metrics, data.power_names))

    summary = {
        'framework': 'counterfactual_report_input_ablation',
        'result_directory': str(result_dir),
        'reference_mpc_run': {
            'result_directory': str(reference_dir),
            'counterfactual_transition_count': int(len(rows)),
            'mpc_action_trace_sha256': expected_action_hash,
        },
        'purpose': (
            'Compare report(x, MPC action, d) with report(x, recorded action, d) '
            'while holding rows, state, disturbance, and saved MPC action fixed.'
        ),
        'case_results': case_results,
    }
    _write_json(result_dir / 'counterfactual_report_input_ablation.json', summary)
    _write_csv(result_dir / 'counterfactual_report_input_ablation_summary.csv', pd.DataFrame(summary_rows))
    LOGGER.info('Counterfactual report-input ablation complete: artifacts=%s', result_dir)
    return summary
