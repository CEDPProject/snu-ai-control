"""Run the supplementary exact-action-stratified row MAPE evaluation."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform

import pandas as pd
from omegaconf import OmegaConf

from controllers.action_stratified_holdout import (
    build_action_stratified_row_holdout,
)
from controllers.candidate_pool_evaluation import (
    evaluate_fixed_action_candidate_pool_holdout,
)
from data_provider.control_data import load_control_data
from data_provider.dataset_versions import dataset_label_from_cfg
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC.action_stratified')


def _write_json(path: Path, value) -> None:
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def _result_directory(cfg) -> Path:
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'action_stratified_evaluations'
        / f'action_stratified_{timestamp}_{os.getpid()}'
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    log_path = result_dir / 'logs' / 'action_stratified_evaluation.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _candidate_summary(result) -> dict:
    selection = result.get('selection') or {}
    test = result.get('test_evaluation') or {}
    metric = result.get('primary_metric') or {}
    return {
        'reportable': bool(metric.get('reportable')),
        'mape_percent': metric.get('value'),
        'target_mape_percent': metric.get('target_percent'),
        'target_achieved': bool(metric.get('target_achieved')),
        'available_candidate_count': selection.get('candidate_count'),
        'eligible_candidate_count': selection.get('eligible_candidate_count'),
        'selected_candidate_count': selection.get('selected_candidate_count'),
        'selected_candidate_ids': selection.get('selected_candidate_ids'),
        'coverage_percent': test.get('pool_exact_match_coverage_percent'),
        'exact_match_count': test.get('pool_exact_match_count'),
        'valid_mape_sample_count': test.get('valid_primary_mape_sample_count'),
        'total_power_model_family': (
            result.get('total_power_model_selection') or {}
        ).get('selected_family'),
        'reason': result.get('reason'),
    }


def run_action_stratified_evaluation(cfg):
    """Evaluate balanced action support without changing the strict protocol."""
    evaluation_cfg = cfg.MPC.evaluation.action_stratified_row_evaluation
    if not bool(evaluation_cfg.enabled):
        raise ValueError(
            'MPC.evaluation.action_stratified_row_evaluation.enabled must be true'
        )

    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=False)
    log_path = _configure_logging(cfg, result_dir)
    OmegaConf.save(cfg, result_dir / 'configs.yaml')
    random_state = (
        int(cfg.DEFAULT.seed)
        if evaluation_cfg.random_state is None
        else int(evaluation_cfg.random_state)
    )
    LOGGER.info('Action-stratified supplementary evaluation started')
    LOGGER.info('Artifacts: %s', result_dir)
    LOGGER.info('Log file: %s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d',
        platform.python_version(),
        platform.platform(),
        os.getpid(),
    )
    LOGGER.warning(
        'Protocol is row-stratified within exact actions: it permits episode '
        'overlap between partitions and is not a future-time generalization KPI'
    )

    data = load_control_data(cfg)
    holdout_cfg = OmegaConf.merge(
        cfg.MPC.evaluation.fixed_optimal_action_holdout,
        evaluation_cfg.get('holdout_overrides', {}),
    )
    candidate_pool_cfg = OmegaConf.merge(
        cfg.MPC.evaluation.fixed_action_candidate_pool,
        evaluation_cfg.get('candidate_pool_overrides', {}),
    )
    holdout = build_action_stratified_row_holdout(
        data,
        holdout_cfg,
        split_rate=cfg.DATASET.split_rate,
        random_state=random_state,
    )
    LOGGER.info(
        'Action-stratified split: train=%d | validation=%d | test=%d | '
        'observed_actions=%d | support_eligible_actions=%d',
        len(holdout.train_rows),
        len(holdout.validation_rows),
        len(holdout.test_rows),
        int(holdout.metadata['total_observed_action_count']),
        int(holdout.metadata['candidate_count']),
    )

    result, prediction_rows = evaluate_fixed_action_candidate_pool_holdout(
        data,
        holdout,
        config=candidate_pool_cfg,
        total_power_report_config=(
            holdout_cfg.total_power_report_model
        ),
        mape_min_actual_power_w=float(
            cfg.MPC.affine_models.mape_min_actual_power_w
        ),
        random_state=random_state,
        report_model_refit_scope=str(
            OmegaConf.select(
                cfg,
                'MPC.evaluation.report_model_refit_scope',
                default='train_validation',
            )
        ),
    )
    result['supplementary_protocol'] = {
        **holdout.metadata['protocol'],
        'official_strict_kpi_replaced': False,
        'reporting_label': (
            'supplementary_action_stratified_row_mape_not_official_strict_kpi'
        ),
    }
    result['split_metadata'] = {
        key: value
        for key, value in holdout.metadata.items()
        if key != 'candidates'
    }
    result['candidate_support'] = holdout.metadata['candidates']
    summary = {
        'experiment': 'action_stratified_row_evaluation',
        'dataset_label': dataset_label_from_cfg(cfg),
        'environment': str(cfg.DATASET.environment_name),
        'result_directory': str(result_dir),
        'supplementary_protocol': result['supplementary_protocol'],
        'candidate_pool': _candidate_summary(result),
        'split_transition_counts': holdout.metadata['split_transition_counts'],
        'artifacts': {
            'evaluation': 'action_stratified_evaluation.json',
            'action_allocation': 'action_split_allocation.csv',
            'candidate_support': 'candidate_support.csv',
            'candidate_selection': 'candidate_selection.csv',
            'test_predictions': 'test_predictions.csv',
        },
    }

    _write_json(result_dir / 'action_stratified_evaluation.json', result)
    _write_json(result_dir / 'summary.json', summary)
    _write_csv(
        result_dir / 'action_split_allocation.csv',
        holdout.metadata['allocation_audit'],
    )
    _write_csv(result_dir / 'candidate_support.csv', holdout.audit_rows)
    _write_csv(
        result_dir / 'candidate_selection.csv',
        pd.json_normalize((result.get('selection') or {}).get('candidates') or []),
    )
    _write_csv(result_dir / 'test_predictions.csv', prediction_rows)

    headline = summary['candidate_pool']
    LOGGER.info(
        'Action-stratified evaluation complete: reportable=%s | MAPE=%s | '
        'coverage=%s | selected_candidates=%s | artifacts=%s',
        headline['reportable'],
        headline['mape_percent'],
        headline['coverage_percent'],
        headline['selected_candidate_count'],
        result_dir,
    )
    return summary
