"""Run evaluator and candidate-pool experiments without test leakage."""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import re

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from controllers.action_crossfit import align_boundaries_to_control_episodes
from controllers.candidate_pool_evaluation import (
    evaluate_fixed_action_candidate_pool_holdout,
    evaluate_fixed_action_candidate_pool_top_k_sweep,
)
from controllers.evaluator_benchmark import (
    benchmark_evaluator_families,
    evaluator_comparison_rows,
)
from controllers.fixed_action_evaluation import (
    build_fixed_action_episode_holdout,
)
from controllers.kpi_coverage_summary import (
    build_kpi_coverage_summary,
    kpi_coverage_summary_rows,
)
from controllers.pipeline import run_mpc_pipeline
from controllers.total_power_evaluation import (
    evaluate_all_chronological_test_total_power,
)
from data_provider.control_data import load_control_data
from data_provider.dataset_versions import dataset_label_from_cfg
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC.experiments')


def _plain(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )



def _artifact_directory(parent: Path, name: str) -> Path:
    """Create a stable, filesystem-safe directory for one experiment."""
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("._")
    if not safe_name:
        raise ValueError("Experiment artifact directory name is empty")
    directory = parent / safe_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_dataframe(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def _write_candidate_artifacts(
    parent: Path,
    scenario_name: str,
    result,
    audit_rows,
) -> Path:
    """Persist one candidate-pool experiment as a self-contained unit."""
    scenario_dir = _artifact_directory(parent, scenario_name)
    result["artifact_directory"] = str(scenario_dir)
    _write_json(scenario_dir / "evaluation.json", result)
    _write_json(
        scenario_dir / "scenario_config.json",
        result.get("experiment_scenario") or {},
    )
    _write_dataframe(scenario_dir / "test_predictions.csv", audit_rows)
    selection = result.get("selection") or {}
    _write_dataframe(
        scenario_dir / "candidate_selection.csv",
        pd.json_normalize(selection.get("candidates") or []),
    )
    _write_json(
        scenario_dir / "selected_actions.json",
        selection.get("selected_actions") or [],
    )
    return scenario_dir


def _write_evaluator_artifacts(parent: Path, result) -> None:
    """Persist comparison plus an independent folder for every evaluator."""
    parent.mkdir(parents=True, exist_ok=True)
    _write_json(parent / "benchmark.json", result)
    comparison_rows = evaluator_comparison_rows(result)
    _write_dataframe(parent / "model_comparison.csv", comparison_rows)

    models_by_family = {
        str(model_result.get("requested_family")): model_result
        for model_result in (result.get("models") or [])
    }
    for comparison_row in comparison_rows:
        family = str(comparison_row.get("requested_family"))
        model_dir = _artifact_directory(parent, family)
        _write_json(
            model_dir / "evaluation.json",
            models_by_family.get(family, {}),
        )
        _write_dataframe(model_dir / "summary.csv", [comparison_row])


def _result_directory(cfg) -> Path:
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run_name = (
        f"{cfg.DEFAULT.exp_name}_{timestamp}_{os.getpid()}"
    )
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'mpc_experiments'
        / run_name
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    log_path = result_dir / 'logs' / 'experiment_matrix.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _chronological_boundaries(cfg, data):
    nominal_train_end, nominal_test_start = data.split_indices(
        cfg.DATASET.split_rate
    )
    if bool(
        cfg.MPC.evaluation.action_options.episode_group_split.enabled
    ):
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
    else:
        train_end, test_start = nominal_train_end, nominal_test_start
    return int(train_end), int(test_start)


def _candidate_pool_experiments(cfg, data, holdout, artifact_root):
    experiment_cfg = cfg.MPC.candidate_pool_experiments
    base = OmegaConf.create(
        _plain(cfg.MPC.evaluation.fixed_action_candidate_pool)
    )
    report_cfg = (
        cfg.MPC.evaluation.fixed_optimal_action_holdout
        .total_power_report_model
    )
    detailed = {}
    audit_by_scenario = {}
    summary_rows = []
    for scenario_index, scenario in enumerate(experiment_cfg.scenarios):
        scenario = _plain(scenario)
        name = str(scenario['name'])
        holdout_overrides = _plain(
            scenario.get('holdout_overrides') or {}
        )
        candidate_pool_overrides = _plain(
            scenario.get('candidate_pool_overrides') or {}
        )
        if holdout_overrides:
            holdout_cfg = OmegaConf.merge(
                cfg.MPC.evaluation.fixed_optimal_action_holdout,
                holdout_overrides,
            )
            scenario_holdout = build_fixed_action_episode_holdout(
                data,
                holdout_cfg,
                split_rate=cfg.DATASET.split_rate,
            )
        else:
            scenario_holdout = holdout
        scenario_cfg = OmegaConf.merge(
            base,
            candidate_pool_overrides,
            {
                'maximum_validation_mape_percent_per_candidate': float(
                    scenario[
                        'maximum_validation_mape_percent_per_candidate'
                    ]
                ),
                'maximum_candidates': int(scenario['maximum_candidates']),
            },
        )
        LOGGER.info(
            'Candidate-pool experiment started: name=%s | '
            'available_candidates=%d | validation_limit=%s | '
            'maximum_candidates=%d | holdout_overrides=%s',
            name,
            int(scenario_holdout.metadata['candidate_count']),
            scenario_cfg.maximum_validation_mape_percent_per_candidate,
            scenario_cfg.maximum_candidates,
            holdout_overrides or None,
        )
        result, audit_rows = evaluate_fixed_action_candidate_pool_holdout(
            data,
            scenario_holdout,
            config=scenario_cfg,
            total_power_report_config=report_cfg,
            mape_min_actual_power_w=float(
                cfg.MPC.affine_models.mape_min_actual_power_w
            ),
            random_state=int(cfg.DEFAULT.seed) + 100 * scenario_index,
            report_model_refit_scope=str(
                OmegaConf.select(
                    cfg,
                    'MPC.evaluation.report_model_refit_scope',
                    default='train_validation',
                )
            ),
        )
        result['experiment_scenario'] = {
            'name': name,
            'description': str(scenario.get('description', '')),
            'maximum_validation_mape_percent_per_candidate': float(
                scenario_cfg.maximum_validation_mape_percent_per_candidate
            ),
            'maximum_candidates': int(
                scenario_cfg.maximum_candidates
            ),
            'available_candidate_count': int(
                scenario_holdout.metadata['candidate_count']
            ),
            'holdout_overrides': holdout_overrides,
            'candidate_pool_overrides': candidate_pool_overrides,
            'effective_holdout_protocol': (
                scenario_holdout.metadata['protocol']
            ),
            'classification': str(
                scenario.get('classification', 'primary_comparison')
            ),
            'selection_uses_test_targets': False,
        }
        _write_candidate_artifacts(
            artifact_root,
            name,
            result,
            audit_rows,
        )
        detailed[name] = result
        audit_by_scenario[name] = audit_rows
        selection = result.get('selection') or {}
        test = result.get('test_evaluation') or {}
        metric = result.get('primary_metric') or {}
        summary_rows.append({
            'scenario': name,
            'description': str(scenario.get('description', '')),
            'classification': str(
                scenario.get('classification', 'primary_comparison')
            ),
            'available_candidate_count': int(
                scenario_holdout.metadata['candidate_count']
            ),
            'candidate_count': selection.get('candidate_count'),
            'eligible_candidate_count': selection.get(
                'eligible_candidate_count'
            ),
            'selected_candidate_count': selection.get(
                'selected_candidate_count'
            ),
            'selected_candidate_ids': ','.join(
                selection.get('selected_candidate_ids') or []
            ),
            'selected_model_family': (
                result.get('total_power_model_selection') or {}
            ).get('selected_family'),
            'pooled_test_mape_percent': metric.get('value'),
            'reportable': metric.get('reportable'),
            'target_achieved': metric.get('target_achieved'),
            'test_exact_match_count': test.get('pool_exact_match_count'),
            'test_exact_match_episode_count': test.get(
                'pool_exact_match_episode_count'
            ),
            'test_exact_match_coverage_percent': test.get(
                'pool_exact_match_coverage_percent'
            ),
            'equal_action_macro_mape_percent': test.get(
                'equal_action_macro_mape_percent'
            ),
            'worst_action_mape_percent': test.get(
                'worst_action_mape_percent'
            ),
            'action_target_pass_rate_percent': test.get(
                'action_target_pass_rate_percent'
            ),
        })
        LOGGER.info(
            'Candidate-pool experiment completed: name=%s | selected=%s | '
            'mape=%s | coverage=%s',
            name,
            selection.get('selected_candidate_count'),
            metric.get('value'),
            test.get('pool_exact_match_coverage_percent'),
        )
    return detailed, audit_by_scenario, summary_rows


def _candidate_pool_top_k_sweep(cfg, data, artifact_root):
    sweep_cfg = cfg.MPC.candidate_pool_top_k_sweep
    if not bool(sweep_cfg.enabled):
        LOGGER.info('Candidate-pool Top-K sweep disabled')
        return None

    scenario_name = str(sweep_cfg.name)
    holdout_cfg = OmegaConf.merge(
        cfg.MPC.evaluation.fixed_optimal_action_holdout,
        sweep_cfg.get('holdout_overrides', {}),
    )
    holdout = build_fixed_action_episode_holdout(
        data,
        holdout_cfg,
        split_rate=cfg.DATASET.split_rate,
    )
    evaluation_cfg = OmegaConf.merge(
        cfg.MPC.evaluation.fixed_action_candidate_pool,
        sweep_cfg.get('candidate_pool_overrides', {}),
        {
            'minimum_k': int(sweep_cfg.minimum_k),
            'maximum_k': int(sweep_cfg.maximum_k),
            'step': int(sweep_cfg.step),
            'k_selection_rule': str(sweep_cfg.k_selection_rule),
            'fallback_rule': str(sweep_cfg.fallback_rule),
            'minimum_pooled_validation_mape_samples': int(
                sweep_cfg.minimum_pooled_validation_mape_samples
            ),
        },
    )
    LOGGER.info(
        'Candidate-pool Top-K sweep started: name=%s | available=%d | '
        'k=%d..%d | step=%d',
        scenario_name,
        int(holdout.metadata['candidate_count']),
        int(evaluation_cfg.minimum_k),
        int(evaluation_cfg.maximum_k),
        int(evaluation_cfg.step),
    )
    result, audit_rows = evaluate_fixed_action_candidate_pool_top_k_sweep(
        data,
        holdout,
        config=evaluation_cfg,
        total_power_report_config=(
            cfg.MPC.evaluation.fixed_optimal_action_holdout
            .total_power_report_model
        ),
        mape_min_actual_power_w=float(
            cfg.MPC.affine_models.mape_min_actual_power_w
        ),
        random_state=int(cfg.DEFAULT.seed) + 10_000,
        report_model_refit_scope=str(
            OmegaConf.select(
                cfg,
                'MPC.evaluation.report_model_refit_scope',
                default='train_validation',
            )
        ),
    )
    result['experiment_scenario'] = {
        'name': scenario_name,
        'description': str(sweep_cfg.get('description', '')),
        'classification': 'validation_selected_top_k_sweep',
        'available_candidate_count': int(
            holdout.metadata['candidate_count']
        ),
        'holdout_overrides': _plain(
            sweep_cfg.get('holdout_overrides', {})
        ),
        'candidate_pool_overrides': _plain(
            sweep_cfg.get('candidate_pool_overrides', {})
        ),
        'minimum_k': int(evaluation_cfg.minimum_k),
        'maximum_k': int(evaluation_cfg.maximum_k),
        'step': int(evaluation_cfg.step),
        'selection_uses_test_targets': False,
    }

    scenario_dir = _artifact_directory(artifact_root, scenario_name)
    result['artifact_directory'] = str(scenario_dir)
    _write_json(scenario_dir / 'evaluation.json', result)
    _write_json(
        scenario_dir / 'scenario_config.json',
        result['experiment_scenario'],
    )
    _write_dataframe(scenario_dir / 'top_k_curve.csv', result.get('curve') or [])
    _write_dataframe(
        scenario_dir / 'candidate_order.csv',
        pd.json_normalize(result.get('candidate_order') or []),
    )
    _write_dataframe(
        scenario_dir / 'all_candidate_test_predictions.csv',
        audit_rows,
    )
    _write_dataframe(
        scenario_dir / 'selected_k_test_predictions.csv',
        [
            row
            for row in audit_rows
            if row.get('included_in_official_selected_k')
        ],
    )
    _write_json(
        scenario_dir / 'selected_k.json',
        {
            'official_selection': result.get('official_selection'),
            'primary_metric': result.get('primary_metric'),
        },
    )
    official_selection = result.get('official_selection') or {}
    metric = result.get('primary_metric') or {}
    LOGGER.info(
        'Candidate-pool Top-K sweep completed: selected_k=%s | '
        'official_test_mape=%s | target_achieved=%s | artifacts=%s',
        official_selection.get('selected_k'),
        metric.get('value'),
        metric.get('target_achieved'),
        scenario_dir,
    )
    return result


def _run_full_mpc_for_selected(cfg, selected_family, result_dir):
    full_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False)
    )
    full_cfg.MPC.independent_evaluator.model_family = selected_family
    auxiliary_family = str(
        full_cfg.MPC.evaluator_benchmark.full_mpc_auxiliary_evaluator_family
    )
    full_cfg.MPC.evaluation.fixed_optimal_action_holdout.model_family = (
        auxiliary_family
    )
    full_cfg.MPC.evaluation.selected_action_crossfit.model_family = (
        auxiliary_family
    )
    full_cfg.DEFAULT.exp_name = (
        f'evaluator_{selected_family.replace(":", "_")}'
    )
    full_result_dir = (
        result_dir
        / f'full_mpc_{selected_family.replace(":", "_")}'
    )
    LOGGER.info(
        'Full MPC run for selected evaluator started: family=%s | artifacts=%s',
        selected_family,
        full_result_dir,
    )
    summary = run_mpc_pipeline(full_cfg, full_result_dir)
    LOGGER.info(
        'Full MPC run for selected evaluator completed: family=%s',
        selected_family,
    )
    # strict 대표 지표에는 coverage/sample이 담기지 않으므로, candidate-pool 평가에서
    # 실제 평가된 행 수와 coverage를 함께 꺼내 kpi_coverage_summary가 쓰게 한다.
    candidate_pool_eval = summary.get('fixed_action_candidate_pool_evaluation') or {}
    pool_test = candidate_pool_eval.get('test_evaluation') or {}
    pool_selection = candidate_pool_eval.get('selection') or {}
    counterfactual = (
        summary.get('mpc_counterfactual_total_power_report_evaluation') or {}
    )
    counterfactual_metric = counterfactual.get('primary_metric') or {}
    counterfactual_cases = counterfactual.get('mape_cases') or {}
    counterfactual_test = counterfactual.get('test_evaluation') or {}
    counterfactual_reduction = (
        counterfactual.get('estimated_energy_reduction') or {}
    )
    report_to_report_reduction = (
        counterfactual.get('report_to_report_predicted_energy_reduction') or {}
    )
    return {
        'selected_family': selected_family,
        'result_directory': str(full_result_dir),
        'official_mape': summary['primary_metric'],
        'strict_coverage_percent': pool_test.get('pool_exact_match_coverage_percent'),
        'strict_evaluated_sample_count': pool_test.get('valid_primary_mape_sample_count'),
        'strict_selected_candidate_count': pool_selection.get('selected_candidate_count'),
        'energy_reduction_percent': summary['energy_reduction_percent'],
        'mpc_counterfactual_total_power_mape_percent': (
            counterfactual_metric.get('value')
        ),
        'mpc_counterfactual_reportable': bool(
            counterfactual_metric.get('reportable', False)
        ),
        'mpc_counterfactual_action_changed_percent': (
            counterfactual_test.get('action_changed_percent')
        ),
        'mpc_counterfactual_estimated_reduction_percent': (
            counterfactual_reduction.get('estimated_reduction_percent')
        ),
        'mpc_report_to_report_total_power_mape_percent': (
            (counterfactual_cases.get(
                'mpc_action_vs_recorded_action_report'
            ) or {}).get('value')
        ),
        'mpc_report_to_report_reportable': bool(
            (counterfactual_cases.get(
                'mpc_action_vs_recorded_action_report'
            ) or {}).get('reportable', False)
        ),
        'mpc_report_to_report_predicted_reduction_percent': (
            report_to_report_reduction.get('estimated_reduction_percent')
        ),
        'quality_assessment': summary['quality_assessment'],
    }


def run_evaluator_benchmark(cfg):
    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=False)
    log_path = _configure_logging(cfg, result_dir)
    LOGGER.info('MPC experiment matrix started')
    LOGGER.info('Artifacts: %s', result_dir)
    LOGGER.info('Log file: %s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d',
        platform.python_version(),
        platform.platform(),
        os.getpid(),
    )
    OmegaConf.save(cfg, result_dir / 'configs.yaml')
    all_data_dir = _artifact_directory(result_dir, '01_all_data')
    candidate_root = _artifact_directory(
        result_dir, '02_candidate_pool_experiments'
    )
    evaluator_root = _artifact_directory(result_dir, '03_evaluators')
    full_mpc_root = _artifact_directory(result_dir, '04_full_mpc')


    data = load_control_data(cfg)
    train_end, test_start = _chronological_boundaries(cfg, data)
    LOGGER.info(
        'Chronological partitions: train=%d | validation=%d | test=%d',
        train_end,
        test_start - train_end,
        len(data.states) - test_start,
    )
    train_rows = np.arange(0, train_end - 1, dtype=int)
    validation_rows = np.arange(
        train_end,
        test_start - 1,
        dtype=int,
    )
    test_rows = np.arange(
        test_start,
        len(data.states) - 1,
        dtype=int,
    )

    all_data_cfg = cfg.MPC.all_test_data_evaluation
    all_data_result = None
    all_data_audit_rows = []
    if bool(all_data_cfg.enabled):
        LOGGER.info(
            'All chronological test-row total-power evaluation started'
        )
        (
            all_data_result,
            all_data_audit_rows,
        ) = evaluate_all_chronological_test_total_power(
            data,
            training_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            report_model_config=(
                cfg.MPC.evaluation.fixed_optimal_action_holdout
                .total_power_report_model
            ),
            minimum_actual_power_w=float(
                cfg.MPC.affine_models.mape_min_actual_power_w
            ),
            target_mape_percent=float(all_data_cfg.target_mape_percent),
            minimum_validation_samples=int(
                all_data_cfg.minimum_validation_mape_samples
            ),
            minimum_test_samples=int(
                all_data_cfg.minimum_test_mape_samples
            ),
            random_state=int(cfg.DEFAULT.seed),
            report_model_refit_scope=str(
                OmegaConf.select(
                    cfg,
                    'MPC.evaluation.report_model_refit_scope',
                    default='train_validation',
                )
            ),
        )
        LOGGER.info(
            'All chronological test-row evaluation completed: '
            'family=%s | mape=%s',
            all_data_result['model_selection']['selected_family'],
            all_data_result['primary_metric']['value'],
        )
        _write_json(all_data_dir / 'evaluation.json', all_data_result)
        _write_dataframe(
            all_data_dir / 'test_predictions.csv',
            all_data_audit_rows,
        )

    holdout = build_fixed_action_episode_holdout(
        data,
        cfg.MPC.evaluation.fixed_optimal_action_holdout,
        split_rate=cfg.DATASET.split_rate,
    )
    (
        candidate_results,
        candidate_audit,
        candidate_summary_rows,
    ) = _candidate_pool_experiments(
        cfg, data, holdout, candidate_root
    )
    _write_json(
        candidate_root / 'all_scenarios.json', candidate_results
    )
    _write_dataframe(
        candidate_root / 'scenario_comparison.csv',
        candidate_summary_rows,
    )
    top_k_sweep_result = _candidate_pool_top_k_sweep(
        cfg, data, candidate_root
    )

    evaluator_benchmark = benchmark_evaluator_families(
        cfg,
        data,
        train_end=train_end,
        test_start=test_start,
    )
    _write_evaluator_artifacts(evaluator_root, evaluator_benchmark)
    selected_family = str(evaluator_benchmark['selected_family'])

    full_mpc_result = None
    if bool(cfg.MPC.evaluator_benchmark.run_full_mpc_for_selected):
        full_mpc_result = _run_full_mpc_for_selected(
            cfg,
            selected_family,
            full_mpc_root,
        )

    # strict 대표 지표와 완화 provisional 값을 coverage·신뢰등급과 함께 요약한다.
    # support가 부족해 strict가 null이 되는 run도 provisional 값을 남겨 해석 가능하게
    # 한다. provisional은 절대 대표값으로 승격하지 않는다(helper가 claim_boundary 부착).
    kpi_coverage = None
    provisional_summary_cfg = cfg.MPC.get('provisional_kpi_summary')
    if provisional_summary_cfg is not None and bool(
        provisional_summary_cfg.get('enabled', False)
    ):
        kpi_coverage = build_kpi_coverage_summary(
            full_mpc_result=full_mpc_result,
            candidate_summary_rows=candidate_summary_rows,
            config=OmegaConf.to_container(provisional_summary_cfg, resolve=True),
        )
        summary_rows = kpi_coverage_summary_rows(kpi_coverage)
        if summary_rows:
            _write_dataframe(
                candidate_root / 'kpi_coverage_summary.csv', summary_rows
            )
        strict = (kpi_coverage or {}).get('strict') or {}
        provisional = (kpi_coverage or {}).get('provisional') or {}
        LOGGER.info(
            'KPI coverage summary: strict=%s (tier=%s, cov=%s) | '
            'provisional=%s (tier=%s, cov=%s, source=%s)',
            strict.get('mape_percent'),
            strict.get('confidence_tier'),
            strict.get('coverage_percent'),
            provisional.get('mape_percent'),
            provisional.get('confidence_tier'),
            provisional.get('coverage_percent'),
            provisional.get('source'),
        )

    summary = {
        'protocol': {
            'name': 'mpc_evaluator_and_action_scope_experiment_matrix',
            'test_targets_used_for_model_or_action_selection': False,
            'all_data_meaning': (
                'all valid chronological test transitions after frozen '
                'validation-only model selection; never train on test'
            ),
            'candidate_pool_ordering': (
                'train-only action-median validation MAPE, test support, '
                'then observed-frequency rank'
            ),
            'expanded_pool_policy': (
                'validation threshold is relaxed only in declared expanded '
                'scenarios; test targets remain unused'
            ),
            'top_k_sweep_policy': (
                'K is selected by validation only; the selected-K test MAPE '
                'is official and every other K test MAPE is sensitivity-only'
            ),
        },
        'result_directory': str(result_dir),
        'artifact_directories': {
            'all_data': str(all_data_dir),
            'candidate_pool_experiments': str(candidate_root),
            'evaluators': str(evaluator_root),
            'full_mpc': str(full_mpc_root),
        },
        'chronological_split': {
            'train_end_exclusive': train_end,
            'validation_start': train_end,
            'test_start': test_start,
            'test_end_exclusive': int(len(data.states)),
        },
        'all_chronological_test_total_power': all_data_result,
        'candidate_pool_experiments': candidate_results,
        'candidate_pool_top_k_sweep': top_k_sweep_result,
        'evaluator_benchmark': evaluator_benchmark,
        'full_mpc_selected_evaluator': full_mpc_result,
        'kpi_coverage_summary': kpi_coverage,
    }

    _write_json(result_dir / 'experiment_matrix.json', summary)
    LOGGER.info(
        'MPC experiment matrix completed: selected_evaluator=%s | artifacts=%s',
        selected_family,
        result_dir,
    )
    return summary
