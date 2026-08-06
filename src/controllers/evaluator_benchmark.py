"""Validation-selected benchmark for interchangeable MPC plant evaluators."""

import gc
import logging
from typing import Dict, Sequence

import numpy as np
from omegaconf import OmegaConf

from controllers.evaluation import evaluate_recursive_rollouts
from controllers.pipeline import _one_step_metrics, _select_evaluation_windows
from dynamics.evaluator_factory import (
    create_independent_evaluator,
    evaluator_label,
)


LOGGER = logging.getLogger('MPC.evaluator_benchmark')


def _plain(config) -> Dict[str, object]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _release_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _finite_minimum(values):
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return min(finite) if finite else None


def _validation_gates(evaluator_cfg, one_step, recursive):
    minimum_r2 = _finite_minimum(one_step['power']['r2_per_output'])
    recursive_wape = recursive['power']['wape_percent']
    false_safe = recursive['constraint_prediction'][
        'maximum_state_false_safe_rate_percent'
    ]
    r2_required = float(evaluator_cfg.minimum_test_power_r2)
    wape_required = float(
        evaluator_cfg.maximum_rollout_power_wape_percent
    )
    false_safe_required = float(
        evaluator_cfg.maximum_constraint_false_safe_rate_percent
    )
    gates = {
        'power_r2': {
            'observed': minimum_r2,
            'required_minimum': r2_required,
            'passed': bool(
                minimum_r2 is not None and minimum_r2 >= r2_required
            ),
        },
        'recursive_power_wape': {
            'observed_percent': recursive_wape,
            'required_maximum_percent': wape_required,
            'passed': bool(
                recursive_wape is not None
                and recursive_wape <= wape_required
            ),
        },
        'constraint_false_safe': {
            'observed_maximum_percent': false_safe,
            'required_maximum_percent': false_safe_required,
            'not_applicable_when_no_actual_violation': True,
            'passed': bool(
                false_safe is None or false_safe <= false_safe_required
            ),
        },
    }
    return {
        'passed': all(gate['passed'] for gate in gates.values()),
        'failed_gate_count': int(
            sum(not gate['passed'] for gate in gates.values())
        ),
        'gates': gates,
    }


def _selection_key(item):
    validation = item['validation']
    quality = validation['quality_gates']
    recursive = validation['recursive']
    one_step = validation['one_step']
    recursive_total_wape = recursive['total_power']['wape_percent']
    recursive_total_mape = recursive['total_power']['mape_percent']
    one_step_total_mape = one_step['total_power']['mape_percent']
    return (
        int(quality['failed_gate_count']),
        float('inf')
        if recursive_total_wape is None
        else float(recursive_total_wape),
        float('inf')
        if recursive_total_mape is None
        else float(recursive_total_mape),
        float('inf')
        if one_step_total_mape is None
        else float(one_step_total_mape),
        int(item['priority']),
    )


def _metric_summary(metrics):
    return {
        'power_component_r2': metrics['power']['r2_per_output'],
        'power_component_wape_percent': metrics['power']['wape_percent'],
        'power_component_mape_percent': metrics['power']['mape_percent'],
        'total_power_r2': metrics['total_power']['r2_per_output'][0],
        'total_power_wape_percent': metrics['total_power']['wape_percent'],
        'total_power_mape_percent': metrics['total_power']['mape_percent'],
        'total_power_valid_mape_samples': metrics['total_power'][
            'mape_valid_samples_per_output'
        ][0],
    }


def benchmark_evaluator_families(
    cfg,
    data,
    *,
    train_end: int,
    test_start: int,
):
    """Select evaluators on validation only and report frozen test results."""

    experiment_cfg = cfg.MPC.evaluator_benchmark
    families = [str(value) for value in experiment_cfg.model_families]
    if not families:
        raise ValueError('MPC.evaluator_benchmark.model_families is empty')
    if len(set(families)) != len(families):
        raise ValueError('Evaluator benchmark families must be unique')

    duration_steps = int(cfg.MPC.evaluation.closed_loop_steps)
    stride = int(cfg.MPC.evaluation.stride)
    validation_starts = _select_evaluation_windows(
        data,
        train_end,
        duration_steps,
        stride,
        int(experiment_cfg.validation_max_windows),
        int(cfg.MPC.prediction_horizon),
        'persistence',
        end_exclusive=test_start,
    )
    test_starts = _select_evaluation_windows(
        data,
        test_start,
        duration_steps,
        stride,
        int(experiment_cfg.test_max_windows),
        int(cfg.MPC.prediction_horizon),
        'persistence',
    )
    mape_threshold = float(
        cfg.MPC.affine_models.mape_min_actual_power_w
    )
    lead_times = [
        int(value) for value in cfg.MPC.evaluation.lead_time_steps
    ]

    results = []
    for priority, family in enumerate(families):
        LOGGER.info(
            'Evaluator validation benchmark %d/%d started: family=%s',
            priority + 1,
            len(families),
            family,
        )
        item = {
            'priority': int(priority),
            'requested_family': family,
            'validation_fit_succeeded': False,
            'validation_failure': None,
            'selected_by_validation': False,
            'validation': None,
            'test_fit_succeeded': False,
            'test_failure': None,
            'test': None,
        }
        evaluator = None
        try:
            evaluator = create_independent_evaluator(
                cfg.MPC.independent_evaluator,
                data,
                train_end,
                random_state=int(cfg.DEFAULT.seed) + priority,
                model_family=family,
            )
            label = evaluator_label(evaluator)
            one_step = _one_step_metrics(
                data,
                evaluator,
                train_end,
                test_start - 1,
                mape_threshold,
            )
            recursive = evaluate_recursive_rollouts(
                evaluator,
                data,
                validation_starts,
                duration_steps,
                cfg.MPC.state_constraints,
                mape_threshold,
                lead_times,
            )
            quality = _validation_gates(
                cfg.MPC.independent_evaluator,
                one_step,
                recursive,
            )
            item.update({
                'evaluator_label': label,
                'validation_fit_succeeded': True,
                'validation': {
                    'training_scope': 'chronological train only',
                    'one_step': one_step,
                    'recursive': recursive,
                    'quality_gates': quality,
                    'selection_summary': {
                        'one_step': _metric_summary(one_step),
                        'recursive': _metric_summary(recursive),
                        'maximum_false_safe_rate_percent': recursive[
                            'constraint_prediction'
                        ]['maximum_state_false_safe_rate_percent'],
                    },
                    'metadata': evaluator.to_metadata(),
                },
            })
            LOGGER.info(
                'Evaluator validation benchmark completed: family=%s | '
                'failed_gates=%d | recursive_total_wape=%s | '
                'recursive_total_mape=%s',
                family,
                quality['failed_gate_count'],
                recursive['total_power']['wape_percent'],
                recursive['total_power']['mape_percent'],
            )
        except Exception as error:
            item['validation_failure'] = (
                f'{type(error).__name__}: {error}'
            )
            LOGGER.exception(
                'Evaluator validation benchmark failed: family=%s',
                family,
            )
        finally:
            if evaluator is not None:
                _release_model(evaluator)
        results.append(item)

    successful = [
        item for item in results if item['validation_fit_succeeded']
    ]
    if not successful:
        raise RuntimeError('Every evaluator family failed validation fitting')
    selected = min(successful, key=_selection_key)
    selected['selected_by_validation'] = True
    selected_family = str(selected['requested_family'])
    LOGGER.info(
        'Evaluator selected using validation only: family=%s | score=%s',
        selected_family,
        _selection_key(selected),
    )

    evaluate_all = bool(experiment_cfg.evaluate_all_models_on_test)
    for item in results:
        if not item['validation_fit_succeeded']:
            continue
        if not evaluate_all and not item['selected_by_validation']:
            continue
        family = str(item['requested_family'])
        evaluator = None
        LOGGER.info(
            'Evaluator frozen test evaluation started: family=%s | '
            'training_scope=train+validation',
            family,
        )
        try:
            evaluator = create_independent_evaluator(
                cfg.MPC.independent_evaluator,
                data,
                test_start,
                random_state=int(cfg.DEFAULT.seed) + 1000 + int(item['priority']),
                model_family=family,
            )
            one_step = _one_step_metrics(
                data,
                evaluator,
                test_start,
                len(data.states) - 1,
                mape_threshold,
            )
            recursive = evaluate_recursive_rollouts(
                evaluator,
                data,
                test_starts,
                duration_steps,
                cfg.MPC.state_constraints,
                mape_threshold,
                lead_times,
            )
            item.update({
                'test_fit_succeeded': True,
                'test': {
                    'training_scope': 'chronological train + validation',
                    'one_step': one_step,
                    'recursive': recursive,
                    'quality_gates': _validation_gates(
                        cfg.MPC.independent_evaluator,
                        one_step,
                        recursive,
                    ),
                    'summary': {
                        'one_step': _metric_summary(one_step),
                        'recursive': _metric_summary(recursive),
                        'maximum_false_safe_rate_percent': recursive[
                            'constraint_prediction'
                        ]['maximum_state_false_safe_rate_percent'],
                    },
                    'metadata': evaluator.to_metadata(),
                },
            })
            LOGGER.info(
                'Evaluator frozen test evaluation completed: family=%s | '
                'recursive_total_wape=%s | recursive_total_mape=%s',
                family,
                recursive['total_power']['wape_percent'],
                recursive['total_power']['mape_percent'],
            )
        except Exception as error:
            item['test_failure'] = f'{type(error).__name__}: {error}'
            LOGGER.exception(
                'Evaluator frozen test evaluation failed: family=%s',
                family,
            )
        finally:
            if evaluator is not None:
                _release_model(evaluator)

    return {
        'protocol': {
            'name': 'validation_selected_independent_evaluator_benchmark',
            'selection_uses_test_targets': False,
            'validation_training_scope': 'chronological train only',
            'validation_scoring_scope': (
                'chronological validation one-step and 60-step recursive rollouts'
            ),
            'selection_rule': (
                'fewest failed validation quality gates, then lowest recursive '
                'total-power WAPE, recursive total-power MAPE, one-step '
                'total-power MAPE, and declared priority'
            ),
            'test_refit_scope': 'chronological train + validation',
            'test_models_reported': (
                'all validation-successful families'
                if evaluate_all
                else 'validation-selected family only'
            ),
            'test_results_used_for_selection': False,
            'claim_boundary': (
                'Surrogate-model fidelity under recorded controls; full MPC '
                'policy performance requires a separate closed-loop run'
            ),
        },
        'partitions': {
            'train_end_exclusive': int(train_end),
            'validation_start': int(train_end),
            'test_start': int(test_start),
            'test_end_exclusive': int(len(data.states)),
            'validation_recursive_window_count': int(len(validation_starts)),
            'test_recursive_window_count': int(len(test_starts)),
            'recursive_window_steps': duration_steps,
        },
        'selected_family': selected_family,
        'selected_evaluator_label': selected.get('evaluator_label'),
        'models': results,
    }


def evaluator_comparison_rows(benchmark) -> Sequence[Dict[str, object]]:
    rows = []
    for item in benchmark['models']:
        validation = item.get('validation') or {}
        test = item.get('test') or {}
        validation_summary = validation.get('selection_summary') or {}
        test_summary = test.get('summary') or {}
        row = {
            'priority': item['priority'],
            'requested_family': item['requested_family'],
            'evaluator_label': item.get('evaluator_label'),
            'selected_by_validation': item['selected_by_validation'],
            'validation_fit_succeeded': item['validation_fit_succeeded'],
            'validation_failure': item.get('validation_failure'),
            'validation_failed_gate_count': (
                (validation.get('quality_gates') or {}).get(
                    'failed_gate_count'
                )
            ),
            'validation_recursive_total_power_wape_percent': (
                (validation_summary.get('recursive') or {}).get(
                    'total_power_wape_percent'
                )
            ),
            'validation_recursive_total_power_mape_percent': (
                (validation_summary.get('recursive') or {}).get(
                    'total_power_mape_percent'
                )
            ),
            'validation_one_step_total_power_mape_percent': (
                (validation_summary.get('one_step') or {}).get(
                    'total_power_mape_percent'
                )
            ),
            'validation_maximum_false_safe_rate_percent': (
                validation_summary.get('maximum_false_safe_rate_percent')
            ),
            'test_fit_succeeded': item['test_fit_succeeded'],
            'test_failure': item.get('test_failure'),
            'test_failed_gate_count': (
                (test.get('quality_gates') or {}).get(
                    'failed_gate_count'
                )
            ),
            'test_recursive_total_power_wape_percent': (
                (test_summary.get('recursive') or {}).get(
                    'total_power_wape_percent'
                )
            ),
            'test_recursive_total_power_mape_percent': (
                (test_summary.get('recursive') or {}).get(
                    'total_power_mape_percent'
                )
            ),
            'test_one_step_total_power_mape_percent': (
                (test_summary.get('one_step') or {}).get(
                    'total_power_mape_percent'
                )
            ),
            'test_maximum_false_safe_rate_percent': (
                test_summary.get('maximum_false_safe_rate_percent')
            ),
        }
        rows.append(row)
    return rows
