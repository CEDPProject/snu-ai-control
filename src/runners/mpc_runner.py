from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform

from omegaconf import OmegaConf

from controllers.pipeline import run_mpc_pipeline
from data_provider.dataset_versions import dataset_label_from_cfg, versions_from_cfg
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC')


def _write_json(path: Path, value) -> None:
    """Write a small, auditable post-MPC artifact without NaN JSON values."""
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _run_counterfactual_input_ablation_if_enabled(cfg, result_dir: Path):
    """Run the three report-input cases against this run's frozen MPC actions.

    The counterfactual runner must execute only after the pipeline has written
    ``mpc_evaluation.json`` and its action-trace CSV.  It refits report models
    only; it never re-solves the MILP, so its three input cases remain directly
    comparable.
    """
    sweep_cfg = cfg.MPC.counterfactual_report_input_ablation
    if not bool(sweep_cfg.get('run_after_mpc', False)):
        LOGGER.info('Counterfactual report-input ablation disabled for this MPC run')
        return None

    from runners.counterfactual_report_input_ablation_runner import (
        run_counterfactual_report_input_ablation,
    )

    sweep_run_cfg = OmegaConf.merge(
        cfg,
        {
            'MPC': {
                'counterfactual_report_input_ablation': {
                    'reference_mpc_result_dir': str(result_dir),
                },
            },
        },
    )
    LOGGER.info(
        'Counterfactual report-input ablation started: reference=%s | '
        'cases=%s | MILP action trace is frozen',
        result_dir,
        [case.name for case in sweep_run_cfg.MPC.report_model_input_ablation.cases],
    )
    sweep_summary = run_counterfactual_report_input_ablation(sweep_run_cfg)
    sidecar = {
        'purpose': (
            'Compare all configured report-model input bundles using the same '
            'completed Full-guard MPC action trace. MILP is not re-solved.'
        ),
        'reference_mpc_result_directory': str(result_dir),
        'counterfactual_report_input_ablation_result_directory': sweep_summary[
            'result_directory'
        ],
        'summary_csv': str(
            Path(sweep_summary['result_directory'])
            / 'counterfactual_report_input_ablation_summary.csv'
        ),
        'case_names': sorted(sweep_summary['case_results']),
        'counterfactual_transition_count': sweep_summary['reference_mpc_run'][
            'counterfactual_transition_count'
        ],
        'mpc_action_trace_sha256': sweep_summary['reference_mpc_run'][
            'mpc_action_trace_sha256'
        ],
    }
    _write_json(
        result_dir / 'mpc_counterfactual_report_input_ablation_manifest.json',
        sidecar,
    )
    LOGGER.info(
        'Counterfactual report-input ablation completed: cases=%s | summary=%s',
        sidecar['case_names'],
        sidecar['summary_csv'],
    )
    return sidecar


def _result_directory(cfg) -> Path:
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'mpc'
        / cfg.DEFAULT.exp_name
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    log_path = result_dir / 'logs' / f'mpc_{timestamp}_{os.getpid()}.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _log_traceback_to_file(message: str) -> None:
    file_logger = logging.getLogger('MPC.file_only')
    file_logger.handlers.clear()
    file_logger.propagate = False
    file_logger.setLevel(logging.ERROR)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            file_logger.addHandler(handler)
    file_logger.exception(message)


def run_mpc(cfg):
    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(cfg, result_dir)

    LOGGER.info('MPC run started')
    LOGGER.info('Log file: %s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d',
        platform.python_version(),
        platform.platform(),
        os.getpid(),
    )
    LOGGER.info(
        'Experiment: data=%s | versions=%s | environment=%s | name=%s | evaluator=%s',
        dataset_label_from_cfg(cfg),
        list(versions_from_cfg(cfg)),
        cfg.DATASET.environment_name,
        cfg.DEFAULT.exp_name,
        cfg.MPC.independent_evaluator.model_family,
    )
    LOGGER.info(
        'MPC configuration: interval=%d minute(s) | prediction_horizon=%d | '
        'control_horizon=%d | closed_loop_steps=%d | max_windows=%d',
        int(cfg.MPC.sample_interval_minutes),
        int(cfg.MPC.prediction_horizon),
        int(cfg.MPC.control_horizon),
        int(cfg.MPC.evaluation.closed_loop_steps),
        int(cfg.MPC.evaluation.max_windows),
    )
    LOGGER.info(
        'Compute: affine models, default sklearn evaluators, and SciPy MILP use CPU; '
        'neural evaluators use their configured device'
    )

    try:
        summary = run_mpc_pipeline(cfg, result_dir)
        _run_counterfactual_input_ablation_if_enabled(cfg, result_dir)
    except KeyboardInterrupt:
        LOGGER.warning('MPC run interrupted by the user')
        raise
    except Exception as exc:
        LOGGER.error(
            'MPC run failed: %s: %s',
            type(exc).__name__,
            exc,
        )
        _log_traceback_to_file('MPC failure traceback')
        raise

    LOGGER.info(
        'MPC run completed: evaluated_windows=%d | artifacts=%s',
        int(summary['evaluated_windows']),
        result_dir,
    )
