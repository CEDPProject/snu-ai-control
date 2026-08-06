from datetime import datetime
import logging
import os
from pathlib import Path
import platform

from controllers.pipeline import run_mpc_pipeline
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC')


def _result_directory(cfg) -> Path:
    return (
        Path(cfg.RESULT.savedir)
        / cfg.DATASET.data_version
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
        'Experiment: data=%s | environment=%s | name=%s | evaluator=%s',
        cfg.DATASET.data_version,
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
