"""Factory for the independent plant evaluator selected by MPC configuration."""

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from omegaconf import OmegaConf

from dynamics.neural_evaluators import (
    ForecastingSequenceEvaluator,
    NeuralResidualEvaluator,
)
from dynamics.nonlinear import IndependentTreeEvaluator
from dynamics.sklearn_evaluators import IndependentSklearnEvaluator


SKLEARN_FAMILIES = {
    'extra_trees_residual_state',
    'random_forest_residual_state',
    'hist_gradient_boosting_residual_state',
}


def _mapping(value) -> dict:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _section(configuration, name: str) -> dict:
    if configuration is None:
        return {}
    if OmegaConf.is_config(configuration):
        return _mapping(configuration.get(name))
    return _mapping(configuration.get(name))


def _model_settings(path: Optional[str] = None) -> Mapping:
    settings_path = (
        Path(path).expanduser().resolve()
        if path
        else Path(__file__).resolve().parents[1] / 'configs' / 'model_setting.yaml'
    )
    if not settings_path.is_file():
        raise FileNotFoundError(f'Forecast model settings not found: {settings_path}')
    return OmegaConf.load(settings_path)


def available_evaluator_families(model_setting_path: Optional[str] = None):
    forecasting_models = tuple(_model_settings(model_setting_path).keys())
    return {
        'sklearn': sorted(SKLEARN_FAMILIES),
        'neural_residual': ['neural_residual'],
        'forecasting_sequence': list(forecasting_models),
        'forecasting_selector_examples': [
            'forecasting_sequence',
            *[f'forecasting:{name}' for name in forecasting_models],
        ],
    }


def create_independent_evaluator(
    evaluator_cfg,
    data,
    training_end_exclusive: int,
    *,
    random_state: int,
    model_family: Optional[str] = None,
    training_rows=None,
):
    """Fit an evaluator on a chronological prefix or explicit transition rows.

    ``training_rows`` is reserved for the outcome-blind action-episode holdout.
    The ordinary MPC model validation path continues to use the chronological
    boundary, so its temporal semantics are unchanged.
    """

    family = str(model_family or evaluator_cfg.model_family)
    explicit_rows = (
        None
        if training_rows is None
        else data.valid_transition_rows(training_rows)
    )
    if explicit_rows is not None and not len(explicit_rows):
        raise ValueError('Explicit evaluator training rows contain no transitions')
    training = (
        data.model_arrays(0, int(training_end_exclusive) - 1)
        if explicit_rows is None
        else data.model_arrays_for_rows(explicit_rows)
    )
    names = {
        'state_names': data.state_names,
        'control_feature_names': data.encoder.feature_names,
        'disturbance_names': data.disturbance_names,
        'power_names': data.power_names,
    }

    if family == 'extra_trees_residual_state':
        return IndependentTreeEvaluator.fit(
            *training,
            **names,
            n_estimators=int(evaluator_cfg.n_estimators),
            min_samples_leaf=int(evaluator_cfg.min_samples_leaf),
            max_features=float(evaluator_cfg.max_features),
            n_jobs=int(evaluator_cfg.n_jobs),
            random_state=int(random_state),
        )

    if family in SKLEARN_FAMILIES:
        return IndependentSklearnEvaluator.fit(
            *training,
            **names,
            model_family=family,
            n_estimators=int(evaluator_cfg.n_estimators),
            min_samples_leaf=int(evaluator_cfg.min_samples_leaf),
            max_features=float(evaluator_cfg.max_features),
            max_iter=int(evaluator_cfg.get('max_iter', 100)),
            n_jobs=int(evaluator_cfg.n_jobs),
            random_state=int(random_state),
        )

    if family == 'neural_residual':
        parameters = _section(evaluator_cfg, 'neural_residual')
        return NeuralResidualEvaluator.fit(
            *training,
            **names,
            random_state=int(random_state),
            **parameters,
        )

    settings_path = evaluator_cfg.get('model_setting_path')
    settings = _model_settings(settings_path)
    if family.startswith('forecasting:'):
        model_name = family.split(':', 1)[1]
    elif family == 'forecasting_sequence':
        model_name = str(evaluator_cfg.forecasting.model_name)
    elif family in settings:
        model_name = family
    else:
        available = available_evaluator_families(settings_path)
        raise ValueError(
            f'Unknown evaluator model_family {family!r}. Available: {available}'
        )
    if model_name not in settings:
        raise ValueError(
            f'Forecast evaluator model {model_name!r} is absent from model settings'
        )
    forecasting = _section(evaluator_cfg, 'forecasting')
    configured_overrides = _mapping(forecasting.pop('model_parameters', {}))
    model_parameters = OmegaConf.to_container(
        OmegaConf.merge(settings[model_name], configured_overrides),
        resolve=True,
    )
    forecasting.pop('model_name', None)
    return ForecastingSequenceEvaluator.fit(
        data,
        0,
        (
            int(training_end_exclusive) - 1
            if explicit_rows is None
            else len(data.states) - 1
        ),
        model_name=model_name,
        model_parameters=model_parameters,
        random_state=int(random_state),
        eligible_endpoints=(
            None
            if explicit_rows is None
            else np.asarray(explicit_rows, dtype=int)
        ),
        **forecasting,
    )


def evaluator_label(evaluator) -> str:
    metadata = evaluator.to_metadata()
    family = str(metadata['model_family'])
    model_name = metadata.get('forecasting_model_name')
    return f'{family}:{model_name}' if model_name else family
