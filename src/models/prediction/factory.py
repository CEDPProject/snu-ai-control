"""Lazy, explicit registry for the historical forecasting task."""

from importlib import import_module

MODEL_NAMES = (
    'DLinear', 'PatchTST', 'TimeXer', 'iTransformer', 'MICN',
    'TimesNet', 'TSMixer', 'FreTS', 'Koopa', 'TimeMixer',
)


def create_model(modelname: str, params: dict | None = None):
    if modelname not in MODEL_NAMES:
        raise ValueError(f'Unknown prediction model {modelname!r}: {MODEL_NAMES}')
    return import_module(f'models.prediction.{modelname}').Model(
        {} if params is None else params
    )
