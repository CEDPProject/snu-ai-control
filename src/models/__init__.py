"""Separate prediction/control namespaces, with legacy forecast compatibility."""

from importlib import import_module
from .prediction.factory import MODEL_NAMES, create_model


def __getattr__(name):
    if name in MODEL_NAMES:
        return import_module(f'models.prediction.{name}').Model
    raise AttributeError(name)


__all__ = ['create_model', *MODEL_NAMES]
