"""Power forecasting models; deliberately independent of control dynamics."""

from .factory import create_model

__all__ = ['create_model']
