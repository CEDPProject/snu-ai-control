"""Compatibility entry point for existing forecast callers."""

from .prediction.factory import create_model

__all__ = ['create_model']
