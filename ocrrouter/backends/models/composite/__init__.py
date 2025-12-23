"""Composite backend for mix-and-match layout detection and OCR."""

from .backend import CompositeBackend
from .client import CompositeClient

__all__ = [
    "CompositeBackend",
    "CompositeClient",
]
