"""Data writers for various output destinations."""

from .base import DataWriter
from .local import FileBasedDataWriter


__all__ = [
    "DataWriter",
    "FileBasedDataWriter",
]
