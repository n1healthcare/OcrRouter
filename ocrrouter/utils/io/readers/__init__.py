"""Data readers for various input sources."""

from .base import DataReader
from .local import FileBasedDataReader


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "HttpReader":
        from .http import HttpReader

        return HttpReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataReader",
    "FileBasedDataReader",
    "HttpReader",
]
