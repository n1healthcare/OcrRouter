"""Data writers for various output destinations."""

from .base import DataWriter
from .local import FileBasedDataWriter


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "S3DataWriter":
        from .s3 import S3DataWriter

        return S3DataWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataWriter",
    "FileBasedDataWriter",
    "S3DataWriter",
]
