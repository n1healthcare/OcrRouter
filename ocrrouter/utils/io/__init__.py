"""I/O utilities for reading from and writing to various sources."""

from .readers import DataReader, FileBasedDataReader
from .writers import DataWriter, FileBasedDataWriter
from .exceptions import FileNotExisted, InvalidConfig, InvalidParams, EmptyData
from .schemas import PageInfo


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "HttpReader":
        from .readers.http import HttpReader

        return HttpReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataReader",
    "FileBasedDataReader",
    "DataWriter",
    "FileBasedDataWriter",
    "HttpReader",
    "FileNotExisted",
    "InvalidConfig",
    "InvalidParams",
    "EmptyData",
    "PageInfo",
]
