"""I/O utilities for reading from and writing to various sources."""

from .exceptions import EmptyData, FileNotExisted, InvalidConfig, InvalidParams
from .readers import DataReader, FileBasedDataReader
from .schemas import PageInfo
from .writers import DataWriter, FileBasedDataWriter


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
