"""I/O utilities for reading from and writing to various sources."""

from .readers import DataReader, FileBasedDataReader
from .writers import DataWriter, FileBasedDataWriter
from .exceptions import FileNotExisted, InvalidConfig, InvalidParams, EmptyData
from .schemas import S3Config, PageInfo


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "S3DataReader":
        from .readers.s3 import S3DataReader

        return S3DataReader
    elif name == "HttpReader":
        from .readers.http import HttpReader

        return HttpReader
    elif name == "S3DataWriter":
        from .writers.s3 import S3DataWriter

        return S3DataWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataReader",
    "FileBasedDataReader",
    "DataWriter",
    "FileBasedDataWriter",
    "S3DataReader",
    "HttpReader",
    "S3DataWriter",
    "FileNotExisted",
    "InvalidConfig",
    "InvalidParams",
    "EmptyData",
    "S3Config",
    "PageInfo",
]
