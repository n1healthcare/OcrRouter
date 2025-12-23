"""Local file system data reader."""

import os

from .base import DataReader


class FileBasedDataReader(DataReader):
    """File-based data reader for local filesystem."""

    def __init__(self, parent_dir: str = ""):
        """Initialize with parent directory.

        Args:
            parent_dir (str, optional): the parent directory for relative paths. Defaults to ''.
        """
        self._parent_dir = parent_dir

    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Read file at offset and limit.

        Args:
            path (str): the path of file, joined with parent_dir if relative.
            offset (int, optional): the number of bytes skipped. Defaults to 0.
            limit (int, optional): the length of bytes to read. Defaults to -1 (all).

        Returns:
            bytes: the content of file
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        with open(fn_path, "rb") as f:
            f.seek(offset)
            if limit == -1:
                return f.read()
            else:
                return f.read(limit)
