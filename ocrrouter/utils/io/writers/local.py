"""Local file system data writer."""

import os

from .base import DataWriter


class FileBasedDataWriter(DataWriter):
    """File-based data writer for local filesystem."""

    def __init__(self, parent_dir: str = "") -> None:
        """Initialize with parent directory.

        Args:
            parent_dir (str, optional): the parent directory for relative paths. Defaults to ''.
        """
        self._parent_dir = parent_dir

    def write(self, path: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, joined with parent_dir if relative.
            data (bytes): the data to write
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        dir_name = os.path.dirname(fn_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(fn_path, "wb") as f:
            f.write(data)
