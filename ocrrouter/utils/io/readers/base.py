"""Base interface for data readers."""

from abc import ABC, abstractmethod


class DataReader(ABC):
    """Abstract base class for data readers."""

    def read(self, path: str) -> bytes:
        """Read the file.

        Args:
            path (str): file path to read

        Returns:
            bytes: the content of the file
        """
        return self.read_at(path)

    @abstractmethod
    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Read the file at offset and limit.

        Args:
            path (str): the file path
            offset (int, optional): the number of bytes skipped. Defaults to 0.
            limit (int, optional): the length of bytes want to read. Defaults to -1.

        Returns:
            bytes: the content of the file
        """
        pass
