"""Base interface for data writers."""

from abc import ABC, abstractmethod


class DataWriter(ABC):
    """Abstract base class for data writers."""

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write binary data to the path.

        Args:
            path (str): file path to write to
            data (bytes): the data to write
        """
        pass

    def write_string(self, path: str, data: str) -> None:
        """Write string data to the path.

        Args:
            path (str): file path to write to
            data (str): the string data to write
        """
        return self.write(path, data.encode("utf-8"))
