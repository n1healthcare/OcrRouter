"""HTTP data reader implementation.

This module provides an HTTP/HTTPS data reader for fetching remote files.
"""

from typing import Any

from .base import DataReader


class HttpReader(DataReader):
    """HTTP/HTTPS data reader for remote files.

    This reader fetches data from HTTP/HTTPS URLs.

    Args:
        base_url: Optional base URL to prepend to all paths.
        timeout: Request timeout in seconds.
        headers: Optional headers to include in requests.
        **kwargs: Additional configuration passed to requests.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: int = 30,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        self._kwargs = kwargs

    def _get_url(self, path: str) -> str:
        """Get the full URL for a path."""
        if path.startswith(("http://", "https://")):
            return path
        if self.base_url:
            return f"{self.base_url}/{path}"
        return path

    def read(self, path: str) -> bytes:
        """Read bytes from an HTTP URL.

        Args:
            path: URL or path (relative to base_url).

        Returns:
            The response content as bytes.
        """
        import requests

        url = self._get_url(path)
        response = requests.get(
            url,
            headers=self.headers,
            timeout=self.timeout,
            **self._kwargs,
        )
        response.raise_for_status()
        return response.content

    def read_string(self, path: str, encoding: str = "utf-8") -> str:
        """Read a string from an HTTP URL.

        Args:
            path: URL or path (relative to base_url).
            encoding: String encoding.

        Returns:
            The response content as a string.
        """
        return self.read(path).decode(encoding)


__all__ = ["HttpReader"]
