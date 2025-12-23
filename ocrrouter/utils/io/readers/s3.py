"""S3 data reader implementation.

This module provides an S3-compatible data reader for cloud storage.
"""

from typing import Any

from .base import DataReader


class S3DataReader(DataReader):
    """S3-compatible data reader for cloud storage.

    This reader downloads data from S3-compatible object storage services
    (AWS S3, MinIO, etc.).

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for all objects.
        client: Optional pre-configured S3 client.
        **kwargs: Additional configuration passed to boto3.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any = None,
        **kwargs: Any,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client = client
        self._kwargs = kwargs

    @property
    def client(self):
        """Get or create the S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", **self._kwargs)
        return self._client

    def _get_key(self, path: str) -> str:
        """Get the full S3 key for a path."""
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def read(self, path: str) -> bytes:
        """Read bytes from S3.

        Args:
            path: Object key (relative to prefix).

        Returns:
            The object content as bytes.
        """
        key = self._get_key(path)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def read_string(self, path: str, encoding: str = "utf-8") -> str:
        """Read a string from S3.

        Args:
            path: Object key (relative to prefix).
            encoding: String encoding.

        Returns:
            The object content as a string.
        """
        return self.read(path).decode(encoding)


__all__ = ["S3DataReader"]
