"""S3 data writer implementation.

This module provides an S3-compatible data writer for cloud storage.
"""

from typing import Any

from .base import DataWriter


class S3DataWriter(DataWriter):
    """S3-compatible data writer for cloud storage.

    This writer uploads data to S3-compatible object storage services
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

    def write(self, path: str, data: bytes) -> str:
        """Write bytes to S3.

        Args:
            path: Object key (relative to prefix).
            data: Bytes to write.

        Returns:
            The full S3 URI of the written object.
        """
        key = self._get_key(path)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def write_string(self, path: str, data: str, encoding: str = "utf-8") -> str:
        """Write a string to S3.

        Args:
            path: Object key (relative to prefix).
            data: String to write.
            encoding: String encoding.

        Returns:
            The full S3 URI of the written object.
        """
        return self.write(path, data.encode(encoding))


__all__ = ["S3DataWriter"]
