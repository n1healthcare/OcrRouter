"""Observability module for OcrRouter using Langfuse."""

from .langfuse_client import (
    set_langfuse_client,
    generate_session_id,
    get_langfuse_client,
    get_langfuse_handler,
)
from langfuse import observe

__all__ = [
    "set_langfuse_client",
    "get_langfuse_client",
    "get_langfuse_handler",
    "generate_session_id",
    "observe",
]
