"""Observability module for OcrRouter using Langfuse."""

from langfuse import observe

from .langfuse_client import (
    generate_session_id,
    get_langfuse_client,
    get_langfuse_handler,
    set_langfuse_client,
)

__all__ = [
    "set_langfuse_client",
    "get_langfuse_client",
    "get_langfuse_handler",
    "generate_session_id",
    "observe",
]
