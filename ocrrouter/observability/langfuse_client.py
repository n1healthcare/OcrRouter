"""Langfuse client utilities.

This module provides utilities for Langfuse integration. The Langfuse client
should be created and owned by the parent application, then passed to
DocumentPipeline via the `langfuse` parameter.
"""

import uuid
from typing import Optional, Any

from loguru import logger


# Module-level client storage
_langfuse_client: Optional[Any] = None


def get_langfuse_client() -> Optional[Any]:
    """Get the currently configured Langfuse client.

    Returns None if no Langfuse client has been set.

    Returns:
        Langfuse client instance or None
    """
    return _langfuse_client


def set_langfuse_client(client: Optional[Any]) -> None:
    """Set the Langfuse client for observability.

    This is called internally by DocumentPipeline when a langfuse client
    is passed. The client should be created by the parent application.

    Args:
        client: Langfuse client instance, or None to disable tracing.
    """
    global _langfuse_client
    _langfuse_client = client
    if client is not None:
        logger.debug("Langfuse client set")


def get_langfuse_handler() -> Optional[Any]:
    """Get Langfuse CallbackHandler for LangChain integration.

    Creates a new CallbackHandler instance each time it's called.
    The handler will automatically nest under the currently active span.

    Returns:
        CallbackHandler instance if Langfuse client is available, None otherwise
    """
    client = get_langfuse_client()
    if not client:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as e:
        logger.warning("Failed to create Langfuse CallbackHandler: %s", e)
        return None


def generate_session_id(name: Optional[str] = None, prefix: str = "ocrRouter") -> str:
    """Generate a session ID for grouping traces.

    Args:
        name: Optional identifier (e.g., document name, batch ID)
        prefix: Prefix for the session ID (default: "ocrRouter")

    Returns:
        Session ID string

    Examples:
        - With name: "ocrRouter-document-report.pdf-a1b2c3d4"
        - Without name: "ocrRouter-a1b2c3d4"
    """
    short_uuid = str(uuid.uuid4())[:8]

    if name:
        # Clean name for session ID
        clean_name = name.replace(" ", "_").replace("/", "_")
        return f"{prefix}-{clean_name}-{short_uuid}"
    else:
        return f"{prefix}-{short_uuid}"
