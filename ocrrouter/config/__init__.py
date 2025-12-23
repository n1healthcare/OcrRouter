"""Centralized configuration module for OcrRouter.

This module provides a unified interface for managing application settings,
including VLM configuration, processing options, and output modes.

Example usage:
    ```python
    from ocrrouter.config import Settings

    settings = Settings(
        backend="deepseek",
        openai_api_key="sk-...",
    )
    ```
"""

from .settings import Settings

__all__ = [
    "Settings",
]
