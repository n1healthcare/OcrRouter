"""Backend implementations for document processing.

This module provides backends for extracting structured content from PDF documents
using Vision Language Models (VLMs).

Available backends:
- MinerU: Uses the MinerU VLM for two-step extraction
- DotsOCR: Uses DotsOCR for one-step or two-step extraction
- DeepSeek: Uses DeepSeek-OCR with grounding mode

Usage:
    >>> from ocrrouter.backends import get_backend
    >>> from ocrrouter.config import Settings
    >>> settings = Settings(backend="mineru", openai_api_key="sk-...")
    >>> backend = get_backend("mineru", settings=settings)
    >>> middle_json, results = await backend.analyze(pdf_bytes, image_writer)
"""

from .factory import get_backend

# Re-export base classes for type hints and subclassing
from .models.base import (
    BaseModelBackend,
    BaseModelClient,
    BasePreprocessor,
    BasePostprocessor,
)

# Re-export common utilities
from .utils import (
    BlockType,
    BLOCK_TYPES,
    ContentBlock,
    SamplingParams,
    result_to_middle_json,
)

__all__ = [
    # Factory
    "get_backend",
    # Base classes
    "BaseModelBackend",
    "BaseModelClient",
    "BasePreprocessor",
    "BasePostprocessor",
    # Common utilities
    "BlockType",
    "BLOCK_TYPES",
    "ContentBlock",
    "SamplingParams",
    "result_to_middle_json",
]
