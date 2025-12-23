"""DeepSeek-OCR backend for document processing using Vision Language Models."""

from .backend import DeepSeekBackend
from .client import DeepSeekClient

__all__ = ["DeepSeekBackend", "DeepSeekClient"]
