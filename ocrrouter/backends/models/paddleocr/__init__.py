"""PaddleOCR backend for document processing using Vision Language Models."""

from .backend import PaddleOCRBackend
from .client import PaddleOCRClient

__all__ = ["PaddleOCRBackend", "PaddleOCRClient"]
