"""Pipeline module for document processing."""

from .entry_point import process_document
from .pipeline import DocumentPipeline

__all__ = [
    "DocumentPipeline",
    "process_document",
]
