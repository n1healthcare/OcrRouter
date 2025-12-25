"""Simple entry point for document processing."""

from typing import Any

from ocrrouter.config import Settings
from .pipeline import DocumentPipeline


def process_document(
    input_path: str,
    output_dir: str | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> dict:
    """Process a document through the pipeline.

    Args:
        input_path: Path to the input PDF or image file.
        output_dir: Directory for output files. If None, a temporary
            directory is used and only images are saved (for downstream use).
            Other output files (markdown, JSON, etc.) are skipped.
        settings: Optional Settings object with configuration.
        **overrides: Configuration overrides (backend, openai_api_key, etc.).

    Returns:
        dict: Processing results containing markdown, middle_json, etc.

    Example:
        >>> from ocrrouter import process_document
        >>> # Development: just get the result (temp directory)
        >>> result = process_document("document.pdf")
        >>> print(result["markdown"])
        >>>
        >>> # Production: persist to disk
        >>> result = process_document(
        ...     "document.pdf",
        ...     "output/",
        ...     backend="deepseek",
        ...     openai_api_key="sk-...",
        ... )
    """
    pipeline = DocumentPipeline(settings=settings, **overrides)
    return pipeline.process(input_path, output_dir)


__all__ = [
    "process_document",
    "DocumentPipeline",
]
