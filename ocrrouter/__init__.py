"""OCRRouter - PDF and image to Markdown conversion with advanced parsing capabilities.

This package provides tools for converting PDF and image files to Markdown,
with support for complex document elements like formulas, tables, and code blocks.

Example usage:
    ```python
    from ocrrouter import DocumentPipeline

    # Create a document processing pipeline with explicit configuration
    pipeline = DocumentPipeline(
        backend="deepseek",
        openai_base_url="https://api.example.com",
        openai_api_key="sk-...",
    )

    # Process a document
    result = pipeline.process("document.pdf", "output_dir")
    print(result["markdown"])
    ```

Example usage with Settings object:
    ```python
    from ocrrouter import DocumentPipeline, Settings

    # Create reusable settings
    settings = Settings(
        backend="mineru",
        openai_api_key="sk-...",
        max_concurrency=50,
    )

    # Create multiple pipelines with same settings
    pipeline1 = DocumentPipeline(settings=settings)
    pipeline2 = DocumentPipeline(settings=settings, max_concurrency=10)  # override
    ```

Example usage (Async):
    ```python
    from ocrrouter import DocumentPipeline

    pipeline = DocumentPipeline(backend="deepseek", openai_api_key="sk-...")
    result = await pipeline.aio_process("document.pdf", "output_dir")
    ```

Example usage (Direct backend access):
    ```python
    from ocrrouter import get_backend, Settings

    settings = Settings(openai_api_key="sk-...")
    backend = get_backend("mineru", settings=settings)
    middle_json, results = await backend.analyze(pdf_bytes, image_writer)
    ```
"""

from .version import __version__

# Core Pipeline API
from .pipeline import DocumentPipeline, process_document

# Backend factory
from .backends import get_backend

# Configuration
from .config import Settings

__all__ = [
    # Version
    "__version__",
    # Core Pipeline API
    "DocumentPipeline",
    "process_document",
    # Backend factory
    "get_backend",
    # Configuration
    "Settings",
]
