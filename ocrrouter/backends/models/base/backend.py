"""Base model backend interface for document processing."""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from ocrrouter.utils.io.writers import DataWriter
from ocrrouter.backends.utils import ContentBlock


class BaseModelBackend(ABC):
    """Abstract base class for model backends with standardized interface.

    This class merges the original Backend and BaseVLMBackend, providing:
    - Common initialization parameters
    - Standardized method signatures for document processing
    - Unified interface across all model backends (MinerU, DotsOCR, DeepSeek)

    All backend implementations must inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    async def analyze(
        self,
        pdf_bytes: bytes,
        image_writer: DataWriter | None = None,
        **kwargs: Any,
    ) -> tuple[dict, list]:
        """Analyze a PDF document and extract structured content.

        This is the main entry point for full document processing.

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            **kwargs: Additional backend-specific options.

        Returns:
            A tuple of (middle_json, model_results) where:
            - middle_json: Structured document representation
            - model_results: Raw inference outputs from the model
        """
        pass

    async def layout_detect(
        self,
        images: list[Image.Image],
    ) -> list[list[ContentBlock]]:
        """Layout detection only.

        Detects document structure (text blocks, tables, images, equations, etc.)
        without extracting content.

        Args:
            images: List of PIL images of document pages.

        Returns:
            List of ContentBlock lists (one per page), WITHOUT content
            (type, bbox, and angle only).
        """
        raise NotImplementedError("Subclasses should implement layout_detect()")

    async def content_extract(
        self,
        images: list[Image.Image],
        types: list[str] | str = "text",
    ) -> list[str | None]:
        """Simple OCR extraction (standalone, no ContentBlock wrapping).

        Extracts text content from pre-cropped images without layout detection.

        Args:
            images: List of pre-cropped PIL images.
            types: Content type(s) for extraction (text, table, equation).

        Returns:
            List of extracted text strings.
        """
        raise NotImplementedError("Subclasses should implement content_extract()")
