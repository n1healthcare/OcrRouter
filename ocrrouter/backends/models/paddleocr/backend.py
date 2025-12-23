"""PaddleOCR backend implementation for document processing."""

from typing import Any, Literal

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.utils.io.writers import DataWriter
from ocrrouter.preprocessor.utils.pdf_image_tools import load_images_from_pdf
from ocrrouter.utils.enum_class import ImageType
from ocrrouter.backends.models.base import BaseModelBackend
from ocrrouter.backends.utils import result_to_middle_json

from .client import PaddleOCRClient


class PaddleOCRBackend(BaseModelBackend):
    """PaddleOCR backend for document analysis using Vision Language Models.

    IMPORTANT: This backend only supports 'ocr_only' mode.
    It does NOT support layout detection.

    For layout + OCR, use CompositeBackend with:
    - layout_model="mineru" (or "deepseek", "dotsocr")
    - ocr_model="paddleocr"

    Example:
        >>> from ocrrouter import Settings
        >>> settings = Settings(openai_api_key="sk-...", output_mode="ocr_only")
        >>> backend = PaddleOCRBackend(settings)
        >>> middle_json, results = await backend.analyze(pdf_bytes)
    """

    def __init__(self, settings: Settings):
        """Initialize the PaddleOCR backend.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings
        self._client = None

    @property
    def client(self) -> PaddleOCRClient:
        """Get the PaddleOCR client instance."""
        if self._client is None:
            self._client = PaddleOCRClient(self._settings)
        return self._client

    async def analyze(
        self,
        pdf_bytes: bytes,
        image_writer: DataWriter | None = None,
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
        table_merge_enable: bool | None = None,
        output_mode: Literal["all", "layout_only", "ocr_only"] | None = None,
        **kwargs: Any,
    ) -> tuple[dict, list]:
        """Analyze a PDF document and extract structured content.

        IMPORTANT: Only 'ocr_only' mode is supported.

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            table_merge_enable: Whether cross-page table merging is enabled.
            output_mode: Output mode. Only 'ocr_only' is supported.
            **kwargs: Additional options.

        Returns:
            A tuple of (middle_json, model_results).

        Raises:
            ValueError: If output_mode is 'layout_only' or 'all'.
        """
        # Get output_mode from kwargs or settings
        output_mode = output_mode or self._settings.output_mode

        # Validate output_mode - PaddleOCR only supports OCR
        if output_mode == "layout_only":
            raise ValueError(
                "PaddleOCR does not support 'layout_only' mode. "
                "Use 'ocr_only' or use CompositeBackend with a different layout_model."
            )
        if output_mode == "all":
            raise ValueError(
                "PaddleOCR does not support 'all' mode (layout + OCR). "
                "Use 'ocr_only' or use CompositeBackend with layout_model and ocr_model='paddleocr'."
            )

        # Load images from PDF
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

        # Run full-page OCR with content block parsing
        logger.debug("Running PaddleOCR full-page OCR (ocr_only mode)")
        results = await self.client.aio_batch_full_page_ocr(
            images=images_pil_list,
            page_images=images_pil_list,
        )

        # Resolve table_merge_enable from settings if not provided
        if table_merge_enable is None:
            table_merge_enable = self._settings.table_merge_enable

        # Convert to middle JSON format
        middle_json = result_to_middle_json(
            results,
            images_list,
            pdf_doc,
            image_writer,
            formula_enable=formula_enable,
            table_enable=table_enable,
            table_merge_enable=table_merge_enable,
        )

        return middle_json, results

    async def layout_detect(self, images_pil_list: list) -> list:
        """Detect layout blocks in document page images.

        NOT SUPPORTED by PaddleOCR.

        Raises:
            NotImplementedError: Always, as PaddleOCR only supports OCR.
        """
        raise NotImplementedError(
            "PaddleOCR does not support layout detection. "
            "Use a different backend (mineru, deepseek, dotsocr) for layout detection."
        )

    async def content_extract(
        self,
        images_pil_list: list,
        types: list[str] | str = "text",
    ) -> list:
        """Extract content from image regions.

        Args:
            images_pil_list: List of PIL images of content regions.
            types: Type(s) of content to extract.

        Returns:
            List of extracted content strings.
        """
        return await self.client.aio_batch_content_extract(
            images=images_pil_list,
            types=types,
        )
