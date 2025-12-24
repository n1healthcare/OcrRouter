"""MinerU backend implementation for document processing."""

from typing import Any, Literal

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.utils.io.writers import DataWriter
from ocrrouter.preprocessor.utils.pdf_image_tools import load_images_from_pdf
from ocrrouter.utils.enum_class import ImageType
from ocrrouter.backends.models.base import BaseModelBackend
from ocrrouter.backends.utils import result_to_middle_json

from .client import MinerUClient


class MinerUBackend(BaseModelBackend):
    """MinerU backend for document analysis using Vision Language Models.

    This backend uses the MinerU VLM client to:
    1. Detect document layout (text, tables, equations, images)
    2. Extract content from each detected region
    3. Convert results to structured JSON format

    Example:
        >>> from ocrrouter import Settings
        >>> settings = Settings(openai_api_key="sk-...")
        >>> backend = MinerUBackend(settings)
        >>> middle_json, results = await backend.analyze(pdf_bytes, image_writer)
    """

    def __init__(self, settings: Settings):
        """Initialize the MinerU backend.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings
        self._client = None

    @property
    def client(self) -> MinerUClient:
        """Get the MinerU client instance."""
        if self._client is None:
            self._client = MinerUClient(self._settings)
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

        This is the main entry point for document processing. It:
        1. Loads images from the PDF
        2. Runs the VLM model for layout detection and content extraction
        3. Converts the results to middle JSON format

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            table_merge_enable: Whether cross-page table merging is enabled.
            output_mode: Output mode controlling processing:
                - 'all': Layout + OCR (default behavior)
                - 'layout_only': Layout detection only, no OCR
                - 'ocr_only': NOT SUPPORTED - MinerU does not support full-page OCR
            **kwargs: Additional options.

        Returns:
            A tuple of (middle_json, model_results) where:
            - middle_json: Structured document representation
            - model_results: Raw inference outputs from the model

        Raises:
            ValueError: If output_mode is 'ocr_only' (not supported by MinerU backend).
        """
        # Get output_mode from kwargs or settings
        output_mode = output_mode or self._settings.output_mode

        # Validate output_mode - MinerU does not support ocr_only
        if output_mode == "ocr_only":
            raise ValueError(
                "MinerU backend does not support 'ocr_only' mode. "
                "MinerU requires layout detection for content extraction. "
                "Use 'all' or 'layout_only' mode, or switch to deepseek/dotsocr backend."
            )

        # Load images from PDF
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

        # Run VLM model based on output_mode
        if output_mode == "layout_only":
            # Layout detection only - skip OCR
            logger.debug("Running layout detection only (layout_only mode)")
            results = await self.client.aio_batch_layout_detect(images=images_pil_list)
        else:  # "all" mode - default behavior
            # Run VLM model for layout detection and content extraction
            results = await self.client.aio_batch_two_step_extract(
                images=images_pil_list
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

        Args:
            images_pil_list: List of PIL images of document pages.

        Returns:
            List of layout detection results for each page.
        """
        return await self.client.aio_batch_layout_detect(images=images_pil_list)

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
