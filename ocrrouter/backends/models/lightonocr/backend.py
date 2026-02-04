"""LightOnOCR backend implementation for document processing."""

from typing import Any, Literal

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.utils.io.writers import DataWriter
from ocrrouter.preprocessor.utils.pdf_image_tools import load_images_from_pdf
from ocrrouter.utils.enum_class import ImageType
from ocrrouter.backends.models.base import BaseModelBackend
from ocrrouter.backends.utils import result_to_middle_json

from .client import LightOnOCRClient
from .utils import remove_image_markers


class LightOnOCRBackend(BaseModelBackend):
    """LightOnOCR backend for document analysis using Vision Language Models.

    This backend uses the LightOnOCR-2-1B-bbox-soup model which:
    1. Performs OCR and outputs text in markdown format
    2. Detects images and returns bounding boxes in format: ![image](image_N.png) x1,y1,x2,y2
    3. Coordinates are normalized to 0-1000 scale

    Output modes:
    - 'all': Returns both image blocks and a single text block with full OCR
    - 'layout_only': Returns only image blocks (no text content)
    - 'ocr_only': Returns only the text block with full OCR

    Example:
        >>> from ocrrouter import Settings
        >>> settings = Settings(openai_api_key="sk-...")
        >>> backend = LightOnOCRBackend(settings)
        >>> middle_json, results = await backend.analyze(pdf_bytes, image_writer)
    """

    def __init__(self, settings: Settings):
        """Initialize the LightOnOCR backend.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings
        self._client = None

    @property
    def client(self) -> LightOnOCRClient:
        """Get the LightOnOCR client instance (lazy-loaded)."""
        if self._client is None:
            self._client = LightOnOCRClient(self._settings)
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
        2. Runs the LightOnOCR model for OCR and image detection
        3. Converts the results to middle JSON format

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            formula_enable: Whether formula extraction is enabled (unused).
            table_enable: Whether table extraction is enabled (unused).
            table_merge_enable: Whether cross-page table merging is enabled.
            output_mode: Output mode controlling processing:
                - 'all': OCR + image detection (default behavior)
                - 'layout_only': Image detection only, no OCR text
                - 'ocr_only': Full-page OCR, no image detection
            **kwargs: Additional options.

        Returns:
            A tuple of (middle_json, model_results) where:
            - middle_json: Structured document representation
            - model_results: Raw inference outputs from the model
        """
        from ocrrouter.backends.utils import ContentBlock

        # Get output_mode from kwargs or settings
        output_mode = output_mode or self._settings.output_mode

        # Load images from PDF
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

        # Run LightOnOCR model based on output_mode
        if output_mode == "layout_only":
            # Layout detection only - get image bboxes only
            logger.debug("Running layout detection only (layout_only mode)")
            results = await self.client.aio_batch_layout_detect(
                images=images_pil_list,
                output_mode="layout_only",
            )

        elif output_mode == "ocr_only":
            # Full-page OCR - skip layout detection
            logger.debug("Running full-page OCR only (ocr_only mode)")
            ocr_texts = await self.client.aio_batch_content_extract(
                images=images_pil_list
            )
            # Convert OCR texts to full-page ContentBlocks
            # Strip image markers since LightOnOCR outputs them by default
            results = []
            for text in ocr_texts:
                if text:
                    # Remove embedded image markers from the text
                    cleaned_text = remove_image_markers(text)
                    if cleaned_text:
                        # Create a full-page text block
                        block = ContentBlock(
                            type="text",
                            bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                            content=cleaned_text,
                        )
                        results.append([block])
                    else:
                        results.append([])
                else:
                    results.append([])

        else:  # "all" mode - default behavior
            # Run LightOnOCR model for OCR with layout detection
            results = await self.client.aio_batch_ocr_with_layout(
                images=images_pil_list,
                output_mode="all",
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
        """Detect layout blocks (images) in document page images.

        Args:
            images_pil_list: List of PIL images of document pages.

        Returns:
            List of layout detection results for each page.
        """
        return await self.client.aio_batch_layout_detect(
            images=images_pil_list,
            output_mode="layout_only",
        )

    async def content_extract(
        self,
        images_pil_list: list,
        types: list[str] | str = "text",
    ) -> list:
        """Extract content from image regions.

        Args:
            images_pil_list: List of PIL images of content regions.
            types: Type(s) of content to extract (unused by LightOnOCR).

        Returns:
            List of extracted content strings.
        """
        return await self.client.aio_batch_content_extract(
            images=images_pil_list,
            types=types,
        )
