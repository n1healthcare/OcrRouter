"""Composite backend implementation for document processing."""

from typing import Any, Literal

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.utils.io.writers import DataWriter
from ocrrouter.preprocessor.utils.pdf_image_tools import load_images_from_pdf
from ocrrouter.utils.enum_class import ImageType
from ocrrouter.backends.models.base import BaseModelBackend
from ocrrouter.backends.utils import result_to_middle_json, ContentBlock

from .client import CompositeClient


class CompositeBackend(BaseModelBackend):
    """Composite backend for document analysis using mix-and-match VLM models.

    This backend enables using different models for layout detection and OCR:
    - Layout detection from: mineru, deepseek, or dotsocr
    - OCR extraction from: mineru, deepseek, or dotsocr

    Supported combinations:
    | Layout Model | OCR Model |
    |--------------|-----------|
    | mineru       | deepseek  |
    | mineru       | dotsocr   |
    | deepseek     | mineru    |
    | deepseek     | dotsocr   |
    | dotsocr      | mineru    |
    | dotsocr      | deepseek  |

    Example:
        >>> from ocrrouter import Settings
        >>> settings = Settings(layout_model="mineru", ocr_model="deepseek", ...)
        >>> backend = CompositeBackend(settings)
        >>> middle_json, results = await backend.analyze(pdf_bytes, image_writer)
    """

    def __init__(self, settings: Settings):
        """Initialize the Composite backend.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings
        self._client = None

    @property
    def client(self) -> CompositeClient:
        """Get the Composite client instance."""
        if self._client is None:
            self._client = CompositeClient(self._settings)
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
        2. Runs layout detection using the configured layout_model
        3. Runs OCR extraction using the configured ocr_model
        4. Converts the results to middle JSON format

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            table_merge_enable: Whether cross-page table merging is enabled.
            output_mode: Output mode controlling processing:
                - 'all': Layout + OCR (default behavior)
                - 'layout_only': Layout detection only, no OCR
                - 'ocr_only': Full-page OCR, no layout detection
            **kwargs: Additional options.

        Returns:
            A tuple of (middle_json, model_results) where:
            - middle_json: Structured document representation
            - model_results: Raw inference outputs from the model
        """
        # Get output_mode from kwargs or settings
        output_mode = output_mode or self._settings.output_mode

        # Load images from PDF
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

        # Log the configuration
        layout_model = self._settings.layout_model
        ocr_model = self._settings.ocr_model
        logger.debug(
            f"Composite backend: layout_model={layout_model}, ocr_model={ocr_model}"
        )

        # Run based on output_mode
        if output_mode == "layout_only":
            # Layout detection only - skip OCR
            logger.debug("Running layout detection only (layout_only mode)")
            results = await self.client.aio_batch_layout_detect(images=images_pil_list)

        elif output_mode == "ocr_only":
            # Full-page OCR - skip layout detection
            logger.debug("Running full-page OCR only (ocr_only mode)")
            ocr_texts = await self.client.aio_batch_content_extract(
                images=images_pil_list
            )
            # Convert OCR texts to full-page ContentBlocks
            results = []
            for text in ocr_texts:
                if text:
                    # Create a full-page text block
                    block = ContentBlock(
                        type="text",
                        bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                        content=text,
                    )
                    results.append([block])
                else:
                    results.append([])

        else:  # "all" mode - default behavior
            # Run composite two-step extraction
            logger.debug(
                f"Running composite two-step extraction (layout={layout_model}, ocr={ocr_model})"
            )
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

        Uses the configured layout_model for detection.

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

        Uses the configured ocr_model for extraction.

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
