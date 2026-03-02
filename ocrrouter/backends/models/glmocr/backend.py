"""GLM-OCR backend implementation for document processing."""

from typing import Any, Literal

from loguru import logger

from ocrrouter.backends.models.base import BaseModelBackend
from ocrrouter.backends.utils import result_to_middle_json
from ocrrouter.config import Settings
from ocrrouter.preprocessor.utils.pdf_image_tools import load_images_from_pdf
from ocrrouter.utils.enum_class import ImageType
from ocrrouter.utils.io.writers import DataWriter

from .client import GlmOCRClient


class GlmOCRBackend(BaseModelBackend):
    """GLM-OCR backend for document analysis using Vision Language Models.

    GLM-OCR is a multimodal OCR model from Zhipu AI built on the GLM-V architecture.
    It uses Multi-Token Prediction (MTP) and supports efficient inference via vLLM/SGLang.

    Unlike DeepSeek-OCR which uses grounding mode for layout detection,
    GLM-OCR returns structured markdown or JSON directly. For layout-aware
    processing, it can be combined with PP-DocLayoutV3 for region detection.

    This backend operates in direct OCR mode (no layout detection), where the
    entire page is sent to the VLM for recognition.

    Example:
        >>> from ocrrouter import Settings
        >>> settings = Settings(openai_api_key="sk-...")
        >>> backend = GlmOCRBackend(settings)
        >>> middle_json, results = await backend.analyze(pdf_bytes, image_writer)
    """

    def __init__(self, settings: Settings):
        """Initialize the GLM-OCR backend.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings
        self._client = None

    @property
    def client(self) -> GlmOCRClient:
        """Get the GLM-OCR client instance."""
        if self._client is None:
            self._client = GlmOCRClient(self._settings)
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
        2. Runs the GLM-OCR model for content extraction
        3. Converts the results to middle JSON format

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            table_merge_enable: Whether cross-page table merging is enabled.
            output_mode: Output mode controlling processing:
                - 'all': Full-page OCR (default behavior for GLM-OCR)
                - 'layout_only': Not supported (GLM-OCR doesn't have native layout detection)
                - 'ocr_only': Full-page OCR, same as 'all' for this backend
            **kwargs: Additional options.

        Returns:
            A tuple of (middle_json, model_results) where:
            - middle_json: Structured document representation
            - model_results: Raw inference outputs from the model
        """
        from ocrrouter.backends.utils import ContentBlock

        # Get output_mode from kwargs or settings
        output_mode = output_mode or self._settings.output_mode

        # GLM-OCR doesn't support layout_only mode natively
        if output_mode == "layout_only":
            logger.warning(
                "GLM-OCR backend does not support 'layout_only' mode. "
                "Use composite backend with a layout model if layout detection is needed."
            )
            # Return empty results for layout_only
            images_list, pdf_doc = load_images_from_pdf(
                pdf_bytes, image_type=ImageType.PIL
            )
            results = [[] for _ in images_list]
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

        # Load images from PDF
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

        # Run GLM-OCR model for full-page OCR
        logger.debug("Running GLM-OCR full-page extraction")
        ocr_texts = await self.client.aio_batch_content_extract(images=images_pil_list)

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

    async def content_extract(
        self,
        images_pil_list: list,
        types: list[str] | str = "text",
    ) -> list:
        """Extract content from image regions.

        Args:
            images_pil_list: List of PIL images of content regions.
            types: Type(s) of content to extract (text, table, formula).

        Returns:
            List of extracted content strings.
        """
        return await self.client.aio_batch_content_extract(
            images=images_pil_list,
            types=types,
        )
