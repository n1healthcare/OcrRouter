"""PP-DocLayoutV3 backend implementation for document layout detection.

This backend provides layout detection only - it does not perform OCR.
Use it with the composite backend to combine layout detection with
an OCR model (deepseek, glmocr, paddleocr, etc.).
"""

from typing import Any, Literal

from loguru import logger

from ocrrouter.backends.models.base import BaseModelBackend
from ocrrouter.backends.utils import result_to_middle_json
from ocrrouter.config import Settings
from ocrrouter.preprocessor.utils.pdf_image_tools import load_images_from_pdf
from ocrrouter.utils.enum_class import ImageType
from ocrrouter.utils.io.writers import DataWriter

from .client import PPDocLayoutClient


class PPDocLayoutBackend(BaseModelBackend):
    """PP-DocLayoutV3 backend for document layout detection.

    This backend uses the PP-DocLayoutV3 model from PaddlePaddle (via HuggingFace
    Transformers) to detect document layout regions. It is a **layout-only**
    backend - it detects bounding boxes for document elements but does not
    extract text content.

    Detected element types:
    - Text blocks (paragraphs, titles, abstracts)
    - Tables
    - Images and charts
    - Formulas (display and inline)
    - Headers, footers, page numbers
    - And more (25 classes total)

    For full document processing (layout + OCR), use the composite backend:
        >>> settings = Settings(
        ...     backend="composite",
        ...     layout_model="ppdoclayout",
        ...     ocr_model="glmocr",  # or deepseek, paddleocr, etc.
        ... )

    Example (layout detection only):
        >>> from ocrrouter import Settings
        >>> settings = Settings(backend="ppdoclayout")
        >>> backend = PPDocLayoutBackend(settings)
        >>> middle_json, results = await backend.analyze(
        ...     pdf_bytes, image_writer, output_mode="layout_only"
        ... )
    """

    def __init__(self, settings: Settings):
        """Initialize the PP-DocLayoutV3 backend.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings
        self._client = None

    @property
    def client(self) -> PPDocLayoutClient:
        """Get the PP-DocLayoutV3 client instance."""
        if self._client is None:
            self._client = PPDocLayoutClient(self._settings)
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
        """Analyze a PDF document for layout detection.

        This backend only supports layout detection. For OCR extraction,
        use the composite backend with an OCR model.

        Args:
            pdf_bytes: The PDF file content as bytes.
            image_writer: Writer for saving extracted images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            table_merge_enable: Whether cross-page table merging is enabled.
            output_mode: Output mode (only "layout_only" fully supported):
                - 'layout_only': Layout detection only (recommended)
                - 'all': Same as layout_only (no OCR capability)
                - 'ocr_only': Not supported
            **kwargs: Additional options.

        Returns:
            A tuple of (middle_json, model_results) where:
            - middle_json: Structured document representation with layout info
            - model_results: List of ContentBlock lists per page

        Raises:
            NotImplementedError: If output_mode is "ocr_only"
        """
        output_mode = output_mode or self._settings.output_mode

        if output_mode == "ocr_only":
            raise NotImplementedError(
                "PP-DocLayoutV3 backend does not support 'ocr_only' mode. "
                "Use the composite backend with an OCR model for full processing."
            )

        # Load images from PDF
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

        # Run layout detection
        logger.debug("Running PP-DocLayoutV3 layout detection")
        results = await self.client.aio_batch_layout_detect(images=images_pil_list)

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
            List of ContentBlock lists, one per page.
        """
        return await self.client.aio_batch_layout_detect(images=images_pil_list)

    async def content_extract(
        self,
        images_pil_list: list,
        types: list[str] | str = "text",
    ) -> list:
        """Extract content (not supported - layout only).

        Raises:
            NotImplementedError: PP-DocLayoutV3 is layout detection only.
        """
        raise NotImplementedError(
            "PP-DocLayoutV3 is a layout detection model only. "
            "Use the composite backend with an OCR model for content extraction."
        )
