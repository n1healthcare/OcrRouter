"""Composite client for mix-and-match layout detection and OCR."""

import asyncio
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client
from ocrrouter.backends.utils import (
    ContentBlock,
    gather_tasks,
    get_rgb_image,
)


class CompositeClient:
    """Composite client that combines layout detection from one model with OCR from another.

    This client enables mix-and-match combinations of different VLM models:
    - Use one model's layout detection capabilities
    - Use another model's OCR extraction capabilities

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
        >>> client = CompositeClient(layout_model="mineru", ocr_model="deepseek")
        >>> blocks = await client.aio_two_step_extract(image)
    """

    def __init__(
        self,
        settings: Settings,
        layout_model: Literal["mineru", "deepseek", "dotsocr"] | None = None,
        ocr_model: Literal["mineru", "deepseek", "dotsocr", "paddleocr", "generalvlm"]
        | None = None,
        executor: Executor | None = None,
        use_tqdm: bool | None = None,
    ) -> None:
        """Initialize the composite client.

        Args:
            settings: Settings object with configuration.
            layout_model: Model to use for layout detection. Defaults to settings.layout_model.
            ocr_model: Model to use for OCR extraction. Defaults to settings.ocr_model.
            executor: Executor for CPU-bound operations.
            use_tqdm: Whether to show progress bars.
        """
        self._settings = settings

        # Use settings.use_tqdm if use_tqdm is not explicitly provided
        if use_tqdm is None:
            use_tqdm = settings.use_tqdm

        # Use settings values if not explicitly provided
        layout_model = (
            layout_model if layout_model is not None else settings.layout_model
        )
        ocr_model = ocr_model if ocr_model is not None else settings.ocr_model

        self.layout_model = layout_model
        self.ocr_model = ocr_model
        self.max_concurrency = settings.max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = settings.debug

        # Create layout client
        self.layout_client = self._create_client(
            settings=settings,
            model_type=layout_model,
            executor=executor,
            use_tqdm=use_tqdm,
        )

        # Create OCR client
        self.ocr_client = self._create_client(
            settings=settings,
            model_type=ocr_model,
            executor=executor,
            use_tqdm=use_tqdm,
        )

    def _create_client(
        self,
        settings: Settings,
        model_type: str,
        executor: Executor | None,
        use_tqdm: bool,
    ):
        """Create a client instance based on model type.

        Args:
            settings: Settings object with configuration.
            model_type: Type of model ("mineru", "deepseek", "dotsocr", "paddleocr", "generalvlm")
            executor: Executor for CPU-bound operations.
            use_tqdm: Whether to show progress bars.

        Returns:
            Client instance for the specified model
        """
        if model_type == "mineru":
            from ocrrouter.backends.models.mineru.client import MinerUClient

            return MinerUClient(
                settings=settings,
                executor=executor,
                use_tqdm=use_tqdm,
            )
        elif model_type == "deepseek":
            from ocrrouter.backends.models.deepseek.client import DeepSeekClient

            return DeepSeekClient(
                settings=settings,
                executor=executor,
                use_tqdm=use_tqdm,
            )
        elif model_type == "dotsocr":
            from ocrrouter.backends.models.dotsocr.client import DotsOCRClient

            return DotsOCRClient(
                settings=settings,
                executor=executor,
                use_tqdm=use_tqdm,
            )
        elif model_type == "paddleocr":
            from ocrrouter.backends.models.paddleocr.client import PaddleOCRClient

            return PaddleOCRClient(
                settings=settings,
                executor=executor,
                use_tqdm=use_tqdm,
            )
        elif model_type == "generalvlm":
            from ocrrouter.backends.models.generalvlm.client import GeneralVLMClient

            return GeneralVLMClient(
                settings=settings,
                executor=executor,
                use_tqdm=use_tqdm,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Detect layout blocks in a document page image.

        Uses the layout_client for detection.

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with type and bbox (content set to None)
        """
        return await self.layout_client.aio_layout_detect(image, priority, semaphore)

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection for multiple images.

        Args:
            images: List of PIL Images
            priority: Priority value(s) for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock lists, one per image
        """
        return await self.layout_client.aio_batch_layout_detect(
            images, priority, semaphore
        )

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Extract content from an image region.

        Uses the ocr_client for extraction.

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, equation)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            Extracted text content, or None on failure
        """
        return await self.ocr_client.aio_content_extract(
            image, type, priority, semaphore
        )

    async def aio_batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[str | None]:
        """Batch content extraction from multiple images.

        Args:
            images: List of PIL Images
            types: Content type(s) for each image
            priority: Priority value(s) for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of extracted text content
        """
        return await self.ocr_client.aio_batch_content_extract(
            images, types, priority, semaphore
        )

    async def _do_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core two-step extraction logic (shared by traced and non-traced paths).

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with type, bbox, and content
        """
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        # Layout detection using layout_client's core method
        blocks = await self.layout_client._do_layout_detect(image, priority, semaphore)

        if blocks is None:
            raise RuntimeError("Layout detection failed for the page")

        if not blocks:
            return blocks

        # Prepare block images for OCR
        image = get_rgb_image(image)
        width, height = image.size

        block_images: list[Image.Image] = []
        block_types: list[str] = []
        block_indices: list[int] = []

        for idx, block in enumerate(blocks):
            if block.type == "image":
                continue

            x1, y1, x2, y2 = block.bbox
            crop_box = (
                max(0, int(x1 * width)),
                max(0, int(y1 * height)),
                min(width, int(x2 * width)),
                min(height, int(y2 * height)),
            )

            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue

            block_images.append(image.crop(crop_box))
            block_types.append(block.type)
            block_indices.append(idx)

        # Extract content using ocr_client's core method
        if block_images:
            contents = await gather_tasks(
                tasks=[
                    self.ocr_client._do_content_extract(img, t, priority, semaphore)
                    for img, t in zip(block_images, block_types)
                ],
                use_tqdm=False,
            )

            for idx, content in zip(block_indices, contents):
                blocks[idx].content = content

        return blocks

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Extract layout first with layout_client, then content with ocr_client.

        Two-step extraction:
        1. Use layout_client to detect layout blocks
        2. Crop each block region from the original image
        3. Use ocr_client to extract content from each cropped region
        4. Map content back to blocks

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control
            page_idx: Optional page index for creating page-level span

        Returns:
            List of ContentBlock with type, bbox, and content

        Raises:
            RuntimeError: If layout detection fails for the page
        """
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        langfuse = get_langfuse_client()

        if langfuse and page_idx is not None:
            with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                with langfuse.start_as_current_span(
                    name=f"layout-{self.layout_model}-detection"
                ):
                    blocks = await self.layout_client._do_layout_detect(
                        image, priority, semaphore
                    )

                if blocks is None:
                    raise RuntimeError("Layout detection failed for the page")

                if not blocks:
                    return blocks

                with langfuse.start_as_current_span(
                    name=f"ocr-{self.ocr_model}-extraction"
                ):
                    image = get_rgb_image(image)
                    width, height = image.size

                    block_images: list[Image.Image] = []
                    block_types: list[str] = []
                    block_indices: list[int] = []

                    for idx, block in enumerate(blocks):
                        if block.type == "image":
                            continue

                        x1, y1, x2, y2 = block.bbox
                        crop_box = (
                            max(0, int(x1 * width)),
                            max(0, int(y1 * height)),
                            min(width, int(x2 * width)),
                            min(height, int(y2 * height)),
                        )

                        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                            continue

                        block_images.append(image.crop(crop_box))
                        block_types.append(block.type)
                        block_indices.append(idx)

                    if block_images:
                        contents = await gather_tasks(
                            tasks=[
                                self.ocr_client._do_content_extract(
                                    img, t, priority, semaphore
                                )
                                for img, t in zip(block_images, block_types)
                            ],
                            use_tqdm=False,
                        )

                        for idx, content in zip(block_indices, contents):
                            blocks[idx].content = content

                    return blocks
        else:
            return await self._do_two_step_extract(image, priority, semaphore)

    async def aio_batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch two-step extraction for multiple images.

        Args:
            images: List of PIL Images
            priority: Priority value(s) for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock lists, one per image
        """
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_two_step_extract(
                    img, p, semaphore, page_idx=idx if total_pages > 1 else None
                )
                for idx, (img, p) in enumerate(zip(images, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Composite Two-Step Extraction",
        )
