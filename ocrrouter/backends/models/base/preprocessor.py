"""Base preprocessor for image preparation before model inference."""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Executor

from PIL import Image

from ocrrouter.backends.utils import ContentBlock


class BasePreprocessor(ABC):
    """Abstract base class for image preprocessing before model inference.

    Preprocessors handle:
    - Image resizing and format conversion for layout detection
    - Block cropping and rotation for OCR
    - Any model-specific image preparation requirements

    Subclasses must implement:
    - prepare_for_layout(): Prepare full page image for layout detection
    - prepare_for_ocr(): Prepare cropped block image for OCR
    """

    @abstractmethod
    def prepare_for_layout(self, image: Image.Image) -> bytes:
        """Prepare image for layout detection.

        Applies model-specific preprocessing such as:
        - Resizing to expected dimensions
        - Color space conversion
        - Format conversion

        Args:
            image: PIL Image of document page.

        Returns:
            Image bytes ready for model inference.
        """
        pass

    @abstractmethod
    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock,
    ) -> bytes:
        """Prepare cropped block image for OCR.

        Handles:
        - Cropping the block region from the full page image
        - Rotation based on block angle
        - Resizing if needed by the model

        Args:
            image: Full page PIL Image.
            block: ContentBlock with bbox and angle information.

        Returns:
            Cropped and prepared image bytes.
        """
        pass

    async def aio_prepare_for_layout(
        self,
        executor: Executor | None,
        image: Image.Image,
    ) -> bytes:
        """Async wrapper for prepare_for_layout.

        Args:
            executor: Optional executor for running in thread pool.
            image: PIL Image of document page.

        Returns:
            Image bytes ready for model inference.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_layout, image)

    async def aio_prepare_for_ocr(
        self,
        executor: Executor | None,
        image: Image.Image,
        block: ContentBlock,
    ) -> bytes:
        """Async wrapper for prepare_for_ocr.

        Args:
            executor: Optional executor for running in thread pool.
            image: Full page PIL Image.
            block: ContentBlock with bbox and angle information.

        Returns:
            Cropped and prepared image bytes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_ocr, image, block)

    async def aio_prepare_batch_for_layout(
        self,
        executor: Executor | None,
        images: list[Image.Image],
    ) -> list[bytes]:
        """Async batch preparation for layout detection.

        Args:
            executor: Optional executor for running in thread pool.
            images: List of PIL Images.

        Returns:
            List of image bytes ready for model inference.
        """
        tasks = [self.aio_prepare_for_layout(executor, image) for image in images]
        return await asyncio.gather(*tasks)

    async def aio_prepare_batch_for_ocr(
        self,
        executor: Executor | None,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> list[bytes]:
        """Async batch preparation for OCR.

        Args:
            executor: Optional executor for running in thread pool.
            image: Full page PIL Image.
            blocks: List of ContentBlocks to prepare.

        Returns:
            List of cropped and prepared image bytes.
        """
        tasks = [self.aio_prepare_for_ocr(executor, image, block) for block in blocks]
        return await asyncio.gather(*tasks)
