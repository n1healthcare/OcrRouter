"""LightOnOCR client for VLM inference."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Literal

from PIL import Image

from ocrrouter.backends.utils import (
    ContentBlock,
    SamplingParams,
    gather_tasks,
    get_png_bytes,
    get_rgb_image,
    new_vlm_client,
)
from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client

from .postprocessor import LightOnOCRPostprocessor
from .preprocessor import LightOnOCRPreprocessor


class LightOnOCRSamplingParams(SamplingParams):
    """Sampling parameters optimized for LightOnOCR."""

    def __init__(
        self,
        temperature: float | None = 0.2,
        top_p: float | None = 0.9,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        max_new_tokens: int | None = 8192,
    ):
        super().__init__(
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )


# LightOnOCR prompts - model works without explicit prompt
DEFAULT_PROMPTS: dict[str, str] = {
    "[default]": "",  # LightOnOCR works without prompt
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "[default]": LightOnOCRSamplingParams(),
}


class LightOnOCRClient:
    """LightOnOCR client for document OCR with image localization.

    This client uses the LightOnOCR-2-1B-bbox-soup model which:
    - Outputs OCR text in markdown format
    - Includes embedded image bounding boxes in format: ![image](image_N.png) x1,y1,x2,y2
    - Coordinates are normalized to 0-1000

    Example:
        >>> client = LightOnOCRClient(settings)
        >>> blocks = await client.aio_layout_detect(image)
        >>> # blocks is a list of ContentBlock with type, bbox, and content
    """

    def __init__(
        self,
        settings: Settings,
        backend: Literal["http-client"] = "http-client",
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        sampling_params: dict[str, SamplingParams] = DEFAULT_SAMPLING_PARAMS,
        executor: Executor | None = None,
        use_tqdm: bool | None = None,
    ) -> None:
        if backend != "http-client":
            raise ValueError(f"Unsupported backend: {backend}. Only 'http-client' is supported.")

        self._settings = settings

        # Use settings.use_tqdm if use_tqdm is not explicitly provided
        if use_tqdm is None:
            use_tqdm = settings.use_tqdm

        # Create VLM client
        # LightOnOCR does not require system or user prompts
        self.client = new_vlm_client(
            backend=backend,
            model_name=settings.lightonocr_model_name,
            server_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            system_prompt="",  # LightOnOCR only uses images
            allow_truncated_content=True,
            max_concurrency=settings.max_concurrency,
            http_timeout=settings.http_timeout,
            debug=settings.debug,
            debug_dir=str(settings.debug_dir) if settings.debug_dir else None,
            max_retries=settings.max_retries,
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = LightOnOCRPreprocessor()
        self.postprocessor = LightOnOCRPostprocessor(debug=settings.debug)

        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.max_concurrency = settings.max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = settings.debug

    async def _do_layout_detect(
        self,
        image: Image.Image,
        output_mode: Literal["all", "layout_only", "ocr_only"] = "all",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core layout detection logic.

        Args:
            image: PIL Image of document page.
            output_mode: Output mode for postprocessor.
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of ContentBlock with type, bbox, and content.
        """
        # Preprocess image
        layout_image = await self.preprocessor.aio_prepare_for_layout(self.executor, image)

        prompt = self.prompts.get("[default]", "")
        params = self.sampling_params.get("[default]")

        if semaphore is None:
            output = await self.client.aio_predict(layout_image, prompt, params, priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(layout_image, prompt, params, priority)

        return await self.postprocessor.aio_parse_layout_output(
            self.executor,
            output,
            output_mode=output_mode,
        )

    async def aio_layout_detect(
        self,
        image: Image.Image,
        output_mode: Literal["all", "layout_only", "ocr_only"] = "layout_only",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout blocks in a document page image.

        Returns image blocks detected in the document.

        Args:
            image: PIL Image of document page.
            output_mode: Output mode (default 'layout_only' for image bboxes only).
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.
            page_idx: Optional page index for creating page-level span.

        Returns:
            List of ContentBlock with type and bbox.
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(name="layout-lightonocr-detection"):
                        return await self._do_layout_detect(image, output_mode, priority, semaphore)
            else:
                with langfuse.start_as_current_span(name="layout-lightonocr-detection"):
                    return await self._do_layout_detect(image, output_mode, priority, semaphore)
        else:
            return await self._do_layout_detect(image, output_mode, priority, semaphore)

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        output_mode: Literal["all", "layout_only", "ocr_only"] = "layout_only",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection for multiple images.

        Args:
            images: List of PIL Images.
            output_mode: Output mode for all images.
            priority: Priority value(s) for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of ContentBlock lists, one per image.
        """
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_layout_detect(
                    img,
                    output_mode,
                    p,
                    semaphore,
                    page_idx=idx if total_pages > 1 else None,
                )
                for idx, (img, p) in enumerate(zip(images, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Detection",
        )

    async def _do_content_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Core content extraction logic.

        Args:
            image: PIL Image to extract content from.
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            Extracted text content, or None on failure.
        """
        image = get_rgb_image(image)
        image_bytes = get_png_bytes(image)

        prompt = self.prompts.get("[default]", "")
        params = self.sampling_params.get("[default]")

        if semaphore is None:
            output = await self.client.aio_predict(image_bytes, prompt, params, priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(image_bytes, prompt, params, priority)

        return output.strip() if output else None

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> str | None:
        """Extract content from an image using OCR.

        Args:
            image: PIL Image to extract content from.
            type: Content type (unused, LightOnOCR handles all types).
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.
            page_idx: Optional page index for creating page-level span.

        Returns:
            Extracted text content, or None on failure.
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(name="ocr-lightonocr-extraction"):
                        return await self._do_content_extract(image, priority, semaphore)
            else:
                with langfuse.start_as_current_span(name="ocr-lightonocr-extraction"):
                    return await self._do_content_extract(image, priority, semaphore)
        else:
            return await self._do_content_extract(image, priority, semaphore)

    async def aio_batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[str | None]:
        """Batch content extraction from multiple images.

        Args:
            images: List of PIL Images.
            types: Content type(s) for each image (unused).
            priority: Priority value(s) for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of extracted text content.
        """
        if isinstance(types, str):
            types = [types] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_content_extract(img, t, p, semaphore, page_idx=idx if total_pages > 1 else None)
                for idx, (img, t, p) in enumerate(zip(images, types, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Content Extraction",
        )

    async def _do_ocr_with_layout(
        self,
        image: Image.Image,
        output_mode: Literal["all", "layout_only", "ocr_only"] = "all",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """OCR with layout detection in a single call.

        LightOnOCR performs both OCR and image detection in a single inference.

        Args:
            image: PIL Image of document page.
            output_mode: Output mode for postprocessor.
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of ContentBlock with type, bbox, and content.
        """
        return await self._do_layout_detect(image, output_mode, priority, semaphore)

    async def aio_ocr_with_layout(
        self,
        image: Image.Image,
        output_mode: Literal["all", "layout_only", "ocr_only"] = "all",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """OCR with layout detection in a single call.

        LightOnOCR performs both OCR and image detection in a single inference.

        Args:
            image: PIL Image of document page.
            output_mode: Output mode for postprocessor.
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.
            page_idx: Optional page index for creating page-level span.

        Returns:
            List of ContentBlock with type, bbox, and content.
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(name="ocr-lightonocr-with-layout"):
                        return await self._do_ocr_with_layout(image, output_mode, priority, semaphore)
            else:
                with langfuse.start_as_current_span(name="ocr-lightonocr-with-layout"):
                    return await self._do_ocr_with_layout(image, output_mode, priority, semaphore)
        else:
            return await self._do_ocr_with_layout(image, output_mode, priority, semaphore)

    async def aio_batch_ocr_with_layout(
        self,
        images: list[Image.Image],
        output_mode: Literal["all", "layout_only", "ocr_only"] = "all",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch OCR with layout detection for multiple images.

        Args:
            images: List of PIL Images.
            output_mode: Output mode for all images.
            priority: Priority value(s) for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of ContentBlock lists, one per image.
        """
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_ocr_with_layout(
                    img,
                    output_mode,
                    p,
                    semaphore,
                    page_idx=idx if total_pages > 1 else None,
                )
                for idx, (img, p) in enumerate(zip(images, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="OCR with Layout",
        )
