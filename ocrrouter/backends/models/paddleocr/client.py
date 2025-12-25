"""PaddleOCR client for VLM inference."""

import asyncio
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client
from .preprocessor import PaddleOCRPreprocessor
from .postprocessor import PaddleOCRPostprocessor
from ocrrouter.backends.utils import (
    ContentBlock,
    SamplingParams,
    new_vlm_client,
    gather_tasks,
    get_png_bytes,
    get_rgb_image,
)


class PaddleOCRSamplingParams(SamplingParams):
    """Sampling parameters optimized for PaddleOCR."""

    def __init__(
        self,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        max_new_tokens: int | None = 2048,
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


# PaddleOCR task-specific prompts
DEFAULT_PROMPTS: dict[str, str] = {
    "[default]": "OCR:",
    "text": "OCR:",
    "table": "Table Recognition:",
    "equation": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

DEFAULT_SAMPLING_PARAMS = PaddleOCRSamplingParams()


class PaddleOCRClient:
    """PaddleOCR client for document OCR.

    This client only supports OCR extraction, not layout detection.

    Example:
        >>> client = PaddleOCRClient()
        >>> blocks = await client.aio_full_page_ocr(image)
        >>> # blocks is a list of ContentBlock with type and content
    """

    def __init__(
        self,
        settings: Settings,
        backend: Literal["http-client"] = "http-client",
        prompt: str | None = None,
        prompts: dict[str, str] | None = None,
        sampling_params: SamplingParams | None = None,
        executor: Executor | None = None,
        use_tqdm: bool | None = None,
    ) -> None:
        if backend != "http-client":
            raise ValueError(
                f"Unsupported backend: {backend}. Only 'http-client' is supported."
            )

        self._settings = settings

        # Use settings.use_tqdm if use_tqdm is not explicitly provided
        if use_tqdm is None:
            use_tqdm = settings.use_tqdm

        # Use provided prompt or default
        if prompt is None:
            prompt = DEFAULT_PROMPTS["[default]"]

        # Create VLM client with PaddleOCR configuration
        self.client = new_vlm_client(
            backend=backend,
            model_name=settings.paddleocr_model_name,
            server_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            system_prompt="",  # PaddleOCR uses empty system prompt
            allow_truncated_content=True,
            max_concurrency=settings.max_concurrency,
            http_timeout=settings.http_timeout,
            debug=settings.debug,
            debug_dir=str(settings.debug_dir) if settings.debug_dir else None,
            max_retries=settings.max_retries,
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = PaddleOCRPreprocessor()
        self.postprocessor = PaddleOCRPostprocessor(debug=settings.debug)

        self.backend = backend
        self.prompt = prompt
        self.prompts = prompts or DEFAULT_PROMPTS
        self.sampling_params = sampling_params or DEFAULT_SAMPLING_PARAMS
        self.max_concurrency = settings.max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = settings.debug

    async def _do_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Core content extraction logic (shared by traced and non-traced paths).

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, equation, chart)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            Extracted text content, or None on failure
        """
        # Apply preprocessing with smart_resize
        image_bytes = self.preprocessor.prepare_for_ocr(image)

        prompt = self.prompts.get(type) or self.prompt

        if semaphore is None:
            output = await self.client.aio_predict(
                image_bytes, prompt, self.sampling_params, priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    image_bytes, prompt, self.sampling_params, priority
                )

        # Post-process output
        if output:
            output = self.postprocessor._clean_ocr_output(output)
            # Convert OTSL to HTML for table type
            if type == "table":
                output = self.postprocessor._convert_table_to_html(output)

        return output.strip() if output else None

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> str | None:
        """Extract OCR content from an image.

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, equation, chart)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control
            page_idx: Optional page index for creating page-level span

        Returns:
            Extracted text content, or None on failure
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(
                        name="ocr-paddleocr-extraction"
                    ):
                        return await self._do_content_extract(
                            image, type, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="ocr-paddleocr-extraction"):
                    return await self._do_content_extract(
                        image, type, priority, semaphore
                    )
        else:
            return await self._do_content_extract(image, type, priority, semaphore)

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
        if isinstance(types, str):
            types = [types] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_content_extract(
                    img, t, p, semaphore, page_idx=idx if total_pages > 1 else None
                )
                for idx, (img, t, p) in enumerate(zip(images, types, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="PaddleOCR Extraction",
        )

    async def _do_full_page_ocr(
        self,
        image: Image.Image,
        page_image: Image.Image | None = None,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core full-page OCR logic (shared by traced and non-traced paths).

        Args:
            image: PIL Image of document page
            page_image: Original page image (for reference, can be same as image)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with type, bbox, and content
        """
        # Apply preprocessing with smart_resize
        image_bytes = self.preprocessor.prepare_for_ocr(image)

        if semaphore is None:
            output = await self.client.aio_predict(
                image_bytes, self.prompt, self.sampling_params, priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    image_bytes, self.prompt, self.sampling_params, priority
                )

        # Parse output into ContentBlocks
        blocks = await self.postprocessor.aio_parse_layout_output(
            self.executor,
            output,
            page_image=page_image or image,
        )

        # Apply post-processing
        return await self.postprocessor.aio_post_process_blocks(self.executor, blocks)

    async def aio_full_page_ocr(
        self,
        image: Image.Image,
        page_image: Image.Image | None = None,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Full-page OCR that returns ContentBlocks.

        Args:
            image: PIL Image of document page
            page_image: Original page image (for reference, can be same as image)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control
            page_idx: Optional page index for creating page-level span

        Returns:
            List of ContentBlock with type, bbox, and content
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(
                        name="ocr-paddleocr-extraction"
                    ):
                        return await self._do_full_page_ocr(
                            image, page_image, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="ocr-paddleocr-extraction"):
                    return await self._do_full_page_ocr(
                        image, page_image, priority, semaphore
                    )
        else:
            return await self._do_full_page_ocr(image, page_image, priority, semaphore)

    async def aio_batch_full_page_ocr(
        self,
        images: list[Image.Image],
        page_images: list[Image.Image] | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch full-page OCR for multiple images.

        Args:
            images: List of PIL Images
            page_images: Original page images (can be same as images)
            priority: Priority value(s) for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock lists, one per image
        """
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        page_images = page_images or images

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_full_page_ocr(
                    img,
                    page_img,
                    p,
                    semaphore,
                    page_idx=idx if total_pages > 1 else None,
                )
                for idx, (img, page_img, p) in enumerate(
                    zip(images, page_images, priority)
                )
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="PaddleOCR Full-Page OCR",
        )
