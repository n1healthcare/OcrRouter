"""DeepSeek-OCR client for VLM inference."""

import asyncio
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client
from .preprocessor import DeepSeekPreprocessor
from .postprocessor import DeepSeekPostprocessor
from ocrrouter.backends.utils import (
    ContentBlock,
    SamplingParams,
    new_vlm_client,
    gather_tasks,
    get_png_bytes,
    get_rgb_image,
)


class DeepSeekSamplingParams(SamplingParams):
    """Sampling parameters optimized for DeepSeek-OCR."""

    def __init__(
        self,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        max_new_tokens: int | None = None,
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


# DeepSeek prompts
DEFAULT_PROMPTS: dict[str, str] = {
    "[layout]": "<|grounding|>Convert the document to markdown.",
    "[two_step]": "<|grounding|>Convert the document to markdown.",
    "[default]": "Free OCR.",
    "table": "Free OCR.",
    "equation": "Free OCR.",
    "text": "Free OCR.",
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "[layout]": DeepSeekSamplingParams(),
    "[two_step]": DeepSeekSamplingParams(),
    "[default]": DeepSeekSamplingParams(),
}

# DeepSeek-specific vLLM extra body parameters
DEEPSEEK_VLLM_XARGS = {
    "ngram_size": 30,
    "window_size": 90,
    "whitelist_token_ids": [128821, 128822],
}


class DeepSeekClient:
    """DeepSeek-OCR client for document analysis.

    This client uses DeepSeek-OCR model which can return both layout detection
    and content extraction in a single API call using the grounding prompt.

    Example:
        >>> client = DeepSeekClient()
        >>> blocks = await client.aio_two_step_extract(image)
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
            raise ValueError(
                f"Unsupported backend: {backend}. Only 'http-client' is supported."
            )

        self._settings = settings

        # Use settings.use_tqdm if use_tqdm is not explicitly provided
        if use_tqdm is None:
            use_tqdm = settings.use_tqdm

        # Create VLM client with DeepSeek-specific configuration
        # Note: DeepSeek-OCR does not use system messages, only user messages
        self.client = new_vlm_client(
            backend=backend,
            model_name=settings.deepseek_model_name,
            server_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            system_prompt="",  # DeepSeek-OCR only uses user messages
            allow_truncated_content=True,
            max_concurrency=settings.max_concurrency,
            http_timeout=settings.http_timeout,
            debug=settings.debug,
            debug_dir=str(settings.debug_dir) if settings.debug_dir else None,
            max_retries=settings.max_retries,
            extra_body={"vllm_xargs": DEEPSEEK_VLLM_XARGS},
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = DeepSeekPreprocessor()
        self.postprocessor = DeepSeekPostprocessor(debug=settings.debug)

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
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core layout detection logic (shared by traced and non-traced paths).

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with type and bbox (content set to None)
        """
        layout_image = await self.preprocessor.aio_prepare_for_layout(
            self.executor, image
        )
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get(
            "[default]"
        )

        if semaphore is None:
            output = await self.client.aio_predict(
                layout_image, prompt, params, priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    layout_image, prompt, params, priority
                )

        blocks = await self.postprocessor.aio_parse_layout_output(self.executor, output)

        # For layout-only detection, clear content
        for block in blocks:
            block.content = None

        return blocks

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout blocks in a document page image.

        Uses grounding mode but returns blocks without content.

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control
            page_idx: Optional page index for creating page-level span

        Returns:
            List of ContentBlock with type and bbox (content set to None)
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(
                        name="layout-deepseek-detection"
                    ):
                        return await self._do_layout_detect(image, priority, semaphore)
            else:
                with langfuse.start_as_current_span(name="layout-deepseek-detection"):
                    return await self._do_layout_detect(image, priority, semaphore)
        else:
            return await self._do_layout_detect(image, priority, semaphore)

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
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_layout_detect(
                    img, p, semaphore, page_idx=idx if total_pages > 1 else None
                )
                for idx, (img, p) in enumerate(zip(images, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Detection",
        )

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
            type: Content type (text, table, equation)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            Extracted text content, or None on failure
        """
        image = get_rgb_image(image)
        image_bytes = get_png_bytes(image)

        prompt = self.prompts.get(type) or self.prompts["[default]"]
        params = self.sampling_params.get(type) or self.sampling_params.get("[default]")

        if semaphore is None:
            output = await self.client.aio_predict(
                image_bytes, prompt, params, priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    image_bytes, prompt, params, priority
                )

        # Clean up end-of-sentence markers if present
        if output and "<｜end▁of▁sentence｜>" in output:
            output = output.replace("<｜end▁of▁sentence｜>", "")

        # Convert LaTeX delimiters
        if output:
            output = output.replace(r"\[", "$$").replace(r"\]", "$$")

        return output.strip() if output else None

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> str | None:
        """Extract content from an image region using Free OCR mode.

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, equation)
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
                    with langfuse.start_as_current_span(name="ocr-deepseek-extraction"):
                        return await self._do_content_extract(
                            image, type, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="ocr-deepseek-extraction"):
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
            tqdm_desc="Content Extraction",
        )

    async def _do_two_step_layout(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core layout detection for two-step extract (shared logic).

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with layout info and raw content
        """
        layout_image = await self.preprocessor.aio_prepare_for_layout(
            self.executor, image
        )
        prompt = self.prompts.get("[two_step]") or self.prompts["[default]"]
        params = self.sampling_params.get("[two_step]") or self.sampling_params.get(
            "[default]"
        )

        if semaphore is None:
            output = await self.client.aio_predict(
                layout_image, prompt, params, priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    layout_image, prompt, params, priority
                )

        # Clean up end-of-sentence markers if present
        if output and "<｜end▁of▁sentence｜>" in output:
            output = output.replace("<｜end▁of▁sentence｜>", "")

        return await self.postprocessor.aio_parse_layout_output(self.executor, output)

    async def _do_two_step_ocr(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Core OCR post-processing for two-step extract (shared logic).

        Args:
            blocks: List of ContentBlock from layout detection

        Returns:
            List of ContentBlock with post-processed content
        """
        return await self.postprocessor.aio_post_process_blocks(self.executor, blocks)

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Extract both layout and content in a single call with optional page-level tracing.

        DeepSeek's grounding mode returns layout detection results along with
        the extracted content in a single API call, making this more efficient
        than a true two-step approach.

        Args:
            image: PIL Image of document page
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
                        name="layout-deepseek-detection"
                    ):
                        blocks = await self._do_two_step_layout(
                            image, priority, semaphore
                        )
                    with langfuse.start_as_current_span(name="ocr-deepseek-extraction"):
                        return await self._do_two_step_ocr(blocks)
            else:
                with langfuse.start_as_current_span(name="layout-deepseek-detection"):
                    blocks = await self._do_two_step_layout(image, priority, semaphore)
                with langfuse.start_as_current_span(name="ocr-deepseek-extraction"):
                    return await self._do_two_step_ocr(blocks)
        else:
            blocks = await self._do_two_step_layout(image, priority, semaphore)
            return await self._do_two_step_ocr(blocks)

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
            tqdm_desc="Two Step Extraction",
        )
