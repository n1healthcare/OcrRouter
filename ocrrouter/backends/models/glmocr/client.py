"""GLM-OCR client for VLM inference."""

import asyncio
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client
from .preprocessor import GlmOCRPreprocessor
from .postprocessor import GlmOCRPostprocessor
from ocrrouter.backends.utils import (
    SamplingParams,
    new_vlm_client,
    gather_tasks,
    get_png_bytes,
    get_rgb_image,
)


class GlmOCRSamplingParams(SamplingParams):
    """Sampling parameters optimized for GLM-OCR.

    GLM-OCR uses specific parameters for best OCR quality:
    - Lower temperature (0.01) for deterministic output
    - High top_p (0.9) with top_k (50) for diverse but focused sampling
    - Repetition penalty (1.1) to avoid repeated text
    """

    def __init__(
        self,
        temperature: float | None = 0.01,
        top_p: float | None = 0.9,
        top_k: int | None = 50,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = 1.1,
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


# GLM-OCR prompts
DEFAULT_PROMPTS: dict[str, str] = {
    "[default]": (
        "Recognize the text in the image and output in Markdown format. "
        "Preserve the original layout (headings/paragraphs/tables/formulas). "
        "Do not fabricate content that does not exist in the image."
    ),
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "equation": "Formula Recognition:",
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "[default]": GlmOCRSamplingParams(),
}


class GlmOCRClient:
    """GLM-OCR client for document analysis.

    This client uses the GLM-OCR model which returns structured markdown
    or text content directly. Unlike DeepSeek-OCR, it does not use grounding
    mode for layout detection.

    GLM-OCR features:
    - Multi-Token Prediction (MTP) for efficient inference
    - Built on GLM-V encoder-decoder architecture
    - 0.9B parameters, optimized for vLLM/SGLang deployment
    - Supports text, table, and formula recognition

    Example:
        >>> client = GlmOCRClient(settings)
        >>> text = await client.aio_content_extract(image)
        >>> # text is the extracted markdown content
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

        # Create VLM client with GLM-OCR configuration
        self.client = new_vlm_client(
            backend=backend,
            model_name=settings.glmocr_model_name,
            server_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            system_prompt="",  # GLM-OCR uses user messages only
            allow_truncated_content=True,
            max_concurrency=settings.max_concurrency,
            http_timeout=settings.http_timeout,
            debug=settings.debug,
            debug_dir=str(settings.debug_dir) if settings.debug_dir else None,
            max_retries=settings.max_retries,
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = GlmOCRPreprocessor()
        self.postprocessor = GlmOCRPostprocessor(debug=settings.debug)

        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
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
        """Core content extraction logic.

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, formula)
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            Extracted text content, or None on failure
        """
        # Prepare image
        image_bytes = await self.preprocessor.aio_prepare_for_ocr(
            self.executor, image, None
        )

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

        # Post-process output
        if output:
            output = self.postprocessor.clean_output(output)

        return output.strip() if output else None

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> str | None:
        """Extract content from an image using GLM-OCR.

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, formula)
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
                    with langfuse.start_as_current_span(name="ocr-glmocr-extraction"):
                        return await self._do_content_extract(
                            image, type, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="ocr-glmocr-extraction"):
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
            tqdm_desc="GLM-OCR Extraction",
        )
