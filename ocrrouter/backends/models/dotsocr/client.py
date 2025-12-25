"""DotsOCR client for VLM inference."""

import asyncio
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client
from .preprocessor import DotsOCRPreprocessor
from .postprocessor import DotsOCRPostprocessor
from ocrrouter.backends.utils import (
    ContentBlock,
    SamplingParams,
    new_vlm_client,
    gather_tasks,
    get_png_bytes,
    get_rgb_image,
)


class DotsOCRSamplingParams(SamplingParams):
    """Sampling parameters optimized for DotsOCR."""

    def __init__(
        self,
        temperature: float | None = 0.1,
        top_p: float | None = 1.0,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        max_new_tokens: int | None = 16384,
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


# DotsOCR prompts for different modes
DEFAULT_PROMPTS: dict[str, str] = {
    "[layout_all]": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.""",
    "[layout_only]": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
    "[ocr]": """Extract the text content from this image.""",
    "[default]": """Extract the text content from this image.""",
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "[layout_all]": DotsOCRSamplingParams(),
    "[layout_only]": DotsOCRSamplingParams(),
    "[ocr]": DotsOCRSamplingParams(),
    "[default]": DotsOCRSamplingParams(),
}


class DotsOCRClient:
    """DotsOCR client for document analysis.

    This client supports multiple extraction modes:
    - One-step: Layout + OCR in a single API call (prompt_layout_all_en)
    - Two-step: Layout detection first, then OCR for each block
    - Layout-only: Just layout detection (prompt_layout_only_en)
    - OCR-only: Just text extraction (prompt_ocr)

    Example:
        >>> client = DotsOCRClient()
        >>> blocks = await client.aio_one_step_extract(image)
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

        # Create VLM client
        # Note: DotsOCR does not use system messages, only user messages
        self.client = new_vlm_client(
            backend=backend,
            model_name=settings.dotsocr_model_name,
            server_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            system_prompt="",  # DotsOCR only uses user messages
            allow_truncated_content=True,
            max_concurrency=settings.max_concurrency,
            http_timeout=settings.http_timeout,
            debug=settings.debug,
            debug_dir=str(settings.debug_dir) if settings.debug_dir else None,
            max_retries=settings.max_retries,
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = DotsOCRPreprocessor()
        self.postprocessor = DotsOCRPostprocessor(debug=settings.debug)

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
        (
            layout_image,
            orig_width,
            orig_height,
            resized_width,
            resized_height,
        ) = await self.preprocessor.aio_prepare_for_layout(self.executor, image)

        prompt = self.prompts.get("[layout_only]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout_only]") or self.sampling_params.get(
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

        return await self.postprocessor.aio_parse_layout_output(
            self.executor,
            output,
            orig_width=orig_width,
            orig_height=orig_height,
            resized_width=resized_width,
            resized_height=resized_height,
            include_content=False,  # include_content=False for layout-only
        )

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout blocks in a document page image.

        Uses layout-only mode, returns blocks without content.

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
                        name="layout-dotsocr-detection"
                    ):
                        return await self._do_layout_detect(image, priority, semaphore)
            else:
                with langfuse.start_as_current_span(name="layout-dotsocr-detection"):
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
            type: Content type (text, table, equation) - currently unused
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            Extracted text content, or None on failure
        """
        image = get_rgb_image(image)
        image_bytes = get_png_bytes(image)

        prompt = self.prompts.get("[ocr]") or self.prompts["[default]"]
        params = self.sampling_params.get("[ocr]") or self.sampling_params.get(
            "[default]"
        )

        if semaphore is None:
            output = await self.client.aio_predict(
                image_bytes, prompt, params, priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    image_bytes, prompt, params, priority
                )

        return output.strip() if output else None

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> str | None:
        """Extract content from an image region using OCR mode.

        Args:
            image: PIL Image to extract content from
            type: Content type (text, table, equation) - currently unused
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
                    with langfuse.start_as_current_span(name="ocr-dotsocr-extraction"):
                        return await self._do_content_extract(
                            image, type, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="ocr-dotsocr-extraction"):
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

    async def _do_one_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core one-step extraction logic (shared by traced and non-traced paths).

        Args:
            image: PIL Image of document page
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with type, bbox, and content
        """
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        (
            layout_image,
            orig_width,
            orig_height,
            resized_width,
            resized_height,
        ) = await self.preprocessor.aio_prepare_for_layout(self.executor, image)

        prompt = self.prompts.get("[layout_all]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout_all]") or self.sampling_params.get(
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

        blocks = await self.postprocessor.aio_parse_layout_output(
            self.executor,
            output,
            orig_width=orig_width,
            orig_height=orig_height,
            resized_width=resized_width,
            resized_height=resized_height,
            include_content=True,
        )

        return await self.postprocessor.aio_post_process_blocks(self.executor, blocks)

    async def aio_one_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Extract both layout and content in a single call with optional page-level tracing.

        Uses prompt_layout_all_en which returns layout detection results
        along with extracted content in a single API call.

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
                        name="layout-dotsocr-detection"
                    ):
                        return await self._do_one_step_extract(
                            image, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="layout-dotsocr-detection"):
                    return await self._do_one_step_extract(image, priority, semaphore)
        else:
            return await self._do_one_step_extract(image, priority, semaphore)

    async def aio_batch_one_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch one-step extraction for multiple images.

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
                self.aio_one_step_extract(
                    img, p, semaphore, page_idx=idx if total_pages > 1 else None
                )
                for idx, (img, p) in enumerate(zip(images, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="One Step Extraction",
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

        # Step 1: Layout detection
        blocks = await self._do_layout_detect(image, priority, semaphore)

        if not blocks:
            return blocks

        # Step 2: Extract content for each block
        image = get_rgb_image(image)
        width, height = image.size

        # Prepare block images for OCR
        block_images: list[Image.Image] = []
        block_indices: list[int] = []

        for idx, block in enumerate(blocks):
            # Skip image blocks (no OCR needed)
            if block.type == "image":
                continue

            # Crop block region
            x1, y1, x2, y2 = block.bbox
            crop_box = (
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height),
            )
            block_image = image.crop(crop_box)
            block_images.append(block_image)
            block_indices.append(idx)

        # Extract content for all blocks (using _do_content_extract to avoid nested spans)
        if block_images:
            contents = await gather_tasks(
                tasks=[
                    self._do_content_extract(img, "text", priority, semaphore)
                    for img in block_images
                ],
                use_tqdm=False,
            )

            # Assign content to blocks
            for idx, content in zip(block_indices, contents):
                blocks[idx].content = content

        # Apply post-processing
        return await self.postprocessor.aio_post_process_blocks(self.executor, blocks)

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Extract layout first, then content for each block with optional page-level tracing.

        Two-step extraction:
        1. Use prompt_layout_only_en to get layout blocks
        2. Crop each block and use prompt_ocr to extract content

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
                        name="layout-dotsocr-detection"
                    ):
                        blocks = await self._do_layout_detect(
                            image,
                            priority,
                            semaphore or asyncio.Semaphore(self.max_concurrency),
                        )

                    if not blocks:
                        return blocks

                    with langfuse.start_as_current_span(name="ocr-dotsocr-extraction"):
                        # Extract content for each block
                        image = get_rgb_image(image)
                        width, height = image.size
                        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

                        block_images: list[Image.Image] = []
                        block_indices: list[int] = []

                        for idx, block in enumerate(blocks):
                            if block.type == "image":
                                continue
                            x1, y1, x2, y2 = block.bbox
                            crop_box = (
                                int(x1 * width),
                                int(y1 * height),
                                int(x2 * width),
                                int(y2 * height),
                            )
                            block_images.append(image.crop(crop_box))
                            block_indices.append(idx)

                        if block_images:
                            contents = await gather_tasks(
                                tasks=[
                                    self._do_content_extract(
                                        img, "text", priority, semaphore
                                    )
                                    for img in block_images
                                ],
                                use_tqdm=False,
                            )
                            for idx, content in zip(block_indices, contents):
                                blocks[idx].content = content

                        return await self.postprocessor.aio_post_process_blocks(
                            self.executor, blocks
                        )
            else:
                with langfuse.start_as_current_span(name="layout-dotsocr-detection"):
                    blocks = await self._do_layout_detect(
                        image,
                        priority,
                        semaphore or asyncio.Semaphore(self.max_concurrency),
                    )

                if not blocks:
                    return blocks

                with langfuse.start_as_current_span(name="ocr-dotsocr-extraction"):
                    # Extract content for each block
                    image = get_rgb_image(image)
                    width, height = image.size
                    semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

                    block_images: list[Image.Image] = []
                    block_indices: list[int] = []

                    for idx, block in enumerate(blocks):
                        if block.type == "image":
                            continue
                        x1, y1, x2, y2 = block.bbox
                        crop_box = (
                            int(x1 * width),
                            int(y1 * height),
                            int(x2 * width),
                            int(y2 * height),
                        )
                        block_images.append(image.crop(crop_box))
                        block_indices.append(idx)

                    if block_images:
                        contents = await gather_tasks(
                            tasks=[
                                self._do_content_extract(
                                    img, "text", priority, semaphore
                                )
                                for img in block_images
                            ],
                            use_tqdm=False,
                        )
                        for idx, content in zip(block_indices, contents):
                            blocks[idx].content = content

                    return await self.postprocessor.aio_post_process_blocks(
                        self.executor, blocks
                    )
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
            tqdm_desc="Two Step Extraction",
        )
