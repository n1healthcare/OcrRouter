import asyncio
from concurrent.futures import Executor
from typing import Sequence

from PIL import Image

from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client
from .preprocessor import MinerUPreprocessor
from .postprocessor import MinerUPostprocessor
from ocrrouter.backends.utils import (
    DEFAULT_SYSTEM_PROMPT,
    SamplingParams,
    new_vlm_client,
    gather_tasks,
    ContentBlock,
)


class MinerUSamplingParams(SamplingParams):
    def __init__(
        self,
        temperature: float | None = 0.0,
        top_p: float | None = 0.01,
        top_k: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        no_repeat_ngram_size: int | None = 100,
        max_new_tokens: int | None = None,
        skip_special_tokens: bool | None = False,
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
            skip_special_tokens,
        )


DEFAULT_PROMPTS: dict[str, str] = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": MinerUSamplingParams(),
}


class MinerUClient:
    def __init__(
        self,
        settings: Settings,
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_SAMPLING_PARAMS,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        incremental_priority: bool = False,
        executor: Executor | None = None,
        use_tqdm: bool | None = None,
    ) -> None:
        self._settings = settings

        # Use settings.use_tqdm if use_tqdm is not explicitly provided
        if use_tqdm is None:
            use_tqdm = settings.use_tqdm

        self.client = new_vlm_client(
            backend="http-client",
            model_name=settings.mineru_model_name,
            server_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            system_prompt=system_prompt,
            allow_truncated_content=True,
            max_concurrency=settings.max_concurrency,
            http_timeout=settings.http_timeout,
            debug=settings.debug,
            debug_dir=str(settings.debug_dir) if settings.debug_dir else None,
            max_retries=settings.max_retries,
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = MinerUPreprocessor(
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            prompts=prompts,
            sampling_params=sampling_params,
        )
        self.postprocessor = MinerUPostprocessor(
            handle_equation_block=handle_equation_block,
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            debug=settings.debug,
        )

        self.prompts = prompts
        self.sampling_params = sampling_params
        self.incremental_priority = incremental_priority
        self.max_concurrency = settings.max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = settings.debug

        # http-client always uses concurrent batching mode
        self.batching_mode = "concurrent"

    async def _do_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core layout detection logic (shared by traced and non-traced paths).

        Args:
            image: PIL Image to process
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with type and bbox
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
        return await self.postprocessor.aio_parse_layout_output(self.executor, output)

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout blocks in a document page image.

        Args:
            image: PIL Image to process
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control
            page_idx: Optional page index for creating page-level span

        Returns:
            List of ContentBlock with type and bbox
        """
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(name="layout-mineru-detection"):
                        return await self._do_layout_detect(image, priority, semaphore)
            else:
                with langfuse.start_as_current_span(name="layout-mineru-detection"):
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
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
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
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        loop = asyncio.get_running_loop()
        block_images, prompts, params, _ = await loop.run_in_executor(
            self.executor, self.preprocessor.prepare_blocks_for_ocr, image, blocks
        )
        if not (block_images and prompts and params):
            return None
        if semaphore is None:
            output = await self.client.aio_predict(
                block_images[0], prompts[0], params[0], priority
            )
        else:
            async with semaphore:
                output = await self.client.aio_predict(
                    block_images[0], prompts[0], params[0], priority
                )
        blocks[0].content = output
        blocks = await self.postprocessor.aio_post_process_blocks(self.executor, blocks)
        return blocks[0].content if blocks else None

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> str | None:
        """Extract content from an image region.

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
                    with langfuse.start_as_current_span(name="ocr-mineru-extraction"):
                        return await self._do_content_extract(
                            image, type, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="ocr-mineru-extraction"):
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
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
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

    async def _do_two_step_ocr(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core OCR extraction for two-step extract (shared logic).

        Args:
            image: PIL Image of document page
            blocks: Layout blocks from layout detection
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control

        Returns:
            List of ContentBlock with content extracted
        """
        loop = asyncio.get_running_loop()
        block_images, prompts, params, indices = await loop.run_in_executor(
            self.executor,
            self.preprocessor.prepare_blocks_for_ocr,
            image,
            blocks,
        )
        outputs = await self.client.aio_batch_predict(
            block_images, prompts, params, priority, semaphore=semaphore
        )
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        return await self.postprocessor.aio_post_process_blocks(self.executor, blocks)

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Two-step extraction with optional page-level tracing.

        Args:
            image: PIL Image to process
            priority: Optional priority for request ordering
            semaphore: Optional semaphore for concurrency control
            page_idx: Optional page index for creating page-level span

        Returns:
            List of ContentBlock with type, bbox, and content
        """
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        langfuse = get_langfuse_client()

        if langfuse:
            if page_idx is not None:
                with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                    with langfuse.start_as_current_span(name="layout-mineru-detection"):
                        blocks = await self._do_layout_detect(
                            image, priority, semaphore
                        )
                    with langfuse.start_as_current_span(name="ocr-mineru-extraction"):
                        return await self._do_two_step_ocr(
                            image, blocks, priority, semaphore
                        )
            else:
                with langfuse.start_as_current_span(name="layout-mineru-detection"):
                    blocks = await self._do_layout_detect(image, priority, semaphore)
                with langfuse.start_as_current_span(name="ocr-mineru-extraction"):
                    return await self._do_two_step_ocr(
                        image, blocks, priority, semaphore
                    )
        else:
            blocks = await self._do_layout_detect(image, priority, semaphore)
            return await self._do_two_step_ocr(image, blocks, priority, semaphore)

    async def aio_concurrent_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
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

    async def aio_stepping_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        blocks_list = await self.aio_batch_layout_detect(images, priority, semaphore)

        loop = asyncio.get_running_loop()
        all_images: list[bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = await gather_tasks(
            tasks=[
                loop.run_in_executor(
                    self.executor, self.preprocessor.prepare_blocks_for_ocr, *args
                )
                for args in zip(images, blocks_list)
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extract Preparation",
        )
        for img_idx, (block_images, prompts, params, indices) in enumerate(
            prepared_inputs
        ):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = await self.client.aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extraction",
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        return await gather_tasks(
            tasks=[
                self.postprocessor.aio_post_process_blocks(self.executor, blocks)
                for blocks in blocks_list
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Post Processing",
        )

    async def aio_batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        if self.batching_mode == "concurrent":
            return await self.aio_concurrent_two_step_extract(
                images, priority, semaphore
            )
        else:  # self.batching_mode == "stepping"
            return await self.aio_stepping_two_step_extract(images, priority, semaphore)
