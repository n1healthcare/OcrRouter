"""Base postprocessor for processing model outputs."""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import Any

from ocrrouter.backends.utils import ContentBlock


class BasePostprocessor(ABC):
    """Abstract base class for processing model outputs.

    Postprocessors handle:
    - Parsing raw model output strings to ContentBlock lists
    - Applying model-specific post-processing fixes
    - Content cleaning and normalization

    Subclasses must implement:
    - parse_layout_output(): Parse raw model output to ContentBlocks
    - post_process_blocks(): Apply post-processing fixes to blocks
    """

    @abstractmethod
    def parse_layout_output(
        self,
        output: str,
        **context: Any,
    ) -> list[ContentBlock]:
        """Parse raw model output to ContentBlocks.

        Converts the model's raw output string to a list of ContentBlock objects.
        The parsing logic is model-specific (e.g., regex for MinerU, JSON for DotsOCR,
        grounding format for DeepSeek).

        Args:
            output: Raw string output from the model.
            **context: Model-specific context needed for parsing, such as:
                - orig_width, orig_height: Original image dimensions
                - resized_width, resized_height: Resized image dimensions
                - include_content: Whether to include content in parsing

        Returns:
            List of ContentBlock objects (with or without content based on mode).
        """
        pass

    @abstractmethod
    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing fixes to blocks.

        Performs model-specific post-processing such as:
        - Equation LaTeX fixes (MinerU)
        - Table HTML extraction from captions (DeepSeek)
        - Content cleaning and deduplication (DotsOCR)

        Args:
            blocks: List of ContentBlock objects to process.

        Returns:
            Processed list of ContentBlock objects.
        """
        pass

    async def aio_parse_layout_output(
        self,
        executor: Executor | None,
        output: str,
        **context: Any,
    ) -> list[ContentBlock]:
        """Async wrapper for parse_layout_output.

        Args:
            executor: Optional executor for running in thread pool.
            output: Raw string output from the model.
            **context: Model-specific context for parsing.

        Returns:
            List of ContentBlock objects.
        """
        loop = asyncio.get_running_loop()

        def _parse():
            return self.parse_layout_output(output, **context)

        return await loop.run_in_executor(executor, _parse)

    async def aio_post_process_blocks(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Async wrapper for post_process_blocks.

        Args:
            executor: Optional executor for running in thread pool.
            blocks: List of ContentBlock objects to process.

        Returns:
            Processed list of ContentBlock objects.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process_blocks, blocks)
