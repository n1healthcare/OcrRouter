"""GLM-OCR postprocessor for processing model outputs.

GLM-OCR returns markdown text directly, so postprocessing is simpler
than grounding-based models. Main tasks are:
- Cleaning up output text
- Removing repeated content
- Normalizing formatting
"""

import re

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock


class GlmOCRPostprocessor(BasePostprocessor):
    """GLM-OCR-specific postprocessor for output processing.

    Handles:
    - Cleaning up model output text
    - Removing repeated punctuation/content
    - Normalizing LaTeX delimiters
    """

    def __init__(self, debug: bool = False):
        """Initialize the GLM-OCR postprocessor.

        Args:
            debug: Enable debug output.
        """
        self.debug = debug

    def clean_output(self, output: str) -> str:
        """Clean up model output text.

        Args:
            output: Raw model output string.

        Returns:
            Cleaned output string.
        """
        if not output:
            return ""

        # Remove leading/trailing literal \t
        output = re.sub(r"^(\\t)+", "", output).lstrip()
        output = re.sub(r"(\\t)+$", "", output).rstrip()

        # Remove repeated punctuation
        output = re.sub(r"(\.)\1{2,}", r"\1\1\1", output)
        output = re.sub(r"(·)\1{2,}", r"\1\1\1", output)
        output = re.sub(r"(_)\1{2,}", r"\1\1\1", output)
        output = re.sub(r"(\\_)\1{2,}", r"\1\1\1", output)

        # Normalize LaTeX delimiters
        output = output.replace(r"\[", "$$").replace(r"\]", "$$")
        output = output.replace(r"\(", "$").replace(r"\)", "$")

        # Remove repeated substrings for long content
        if len(output) >= 2048:
            output = self._clean_repeated_content(output)

        return output.strip()

    def _clean_repeated_content(self, content: str) -> str:
        """Remove repeated substrings from content.

        This handles cases where the model repeats content multiple times,
        which can happen with long documents.

        Args:
            content: Text content to clean.

        Returns:
            Cleaned content with repeats removed.
        """
        # Simple heuristic: find repeated patterns of 50+ chars
        # that appear more than twice consecutively
        pattern = re.compile(r"(.{50,}?)\1{2,}", re.DOTALL)
        cleaned = pattern.sub(r"\1", content)

        # Also handle shorter repeated patterns (20+ chars)
        pattern_short = re.compile(r"(.{20,}?)\1{3,}", re.DOTALL)
        cleaned = pattern_short.sub(r"\1", cleaned)

        return cleaned

    def parse_layout_output(
        self,
        output: str,
        **context,
    ) -> list[ContentBlock]:
        """Parse model output to ContentBlocks.

        Note: GLM-OCR doesn't have native layout detection output format.
        This method returns a single full-page text block.

        Args:
            output: Raw model output string.
            **context: Unused context.

        Returns:
            List containing a single ContentBlock for the full page.
        """
        cleaned_output = self.clean_output(output)

        if not cleaned_output:
            return []

        # Return single full-page block
        return [
            ContentBlock(
                type="text",
                bbox=[0.0, 0.0, 1.0, 1.0],
                angle=None,
                content=cleaned_output,
            )
        ]

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing to blocks.

        For GLM-OCR, this mainly cleans up the content text.

        Args:
            blocks: List of ContentBlock objects.

        Returns:
            Processed list of ContentBlock objects.
        """
        for block in blocks:
            if block.content:
                block.content = self.clean_output(block.content)
        return blocks
