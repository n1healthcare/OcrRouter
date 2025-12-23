"""PaddleOCR-specific postprocessor for parsing and processing model outputs."""

import html
import re
from typing import Any

from PIL import Image

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock, convert_otsl_to_html


def clean_repeated_substrings(text: str) -> str:
    """Clean repeated substrings in text.

    This handles a model bug where text gets repeated excessively.
    Only applies to long outputs (8000+ chars).
    """
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[: n - length * (count - 1)]

    return text


def fix_latex_spacing(content: str) -> str:
    """Remove leading/trailing spaces inside LaTeX delimiters."""
    # Fix display math first ($$...$$)
    content = re.sub(r"\$\$\s+(.+?)\s+\$\$", r"$$\1$$", content, flags=re.DOTALL)
    # Fix inline math ($...$)
    content = re.sub(r"(?<!\$)\$\s+([^$]+?)\s+\$(?!\$)", r"$\1$", content)
    return content


class PaddleOCRPostprocessor(BasePostprocessor):
    """PaddleOCR-specific postprocessor for output parsing.

    Handles:
    - Parsing plain text/markdown OCR output
    - Creating ContentBlocks for text content
    - Content cleaning and normalization

    Note: PaddleOCR returns plain text/markdown without coordinates,
    so no coordinate parsing is needed (unlike Hunyuan).
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def _clean_ocr_output(self, content: str) -> str:
        """Clean raw OCR output."""
        if not content:
            return content

        # Clean repeated substrings
        content = clean_repeated_substrings(content)

        # Fix LaTeX spacing
        content = fix_latex_spacing(content)

        # Unescape HTML entities
        content = html.unescape(content)

        # Clean up end-of-sentence markers
        if "<|end_of_sentence|>" in content:
            content = content.replace("<|end_of_sentence|>", "")

        # Normalize newlines
        content = content.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

        return content

    def _convert_table_to_html(self, otsl_content: str) -> str:
        """Convert OTSL format table content to HTML.

        Args:
            otsl_content: OTSL formatted string from PaddleOCR table recognition.

        Returns:
            HTML table string with proper rowspan/colspan attributes.
        """
        return convert_otsl_to_html(otsl_content)

    def parse_layout_output(
        self,
        output: str,
        page_image: Image.Image | None = None,
        **context: Any,
    ) -> list[ContentBlock]:
        """Parse PaddleOCR output into ContentBlocks.

        Since PaddleOCR returns plain text/markdown without coordinates,
        we create a single full-page text block.

        Args:
            output: Raw OCR output string from PaddleOCR
            page_image: Original page image (for reference, not used directly)
            **context: Additional context (unused)

        Returns:
            List containing a single text ContentBlock
        """
        if not output:
            return []

        # Clean the output
        output = self._clean_ocr_output(output)

        content = output.strip()
        if not content:
            return []

        # Return single full-page text block
        return [
            ContentBlock(
                type="text",
                bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                content=content,
            )
        ]

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing fixes to blocks.

        Filters out empty text blocks.
        """
        return [
            block
            for block in blocks
            if block.type != "text" or (block.content and block.content.strip())
        ]
