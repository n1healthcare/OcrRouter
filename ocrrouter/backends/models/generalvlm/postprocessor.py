"""General VLM postprocessor for parsing and processing model outputs."""

import html
import re
from typing import Any

from PIL import Image

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock


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


class GeneralVLMPostprocessor(BasePostprocessor):
    """General VLM postprocessor for output parsing.

    Handles:
    - Cleaning raw OCR output
    - Creating ContentBlocks for full-page text
    - Content cleaning and normalization

    Note: Unlike HunyuanOCR, General VLM outputs full-page markdown without
    embedded coordinates, so parsing is simpler.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def _clean_ocr_output(self, content: str) -> str:
        """Clean raw OCR output."""
        if not content:
            return content

        # Try to find opening code block fence and extract content
        # This approach explicitly finds the opening fence, then the LAST closing fence
        # to properly handle nested code blocks
        # Priority order: markdown > md > plain ```
        opening_patterns = [
            r"```markdown\s*\n",
            r"```md\s*\n",
            r"```\s*\n",  # Plain code fence without language specifier
        ]

        for pattern in opening_patterns:
            match = re.search(pattern, content)
            if match:
                # Found opening fence, get position after it
                start_pos = match.end()

                # Find the LAST occurrence of ``` after the opening fence
                last_closing = content.rfind("```", start_pos)

                if last_closing != -1:
                    # Extract content between opening and last closing fence
                    content = content[start_pos:last_closing].strip()
                    break  # Successfully extracted, stop looking

        # Clean repeated substrings
        content = clean_repeated_substrings(content)

        # Fix LaTeX spacing
        content = fix_latex_spacing(content)

        # Unescape HTML entities
        content = html.unescape(content)

        # Clean up common model artifacts
        if "<|end_of_sentence|>" in content:
            content = content.replace("<|end_of_sentence|>", "")

        # Remove image markdown references (images don't exist in OCR output)
        # Replace ![any text](any path) with ![]()
        content = re.sub(r"!\[.*?\]\(.*?\)", "![]()", content)

        # Normalize newlines
        content = content.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

        return content

    def parse_layout_output(
        self,
        output: str,
        page_image: Image.Image | None = None,
        **context: Any,
    ) -> list[ContentBlock]:
        """Parse General VLM OCR output into ContentBlocks.

        General VLM outputs full-page markdown without coordinates.
        Returns a single full-page text ContentBlock.

        Args:
            output: Raw OCR output string from Gemini
            page_image: Original page image (for reference, not used directly)
            **context: Additional context (unused)

        Returns:
            List containing a single full-page text ContentBlock
        """
        if not output:
            return []

        # Clean the output
        content = self._clean_ocr_output(output)

        if not content.strip():
            return []

        # Return single full-page text block
        return [
            ContentBlock(
                type="text",
                bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                content=content.strip(),
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
