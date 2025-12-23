"""Hunyuan-specific postprocessor for parsing and processing model outputs."""

import html
import re
from typing import Any

from PIL import Image

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock


# Coordinate pattern: (x1,y1),(x2,y2) where coordinates are 0-1000
COORD_PATTERN = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")


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


def normalize_coord_to_bbox(x1: int, y1: int, x2: int, y2: int) -> list[float] | None:
    """Convert Hunyuan coordinates (0-1000 range) to normalized [0-1] bbox.

    Args:
        x1, y1, x2, y2: Coordinates in 0-1000 range

    Returns:
        List of normalized coordinates [x1, y1, x2, y2] in 0-1 range,
        or None if coordinates are invalid.
    """
    # Ensure correct ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Validate range
    if x1 < 0 or x2 > 1000 or y1 < 0 or y2 > 1000:
        return None

    # Skip zero-area or tiny boxes
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    # Normalize to 0-1
    return [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]


class HunyuanOCRPostprocessor(BasePostprocessor):
    """Hunyuan-specific postprocessor for output parsing.

    Handles:
    - Parsing OCR output with embedded coordinates
    - Creating separate ContentBlocks for text and images
    - Content cleaning and normalization
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

    def parse_layout_output(
        self,
        output: str,
        page_image: Image.Image | None = None,
        **context: Any,
    ) -> list[ContentBlock]:
        """Parse Hunyuan OCR markdown output into ContentBlocks.

        The output may contain embedded figure coordinates in format:
        (x1,y1),(x2,y2) where coordinates are in 0-1000 range.

        This method:
        1. Finds all coordinate patterns
        2. Splits text at each coordinate occurrence
        3. Creates text ContentBlocks for text segments
        4. Creates image ContentBlocks for each coordinate
        5. Returns ordered list maintaining document flow

        Args:
            output: Raw OCR output string from Hunyuan
            page_image: Original page image (for reference, not used directly)
            **context: Additional context (unused)

        Returns:
            List of ContentBlock objects (text and image blocks)
        """
        if not output:
            return []

        # Clean the output first
        output = self._clean_ocr_output(output)

        blocks: list[ContentBlock] = []

        # Find all coordinate matches with their positions
        matches = list(COORD_PATTERN.finditer(output))

        if not matches:
            # No coordinates found - return single text block
            content = output.strip()
            if content:
                blocks.append(
                    ContentBlock(
                        type="text",
                        bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                        content=content,
                    )
                )
            return blocks

        # Process output with coordinates
        last_end = 0

        for match in matches:
            # Extract text before this coordinate
            text_before = output[last_end : match.start()].strip()

            if text_before:
                # Create text block for content before the image
                blocks.append(
                    ContentBlock(
                        type="text",
                        bbox=[0.0, 0.0, 1.0, 1.0],  # Full page for text
                        content=text_before,
                    )
                )

            # Extract coordinates and create image block
            x1, y1, x2, y2 = map(int, match.groups())
            bbox = normalize_coord_to_bbox(x1, y1, x2, y2)

            if bbox is not None:
                blocks.append(
                    ContentBlock(
                        type="image",
                        bbox=bbox,
                        content=None,  # Image blocks don't have text content
                    )
                )

            last_end = match.end()

        # Handle text after the last coordinate
        text_after = output[last_end:].strip()
        if text_after:
            blocks.append(
                ContentBlock(
                    type="text",
                    bbox=[0.0, 0.0, 1.0, 1.0],  # Full page for text
                    content=text_after,
                )
            )

        return blocks

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing fixes to blocks.

        For Hunyuan, the cleaning is already done in parse_layout_output,
        so this filters out empty text blocks.
        """
        # Filter out empty text blocks
        return [
            block
            for block in blocks
            if block.type != "text" or (block.content and block.content.strip())
        ]
