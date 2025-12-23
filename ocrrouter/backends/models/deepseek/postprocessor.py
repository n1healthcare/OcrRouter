"""DeepSeek-specific postprocessor for parsing and processing model outputs."""

import ast
import re
from typing import Any

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock, BLOCK_TYPES
from .utils.structs import map_deepseek_label


# DeepSeek grounding format regex
# Format: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>content
# or multi-bbox: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], [x1, y1, x2, y2]]<|/det|>content
_GROUNDING_PATTERN = re.compile(r"<\|ref\|>(.+?)<\|/ref\|><\|det\|>(\[.+?\])<\|/det\|>")

# Pattern to extract HTML table from content
_TABLE_HTML_PATTERN = re.compile(r"<table>.*?</table>", re.DOTALL)


def _convert_bbox_deepseek(
    x1: int | str, y1: int | str, x2: int | str, y2: int | str
) -> list[float] | None:
    """Convert DeepSeek bbox (0-999 range) to normalized [0-1] range.

    Args:
        x1, y1, x2, y2: Coordinates in 0-999 range

    Returns:
        List of normalized coordinates [x1, y1, x2, y2] in 0-1 range,
        or None if coordinates are invalid.
    """
    coords = tuple(map(int, (x1, y1, x2, y2)))

    # Validate range
    if any(coord < 0 or coord > 999 for coord in coords):
        return None

    x1, y1, x2, y2 = coords

    # Ensure correct ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Skip zero-area boxes
    if x1 == x2 or y1 == y2:
        return None

    # Normalize to 0-1 (DeepSeek uses 0-999)
    return [x1 / 999.0, y1 / 999.0, x2 / 999.0, y2 / 999.0]


def _parse_coords_array(coords_str: str) -> list[list[int]] | None:
    """Parse coordinates array from string.

    Handles both single bbox: [[x1, y1, x2, y2]]
    and multi-bbox: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]

    Args:
        coords_str: String like "[[134, 213, 836, 241]]" or "[[73, 749, 488, 915], [508, 236, 920, 295]]"

    Returns:
        List of coordinate lists, or None on parse error
    """
    try:
        parsed = ast.literal_eval(coords_str)
        # Check if it's a single bbox wrapped in one list: [[x1, y1, x2, y2]]
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], int):
                # Single bbox without outer wrapper: [x1, y1, x2, y2]
                return [parsed]
            elif isinstance(parsed[0], list):
                # Multiple bboxes or single wrapped: [[x1, y1, x2, y2]] or [[...], [...]]
                return parsed
        return None
    except (ValueError, SyntaxError) as e:
        print(f"Warning: failed to parse coords '{coords_str}': {e}")
        return None


def _extract_table_html_from_caption(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Post-process to move table HTML from caption/footnote to table block.

    DeepSeek can output table HTML as part of table_caption or table_footnote content,
    with the caption appearing either BEFORE or AFTER the table block:

    Case 1 (caption above table):
    - table_caption: "Caption text...\n\n<table>...</table>"
    - table: "" (empty)

    Case 2 (caption below table):
    - table: "" (empty)
    - table_caption: "Caption text...\n\n<table>...</table>"

    Args:
        blocks: List of ContentBlock objects to process

    Returns:
        The same list with table HTML moved to correct blocks
    """
    for i, block in enumerate(blocks):
        # Case 1: Empty table with caption/footnote BELOW containing HTML
        if block.type == "table" and not block.content:
            if i + 1 < len(blocks):
                next_block = blocks[i + 1]
                if (
                    next_block.type in ("table_caption", "table_footnote")
                    and next_block.content
                ):
                    match = _TABLE_HTML_PATTERN.search(next_block.content)
                    if match:
                        # Move HTML to table block
                        block.content = match.group(0)
                        # Keep only caption text
                        caption_text = next_block.content[: match.start()].strip()
                        next_block.content = caption_text if caption_text else None

        # Case 2: Caption/footnote with HTML, empty table BELOW
        elif block.type in ("table_caption", "table_footnote") and block.content:
            match = _TABLE_HTML_PATTERN.search(block.content)
            if match:
                # Find the next table block with empty content
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    if next_block.type == "table" and not next_block.content:
                        # Move HTML to table block
                        next_block.content = match.group(0)
                        # Keep only caption text
                        caption_text = block.content[: match.start()].strip()
                        block.content = caption_text if caption_text else None

    return blocks


class DeepSeekPostprocessor(BasePostprocessor):
    """DeepSeek-specific postprocessor for output parsing.

    Handles:
    - Parsing grounding format output
    - Bbox coordinate conversion (0-999 to 0-1)
    - Table HTML extraction from captions
    """

    def __init__(self, debug: bool = False):
        """Initialize the DeepSeek postprocessor.

        Args:
            debug: Enable debug output.
        """
        self.debug = debug

    def parse_layout_output(
        self,
        output: str,
        **context,
    ) -> list[ContentBlock]:
        """Parse DeepSeek grounding output to ContentBlocks.

        The grounding output format is:
        <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
        Content text here...

        Or with multiple bboxes (for content spanning columns/areas):
        <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], [x1, y1, x2, y2]]<|/det|>
        Content text here...

        Args:
            output: Raw model output string.
            **context: Unused context.

        Returns:
            List of ContentBlock objects with type, bbox, and content.
        """
        blocks: list[ContentBlock] = []

        # Find all grounding markers and their positions
        matches = list(_GROUNDING_PATTERN.finditer(output))

        for i, match in enumerate(matches):
            label_type = match.group(1)
            coords_str = match.group(2)

            # Parse coordinates (handles both single and multi-bbox)
            coords_list = _parse_coords_array(coords_str)
            if coords_list is None:
                print(f"Warning: failed to parse coords in match: {match.group(0)}")
                continue

            # Map DeepSeek label to MinerU block type
            block_type = map_deepseek_label(label_type)
            if block_type not in BLOCK_TYPES:
                print(
                    f"Warning: unknown block type after mapping: {label_type} -> {block_type}"
                )
                block_type = "unknown"

            # Extract content: from end of this match to start of next match (or end of string)
            content_start = match.end()
            if i + 1 < len(matches):
                content_end = matches[i + 1].start()
            else:
                content_end = len(output)

            content = output[content_start:content_end].strip()

            # For image blocks, content is typically empty or whitespace
            if block_type == "image":
                content = None

            # Create ContentBlock for each bbox
            # First bbox gets the content, additional bboxes get empty string
            for bbox_idx, coords in enumerate(coords_list):
                if len(coords) != 4:
                    print(f"Warning: invalid coords length {len(coords)}: {coords}")
                    continue

                x1, y1, x2, y2 = coords
                bbox = _convert_bbox_deepseek(x1, y1, x2, y2)
                if bbox is None:
                    print(f"Warning: invalid bbox: {coords}")
                    continue

                # First bbox gets content, additional bboxes get empty string
                block_content = content if bbox_idx == 0 else ""

                # DeepSeek doesn't provide angle information, default to None
                blocks.append(
                    ContentBlock(block_type, bbox, angle=None, content=block_content)
                )

        return blocks

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing fixes to blocks.

        Applies:
        - Table HTML extraction from caption blocks

        Args:
            blocks: List of ContentBlock objects.

        Returns:
            Processed list of ContentBlock objects.
        """
        # Move table HTML from caption to table blocks
        blocks = _extract_table_html_from_caption(blocks)
        return blocks
