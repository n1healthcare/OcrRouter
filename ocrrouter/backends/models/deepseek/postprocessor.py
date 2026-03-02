"""DeepSeek-OCR 2 postprocessor for parsing and processing model outputs.

Supports two output formats:
- v1 format: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>content
- v2 format: label[[x1, y1, x2, y2]]content

Both formats use bbox range 0-999 (normalized to 0-1 during parsing).
"""

import ast
import re

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import BLOCK_TYPES, ContentBlock

from .utils.structs import map_deepseek_label

# DeepSeek v1 grounding format regex
# Format: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>content
_GROUNDING_PATTERN_V1 = re.compile(r"<\|ref\|>(.+?)<\|/ref\|><\|det\|>(\[.+?\])<\|/det\|>")

# DeepSeek v2 grounding format regex
# Format: label[[x1, y1, x2, y2]]content or label[[x1, y1, x2, y2], [x1, y1, x2, y2]]content
_GROUNDING_PATTERN_V2 = re.compile(r"(\w+)\[\[([\d,\s\[\]]+)\]\]")

# Pattern to extract HTML table from content
_TABLE_HTML_PATTERN = re.compile(r"<table>.*?</table>", re.DOTALL)


def _convert_bbox_deepseek(x1: int | str, y1: int | str, x2: int | str, y2: int | str) -> list[float] | None:
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


def _parse_coords_array(coords_str: str, is_v2: bool = False) -> list[list[int]] | None:
    """Parse coordinates array from string.

    Handles multiple formats:
    - v1 single bbox: [[x1, y1, x2, y2]]
    - v1 multi-bbox: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    - v2 single bbox: x1, y1, x2, y2 (no brackets)
    - v2 multi-bbox: x1, y1, x2, y2], [x1, y1, x2, y2 (captured without outer brackets)

    Args:
        coords_str: Coordinate string in v1 or v2 format
        is_v2: Whether this is v2 format (different parsing)

    Returns:
        List of coordinate lists, or None on parse error
    """
    try:
        if is_v2:
            # v2 format captured from regex (inside the [[ ]])
            # Could be: "x1, y1, x2, y2" (single) or "x1,y1,x2,y2], [x1,y1,x2,y2" (multi, without outer [])
            coords_str = coords_str.strip()

            # Check if this is multi-bbox (contains "], [" pattern)
            if "], [" in coords_str or "],[" in coords_str:
                # Multi-bbox: wrap with [[ ]] to make it a valid nested list
                parsed = ast.literal_eval(f"[[{coords_str}]]")
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], list):
                        return parsed
                return None
            elif coords_str.startswith("["):
                # Single bbox with brackets: [x1, y1, x2, y2]
                parsed = ast.literal_eval(f"[{coords_str}]")
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], list):
                        return parsed
                    else:
                        return [parsed]
                return None
            else:
                # Single bbox without brackets: "x1, y1, x2, y2"
                coords = [int(x.strip()) for x in coords_str.split(",")]
                if len(coords) == 4:
                    return [coords]
                return None
        else:
            # v1 format: "[[x1, y1, x2, y2]]" or "[[...], [...]]"
            parsed = ast.literal_eval(coords_str)
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
                if next_block.type in ("table_caption", "table_footnote") and next_block.content:
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

        Supports two formats:
        - v1: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>content
        - v2: label[[x1, y1, x2, y2]]content

        Args:
            output: Raw model output string.
            **context: Unused context.

        Returns:
            List of ContentBlock objects with type, bbox, and content.
        """
        blocks: list[ContentBlock] = []

        # Try v1 format first
        matches_v1 = list(_GROUNDING_PATTERN_V1.finditer(output))
        matches_v2 = list(_GROUNDING_PATTERN_V2.finditer(output))

        # Use whichever format has more matches (v2 typically)
        if len(matches_v1) >= len(matches_v2) and len(matches_v1) > 0:
            matches = matches_v1
            is_v2 = False
        elif len(matches_v2) > 0:
            matches = matches_v2
            is_v2 = True
        else:
            # No matches found
            return blocks

        for i, match in enumerate(matches):
            label_type = match.group(1)
            coords_str = match.group(2)

            # Parse coordinates (handles both single and multi-bbox)
            coords_list = _parse_coords_array(coords_str, is_v2=is_v2)
            if coords_list is None:
                print(f"Warning: failed to parse coords in match: {match.group(0)}")
                continue

            # Map DeepSeek label to MinerU block type
            block_type = map_deepseek_label(label_type)
            if block_type not in BLOCK_TYPES:
                print(f"Warning: unknown block type after mapping: {label_type} -> {block_type}")
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
                blocks.append(ContentBlock(block_type, bbox, angle=None, content=block_content))

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
