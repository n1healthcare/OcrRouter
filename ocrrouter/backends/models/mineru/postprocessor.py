"""MinerU-specific postprocessor for parsing and processing model outputs."""

import re
from typing import Literal

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock, BLOCK_TYPES, convert_otsl_to_html

# Import individual equation fix functions from the post_process submodules
from .post_process.equation_unbalanced_braces import try_fix_unbalanced_braces
from .post_process.equation_block import do_handle_equation_block
from .post_process.equation_double_subscript import try_fix_equation_double_subscript
from .post_process.equation_fix_eqqcolon import try_fix_equation_eqqcolon
from .post_process.equation_big import try_fix_equation_big
from .post_process.equation_leq import try_fix_equation_leq
from .post_process.equation_left_right import try_match_equation_left_right


# Layout detection regex pattern
# Format: <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|>optional_angle
_LAYOUT_RE = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"

# Angle token mapping
ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}

# Block types that are considered "paratext" (not main content)
PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}


def _convert_bbox(bbox: tuple[int, ...]) -> list[float] | None:
    """Convert MinerU bbox (0-1000 range) to normalized [0-1] range.

    Args:
        bbox: Tuple of (x1, y1, x2, y2) in 0-1000 range.

    Returns:
        List of normalized coordinates, or None if invalid.
    """
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    # Ensure correct ordering
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    # Skip zero-area boxes
    if x1 == x2 or y1 == y2:
        return None
    return [coord / 1000.0 for coord in (x1, y1, x2, y2)]


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    """Parse rotation angle from tail string.

    Args:
        tail: Tail portion of layout line.

    Returns:
        Rotation angle (0, 90, 180, 270) or None.
    """
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


def _process_equation(content: str, debug: bool) -> str:
    """Apply all equation fixes to LaTeX content.

    Args:
        content: LaTeX string to fix.
        debug: Enable debug output.

    Returns:
        Fixed LaTeX string.
    """
    content = try_match_equation_left_right(content, debug=debug)
    content = try_fix_equation_double_subscript(content, debug=debug)
    content = try_fix_equation_eqqcolon(content, debug=debug)
    content = try_fix_equation_big(content, debug=debug)
    content = try_fix_equation_leq(content, debug=debug)
    content = try_fix_unbalanced_braces(content, debug=debug)
    return content


def _add_equation_brackets(content: str) -> str:
    """Add LaTeX display math brackets if not present.

    Args:
        content: LaTeX string.

    Returns:
        LaTeX string wrapped in \\[ ... \\] if not already.
    """
    content = content.strip()
    if not content.startswith("\\["):
        content = f"\\[\n{content}"
    if not content.endswith("\\]"):
        content = f"{content}\n\\]"
    return content


class MinerUPostprocessor(BasePostprocessor):
    """MinerU-specific postprocessor for output parsing and fixes.

    Handles:
    - Parsing layout detection output (custom box format)
    - Applying equation LaTeX fixes
    - Converting table OTSL to HTML
    - Block filtering (paratext, list, equation_block)
    """

    def __init__(
        self,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        debug: bool = False,
    ):
        """Initialize the MinerU postprocessor.

        Args:
            handle_equation_block: Whether to merge equation blocks.
            abandon_list: Whether to remove list blocks.
            abandon_paratext: Whether to remove paratext blocks (header, footer, etc.).
            debug: Enable debug output for equation fixes.
        """
        self.handle_equation_block = handle_equation_block
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.debug = debug

    def parse_layout_output(
        self,
        output: str,
        **context,
    ) -> list[ContentBlock]:
        """Parse MinerU layout detection output to ContentBlocks.

        The MinerU layout format is:
        <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|><|angle|>

        Args:
            output: Raw layout detection output string.
            **context: Unused context (for interface compatibility).

        Returns:
            List of ContentBlock objects without content.
        """
        blocks: list[ContentBlock] = []

        for line in output.split("\n"):
            match = re.match(_LAYOUT_RE, line)
            if not match:
                if line.strip():  # Only warn for non-empty lines
                    print(f"Warning: line does not match layout format: {line}")
                continue

            x1, y1, x2, y2, ref_type, tail = match.groups()
            bbox = _convert_bbox(tuple(map(int, (x1, y1, x2, y2))))
            if bbox is None:
                print(f"Warning: invalid bbox in line: {line}")
                continue

            ref_type = ref_type.lower()
            if ref_type not in BLOCK_TYPES:
                print(f"Warning: unknown block type in line: {line}")
                continue

            angle = _parse_angle(tail)
            if angle is None and tail.strip():
                print(f"Warning: no angle found in line: {line}")

            blocks.append(ContentBlock(ref_type, bbox, angle=angle))

        return blocks

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing fixes to blocks.

        Applies:
        - Table OTSL to HTML conversion
        - Equation LaTeX fixes
        - Equation block handling
        - Block filtering (list, paratext)

        Args:
            blocks: List of ContentBlock objects to process.

        Returns:
            Processed list of ContentBlock objects.
        """
        # Step 1: Process table and equation content
        for block in blocks:
            if block.type == "table" and block.content:
                block.content = convert_otsl_to_html(block.content)
            if block.type == "equation" and block.content:
                block.content = _process_equation(block.content, debug=self.debug)

        # Step 2: Handle equation blocks (merge adjacent equations)
        if self.handle_equation_block:
            blocks = do_handle_equation_block(blocks, debug=self.debug)

        # Step 3: Add equation brackets
        for block in blocks:
            if block.type == "equation" and block.content:
                block.content = _add_equation_brackets(block.content)

        # Step 4: Filter out unwanted blocks
        out_blocks: list[ContentBlock] = []
        for block in blocks:
            if block.type == "equation_block":  # drop equation_block anyway
                continue
            if self.abandon_list and block.type == "list":
                continue
            if self.abandon_paratext and block.type in PARATEXT_TYPES:
                continue
            out_blocks.append(block)

        return out_blocks
