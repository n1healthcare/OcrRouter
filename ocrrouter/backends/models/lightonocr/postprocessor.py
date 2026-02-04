"""LightOnOCR-specific output postprocessing after model inference."""

from typing import Literal

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock
from .utils import parse_image_bboxes, remove_image_markers


class LightOnOCRPostprocessor(BasePostprocessor):
    """LightOnOCR-specific postprocessor for model outputs.

    LightOnOCR outputs markdown text with embedded image bounding boxes in format:
    `![image](image_N.png) x1,y1,x2,y2`

    This postprocessor handles:
    - Extracting image bboxes from the output
    - Creating ContentBlocks for detected images
    - Creating a text ContentBlock for the OCR output
    """

    def __init__(self, debug: bool = False):
        """Initialize the LightOnOCR postprocessor.

        Args:
            debug: Enable debug mode for additional logging.
        """
        self.debug = debug

    def parse_layout_output(
        self,
        output: str,
        output_mode: Literal["all", "layout_only", "ocr_only"] = "all",
        **context,
    ) -> list[ContentBlock]:
        """Parse LightOnOCR layout output to ContentBlock list.

        The output format is markdown with image bboxes:
        ```
        Text content here...

        ![image](image_1.png) 120,50,850,400

        More text...
        ```

        Args:
            output: Raw model output string (markdown with image bboxes).
            output_mode: Output mode controlling what blocks to return:
                - 'all': Return both image blocks and text block
                - 'layout_only': Return only image blocks (no text content)
                - 'ocr_only': Return only text block (no image blocks)
            **context: Additional context (unused).

        Returns:
            List of ContentBlock objects.
        """
        blocks: list[ContentBlock] = []

        if not output:
            return blocks

        output = output.strip()

        # Extract image bboxes
        image_bboxes = parse_image_bboxes(output)

        if output_mode == "layout_only":
            # Return only image bbox ContentBlocks
            for _, bbox in image_bboxes:
                blocks.append(
                    ContentBlock(
                        type="image",
                        bbox=bbox,
                        angle=None,
                        content=None,
                    )
                )
        elif output_mode == "ocr_only":
            # Return single text ContentBlock with full markdown content
            # Remove image markers from the text
            cleaned_text = remove_image_markers(output)
            if cleaned_text:
                blocks.append(
                    ContentBlock(
                        type="text",
                        bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                        angle=None,
                        content=cleaned_text,
                    )
                )
        else:  # "all" mode
            # Return image blocks + text block
            for _, bbox in image_bboxes:
                blocks.append(
                    ContentBlock(
                        type="image",
                        bbox=bbox,
                        angle=None,
                        content=None,
                    )
                )

            # Add full-page text block with cleaned markdown
            cleaned_text = remove_image_markers(output)
            if cleaned_text:
                blocks.append(
                    ContentBlock(
                        type="text",
                        bbox=[0.0, 0.0, 1.0, 1.0],  # Full page
                        angle=None,
                        content=cleaned_text,
                    )
                )

        return blocks

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing to extracted content blocks.

        Currently minimal processing is needed for LightOnOCR outputs.

        Args:
            blocks: List of ContentBlock to post-process.

        Returns:
            List of post-processed ContentBlock.
        """
        for block in blocks:
            if block.content:
                # Clean up any extra whitespace
                block.content = block.content.strip()

        return blocks
