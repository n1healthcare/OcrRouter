"""MinerU-specific image preprocessing before model inference."""

import math
from typing import Literal

from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import ContentBlock, get_png_bytes, get_rgb_image


class MinerUPreprocessor(BasePreprocessor):
    """MinerU-specific preprocessor for image preparation.

    Handles:
    - Image resizing for layout detection (fixed 1036x1036)
    - Block cropping with rotation for content extraction
    - Edge ratio adjustments for extreme aspect ratios
    """

    def __init__(
        self,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50.0,
        prompts: dict[str, str] | None = None,
        sampling_params: dict | None = None,
    ):
        """Initialize the MinerU preprocessor.

        Args:
            layout_image_size: Target size for layout detection images.
            min_image_edge: Minimum edge size for block images.
            max_image_edge_ratio: Maximum aspect ratio before padding.
            prompts: Prompt dictionary for different block types.
            sampling_params: Sampling parameters for different block types.
        """
        self.layout_image_size = layout_image_size
        self.min_image_edge = min_image_edge
        self.max_image_edge_ratio = max_image_edge_ratio
        self.prompts = prompts or {}
        self.sampling_params = sampling_params or {}

    def _resize_by_need(self, image: Image.Image) -> Image.Image:
        """Resize image if needed based on edge ratio and minimum size.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image.
        """
        edge_ratio = max(image.size) / min(image.size)
        if edge_ratio > self.max_image_edge_ratio:
            width, height = image.size
            if width > height:
                new_w, new_h = width, math.ceil(width / self.max_image_edge_ratio)
            else:  # width < height
                new_w, new_h = math.ceil(height / self.max_image_edge_ratio), height
            new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
            new_image.paste(
                image, (int((new_w - width) / 2), int((new_h - height) / 2))
            )
            image = new_image
        if min(image.size) < self.min_image_edge:
            scale = self.min_image_edge / min(image.size)
            new_w, new_h = round(image.width * scale), round(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image

    def prepare_for_layout(self, image: Image.Image) -> bytes:
        """Prepare image for layout detection.

        Converts to RGB and resizes to the fixed layout image size.

        Args:
            image: PIL Image of document page.

        Returns:
            PNG bytes of the prepared image.
        """
        image = get_rgb_image(image)
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        return get_png_bytes(image)

    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock,
    ) -> bytes:
        """Prepare cropped block image for OCR.

        Handles:
        - Cropping the block region from the full page image
        - Rotation based on block angle
        - Edge ratio adjustments

        Args:
            image: Full page PIL Image.
            block: ContentBlock with bbox and angle information.

        Returns:
            PNG bytes of the cropped and prepared image.
        """
        image = get_rgb_image(image)
        width, height = image.size

        x1, y1, x2, y2 = block.bbox
        scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
        block_image = image.crop(scaled_bbox)

        if block.angle in [90, 180, 270]:
            block_image = block_image.rotate(block.angle, expand=True)

        block_image = self._resize_by_need(block_image)
        return get_png_bytes(block_image)

    def prepare_blocks_for_ocr(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[bytes], list[str], list, list[int]]:
        """Prepare multiple blocks for OCR extraction.

        Filters out blocks that don't need OCR (image, list, equation_block)
        and prepares the remaining blocks.

        Args:
            image: Full page PIL Image.
            blocks: List of ContentBlocks to prepare.

        Returns:
            Tuple of (block_images, prompts, sampling_params, indices).
        """
        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[bytes] = []
        block_prompts: list[str] = []
        block_params: list = []
        indices: list[int] = []

        for idx, block in enumerate(blocks):
            # Skip blocks that don't need OCR
            if block.type in ("image", "list", "equation_block"):
                continue

            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)

            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)

            block_image = self._resize_by_need(block_image)
            block_images.append(get_png_bytes(block_image))

            prompt = self.prompts.get(block.type) or self.prompts.get("[default]", "")
            block_prompts.append(prompt)

            params = self.sampling_params.get(block.type) or self.sampling_params.get(
                "[default]"
            )
            block_params.append(params)

            indices.append(idx)

        return block_images, block_prompts, block_params, indices
