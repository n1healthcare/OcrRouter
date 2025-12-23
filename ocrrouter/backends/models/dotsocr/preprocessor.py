"""DotsOCR-specific image preprocessing before model inference."""

import math
from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import get_png_bytes, get_rgb_image


# Image processing constants
IMAGE_FACTOR = 28
MIN_PIXELS = 3136
MAX_PIXELS = 11289600


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Rescales image dimensions so that:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels is within [min_pixels, max_pixels]
    3. Aspect ratio is maintained as closely as possible

    Args:
        height: Original height
        width: Original width
        factor: Dimensions must be divisible by this
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (new_height, new_width)
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, floor_by_factor(int(width / beta), factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, floor_by_factor(int(h_bar / beta), factor))
            w_bar = max(factor, floor_by_factor(int(w_bar / beta), factor))

    return h_bar, w_bar


class DotsOCRPreprocessor(BasePreprocessor):
    """DotsOCR-specific preprocessor for image preparation.

    Handles:
    - Image resizing with smart_resize algorithm
    - RGB conversion
    - Maintaining aspect ratio within constraints
    """

    def __init__(
        self,
        image_factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ):
        """Initialize the DotsOCR preprocessor.

        Args:
            image_factor: Dimensions must be divisible by this factor
            min_pixels: Minimum total pixels
            max_pixels: Maximum total pixels
        """
        self.image_factor = image_factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def prepare_for_layout(
        self, image: Image.Image
    ) -> tuple[bytes, int, int, int, int]:
        """Prepare image for layout detection.

        Applies smart_resize to ensure image fits within pixel constraints
        and dimensions are divisible by IMAGE_FACTOR.

        Args:
            image: PIL Image of document page

        Returns:
            Tuple of (image_bytes, orig_width, orig_height, resized_width, resized_height)
        """
        image = get_rgb_image(image)
        orig_width, orig_height = image.size

        # Apply smart_resize
        resized_height, resized_width = smart_resize(
            orig_height,
            orig_width,
            factor=self.image_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Resize if needed
        if (resized_width, resized_height) != (orig_width, orig_height):
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            )

        return (
            get_png_bytes(image),
            orig_width,
            orig_height,
            resized_width,
            resized_height,
        )

    def prepare_for_ocr(self, image: Image.Image) -> bytes:
        """Prepare image for OCR extraction.

        Args:
            image: PIL Image to prepare

        Returns:
            PNG bytes of the prepared image
        """
        image = get_rgb_image(image)
        return get_png_bytes(image)
