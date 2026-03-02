"""LightOnOCR-specific image preprocessing before model inference."""

from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import ContentBlock, get_png_bytes, get_rgb_image

# LightOnOCR recommended input: 1540px longest dimension
MAX_DIMENSION = 1540


class LightOnOCRPreprocessor(BasePreprocessor):
    """LightOnOCR-specific preprocessor for image preparation.

    Handles:
    - Image resizing to max 1540px longest dimension
    - RGB conversion
    - PNG byte conversion
    """

    def __init__(self, max_dimension: int = MAX_DIMENSION):
        """Initialize the LightOnOCR preprocessor.

        Args:
            max_dimension: Maximum dimension for longest side (default 1540px).
        """
        self.max_dimension = max_dimension

    def prepare_for_layout(self, image: Image.Image) -> bytes:
        """Prepare image for layout detection / OCR.

        Resizes the image so the longest dimension is at most max_dimension,
        maintaining aspect ratio.

        Args:
            image: PIL Image of document page.

        Returns:
            PNG bytes of the prepared image.
        """
        image = get_rgb_image(image)
        max_dim = max(image.size)

        if max_dim > self.max_dimension:
            scale = self.max_dimension / max_dim
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return get_png_bytes(image)

    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock,
    ) -> bytes:
        """Prepare cropped block image for OCR.

        Crops the block region from the full page image and resizes if needed.

        Args:
            image: Full page PIL Image.
            block: ContentBlock with bbox information.

        Returns:
            Cropped and prepared PNG image bytes.
        """
        image = get_rgb_image(image)
        width, height = image.size

        # Crop the block region
        x1, y1, x2, y2 = block.bbox
        crop_box = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )
        cropped = image.crop(crop_box)

        # Resize if needed
        max_dim = max(cropped.size)
        if max_dim > self.max_dimension:
            scale = self.max_dimension / max_dim
            new_size = (int(cropped.width * scale), int(cropped.height * scale))
            cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

        return get_png_bytes(cropped)
