"""GLM-OCR image preprocessing before model inference.

GLM-OCR uses JPEG format by default with configurable pixel ranges.
The model supports dynamic resolution with min/max pixel constraints.
"""

from io import BytesIO

from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import ContentBlock, get_rgb_image


class GlmOCRPreprocessor(BasePreprocessor):
    """GLM-OCR-specific preprocessor for image preparation.

    GLM-OCR works with JPEG images by default and applies pixel
    constraints for optimal model performance.
    """

    def __init__(
        self,
        image_format: str = "JPEG",
        min_pixels: int = 112 * 112,
        max_pixels: int = 14 * 14 * 4 * 1280,
    ):
        """Initialize the GLM-OCR preprocessor.

        Args:
            image_format: Output image format (JPEG, PNG, WEBP).
            min_pixels: Minimum total pixels for the image.
            max_pixels: Maximum total pixels for the image.
        """
        self.image_format = image_format
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds pixel constraints.

        Args:
            image: PIL Image to potentially resize.

        Returns:
            Resized or original PIL Image.
        """
        width, height = image.size
        total_pixels = width * height

        if total_pixels > self.max_pixels:
            # Scale down to fit within max_pixels
            scale = (self.max_pixels / total_pixels) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif total_pixels < self.min_pixels:
            # Scale up to meet min_pixels
            scale = (self.min_pixels / total_pixels) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes in the configured format.

        Args:
            image: PIL Image to convert.

        Returns:
            Image bytes.
        """
        buffered = BytesIO()
        if self.image_format.upper() == "JPEG":
            # JPEG requires RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=95)
        elif self.image_format.upper() == "PNG":
            image.save(buffered, format="PNG")
        elif self.image_format.upper() == "WEBP":
            image.save(buffered, format="WEBP", quality=95)
        else:
            # Default to PNG
            image.save(buffered, format="PNG")
        return buffered.getvalue()

    def prepare_for_layout(self, image: Image.Image) -> bytes:
        """Prepare image for layout detection.

        Note: GLM-OCR doesn't have native layout detection.
        This method is provided for interface compatibility.

        Args:
            image: PIL Image of document page.

        Returns:
            Image bytes.
        """
        image = get_rgb_image(image)
        image = self._resize_if_needed(image)
        return self._to_bytes(image)

    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock | None = None,
    ) -> bytes:
        """Prepare image for OCR.

        If a block is provided, crops the image to the block's bbox.
        Otherwise, processes the full image.

        Args:
            image: Full page PIL Image.
            block: Optional ContentBlock with bbox information.

        Returns:
            Image bytes (JPEG by default).
        """
        image = get_rgb_image(image)

        # Crop if block is provided
        if block is not None and block.bbox:
            width, height = image.size
            x1, y1, x2, y2 = block.bbox
            crop_box = (
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height),
            )
            image = image.crop(crop_box)

        image = self._resize_if_needed(image)
        return self._to_bytes(image)
