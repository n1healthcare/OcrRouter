"""DeepSeek-specific image preprocessing before model inference."""

from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import ContentBlock, get_png_bytes, get_rgb_image


class DeepSeekPreprocessor(BasePreprocessor):
    """DeepSeek-specific preprocessor for image preparation.

    DeepSeek-OCR works best with original resolution images
    (recommended DPI=144 for PDF rendering). No resizing is applied.
    """

    def __init__(self):
        """Initialize the DeepSeek preprocessor."""
        pass

    def prepare_for_layout(self, image: Image.Image) -> bytes:
        """Prepare image for layout detection.

        No resize is applied. DeepSeek-OCR works best with original
        resolution images.

        Args:
            image: PIL Image of document page.

        Returns:
            PNG bytes of the image.
        """
        image = get_rgb_image(image)
        return get_png_bytes(image)

    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock,
    ) -> bytes:
        """Prepare cropped block image for OCR.

        Args:
            image: Full page PIL Image.
            block: ContentBlock with bbox information.

        Returns:
            PNG bytes of the cropped image.
        """
        image = get_rgb_image(image)
        width, height = image.size

        x1, y1, x2, y2 = block.bbox
        crop_box = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )
        block_image = image.crop(crop_box)

        return get_png_bytes(block_image)
