"""Hunyuan-specific image preprocessing before model inference."""

from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import ContentBlock, get_png_bytes, get_rgb_image


class HunyuanOCRPreprocessor(BasePreprocessor):
    """Hunyuan-specific preprocessor for image preparation.

    Hunyuan-OCR works best with original resolution images.
    No special resizing is applied.

    Note: Hunyuan does NOT support layout detection, only OCR.
    """

    def __init__(self):
        """Initialize the Hunyuan preprocessor."""
        pass

    def prepare_for_layout(self, image: Image.Image) -> bytes:
        """Not supported - Hunyuan does not do layout detection.

        Raises:
            NotImplementedError: Always, as Hunyuan only supports OCR.
        """
        raise NotImplementedError(
            "HunyuanOCR does not support layout detection. "
            "Use prepare_for_ocr() for OCR extraction."
        )

    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock | None = None,
    ) -> bytes:
        """Prepare image for OCR.

        Args:
            image: Full page PIL Image or pre-cropped image.
            block: If provided, crop this region from image.

        Returns:
            PNG bytes of the image.
        """
        image = get_rgb_image(image)

        if block is not None:
            # Crop the block region
            width, height = image.size
            x1, y1, x2, y2 = block.bbox
            crop_box = (
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height),
            )
            image = image.crop(crop_box)

        return get_png_bytes(image)
