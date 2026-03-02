"""PP-DocLayoutV3 image preprocessing.

Prepares images for layout detection with PP-DocLayoutV3 model.
"""

from PIL import Image

from ocrrouter.backends.models.base import BasePreprocessor
from ocrrouter.backends.utils import ContentBlock, get_rgb_image


class PPDocLayoutPreprocessor(BasePreprocessor):
    """PP-DocLayoutV3 preprocessor for image preparation.

    PP-DocLayoutV3 works with RGB images. The model's image processor
    handles resizing and normalization internally.
    """

    def __init__(self):
        """Initialize the PP-DocLayoutV3 preprocessor."""
        pass

    def prepare_for_layout(self, image: Image.Image) -> Image.Image:
        """Prepare image for layout detection.

        Args:
            image: PIL Image of document page.

        Returns:
            RGB PIL Image (not bytes, as PP-DocLayoutV3 uses PIL directly).
        """
        return get_rgb_image(image)

    def prepare_for_ocr(
        self,
        image: Image.Image,
        block: ContentBlock | None = None,
    ) -> bytes:
        """Prepare image for OCR (not used for layout-only backend).

        Note: PP-DocLayoutV3 is a layout detection model only.
        This method is provided for interface compatibility.

        Args:
            image: Full page PIL Image.
            block: Optional ContentBlock with bbox information.

        Returns:
            Empty bytes (not used).
        """
        raise NotImplementedError(
            "PP-DocLayoutV3 is a layout detection model only. Use a different backend for OCR extraction."
        )
