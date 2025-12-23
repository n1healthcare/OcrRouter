"""Input handling for document processing pipeline."""

from pathlib import Path

from .utils.guess_suffix_or_lang import guess_suffix_by_bytes
from .utils.pdf_image_tools import images_bytes_to_pdf_bytes


PDF_SUFFIXES = ["pdf"]
IMAGE_SUFFIXES = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]


class InputHandler:
    """Handles reading and validation of input files."""

    def __init__(self):
        self.pdf_suffixes = PDF_SUFFIXES
        self.image_suffixes = IMAGE_SUFFIXES

    def read(self, path: str | Path) -> bytes:
        """Read a file and return its bytes.

        If the file is an image, it will be converted to PDF bytes.

        Args:
            path: Path to the file to read.

        Returns:
            PDF bytes (either original PDF or converted from image).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(str(path), "rb") as input_file:
            file_bytes = input_file.read()

        file_suffix = guess_suffix_by_bytes(file_bytes, path)

        if file_suffix in self.image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in self.pdf_suffixes:
            return file_bytes
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")

    def read_multiple(self, paths: list[str | Path]) -> list[tuple[str, bytes]]:
        """Read multiple files.

        Args:
            paths: List of file paths to read.

        Returns:
            List of tuples (file_name, pdf_bytes).
        """
        results = []
        for path in paths:
            if not isinstance(path, Path):
                path = Path(path)
            file_name = str(path.stem)
            pdf_bytes = self.read(path)
            results.append((file_name, pdf_bytes))
        return results

    def is_supported_file(self, path: str | Path) -> bool:
        """Check if a file type is supported.

        Args:
            path: Path to check.

        Returns:
            True if the file type is supported, False otherwise.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            return False

        try:
            with open(str(path), "rb") as f:
                file_bytes = f.read()
            file_suffix = guess_suffix_by_bytes(file_bytes, path)
            return file_suffix in (self.pdf_suffixes + self.image_suffixes)
        except Exception:
            return False
