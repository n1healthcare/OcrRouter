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

    def read(self, input_data: str | Path | bytes) -> bytes:
        """Read input and return PDF bytes.

        If the input is an image, it will be converted to PDF bytes.
        Uses Google Magika to detect file type from byte content.

        Args:
            input_data: File path (str or Path) or raw file bytes (PDF or image).

        Returns:
            PDF bytes (either original PDF or converted from image).

        Raises:
            FileNotFoundError: If a path is provided and file does not exist.
            ValueError: If the file type is not supported.
        """
        # Handle bytes input directly
        if isinstance(input_data, bytes):
            file_bytes = input_data
            file_suffix = guess_suffix_by_bytes(file_bytes)
        else:
            # Handle path input
            path = Path(input_data) if not isinstance(input_data, Path) else input_data
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with open(str(path), "rb") as f:
                file_bytes = f.read()
            file_suffix = guess_suffix_by_bytes(file_bytes, path)

        if file_suffix in self.image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in self.pdf_suffixes:
            return file_bytes
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")

    def read_multiple(
        self, inputs: list[str | Path | bytes | tuple[str, bytes]]
    ) -> list[tuple[str, bytes]]:
        """Read multiple inputs.

        Args:
            inputs: List of inputs. Each can be:
                - str or Path: File path (filename derived from path stem)
                - bytes: Raw bytes (filename defaults to "document_N")
                - tuple[str, bytes]: (filename, raw_bytes)

        Returns:
            List of tuples (file_name, pdf_bytes).
        """
        results = []
        for i, input_data in enumerate(inputs):
            if isinstance(input_data, tuple):
                file_name, file_bytes = input_data
                pdf_bytes = self.read(file_bytes)
            elif isinstance(input_data, bytes):
                file_name = f"document_{i}"
                pdf_bytes = self.read(input_data)
            else:
                path = (
                    Path(input_data) if not isinstance(input_data, Path) else input_data
                )
                file_name = path.stem
                pdf_bytes = self.read(path)
            results.append((file_name, pdf_bytes))
        return results

    def is_supported(self, input_data: str | Path | bytes) -> bool:
        """Check if input type is supported.

        Args:
            input_data: File path (str or Path) or raw bytes to check.

        Returns:
            True if the input type is supported, False otherwise.
        """
        try:
            if isinstance(input_data, bytes):
                file_bytes = input_data
                file_suffix = guess_suffix_by_bytes(file_bytes)
            else:
                path = (
                    Path(input_data) if not isinstance(input_data, Path) else input_data
                )
                if not path.exists():
                    return False
                with open(str(path), "rb") as f:
                    file_bytes = f.read()
                file_suffix = guess_suffix_by_bytes(file_bytes, path)
            return file_suffix in (self.pdf_suffixes + self.image_suffixes)
        except Exception:
            return False
