"""Preprocessing for document processing pipeline."""

import io

from loguru import logger
import pypdfium2 as pdfium

from ocrrouter.config import Settings
from .utils.pdf_page_id import get_end_page_id


class Preprocessor:
    """Handles document preprocessing including page selection."""

    def __init__(self, settings: Settings):
        """Initialize the preprocessor.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings

    def prepare(
        self,
        pdf_bytes: bytes,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **kwargs,
    ) -> bytes:
        """Prepare PDF bytes for processing.

        Extracts the specified page range from the PDF.

        Args:
            pdf_bytes: The original PDF bytes.
            start_page_id: Starting page index (0-based). Defaults to settings.start_page.
            end_page_id: Ending page index (0-based). Defaults to settings.end_page.

        Returns:
            PDF bytes containing only the specified page range.
        """
        start_page_id = (
            start_page_id if start_page_id is not None else self._settings.start_page
        )
        end_page_id = end_page_id if end_page_id is not None else self._settings.end_page

        return self._extract_page_range(pdf_bytes, start_page_id, end_page_id)

    def _extract_page_range(
        self,
        pdf_bytes: bytes,
        start_page_id: int = 0,
        end_page_id: int | None = None,
    ) -> bytes:
        """Extract a page range from PDF bytes.

        Args:
            pdf_bytes: Original PDF bytes.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based). None means all remaining pages.

        Returns:
            PDF bytes containing only the specified page range.
        """
        pdf = pdfium.PdfDocument(pdf_bytes)
        output_pdf = pdfium.PdfDocument.new()

        try:
            end_page_id = get_end_page_id(end_page_id, len(pdf))

            output_index = 0
            for page_index in range(start_page_id, end_page_id + 1):
                try:
                    output_pdf.import_pages(pdf, pages=[page_index])
                    output_index += 1
                except Exception as page_error:
                    output_pdf.del_page(output_index)
                    logger.warning(
                        f"Failed to import page {page_index}: {page_error}, skipping this page."
                    )
                    continue

            output_buffer = io.BytesIO()
            output_pdf.save(output_buffer)
            output_bytes = output_buffer.getvalue()
        except Exception as e:
            logger.warning(
                f"Error in converting PDF bytes: {e}, Using original PDF bytes."
            )
            output_bytes = pdf_bytes
        finally:
            pdf.close()
            output_pdf.close()

        return output_bytes

    def prepare_multiple(
        self,
        pdf_bytes_list: list[bytes],
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **kwargs,
    ) -> list[bytes]:
        """Prepare multiple PDF documents.

        Args:
            pdf_bytes_list: List of PDF bytes to prepare.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).

        Returns:
            List of prepared PDF bytes.
        """
        return [
            self.prepare(pdf_bytes, start_page_id, end_page_id, **kwargs)
            for pdf_bytes in pdf_bytes_list
        ]
