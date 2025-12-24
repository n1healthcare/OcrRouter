from pathlib import Path

from loguru import logger
from magika import Magika


DEFAULT_LANG = "txt"
PDF_SIG_BYTES = b"%PDF"
magika = Magika()


def guess_language_by_text(code):
    codebytes = code.encode(encoding="utf-8")
    lang = magika.identify_bytes(codebytes).prediction.output.label
    return lang if lang != "unknown" else DEFAULT_LANG


def guess_suffix_by_bytes(file_bytes, file_path=None) -> str:
    suffix = magika.identify_bytes(file_bytes).prediction.output.label
    # Check for PDF signature when Magika misclassifies as ai/html
    if suffix in ["ai", "html"] and file_bytes[:4] == PDF_SIG_BYTES:
        # When path is provided, verify extension matches; otherwise trust signature
        if file_path is None or Path(file_path).suffix.lower() in [".pdf"]:
            suffix = "pdf"
    return suffix


def guess_suffix_by_path(file_path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    suffix = magika.identify_path(file_path).prediction.output.label
    if suffix in ["ai", "html"] and file_path.suffix.lower() in [".pdf"]:
        try:
            with open(file_path, "rb") as f:
                if f.read(4) == PDF_SIG_BYTES:
                    suffix = "pdf"
        except Exception as e:
            logger.warning(
                f"Failed to read file {file_path} for PDF signature check: {e}"
            )
    return suffix
