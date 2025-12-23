"""Label mapping for DotsOCR to MinerU block types."""

from ocrrouter.backends.utils import BlockType

# DotsOCR labels that the model can return
DOTSOCR_LABELS = {
    "text",  # Regular paragraphs/body text
    "picture",  # Figures, photos, diagrams
    "table",  # Tabular data
    "title",  # Main headings
    "section-header",  # Subheadings, section headers
    "caption",  # Captions for tables or images
    "footnote",  # Footnotes for tables or images
    "formula",  # Mathematical equations
    "list-item",  # Bullet points, numbered lists
    "page-header",  # Page headers
    "page-footer",  # Page footers
}

# DotsOCR labels -> MinerU BlockType mapping
DOTSOCR_LABEL_MAP: dict[str, str] = {
    "text": BlockType.TEXT,
    "picture": BlockType.IMAGE,
    "table": BlockType.TABLE,
    "title": BlockType.TEXT,  # Map title to text (model adds heading style in markdown)
    "section-header": BlockType.TEXT,  # Map section-header to text (model adds heading style)
    "caption": BlockType.TEXT,  # Map to text for simplicity; post-processor can reassign
    "footnote": BlockType.TEXT,  # Map to text for simplicity; post-processor can reassign
    "formula": BlockType.EQUATION,
    "list-item": BlockType.TEXT,  # Map list-item to text, mineru assume lists bbox are overlapped
    "page-header": BlockType.HEADER,
    "page-footer": BlockType.FOOTER,
}


def map_dotsocr_label(label: str) -> str:
    """Map a DotsOCR label to MinerU BlockType.

    Args:
        label: DotsOCR label string (case-insensitive)

    Returns:
        MinerU BlockType string. Returns "unknown" if label is not recognized.
    """
    return DOTSOCR_LABEL_MAP.get(label.lower(), BlockType.UNKNOWN)
