"""Label mapping for DeepSeek-OCR to MinerU block types."""

from ocrrouter.backends.utils import BlockType

# DeepSeek OCR labels that the model can return
DEEPSEEK_OCR_LABELS = {
    "text",  # Regular paragraphs/body text
    "image",  # Figures, photos, diagrams
    "sub_title",  # Subheadings, section headers
    "table",  # Tabular data
    "title",  # Main headings
    "table_caption",  # Captions for tables
    "image_caption",  # Captions for images/figures
    "table_footnote",  # Footnotes for tables
    "image_footnote",  # Footnotes for images/figures
    "equation",  # Mathematical equations
    "list",  # Bullet points, numbered lists
    "page_number",  # Page numbers
    "footer",  # Page footers
}

# DeepSeek labels -> MinerU BlockType mapping
DEEPSEEK_LABEL_MAP: dict[str, str] = {
    "text": BlockType.TEXT,
    "image": BlockType.IMAGE,
    "sub_title": BlockType.TEXT,  # Map sub_title to text (deepseek will auto add heading style)
    "table": BlockType.TABLE,
    "title": BlockType.TEXT,  # Map title to text (deepseek will auto add heading style)
    "table_caption": BlockType.TABLE_CAPTION,
    "image_caption": BlockType.IMAGE_CAPTION,  # DeepSeek uses image_caption
    "table_footnote": BlockType.TABLE_FOOTNOTE,
    "image_footnote": BlockType.IMAGE_FOOTNOTE,
    "equation": BlockType.EQUATION,  # DeepSeek uses equation instead of formula
    "list": BlockType.LIST,
    "page_number": BlockType.PAGE_NUMBER,
    "footer": BlockType.FOOTER,
}


def map_deepseek_label(label: str) -> str:
    """Map a DeepSeek label to MinerU BlockType.

    Args:
        label: DeepSeek label string (case-insensitive)

    Returns:
        MinerU BlockType string. Returns "unknown" if label is not recognized.
    """
    return DEEPSEEK_LABEL_MAP.get(label.lower(), BlockType.UNKNOWN)
