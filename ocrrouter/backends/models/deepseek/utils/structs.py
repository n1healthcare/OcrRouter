"""Label mapping for DeepSeek-OCR 2 to MinerU block types.

v2 uses different label names than v1:
- v2: figure_title, table_title (new)
- v2: uses underscore naming (figure_title vs image_caption)
"""

from ocrrouter.backends.utils import BlockType

# DeepSeek OCR labels that the model can return (v1 + v2 combined)
DEEPSEEK_OCR_LABELS = {
    # Common labels
    "text",  # Regular paragraphs/body text
    "image",  # Figures, photos, diagrams
    "table",  # Tabular data
    "equation",  # Mathematical equations
    "list",  # Bullet points, numbered lists
    # v1 labels
    "sub_title",  # Subheadings, section headers
    "title",  # Main headings
    "table_caption",  # Captions for tables
    "image_caption",  # Captions for images/figures
    "table_footnote",  # Footnotes for tables
    "image_footnote",  # Footnotes for images/figures
    "page_number",  # Page numbers
    "footer",  # Page footers
    # v2 labels (new naming convention)
    "figure_title",  # v2: Figure/image captions
    "table_title",  # v2: Table title/caption
    "figure",  # v2: alias for image
    "header",  # v2: Page headers
    "abstract",  # v2: Abstract section
    "reference",  # v2: Reference/bibliography
    "footnote",  # v2: Footnotes
    "code",  # v2: Code blocks
}

# DeepSeek labels -> MinerU BlockType mapping
DEEPSEEK_LABEL_MAP: dict[str, str] = {
    # Common labels
    "text": BlockType.TEXT,
    "image": BlockType.IMAGE,
    "table": BlockType.TABLE,
    "equation": BlockType.EQUATION,
    "list": BlockType.LIST,
    # v1 labels
    "sub_title": BlockType.TEXT,  # Map sub_title to text (deepseek will auto add heading style)
    "title": BlockType.TEXT,  # Map title to text (deepseek will auto add heading style)
    "table_caption": BlockType.TABLE_CAPTION,
    "image_caption": BlockType.IMAGE_CAPTION,
    "table_footnote": BlockType.TABLE_FOOTNOTE,
    "image_footnote": BlockType.IMAGE_FOOTNOTE,
    "page_number": BlockType.PAGE_NUMBER,
    "footer": BlockType.FOOTER,
    # v2 labels (new naming convention)
    "figure_title": BlockType.IMAGE_CAPTION,  # v2: figure captions -> image_caption
    "table_title": BlockType.TABLE_CAPTION,  # v2: table titles -> table_caption
    "figure": BlockType.IMAGE,  # v2: alias for image
    "header": BlockType.HEADER,  # v2: Page headers
    "abstract": BlockType.TEXT,  # v2: Abstract section -> text
    "reference": BlockType.TEXT,  # v2: References -> text
    "footnote": BlockType.PAGE_FOOTNOTE,  # v2: Footnotes -> page_footnote
    "code": BlockType.TEXT,  # v2: Code blocks -> text
}


def map_deepseek_label(label: str) -> str:
    """Map a DeepSeek label to MinerU BlockType.

    Args:
        label: DeepSeek label string (case-insensitive)

    Returns:
        MinerU BlockType string. Returns "unknown" if label is not recognized.
    """
    return DEEPSEEK_LABEL_MAP.get(label.lower(), BlockType.UNKNOWN)
