"""PP-DocLayoutV3 label mappings and configurations.

PP-DocLayoutV3 detects 25 document element types. This module provides
mappings between:
- Model output class IDs and human-readable labels
- Labels and OCR task types (text, table, formula, skip, abandon)
- Labels and OcrRouter ContentBlock types
"""

# PP-DocLayoutV3 class ID to label mapping (25 classes)
DEFAULT_ID2LABEL: dict[int, str] = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}

# Label to task type mapping for OCR
# - text: OCR with text prompt
# - table: OCR with table prompt
# - formula: OCR with formula prompt
# - skip: Keep region but don't OCR (images, charts)
# - abandon: Discard region entirely (headers, footers, page numbers)
DEFAULT_LABEL_TASK_MAPPING: dict[str, list[str]] = {
    "text": [
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
    ],
    "table": [
        "table",
    ],
    "formula": [
        "display_formula",
        "inline_formula",
    ],
    "skip": [
        "chart",
        "image",
    ],
    "abandon": [
        "header",
        "footer",
        "number",
        "footnote",
        "aside_text",
        "reference",
        "footer_image",
        "header_image",
    ],
}

# Default detection threshold
DEFAULT_THRESHOLD: float = 0.3

# Mapping from PP-DocLayoutV3 labels to OcrRouter ContentBlock types
# This maps the 25 PP-DocLayoutV3 labels to the standard ContentBlock types
LABEL_TO_CONTENT_BLOCK_TYPE: dict[str, str] = {
    # Text types -> "text"
    "abstract": "text",
    "algorithm": "text",
    "content": "text",
    "doc_title": "title",
    "figure_title": "image_caption",
    "paragraph_title": "title",
    "reference_content": "text",
    "text": "text",
    "vertical_text": "text",
    "vision_footnote": "page_footnote",
    "seal": "text",
    "formula_number": "text",
    "aside_text": "text",
    # Table types -> "table"
    "table": "table",
    # Formula types -> "equation"
    "display_formula": "equation",
    "inline_formula": "equation",
    # Image types -> "image"
    "chart": "image",
    "image": "image",
    # Skip/abandon types (usually filtered out, but mapped for completeness)
    "header": "header",
    "footer": "footer",
    "number": "page_number",
    "footnote": "page_footnote",
    "reference": "text",
    "footer_image": "image",
    "header_image": "image",
}


def get_task_type_for_label(label: str) -> str | None:
    """Get the OCR task type for a given label.

    Args:
        label: PP-DocLayoutV3 label name

    Returns:
        Task type (text, table, formula, skip, abandon) or None if not found
    """
    for task_type, labels in DEFAULT_LABEL_TASK_MAPPING.items():
        if label in labels:
            return task_type
    return None


def get_content_block_type(label: str) -> str:
    """Get the ContentBlock type for a given label.

    Args:
        label: PP-DocLayoutV3 label name

    Returns:
        ContentBlock type string
    """
    return LABEL_TO_CONTENT_BLOCK_TYPE.get(label, "text")
