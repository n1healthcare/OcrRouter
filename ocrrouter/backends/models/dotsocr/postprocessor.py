"""DotsOCR-specific output postprocessing after model inference."""

import json
import re
from dataclasses import dataclass
from typing import Any

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import BLOCK_TYPES, ContentBlock
from .utils import map_dotsocr_label


@dataclass
class CleanedData:
    """Data structure for cleaned data."""

    case_id: int
    original_type: str
    original_length: int
    cleaned_data: list[dict]
    cleaning_operations: dict[str, Any]
    success: bool


class OutputCleaner:
    """Data Cleaner for malformed JSON responses."""

    def __init__(self):
        self.dict_pattern = re.compile(
            r'\{[^{}]*?"bbox"\s*:\s*\[[^\]]*?\][^{}]*?\}', re.DOTALL
        )
        self.bbox_pattern = re.compile(r'"bbox"\s*:\s*\[([^\]]+)\]')
        self.missing_delimiter_pattern = re.compile(r'\}\s*\{(?!")')

    def clean_list_data(self, data: list[dict], case_id: int = 0) -> CleanedData:
        """Cleans list-type data."""
        cleaned_data = []
        operations = {
            "type": "list",
            "bbox_fixes": 0,
            "removed_items": 0,
            "original_count": len(data),
        }

        for item in data:
            if not isinstance(item, dict):
                operations["removed_items"] += 1
                continue

            if "bbox" in item:
                bbox = item["bbox"]
                if isinstance(bbox, list) and len(bbox) == 3:
                    new_item = {}
                    if "category" in item:
                        new_item["category"] = item["category"]
                    if "text" in item:
                        new_item["text"] = item["text"]
                    if new_item:
                        cleaned_data.append(new_item)
                        operations["bbox_fixes"] += 1
                    else:
                        operations["removed_items"] += 1
                elif isinstance(bbox, list) and len(bbox) == 4:
                    cleaned_data.append(item.copy())
                else:
                    operations["removed_items"] += 1
            else:
                if "category" in item:
                    cleaned_data.append(item.copy())
                else:
                    operations["removed_items"] += 1

        operations["final_count"] = len(cleaned_data)
        return CleanedData(
            case_id=case_id,
            original_type="list",
            original_length=len(data),
            cleaned_data=cleaned_data,
            cleaning_operations=operations,
            success=True,
        )

    def clean_string_data(self, data_str: str, case_id: int = 0) -> CleanedData:
        """Cleans string-type data."""
        operations = {
            "type": "str",
            "original_length": len(data_str),
            "delimiter_fixes": 0,
            "tail_truncated": False,
            "duplicate_dicts_removed": 0,
            "final_objects": 0,
        }

        try:
            # Fix missing delimiters
            data_str, fixes = self._fix_missing_delimiters(data_str)
            operations["delimiter_fixes"] = fixes

            # Truncate incomplete elements
            data_str, truncated = self._truncate_last_incomplete_element(data_str)
            operations["tail_truncated"] = truncated

            # Remove duplicates
            data_str, removed = self._remove_duplicate_dicts(data_str)
            operations["duplicate_dicts_removed"] = removed

            # Ensure JSON format
            data_str = self._ensure_json_format(data_str)

            # Parse final result
            final_data = self._parse_final_json(data_str)

            if final_data is not None:
                operations["final_objects"] = len(final_data)
                return CleanedData(
                    case_id=case_id,
                    original_type="str",
                    original_length=operations["original_length"],
                    cleaned_data=final_data,
                    cleaning_operations=operations,
                    success=True,
                )
            else:
                raise Exception("Could not parse cleaned data")

        except Exception:
            return CleanedData(
                case_id=case_id,
                original_type="str",
                original_length=operations["original_length"],
                cleaned_data=[],
                cleaning_operations=operations,
                success=False,
            )

    def _fix_missing_delimiters(self, text: str) -> tuple[str, int]:
        fixes = 0

        def replace_delimiter(match):
            nonlocal fixes
            fixes += 1
            return "},{"

        text = self.missing_delimiter_pattern.sub(replace_delimiter, text)
        return text, fixes

    def _truncate_last_incomplete_element(self, text: str) -> tuple[str, bool]:
        needs_truncation = len(text) > 50000 or not text.strip().endswith("]")
        if needs_truncation:
            bbox_count = text.count('{"bbox":')
            if bbox_count <= 1:
                return text, False

            last_bbox_pos = text.rfind('{"bbox":')
            if last_bbox_pos > 0:
                truncated_text = text[:last_bbox_pos].rstrip()
                if truncated_text.endswith(","):
                    truncated_text = truncated_text[:-1]
                return truncated_text, True
        return text, False

    def _remove_duplicate_dicts(self, text: str) -> tuple[str, int]:
        dict_matches = list(self.dict_pattern.finditer(text))
        if not dict_matches:
            return text, 0

        unique_dicts = []
        seen = set()
        duplicates = 0

        for match in dict_matches:
            dict_str = match.group()
            if dict_str not in seen:
                unique_dicts.append(dict_str)
                seen.add(dict_str)
            else:
                duplicates += 1

        if duplicates > 0:
            return "[" + ", ".join(unique_dicts) + "]", duplicates
        return text, 0

    def _ensure_json_format(self, text: str) -> str:
        text = text.strip()
        if not text.startswith("["):
            text = "[" + text
        if not text.endswith("]"):
            text = text.rstrip(",").rstrip() + "]"
        return text

    def _parse_final_json(self, text: str) -> list[dict] | None:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            valid_dicts = []
            for match in self.dict_pattern.finditer(text):
                try:
                    dict_obj = json.loads(match.group())
                    valid_dicts.append(dict_obj)
                except json.JSONDecodeError:
                    continue
            if valid_dicts:
                return valid_dicts
        return None

    def remove_duplicate_pairs(self, data_list: list[dict]) -> list[dict]:
        """Removes duplicate category-text pairs and bboxes."""
        if not data_list or len(data_list) <= 1:
            return data_list

        # Count category-text pairs
        category_text_pairs: dict[tuple, list[int]] = {}
        for i, item in enumerate(data_list):
            if isinstance(item, dict) and "category" in item and "text" in item:
                pair_key = (item.get("category", ""), item.get("text", ""))
                if pair_key not in category_text_pairs:
                    category_text_pairs[pair_key] = []
                category_text_pairs[pair_key].append(i)

        # Count bboxes
        bbox_pairs: dict[tuple, list[int]] = {}
        for i, item in enumerate(data_list):
            if isinstance(item, dict) and "bbox" in item:
                bbox = item.get("bbox")
                if isinstance(bbox, list) and len(bbox) > 0:
                    bbox_key = tuple(bbox)
                    if bbox_key not in bbox_pairs:
                        bbox_pairs[bbox_key] = []
                    bbox_pairs[bbox_key].append(i)

        # Identify duplicates to remove
        duplicates_to_remove: set[int] = set()

        for pair_key, positions in category_text_pairs.items():
            if len(positions) >= 5:
                duplicates_to_remove.update(positions[1:])

        for bbox_key, positions in bbox_pairs.items():
            if len(positions) >= 2:
                duplicates_to_remove.update(positions[1:])

        if not duplicates_to_remove:
            return data_list

        return [
            item for i, item in enumerate(data_list) if i not in duplicates_to_remove
        ]

    def clean_model_output(self, model_output: Any) -> list[dict]:
        """Main cleaning method."""
        try:
            if isinstance(model_output, list):
                result = self.clean_list_data(model_output)
            else:
                result = self.clean_string_data(str(model_output))

            if result.success and result.cleaned_data:
                result.cleaned_data = self.remove_duplicate_pairs(result.cleaned_data)

            return result.cleaned_data
        except Exception as e:
            print(f"Cleaning failed: {e}")
            return model_output if isinstance(model_output, list) else []


def has_latex_markdown(text: str) -> bool:
    """Check if string contains LaTeX markdown patterns."""
    if not isinstance(text, str):
        return False

    patterns = [
        r"\$\$.*?\$\$",
        r"\$[^$\n]+?\$",
        r"\\begin\{.*?\}.*?\\end\{.*?\}",
        r"\\[a-zA-Z]+\{.*?\}",
        r"\\[a-zA-Z]+",
        r"\\\[.*?\\\]",
        r"\\\(.*?\\\)",
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    return False


def clean_latex_preamble(latex_text: str) -> str:
    """Remove LaTeX preamble commands."""
    patterns = [
        r"\\documentclass\{[^}]+\}",
        r"\\usepackage\{[^}]+\}",
        r"\\usepackage\[[^\]]*\]\{[^}]+\}",
        r"\\begin\{document\}",
        r"\\end\{document\}",
    ]

    cleaned = latex_text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def get_formula_in_markdown(text: str) -> str:
    """Format formula string into Markdown block."""
    text = text.strip()

    if text.startswith("$$") and text.endswith("$$"):
        text_new = text[2:-2].strip()
        if "$" not in text_new:
            return f"$$\n{text_new}\n$$"
        return text

    if text.startswith("\\[") and text.endswith("\\]"):
        inner = text[2:-2].strip()
        return f"$$\n{inner}\n$$"

    if re.findall(r".*\\\[.*\\\].*", text):
        return text

    if re.findall(r"\$([^$]+)\$", text):
        return text

    if not has_latex_markdown(text):
        return text

    if "usepackage" in text:
        text = clean_latex_preamble(text)

    if text and text[0] == "`" and text[-1] == "`":
        text = text[1:-1]

    return f"$$\n{text}\n$$"


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace."""
    if not text:
        return ""

    text = text.strip()

    if len(text) >= 4 and text[:2] == "`$" and text[-2:] == "$`":
        text = text[1:-1]

    return text


def convert_bbox_dotsocr(
    bbox: list,
    orig_width: int,
    orig_height: int,
    resized_width: int,
    resized_height: int,
) -> list[float] | None:
    """Convert DotsOCR bbox from resized coordinates to normalized [0-1] range.

    Model returns bbox in resized image coordinates. We convert back to
    original coordinates and then normalize.

    Args:
        bbox: Bounding box in resized image pixel coordinates [x1, y1, x2, y2]
        orig_width: Original image width in pixels
        orig_height: Original image height in pixels
        resized_width: Resized image width sent to model
        resized_height: Resized image height sent to model

    Returns:
        List of normalized coordinates [x1, y1, x2, y2] in 0-1 range,
        or None if coordinates are invalid.
    """
    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (ValueError, TypeError):
        return None

    # Scale from resized to original coordinates
    scale_x = orig_width / resized_width
    scale_y = orig_height / resized_height

    x1_orig = x1 * scale_x
    y1_orig = y1 * scale_y
    x2_orig = x2 * scale_x
    y2_orig = y2 * scale_y

    # Ensure correct ordering
    if x2_orig < x1_orig:
        x1_orig, x2_orig = x2_orig, x1_orig
    if y2_orig < y1_orig:
        y1_orig, y2_orig = y2_orig, y1_orig

    # Skip zero-area boxes
    if x1_orig == x2_orig or y1_orig == y2_orig:
        return None

    # Normalize to 0-1 using original dimensions
    normalized = [
        max(0.0, min(1.0, x1_orig / orig_width)),
        max(0.0, min(1.0, y1_orig / orig_height)),
        max(0.0, min(1.0, x2_orig / orig_width)),
        max(0.0, min(1.0, y2_orig / orig_height)),
    ]

    # Validate normalized values
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return None

    return normalized


def clean_json_response(response: str) -> str:
    """Clean potential JSON issues in the response.

    Args:
        response: Raw model response that may contain markdown formatting

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code block markers if present
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()


class DotsOCRPostprocessor(BasePostprocessor):
    """DotsOCR-specific postprocessor for model outputs.

    Handles:
    - JSON cleaning and parsing for layout output
    - Bbox coordinate conversion (resized -> normalized)
    - Text formatting (LaTeX formulas, etc.)
    - Duplicate removal
    """

    def __init__(self, debug: bool = False):
        """Initialize the DotsOCR postprocessor.

        Args:
            debug: Enable debug mode for additional logging
        """
        self.debug = debug
        self.output_cleaner = OutputCleaner()

    def parse_layout_output(
        self,
        output: str,
        orig_width: int,
        orig_height: int,
        resized_width: int,
        resized_height: int,
        include_content: bool = True,
    ) -> list[ContentBlock]:
        """Parse DotsOCR layout output to ContentBlock list.

        The output format is a JSON array:
        [
            {"bbox": [x1, y1, x2, y2], "category": "Text", "text": "..."},
            {"bbox": [x1, y1, x2, y2], "category": "Picture"},
            ...
        ]

        Args:
            output: Raw model output string (JSON array)
            orig_width: Original image width for bbox normalization
            orig_height: Original image height for bbox normalization
            resized_width: Resized image width sent to model
            resized_height: Resized image height sent to model
            include_content: Whether to include text content in ContentBlock

        Returns:
            List of ContentBlock objects with type, bbox, and content
        """
        blocks: list[ContentBlock] = []

        if not output:
            return blocks

        # Clean and parse JSON
        cleaned = clean_json_response(output)

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Use OutputCleaner for robust parsing
            if self.debug:
                print(f"Warning: JSON parse error: {e}, using OutputCleaner")
            items = self.output_cleaner.clean_model_output(cleaned)

        if not isinstance(items, list):
            if self.debug:
                print(f"Warning: expected list, got {type(items)}")
            return blocks

        for item in items:
            if not isinstance(item, dict):
                continue

            # Get bbox
            bbox_raw = item.get("bbox")
            if not bbox_raw:
                continue

            bbox = convert_bbox_dotsocr(
                bbox_raw, orig_width, orig_height, resized_width, resized_height
            )
            if bbox is None:
                if self.debug:
                    print(f"Warning: invalid bbox: {bbox_raw}")
                continue

            # Get category and map to BlockType
            category = item.get("category", "").lower()
            block_type = map_dotsocr_label(category)
            if block_type not in BLOCK_TYPES:
                if self.debug:
                    print(f"Warning: unknown block type: {category} -> {block_type}")
                block_type = "unknown"

            # Get content
            content = None
            if include_content:
                content = item.get("text")
                # For image blocks, content is typically empty
                if block_type == "image":
                    content = None

            # DotsOCR doesn't provide angle information, default to None
            blocks.append(ContentBlock(block_type, bbox, angle=None, content=content))

        return blocks

    def post_process_blocks(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Apply post-processing to extracted content blocks.

        Args:
            blocks: List of ContentBlock to post-process

        Returns:
            List of post-processed ContentBlock
        """
        for block in blocks:
            if block.content:
                # Clean text
                block.content = clean_text(block.content)

                # Format formulas
                if block.type == "equation":
                    block.content = get_formula_in_markdown(block.content)

        return blocks
