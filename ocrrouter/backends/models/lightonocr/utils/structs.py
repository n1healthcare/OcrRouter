"""LightOnOCR-specific data structures and regex patterns."""

import re

from ocrrouter.backends.utils import BlockType

# Regex pattern for parsing image bboxes in LightOnOCR output
# Format: ![image](image_N.png) x1,y1,x2,y2
# Where coordinates are normalized to 0-1000
IMAGE_BBOX_PATTERN = re.compile(r"!\[image\]\(image_(\d+)\.png\)\s*(\d+),(\d+),(\d+),(\d+)")


def parse_image_bboxes(text: str) -> list[tuple[int, list[float]]]:
    """Parse image bboxes from LightOnOCR model output.

    Args:
        text: Raw model output containing image markers with bboxes.

    Returns:
        List of tuples (image_index, normalized_bbox) where bbox is [x1, y1, x2, y2]
        normalized to 0-1 range.
    """
    results = []
    for match in IMAGE_BBOX_PATTERN.finditer(text):
        image_idx = int(match.group(1))
        x1 = int(match.group(2)) / 1000.0
        y1 = int(match.group(3)) / 1000.0
        x2 = int(match.group(4)) / 1000.0
        y2 = int(match.group(5)) / 1000.0

        # Ensure correct ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # Clamp to valid range
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # Skip zero-area boxes
        if x1 >= x2 or y1 >= y2:
            continue

        results.append((image_idx, [x1, y1, x2, y2]))

    return results


def remove_image_markers(text: str) -> str:
    """Remove image markers from text, leaving just the markdown content.

    Args:
        text: Raw model output with image markers.

    Returns:
        Cleaned text without image bbox markers.
    """
    # Remove the image markers with their coordinates
    cleaned = IMAGE_BBOX_PATTERN.sub("", text)
    # Clean up extra whitespace that may result from removal
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# Block type for LightOnOCR outputs
# LightOnOCR only provides text and images, no other layout categories
LIGHTONOCR_BLOCK_TYPES = {
    "text": BlockType.TEXT,
    "image": BlockType.IMAGE,
}
