"""LightOnOCR-specific utilities."""

from .structs import (
    IMAGE_BBOX_PATTERN,
    LIGHTONOCR_BLOCK_TYPES,
    parse_image_bboxes,
    remove_image_markers,
)

__all__ = [
    "IMAGE_BBOX_PATTERN",
    "parse_image_bboxes",
    "remove_image_markers",
    "LIGHTONOCR_BLOCK_TYPES",
]
