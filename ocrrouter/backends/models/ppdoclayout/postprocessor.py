"""PP-DocLayoutV3 postprocessor for processing model outputs.

Handles:
- Parsing raw detection outputs
- Applying NMS (Non-Maximum Suppression)
- Filtering large images
- Merging overlapping bboxes
- Converting to ContentBlock format
"""

import numpy as np
from typing import Any

from ocrrouter.backends.models.base import BasePostprocessor
from ocrrouter.backends.utils import ContentBlock
from .utils import (
    DEFAULT_ID2LABEL,
    DEFAULT_LABEL_TASK_MAPPING,
    LABEL_TO_CONTENT_BLOCK_TYPE,
)


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU (Intersection over Union) of two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Intersection coordinates
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)

    # Intersection area
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

    # Union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def apply_nms(
    boxes: np.ndarray,
    iou_same: float = 0.6,
    iou_diff: float = 0.95,
) -> list[int]:
    """Apply Non-Maximum Suppression with class-aware thresholds.

    Args:
        boxes: Array of shape (N, 6+) with [cls_id, score, x1, y1, x2, y2, ...]
        iou_same: IoU threshold for same class
        iou_diff: IoU threshold for different classes

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    scores = boxes[:, 1]
    indices = np.argsort(scores)[::-1].tolist()
    selected = []

    while indices:
        current = indices.pop(0)
        selected.append(current)

        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = tuple(current_box[2:6])

        remaining = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = tuple(box[2:6])

            iou = compute_iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff

            if iou < threshold:
                remaining.append(i)

        indices = remaining

    return selected


def is_contained(box1: np.ndarray, box2: np.ndarray, threshold: float = 0.8) -> bool:
    """Check if box1 is contained within box2.

    Args:
        box1: [cls_id, score, x1, y1, x2, y2, ...]
        box2: [cls_id, score, x1, y1, x2, y2, ...]
        threshold: Minimum overlap ratio to consider contained

    Returns:
        True if box1 is contained in box2
    """
    x1, y1, x2, y2 = box1[2:6]
    x1_p, y1_p, x2_p, y2_p = box2[2:6]

    box1_area = (x2 - x1) * (y2 - y1)
    if box1_area <= 0:
        return False

    # Intersection
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    overlap_ratio = inter_area / box1_area

    return overlap_ratio >= threshold


class PPDocLayoutPostprocessor(BasePostprocessor):
    """PP-DocLayoutV3 postprocessor for detection output processing.

    Handles:
    - Applying NMS to remove overlapping detections
    - Filtering large images that cover most of the page
    - Merging nested bounding boxes
    - Converting to ContentBlock format with normalized coordinates
    """

    def __init__(
        self,
        id2label: dict[int, str] | None = None,
        label_task_mapping: dict[str, list[str]] | None = None,
        layout_nms: bool = True,
        layout_merge_mode: str = "large",
        debug: bool = False,
    ):
        """Initialize the PP-DocLayoutV3 postprocessor.

        Args:
            id2label: Mapping from class ID to label name.
            label_task_mapping: Mapping from task type to labels.
            layout_nms: Whether to apply NMS.
            layout_merge_mode: Mode for merging nested boxes ("large", "small", "union").
            debug: Enable debug output.
        """
        self.id2label = id2label or DEFAULT_ID2LABEL
        self.label_task_mapping = label_task_mapping or DEFAULT_LABEL_TASK_MAPPING
        self.layout_nms = layout_nms
        self.layout_merge_mode = layout_merge_mode
        self.debug = debug

        # Build reverse mapping for task lookup
        self._label_to_task: dict[str, str] = {}
        for task, labels in self.label_task_mapping.items():
            for label in labels:
                self._label_to_task[label] = task

    def _get_task_type(self, label: str) -> str:
        """Get task type for a label."""
        return self._label_to_task.get(label, "text")

    def _get_content_block_type(self, label: str) -> str:
        """Get ContentBlock type for a label."""
        return LABEL_TO_CONTENT_BLOCK_TYPE.get(label, "text")

    def _filter_large_images(
        self,
        boxes: np.ndarray,
        img_size: tuple[int, int],
    ) -> np.ndarray:
        """Filter out image detections that cover most of the page.

        Args:
            boxes: Detection boxes array
            img_size: (width, height) of the image

        Returns:
            Filtered boxes array
        """
        if len(boxes) <= 1:
            return boxes

        width, height = img_size
        img_area = width * height

        # Use different thresholds based on aspect ratio
        area_threshold = 0.82 if width > height else 0.93

        # Find image class index
        image_labels = ["image", "chart"]
        image_indices = set()
        for cls_id, label in self.id2label.items():
            if label in image_labels:
                image_indices.add(cls_id)

        filtered = []
        for box in boxes:
            cls_id = int(box[0])
            if cls_id in image_indices:
                x1, y1, x2, y2 = box[2:6]
                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > area_threshold * img_area:
                    continue  # Skip large images
            filtered.append(box)

        return np.array(filtered) if filtered else np.array([]).reshape(0, boxes.shape[1])

    def _merge_nested_boxes(
        self,
        boxes: np.ndarray,
        mode: str = "large",
    ) -> np.ndarray:
        """Merge nested bounding boxes.

        Args:
            boxes: Detection boxes array
            mode: "large" keeps larger boxes, "small" keeps smaller ones

        Returns:
            Filtered boxes array
        """
        if len(boxes) <= 1 or mode == "union":
            return boxes

        # Labels to preserve (never filter out)
        preserve_labels = {"image", "chart", "seal"}
        preserve_indices = set()
        for cls_id, label in self.id2label.items():
            if label in preserve_labels:
                preserve_indices.add(cls_id)

        n = len(boxes)
        keep_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep_mask[i]:
                continue

            # Don't filter preserved labels
            if int(boxes[i][0]) in preserve_indices:
                continue

            for j in range(n):
                if i == j or not keep_mask[j]:
                    continue

                if mode == "large":
                    # Remove smaller box if contained
                    if is_contained(boxes[i], boxes[j]):
                        keep_mask[i] = False
                        break
                elif mode == "small":
                    # Remove larger box if it contains another
                    if is_contained(boxes[j], boxes[i]):
                        keep_mask[j] = False

        return boxes[keep_mask]

    def parse_layout_output(
        self,
        output: dict,
        img_size: tuple[int, int] | None = None,
        **context: Any,
    ) -> list[ContentBlock]:
        """Parse PP-DocLayoutV3 detection output to ContentBlocks.

        Args:
            output: Dict with 'scores', 'labels', 'boxes' tensors
            img_size: (width, height) of the original image
            **context: Additional context

        Returns:
            List of ContentBlock objects with normalized bboxes
        """
        import torch

        if not output or "scores" not in output:
            return []

        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        boxes = output["boxes"].cpu().numpy()

        if len(scores) == 0:
            return []

        # Get order sequence if available
        order_seq = output.get("order_seq")
        if order_seq is not None:
            order_seq = order_seq.cpu().numpy()
        else:
            order_seq = np.arange(len(scores))

        # Build boxes array: [cls_id, score, x1, y1, x2, y2, order]
        boxes_array = np.column_stack([
            labels,
            scores,
            boxes,
            order_seq,
        ])

        # Apply NMS
        if self.layout_nms and len(boxes_array) > 0:
            keep_indices = apply_nms(boxes_array[:, :6])
            boxes_array = boxes_array[keep_indices]

        # Filter large images
        if img_size and len(boxes_array) > 0:
            boxes_array = self._filter_large_images(boxes_array, img_size)

        # Merge nested boxes
        if len(boxes_array) > 0:
            boxes_array = self._merge_nested_boxes(boxes_array, self.layout_merge_mode)

        if len(boxes_array) == 0:
            return []

        # Sort by order
        sorted_indices = np.argsort(boxes_array[:, 6])
        boxes_array = boxes_array[sorted_indices]

        # Convert to ContentBlocks
        width, height = img_size or (1, 1)
        content_blocks = []

        for box in boxes_array:
            cls_id = int(box[0])
            score = float(box[1])
            x1, y1, x2, y2 = box[2:6]

            label = self.id2label.get(cls_id, f"class_{cls_id}")
            task_type = self._get_task_type(label)

            # Skip abandoned regions
            if task_type == "abandon":
                continue

            block_type = self._get_content_block_type(label)

            # Normalize bbox to [0, 1]
            bbox = [
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            ]

            content_blocks.append(
                ContentBlock(
                    type=block_type,
                    bbox=bbox,
                    angle=None,
                    content=None,  # Layout only, no content
                )
            )

        return content_blocks

    def post_process_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Apply post-processing to blocks.

        For layout detection, this mainly filters empty/invalid blocks.

        Args:
            blocks: List of ContentBlock objects.

        Returns:
            Processed list of ContentBlock objects.
        """
        valid_blocks = []
        for block in blocks:
            # Validate bbox
            if block.bbox:
                x1, y1, x2, y2 = block.bbox
                if x2 > x1 and y2 > y1:
                    valid_blocks.append(block)
        return valid_blocks
