"""Output handling for document processing pipeline."""

import json
import os
from typing import Literal

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.utils.io.writers import FileBasedDataWriter
from .utils.draw_bbox import draw_layout_bbox


class OutputHandler:
    """Handles writing processing results to output destinations."""

    def __init__(self, settings: Settings):
        """Initialize the output handler.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings

    def write(
        self,
        pdf_file_name: str,
        pdf_bytes: bytes,
        middle_json: dict,
        formatted_output: dict,
        output_dir: str,
        local_image_dir: str,
        model_output: list | None = None,
        draw_layout: bool | None = None,
        dump_md: bool | None = None,
        dump_middle_json: bool | None = None,
        dump_model_output: bool | None = None,
        dump_orig_pdf: bool | None = None,
        dump_content_list: bool | None = None,
        output_mode: Literal["all", "layout_only", "ocr_only"] | None = None,
        **kwargs,
    ) -> str:
        """Write processing results to output directory.

        Args:
            pdf_file_name: Name of the PDF file (without extension).
            pdf_bytes: Original PDF bytes.
            middle_json: The intermediate JSON representation.
            formatted_output: Dictionary with formatted outputs (markdown, content_list, etc.).
            output_dir: Output directory path.
            local_image_dir: Directory for images.
            model_output: Raw model output (optional).
            draw_layout: Whether to draw layout bounding boxes.
            dump_md: Whether to output markdown file.
            dump_middle_json: Whether to output middle JSON.
            dump_model_output: Whether to output model JSON.
            dump_orig_pdf: Whether to save original PDF.
            dump_content_list: Whether to output content list JSON.
            output_mode: Output mode controlling which files to write:
                - 'all': Write all outputs based on individual flags
                - 'layout_only': Skip markdown (no OCR content)
                - 'ocr_only': Only write markdown, skip layout-related outputs

        Returns:
            Path to the output directory.
        """
        # Get output_mode from kwargs or settings
        output_mode = output_mode or self._settings.output_mode

        # Apply settings defaults
        draw_layout = (
            draw_layout if draw_layout is not None else self._settings.draw_layout_bbox
        )
        dump_md = dump_md if dump_md is not None else self._settings.dump_md
        dump_middle_json = (
            dump_middle_json
            if dump_middle_json is not None
            else self._settings.dump_middle_json
        )
        dump_model_output = (
            dump_model_output
            if dump_model_output is not None
            else self._settings.dump_model_output
        )
        dump_orig_pdf = (
            dump_orig_pdf if dump_orig_pdf is not None else self._settings.dump_orig_pdf
        )
        dump_content_list = (
            dump_content_list
            if dump_content_list is not None
            else self._settings.dump_content_list
        )

        # Override flags based on output_mode
        if output_mode == "layout_only":
            # Layout only: skip markdown (no OCR content), keep layout outputs
            dump_md = False
        elif output_mode == "ocr_only":
            # OCR only: only markdown, skip layout-related outputs
            draw_layout = False
            dump_model_output = False
            dump_content_list = False
            dump_middle_json = False

        # Create writer
        md_writer = FileBasedDataWriter(output_dir)

        pdf_info = middle_json.get("pdf_info", [])

        # Draw layout bounding boxes
        if draw_layout:
            draw_layout_bbox(
                pdf_info, pdf_bytes, output_dir, f"{pdf_file_name}_layout.pdf"
            )

        # Save original PDF
        if dump_orig_pdf:
            md_writer.write(f"{pdf_file_name}_origin.pdf", pdf_bytes)

        # Save markdown
        if dump_md and formatted_output.get("markdown"):
            md_writer.write_string(f"{pdf_file_name}.md", formatted_output["markdown"])

        # Save content lists
        if dump_content_list:
            if formatted_output.get("content_list"):
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(
                        formatted_output["content_list"], ensure_ascii=False, indent=4
                    ),
                )
            if formatted_output.get("content_list_v2"):
                md_writer.write_string(
                    f"{pdf_file_name}_content_list_v2.json",
                    json.dumps(
                        formatted_output["content_list_v2"],
                        ensure_ascii=False,
                        indent=4,
                    ),
                )

        # Save middle JSON
        if dump_middle_json:
            md_writer.write_string(
                f"{pdf_file_name}_middle.json",
                json.dumps(middle_json, ensure_ascii=False, indent=4),
            )

        # Save model output
        if dump_model_output and model_output is not None:
            md_writer.write_string(
                f"{pdf_file_name}_model.json",
                json.dumps(model_output, ensure_ascii=False, indent=4),
            )

        return output_dir

    def prepare_output_dirs(
        self,
        output_dir: str,
        pdf_file_name: str,
        parse_method: str = "vlm",
    ) -> tuple[str, str]:
        """Prepare output directories for a document.

        Args:
            output_dir: Base output directory.
            pdf_file_name: Name of the PDF file.
            parse_method: Parsing method name (default: "vlm").

        Returns:
            Tuple of (image_dir, md_dir).
        """
        local_md_dir = os.path.join(output_dir, pdf_file_name, parse_method)
        local_image_dir = os.path.join(local_md_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)
        return local_image_dir, local_md_dir
