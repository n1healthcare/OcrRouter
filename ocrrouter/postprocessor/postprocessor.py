"""Post-processing for document processing pipeline."""

import json
from typing import Literal

from ocrrouter.config import Settings
from ocrrouter.utils.enum_class import MakeMode

from ocrrouter.postprocessor.utils.markdown import union_make


class Postprocessor:
    """Handles post-processing of extracted content."""

    def __init__(self, settings: Settings):
        """Initialize the postprocessor.

        Args:
            settings: Settings object with configuration.
        """
        self._settings = settings

    def format(
        self,
        middle_json: dict,
        make_mode: str | None = None,
        image_dir: str = "",
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
        output_mode: Literal["all", "layout_only", "ocr_only"] | None = None,
        **kwargs,
    ) -> dict:
        """Format the middle JSON into various output formats.

        Args:
            middle_json: The intermediate JSON representation.
            make_mode: Output format mode. Defaults to settings.make_md_mode.
            image_dir: Directory path for images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            output_mode: Output mode controlling formatting:
                - 'all': Generate markdown and content lists
                - 'layout_only': Skip markdown (no OCR content), generate content lists
                - 'ocr_only': Generate markdown only, skip content lists

        Returns:
            Dictionary containing formatted outputs.
        """
        make_mode = make_mode if make_mode is not None else self._settings.make_md_mode
        formula_enable = (
            formula_enable if formula_enable is not None else self._settings.formula_enable
        )
        table_enable = (
            table_enable if table_enable is not None else self._settings.table_enable
        )
        output_mode = output_mode or self._settings.output_mode

        pdf_info = middle_json.get("pdf_info", [])

        result = {
            "middle_json": middle_json,
            "markdown": None,
            "content_list": None,
            "content_list_v2": None,
        }

        # Generate markdown (skip for layout_only mode - no OCR content)
        if output_mode != "layout_only":
            if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
                result["markdown"] = union_make(
                    pdf_info,
                    make_mode,
                    image_dir,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )

        # Generate content list (skip for ocr_only mode - no layout structure)
        if output_mode != "ocr_only":
            result["content_list"] = union_make(
                pdf_info,
                MakeMode.CONTENT_LIST,
                image_dir,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )

            result["content_list_v2"] = union_make(
                pdf_info,
                MakeMode.CONTENT_LIST_V2,
                image_dir,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )

        return result

    def to_markdown(
        self,
        middle_json: dict,
        make_mode: str = MakeMode.MM_MD,
        image_dir: str = "",
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
    ) -> str:
        """Convert middle JSON to markdown string.

        Args:
            middle_json: The intermediate JSON representation.
            make_mode: Markdown mode (mm_markdown or nlp_markdown).
            image_dir: Directory path for images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.

        Returns:
            Markdown string.
        """
        formula_enable = (
            formula_enable if formula_enable is not None else self._settings.formula_enable
        )
        table_enable = (
            table_enable if table_enable is not None else self._settings.table_enable
        )

        pdf_info = middle_json.get("pdf_info", [])
        return union_make(
            pdf_info,
            make_mode,
            image_dir,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )

    def to_content_list(
        self,
        middle_json: dict,
        image_dir: str = "",
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
        version: int = 1,
    ) -> list:
        """Convert middle JSON to content list.

        Args:
            middle_json: The intermediate JSON representation.
            image_dir: Directory path for images.
            formula_enable: Whether formula extraction is enabled.
            table_enable: Whether table extraction is enabled.
            version: Content list version (1 or 2).

        Returns:
            Content list.
        """
        formula_enable = (
            formula_enable if formula_enable is not None else self._settings.formula_enable
        )
        table_enable = (
            table_enable if table_enable is not None else self._settings.table_enable
        )

        pdf_info = middle_json.get("pdf_info", [])
        mode = MakeMode.CONTENT_LIST if version == 1 else MakeMode.CONTENT_LIST_V2
        return union_make(
            pdf_info,
            mode,
            image_dir,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )
