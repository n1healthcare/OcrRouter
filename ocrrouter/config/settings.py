"""Centralized settings management for OcrRouter."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Configuration for OcrRouter.

    Settings are passed explicitly to DocumentPipeline and other components.
    No automatic environment variable or .env file loading.

    Example:
        >>> from ocrrouter import DocumentPipeline, Settings
        >>>
        >>> # Direct constructor arguments
        >>> pipeline = DocumentPipeline(backend="deepseek", openai_api_key="sk-...")
        >>>
        >>> # Or via Settings object
        >>> settings = Settings(backend="deepseek", openai_api_key="sk-...")
        >>> pipeline = DocumentPipeline(settings=settings)
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # ============ API/Network Settings ============
    openai_base_url: str | None = Field(
        default=None,
        description="VLM server URL",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="API authentication key",
    )

    http_timeout: int = Field(
        default=120,
        description="HTTP request timeout in seconds",
    )

    max_concurrency: int = Field(
        default=20,
        description="Maximum concurrent requests",
    )

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed requests",
    )

    # ============ Backend Settings ============
    backend: Literal[
        "mineru", "deepseek", "dotsocr", "composite", "hunyuanocr", "generalvlm"
    ] = Field(
        default="mineru",
        description="Document processing backend to use",
    )

    # ============ Composite Backend Settings ============
    layout_model: Literal["mineru", "deepseek", "dotsocr"] = Field(
        default="mineru",
        description="Model to use for layout detection in composite mode",
    )

    ocr_model: Literal[
        "mineru", "deepseek", "dotsocr", "hunyuanocr", "paddleocr", "generalvlm"
    ] = Field(
        default="mineru",
        description="Model to use for OCR extraction in composite mode",
    )

    # ============ VLM Model Settings ============
    mineru_model_name: str = Field(
        default="mineru-2.5",
        description="MinerU VLM model name",
    )

    deepseek_model_name: str = Field(
        default="deepseek-ocr",
        description="DeepSeek-OCR model name",
    )

    dotsocr_model_name: str = Field(
        default="dots-ocr",
        description="DotsOCR model name",
    )

    dotsocr_extraction_mode: Literal["one_step", "two_step"] = Field(
        default="one_step",
        description="DotsOCR extraction mode: 'one_step' (layout+OCR in one call) or 'two_step' (layout first, then OCR)",
    )

    hunyuan_model_name: str = Field(
        default="hunyuan-ocr",
        description="Hunyuan-OCR model name",
    )

    paddleocr_model_name: str = Field(
        default="paddle-ocr",
        description="PaddleOCR model name",
    )

    generalvlm_model_name: str = Field(
        default="gemini-2.5-pro",
        description="General VLM model name (supports GPT, Claude, Gemini, etc.)",
    )

    # ============ Output Mode ============
    output_mode: Literal["all", "layout_only", "ocr_only"] = Field(
        default="all",
        description=(
            "Output mode controlling processing and outputs: "
            "'all' (layout+OCR, all outputs), "
            "'layout_only' (layout detection only, no markdown), "
            "'ocr_only' (full-page OCR, markdown only). "
            "Note: mineru backend does not support 'ocr_only' mode."
        ),
    )

    # ============ Processing Options ============
    formula_enable: bool = Field(
        default=True,
        description="Enable mathematical formula extraction",
    )

    table_enable: bool = Field(
        default=True,
        description="Enable table extraction and recognition",
    )

    table_merge_enable: bool = Field(
        default=True,
        description="Enable cross-page table merging",
    )

    # ============ Page Range ============
    start_page: int = Field(
        default=0,
        description="Starting page index (0-based)",
    )

    end_page: int | None = Field(
        default=None,
        description="Ending page index (None = all pages)",
    )

    # ============ Output Options ============
    draw_layout_bbox: bool = Field(
        default=True,
        description="Draw layout bounding boxes",
    )

    dump_md: bool = Field(
        default=True,
        description="Output markdown file",
    )

    dump_middle_json: bool = Field(
        default=True,
        description="Output middle JSON file",
    )

    dump_model_output: bool = Field(
        default=True,
        description="Output model JSON file",
    )

    dump_orig_pdf: bool = Field(
        default=False,
        description="Save original PDF",
    )

    dump_content_list: bool = Field(
        default=True,
        description="Output content list JSON",
    )

    make_md_mode: Literal[
        "mm_markdown", "nlp_markdown", "content_list", "content_list_v2"
    ] = Field(
        default="mm_markdown",
        description="Markdown generation mode",
    )

    # ============ Logging ============
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # ============ Progress Display ============
    use_tqdm: bool = Field(
        default=True,
        description="Show progress bars during processing",
    )

    # ============ Debug Settings ============
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    debug_dir: Path | None = Field(
        default=None,
        description="Debug output directory",
    )
