"""Shared utilities for backend implementations."""

from .api_retry import api_retry
from .converter import blocks_to_page_info, result_to_middle_json
from .debug_storage import cleanup_old_debug_files, save_failed_request
from .image_utils import (
    ImageFormatError,
    aio_load_resource,
    gather_tasks,
    get_image_data_url,
    get_image_format,
    get_png_bytes,
    get_rgb_image,
    load_resource,
)
from .otsl2html import convert_otsl_to_html
from .structs import ANGLE_OPTIONS, BLOCK_TYPES, BlockType, ContentBlock
from .vlm_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    HttpVlmClient,
    RequestError,
    SamplingParams,
    ServerError,
    UnsupportedError,
    VlmClient,
    new_vlm_client,
)

__all__ = [
    # Structs
    "BlockType",
    "BLOCK_TYPES",
    "ANGLE_OPTIONS",
    "ContentBlock",
    # API retry
    "api_retry",
    # Converter
    "result_to_middle_json",
    "blocks_to_page_info",
    # Image utils
    "load_resource",
    "aio_load_resource",
    "get_png_bytes",
    "get_image_format",
    "get_image_data_url",
    "get_rgb_image",
    "gather_tasks",
    "ImageFormatError",
    # VLM client
    "VlmClient",
    "HttpVlmClient",
    "SamplingParams",
    "new_vlm_client",
    "UnsupportedError",
    "RequestError",
    "ServerError",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    # OTSL to HTML converter
    "convert_otsl_to_html",
    # Debug storage
    "save_failed_request",
    "cleanup_old_debug_files",
]
