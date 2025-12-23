"""Shared utilities for backend implementations."""

from .structs import BlockType, BLOCK_TYPES, ANGLE_OPTIONS, ContentBlock
from .api_retry import api_retry
from .converter import result_to_middle_json, blocks_to_page_info
from .image_utils import (
    load_resource,
    aio_load_resource,
    get_png_bytes,
    get_image_format,
    get_image_data_url,
    get_rgb_image,
    gather_tasks,
    ImageFormatError,
)
from .vlm_client import (
    VlmClient,
    HttpVlmClient,
    SamplingParams,
    new_vlm_client,
    UnsupportedError,
    RequestError,
    ServerError,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
)
from .otsl2html import convert_otsl_to_html
from .debug_storage import save_failed_request, cleanup_old_debug_files

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
