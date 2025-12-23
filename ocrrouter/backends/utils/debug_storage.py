"""Debug storage utilities for saving failed VLM requests."""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import io
from PIL import Image
from loguru import logger


def save_failed_request(
    *,
    error: Exception,
    image: Image.Image | bytes | None = None,
    prompt: str | None = None,
    messages: list | None = None,
    kwargs: dict | None = None,
    model_name: str | None = None,
    server_url: str | None = None,
    debug_dir: str | Path | None = None,
    error_type: str = "unknown",
) -> Path | None:
    """Save failed VLM request for debugging.

    Args:
        error: The exception that occurred
        image: The image that caused the failure (PIL Image or bytes)
        prompt: The text prompt used
        messages: The full message payload
        kwargs: The invoke kwargs (sampling params, etc.)
        model_name: Name of the model
        server_url: VLM server URL
        debug_dir: Directory to save debug files (defaults to ./debug)
        error_type: Type of error (timeout, connection, etc.)

    Returns:
        Path to the created debug directory, or None if saving failed
    """
    if debug_dir is None:
        debug_dir = Path("debug")
    else:
        debug_dir = Path(debug_dir)

    try:
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        debug_subdir = debug_dir / "failed_requests" / f"{timestamp}_{error_type}"
        debug_subdir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "model_name": model_name,
            "server_url": server_url,
            "prompt": prompt,
            "kwargs": _sanitize_for_json(kwargs) if kwargs else None,
        }

        with open(debug_subdir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save image if provided
        if image is not None:
            if isinstance(image, bytes):
                # Convert bytes to PIL Image and save as PNG
                pil_image = Image.open(io.BytesIO(image))
                pil_image.save(debug_subdir / "image.png", format="PNG")
            elif isinstance(image, Image.Image):
                # Save PIL Image as PNG
                image.save(debug_subdir / "image.png", format="PNG")

        # Save full messages if provided (sanitized)
        if messages is not None:
            sanitized_messages = _sanitize_messages(messages)
            with open(debug_subdir / "messages.json", "w", encoding="utf-8") as f:
                json.dump(sanitized_messages, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved failed request debug info to: {debug_subdir}")
        return debug_subdir

    except Exception as e:
        logger.warning(f"Failed to save debug info: {e}", exc_info=True)
        return None


def _sanitize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def _sanitize_messages(messages: list) -> list:
    """Sanitize messages by truncating base64 image data."""
    import copy
    import re

    sanitized = copy.deepcopy(messages)

    def truncate_base64_in_content(content):
        """Recursively truncate base64 image data in message content."""
        if isinstance(content, dict):
            result = {}
            for key, value in content.items():
                if (
                    key == "url"
                    and isinstance(value, str)
                    and value.startswith("data:image")
                ):
                    match = re.match(r"(data:image/[^;]+;base64,)(.+)", value)
                    if match:
                        prefix = match.group(1)
                        base64_data = match.group(2)
                        result[key] = (
                            f"{prefix}{base64_data[:50]}...<truncated {len(base64_data)} chars>"
                        )
                    else:
                        result[key] = value
                else:
                    result[key] = truncate_base64_in_content(value)
            return result
        elif isinstance(content, list):
            return [truncate_base64_in_content(item) for item in content]
        else:
            return content

    for message in sanitized:
        if hasattr(message, "content"):
            message.content = truncate_base64_in_content(message.content)
        elif isinstance(message, dict) and "content" in message:
            message["content"] = truncate_base64_in_content(message["content"])

    return [_sanitize_for_json(msg) for msg in sanitized]


def cleanup_old_debug_files(
    debug_dir: str | Path | None = None, keep_last_n: int = 100
) -> None:
    """Clean up old debug files, keeping only the most recent N entries.

    Args:
        debug_dir: Directory containing debug files
        keep_last_n: Number of most recent debug entries to keep
    """
    if debug_dir is None:
        debug_dir = Path("debug")
    else:
        debug_dir = Path(debug_dir)

    failed_requests_dir = debug_dir / "failed_requests"
    if not failed_requests_dir.exists():
        return

    try:
        # Get all subdirectories sorted by creation time
        subdirs = sorted(
            [d for d in failed_requests_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_ctime,
            reverse=True,
        )

        # Remove old directories
        for old_dir in subdirs[keep_last_n:]:
            import shutil

            shutil.rmtree(old_dir)
            logger.debug(f"Removed old debug directory: {old_dir}")

    except Exception as e:
        logger.warning(f"Failed to cleanup old debug files: {e}", exc_info=True)
