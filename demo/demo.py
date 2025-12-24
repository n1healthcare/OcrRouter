"""Demo script showing how to use OCRRouter for document parsing.

This demo showcases both the high-level DocumentPipeline API and
the lower-level backend access for more granular control.
"""

import asyncio
import os
from pathlib import Path

from loguru import logger

from ocrrouter import DocumentPipeline, Settings
from ocrrouter.observability import generate_session_id
from ocrrouter.preprocessor.utils import guess_suffix_by_path
from ocrrouter.backends.utils.image_utils import gather_tasks


# Supported file formats
PDF_SUFFIXES = ["pdf"]
IMAGE_SUFFIXES = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]


def create_langfuse_client():
    """Create Langfuse client from environment variables.

    Returns:
        Langfuse client if env vars are set, None otherwise.

    Environment variables:
        LANGFUSE_PUBLIC_KEY: Langfuse public API key
        LANGFUSE_SECRET_KEY: Langfuse secret API key
        LANGFUSE_HOST: Optional Langfuse host URL (default: https://cloud.langfuse.com)
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.debug(
            "Langfuse not configured (missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY)"
        )
        return None

    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("Langfuse client created from environment variables")
        return client
    except ImportError:
        logger.warning("langfuse package not installed, tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to create Langfuse client: {e}")
        return None


async def parse_documents(
    paths: list[Path],
    output_dir: str,
    settings: Settings | None = None,
    langfuse=None,
    session_id: str | None = None,
    max_concurrency: int = 4,
) -> list[dict]:
    """Parse documents using the high-level DocumentPipeline API.

    Args:
        paths: List of document paths to parse (PDF or image files).
        output_dir: Output directory for storing parsed results.
        settings: Settings object with configuration (backend, api keys, etc.).
        langfuse: Optional Langfuse client for observability tracing.
        session_id: Optional session ID for grouping traces in Langfuse.
                   If not provided, auto-generated for this batch.
        max_concurrency: Maximum number of documents to process concurrently.
                        Default is 4.

    Returns:
        List of processing results for each document.

    Example:
        settings = Settings(
            backend="deepseek",
            openai_base_url="http://localhost:30000",
            openai_api_key="your-key",
        )
        langfuse = create_langfuse_client()  # Or None to disable tracing
        await parse_documents(
            [Path("doc.pdf")],
            "./output",
            settings=settings,
            langfuse=langfuse,
            max_concurrency=4,
        )
    """
    # Generate session_id for batch if not provided
    if session_id is None:
        session_id = generate_session_id(name="demo-batch")

    logger.info(f"Processing batch with session_id: {session_id}")
    logger.info(f"Max concurrency: {max_concurrency}")

    # Create pipeline with settings and optional langfuse client
    pipeline = DocumentPipeline(settings=settings, langfuse=langfuse)

    # Log configuration once at the start
    backend_name = settings.backend if settings else "default"
    output_mode = settings.output_mode if settings else "all"
    if backend_name == "composite" and settings:
        logger.info(
            f"Using backend '{backend_name}' (mode: {output_mode}, "
            f"layout: {settings.layout_model}, ocr: {settings.ocr_model})"
        )
    else:
        logger.info(f"Using backend '{backend_name}' (mode: {output_mode})")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async def process_one(path: Path):
        """Process a single document with semaphore control."""
        async with semaphore:
            try:
                logger.info(f"Processing: {path}")
                result = await pipeline.aio_process(str(path), output_dir)
                logger.info(
                    f"Completed: {path} -> {result.get('output_dir', output_dir)}"
                )
                return result
            except Exception as e:
                logger.exception(f"Failed to process {path}: {e}")
                return None

    # Process all documents concurrently with semaphore control
    results = await gather_tasks(
        tasks=[process_one(path) for path in paths],
        use_tqdm=True,
        tqdm_desc="Parsing documents",
    )

    # Filter out None results (failed documents)
    return [r for r in results if r is not None]


def parse_documents_sync(
    paths: list[Path],
    output_dir: str,
    settings: Settings | None = None,
    langfuse=None,
    session_id: str | None = None,
    max_concurrency: int = 4,
) -> list[dict]:
    """Synchronous wrapper for parse_documents.

    Same arguments as parse_documents, but runs synchronously.

    Returns:
        List of processing results for each document.
    """
    return asyncio.run(
        parse_documents(
            paths=paths,
            output_dir=output_dir,
            settings=settings,
            langfuse=langfuse,
            session_id=session_id,
            max_concurrency=max_concurrency,
        )
    )


def get_documents(directory: str) -> list[Path]:
    """Find all supported documents in a directory.

    Args:
        directory: Path to directory containing documents.

    Returns:
        List of Path objects for supported files.
    """
    doc_paths = []
    for doc_path in Path(directory).glob("*"):
        suffix = guess_suffix_by_path(doc_path)
        if suffix in PDF_SUFFIXES + IMAGE_SUFFIXES:
            doc_paths.append(doc_path)
    return doc_paths


if __name__ == "__main__":
    # Configuration - create settings with your configuration
    settings = Settings(
        backend="generalvlm",  # Options: deepseek, mineru, dotsocr, composite, hunyuanocr, generalvlm
        output_mode="ocr_only",  # Options: all, layout_only, ocr_only
        generalvlm_model_name="ministral-3b-2512-openrouter",  # Options: ministral-3b-2512-openrouter, ministral-7b-2512-openrouter
        # layout_model="mineru",  # Options: mineru, deepseek, dotsocr
        # ocr_model="mineru",  # Options: mineru, deepseek, dotsocr, paddleocr, generalvlm
        openai_base_url=os.getenv("OPENAI_BASE_URL"),  # Your VLM server URL
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Your API key
        # start_page=0,  # Optional: starting page
        # end_page=None,  # Optional: ending page (None for all),
        # log_level="DEBUG",  # Set log level to DEBUG for more details
    )

    # Optional: Create Langfuse client from environment variables
    # Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing
    langfuse = create_langfuse_client()

    __dir__ = os.path.dirname(os.path.abspath(__file__))

    pdf_files_dir = os.path.join(__dir__, "input/pdfs")
    output_dir = os.path.join(__dir__, f"output/{settings.backend}")

    # pdf_files_dir = os.path.join(__dir__, "input/test-files")
    # output_dir = os.path.join(__dir__, f"output-test-files/{settings.backend}")

    # Find documents
    doc_paths = get_documents(pdf_files_dir)

    if not doc_paths:
        logger.warning(f"No documents found in {pdf_files_dir}")
        logger.info(f"Supported formats: {PDF_SUFFIXES + IMAGE_SUFFIXES}")
    else:
        logger.info(f"Found {len(doc_paths)} document(s)")

        # Parse documents with explicit settings and optional langfuse
        parse_documents_sync(
            paths=doc_paths,
            output_dir=output_dir,
            settings=settings,
            langfuse=langfuse,
            max_concurrency=4,
        )
    langfuse.shutdown() if langfuse else None
