"""Main document processing pipeline."""

import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.backends import get_backend
from ocrrouter.utils.io.writers import FileBasedDataWriter
from ocrrouter.utils.run_async import run_async
from ocrrouter.observability import set_langfuse_client

from ocrrouter.preprocessor import InputHandler, Preprocessor
from ocrrouter.postprocessor import Postprocessor, OutputHandler


class DocumentPipeline:
    """Main pipeline for document processing.

    This class orchestrates the entire document processing workflow:
    1. Input handling (read and validate files)
    2. Preprocessing (page selection, PDF preparation)
    3. Backend processing (layout detection, content extraction)
    4. Post-processing (format conversion to markdown, etc.)
    5. Output handling (write results to disk)

    Example:
        >>> from ocrrouter import DocumentPipeline, Settings
        >>>
        >>> # Development: just get the result (uses temp directory)
        >>> pipeline = DocumentPipeline(backend="deepseek")
        >>> result = pipeline.process("document.pdf")
        >>> print(result["markdown"])
        >>>
        >>> # Production: persist to disk
        >>> pipeline = DocumentPipeline(
        ...     backend="deepseek",
        ...     openai_base_url="https://api.example.com",
        ...     openai_api_key="sk-...",
        ... )
        >>> result = pipeline.process("document.pdf", "output/")
        >>>
        >>> # Process from bytes (e.g., from API upload)
        >>> with open("document.pdf", "rb") as f:
        ...     pdf_bytes = f.read()
        >>> result = pipeline.process(pdf_bytes, filename="document")  # temp dir
        >>> result = pipeline.process(pdf_bytes, "output/", filename="document")
        >>>
        >>> # With Langfuse observability (parent app owns the client)
        >>> from langfuse import Langfuse
        >>> langfuse = Langfuse(public_key="pk-...", secret_key="sk-...")
        >>> pipeline = DocumentPipeline(settings=settings, langfuse=langfuse)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        langfuse: Any | None = None,
        **overrides: Any,
    ):
        """Initialize the document pipeline.

        Args:
            settings: Settings object with configuration. If not provided,
                a new Settings instance is created from overrides.
            langfuse: Optional Langfuse client for observability. If provided,
                tracing will be enabled using this client. The client should be
                created and configured by the parent application.
            **overrides: Configuration overrides. These are applied on top of
                the settings object, or used to create a new Settings if none provided.
                Common options: backend, openai_base_url, openai_api_key,
                max_concurrency, http_timeout, etc.
        """
        # Create settings from overrides if not provided
        if settings is None:
            self._settings = Settings(**overrides)
        else:
            # Apply overrides on top of provided settings with validation
            if overrides:
                self._settings = Settings.model_validate(
                    {**settings.model_dump(), **overrides}
                )
            else:
                self._settings = settings

        # Configure loguru log level globally
        logger.remove()
        logger.add(sys.stderr, level=self._settings.log_level)

        # Set Langfuse client only if provided (don't clear existing config)
        if langfuse is not None:
            set_langfuse_client(langfuse)

        # Initialize pipeline components with settings
        self.input_handler = InputHandler()
        self.preprocessor = Preprocessor(self._settings)
        self.postprocessor = Postprocessor(self._settings)
        self.output_handler = OutputHandler(self._settings)

        # Backend is lazily initialized
        self._backend = None

    @property
    def settings(self) -> Settings:
        """Get the current settings."""
        return self._settings

    @property
    def backend(self):
        """Get the backend instance, creating it if necessary."""
        if self._backend is None:
            self._backend = get_backend(self._settings.backend, settings=self._settings)
        return self._backend

    async def aio_process(
        self,
        input_data: str | Path | bytes,
        output_dir: str | None = None,
        filename: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **options: Any,
    ) -> dict:
        """Process a document asynchronously.

        Args:
            input_data: Input source - can be:
                - str or Path: Path to the input file (PDF or image)
                - bytes: Raw file bytes (PDF or image, detected via Magika)
            output_dir: Directory to write output files. If None, a temporary
                directory is used and only images are saved (for downstream use).
                Other output files (markdown, JSON, etc.) are skipped.
            filename: Name for the output file (without extension).
                Required when input_data is bytes, optional for paths
                (defaults to path stem).
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            **options: Additional processing options.

        Returns:
            Dictionary containing processing results.

        Raises:
            ValueError: If input_data is bytes and filename is not provided.
        """
        # Track if user provided output_dir (affects file writing behavior)
        user_provided_output = output_dir is not None

        # Use temp directory if output_dir not provided
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="ocrrouter_")
            logger.debug(f"Using temporary output directory: {output_dir}")

        # Step 1: Read input
        if isinstance(input_data, bytes):
            if filename is None:
                raise ValueError("filename is required when input_data is bytes")
            pdf_file_name = filename
            logger.debug(f"Reading input from bytes: {pdf_file_name}")
        else:
            input_path = (
                Path(input_data) if not isinstance(input_data, Path) else input_data
            )
            pdf_file_name = filename if filename is not None else input_path.stem
            logger.debug(f"Reading input: {input_path}")

        pdf_bytes = self.input_handler.read(input_data)

        # Step 2: Preprocess (page selection)
        logger.debug("Preprocessing document...")
        prepared_pdf = self.preprocessor.prepare(
            pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **options,
        )

        # Step 3: Prepare output directories
        local_image_dir, local_md_dir = self.output_handler.prepare_output_dirs(
            output_dir, pdf_file_name, parse_method="vlm"
        )
        image_writer = FileBasedDataWriter(local_image_dir)

        # Step 4: Analyze with backend
        logger.debug(f"Analyzing with {self._settings.backend} backend...")
        middle_json, model_output = await self.backend.analyze(
            prepared_pdf,
            image_writer=image_writer,
            **options,
        )

        # Step 5: Post-process
        logger.debug("Post-processing results...")
        formatted_output = self.postprocessor.format(
            middle_json,
            image_dir="images",
            **options,
        )

        # Step 6: Write output (only if user provided output_dir)
        if user_provided_output:
            logger.debug("Writing output files...")
            self.output_handler.write(
                pdf_file_name=pdf_file_name,
                pdf_bytes=prepared_pdf,
                middle_json=middle_json,
                formatted_output=formatted_output,
                output_dir=local_md_dir,
                local_image_dir=local_image_dir,
                model_output=model_output,
                **options,
            )
        else:
            logger.debug("Skipping output file writing (temp directory mode)")

        if isinstance(input_data, bytes):
            logger.info(f"Completed: {pdf_file_name} (from bytes) -> {local_md_dir}")
        else:
            logger.info(f"Completed: {input_data} -> {local_md_dir}")

        return {
            "pdf_file_name": pdf_file_name,
            "output_dir": local_md_dir,
            "middle_json": middle_json,
            "markdown": formatted_output.get("markdown"),
            "content_list": formatted_output.get("content_list"),
        }

    def process(
        self,
        input_data: str | Path | bytes,
        output_dir: str | None = None,
        filename: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **options: Any,
    ) -> dict:
        """Process a document synchronously.

        This is a synchronous wrapper around aio_process.

        Args:
            input_data: Input source - can be:
                - str or Path: Path to the input file (PDF or image)
                - bytes: Raw file bytes (PDF or image, detected via Magika)
            output_dir: Directory to write output files. If None, a temporary
                directory is used and only images are saved (for downstream use).
                Other output files (markdown, JSON, etc.) are skipped.
            filename: Name for the output file (without extension).
                Required when input_data is bytes, optional for paths
                (defaults to path stem).
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            **options: Additional processing options.

        Returns:
            Dictionary containing processing results.

        Raises:
            ValueError: If input_data is bytes and filename is not provided.
        """
        return run_async(
            self.aio_process(
                input_data,
                output_dir,
                filename=filename,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                **options,
            )
        )

    async def aio_process_batch(
        self,
        inputs: list[str | Path | bytes | tuple[str, bytes]],
        output_dir: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **options: Any,
    ) -> list[dict]:
        """Process multiple documents asynchronously.

        Args:
            inputs: List of inputs. Each can be:
                - str or Path: File path (filename derived from path stem)
                - bytes: Raw bytes (filename defaults to "document_N")
                - tuple[str, bytes]: (filename, raw_bytes)
            output_dir: Directory to write output files. If None, a temporary
                directory is used and only images are saved (for downstream use).
                Other output files (markdown, JSON, etc.) are skipped.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            **options: Additional processing options.

        Returns:
            List of dictionaries containing processing results.
        """
        logger.info(f"Processing batch of {len(inputs)} documents")

        results = []
        for i, input_data in enumerate(inputs):
            # Handle different input types
            if isinstance(input_data, tuple):
                filename, file_bytes = input_data
                result = await self.aio_process(
                    file_bytes,
                    output_dir,
                    filename=filename,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    **options,
                )
            elif isinstance(input_data, bytes):
                result = await self.aio_process(
                    input_data,
                    output_dir,
                    filename=f"document_{i}",
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    **options,
                )
            else:
                result = await self.aio_process(
                    input_data,
                    output_dir,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    **options,
                )
            results.append(result)
        return results

    def process_batch(
        self,
        inputs: list[str | Path | bytes | tuple[str, bytes]],
        output_dir: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **options: Any,
    ) -> list[dict]:
        """Process multiple documents synchronously.

        Args:
            inputs: List of inputs. Each can be:
                - str or Path: File path (filename derived from path stem)
                - bytes: Raw bytes (filename defaults to "document_N")
                - tuple[str, bytes]: (filename, raw_bytes)
            output_dir: Directory to write output files. If None, a temporary
                directory is used and only images are saved (for downstream use).
                Other output files (markdown, JSON, etc.) are skipped.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            **options: Additional processing options.

        Returns:
            List of dictionaries containing processing results.
        """
        return run_async(
            self.aio_process_batch(
                inputs,
                output_dir,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                **options,
            )
        )
