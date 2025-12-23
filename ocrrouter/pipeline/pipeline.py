"""Main document processing pipeline."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.backends import get_backend
from ocrrouter.utils.io.writers import FileBasedDataWriter
from ocrrouter.utils.run_async import run_async
from ocrrouter.observability import (
    generate_session_id,
    get_langfuse_client,
    observe,
    set_langfuse_client,
)

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
        >>> # Simple usage with constructor arguments
        >>> pipeline = DocumentPipeline(
        ...     backend="deepseek",
        ...     openai_base_url="https://api.example.com",
        ...     openai_api_key="sk-...",
        ... )
        >>> result = pipeline.process("document.pdf", "output/")
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

    @observe(name="process-document", capture_input=False, capture_output=False)
    async def aio_process(
        self,
        input_path: str | Path,
        output_dir: str,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        session_id: str | None = None,
        **options: Any,
    ) -> dict:
        """Process a document asynchronously.

        Args:
            input_path: Path to the input file (PDF or image).
            output_dir: Directory to write output files.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            session_id: Optional session ID for grouping traces in batch processing.
            **options: Additional processing options.

        Returns:
            Dictionary containing processing results.
        """
        # Step 1: Read input
        logger.debug(f"Reading input: {input_path}")
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        pdf_file_name = input_path.stem
        pdf_bytes = self.input_handler.read(input_path)

        # Update trace metadata if Langfuse is configured
        langfuse = get_langfuse_client()
        if langfuse is not None:
            # Generate session_id if not provided
            if session_id is None:
                session_id = generate_session_id(name=pdf_file_name)

            langfuse.update_current_trace(
                name=f"document-{pdf_file_name}",
                session_id=session_id,
                tags=["ocrrouter", self._settings.backend],
                metadata={
                    "backend": self._settings.backend,
                    "document_name": pdf_file_name,
                },
            )

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

        # Step 6: Write output
        logger.debug("Writing output...")
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

        logger.info(f"Completed: {input_path} -> {local_md_dir}")

        return {
            "pdf_file_name": pdf_file_name,
            "output_dir": local_md_dir,
            "middle_json": middle_json,
            "markdown": formatted_output.get("markdown"),
            "content_list": formatted_output.get("content_list"),
        }

    def process(
        self,
        input_path: str | Path,
        output_dir: str,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **options: Any,
    ) -> dict:
        """Process a document synchronously.

        This is a synchronous wrapper around aio_process.

        Args:
            input_path: Path to the input file (PDF or image).
            output_dir: Directory to write output files.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            **options: Additional processing options.

        Returns:
            Dictionary containing processing results.
        """
        return run_async(
            self.aio_process(
                input_path,
                output_dir,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                **options,
            )
        )

    async def aio_process_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        session_id: str | None = None,
        **options: Any,
    ) -> list[dict]:
        """Process multiple documents asynchronously.

        Args:
            input_paths: List of paths to input files.
            output_dir: Directory to write output files.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            session_id: Optional session ID for grouping all documents in this batch.
            **options: Additional processing options.

        Returns:
            List of dictionaries containing processing results.
        """
        # Generate shared session_id for batch if Langfuse is configured
        langfuse = get_langfuse_client()
        if langfuse is not None and session_id is None:
            session_id = generate_session_id(name="batch", prefix="mineru-batch")

        logger.info(
            f"Processing batch of {len(input_paths)} documents"
            + (f" with session_id: {session_id}" if session_id else "")
        )

        results = []
        for input_path in input_paths:
            result = await self.aio_process(
                input_path,
                output_dir,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                session_id=session_id,  # Pass shared session_id
                **options,
            )
            results.append(result)
        return results

    def process_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        **options: Any,
    ) -> list[dict]:
        """Process multiple documents synchronously.

        Args:
            input_paths: List of paths to input files.
            output_dir: Directory to write output files.
            start_page_id: Starting page index (0-based).
            end_page_id: Ending page index (0-based).
            **options: Additional processing options.

        Returns:
            List of dictionaries containing processing results.
        """
        return run_async(
            self.aio_process_batch(
                input_paths,
                output_dir,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                **options,
            )
        )
