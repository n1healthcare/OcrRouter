"""Base model client with shared caching and configuration logic."""

import time
from abc import ABC, abstractmethod
from typing import TypeVar

from loguru import logger

from ocrrouter.config import Settings
from ocrrouter.backends.utils import HttpVlmClient


ClientT = TypeVar("ClientT", bound="BaseModelClient")


class BaseModelClient(ABC):
    """Abstract base class for model clients with shared initialization and caching.

    This class provides:
    - Common initialization parameters with settings fallbacks
    - Lazy VLM client initialization via module-level caching
    - Standard interface for model-specific clients to implement

    Subclasses must implement:
    - backend_name: Property returning the name for logging
    - _create_vlm_client(): Factory method to create the backend-specific VLM client
    - _get_cache(): Return the module-level cache dict for client instances
    """

    def __init__(
        self,
        settings: Settings,
        **kwargs,
    ):
        """Initialize the client.

        Args:
            settings: Settings object with configuration.
            **kwargs: Additional configuration.
        """
        # Store settings
        self._settings = settings
        self.server_url = settings.openai_base_url
        self.max_concurrency = settings.max_concurrency
        self.http_timeout = settings.http_timeout
        self.max_retries = settings.max_retries
        self.debug = settings.debug
        self.kwargs = kwargs

        # Client is lazily initialized
        self._vlm_client: HttpVlmClient | None = None

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend for logging."""
        pass

    @abstractmethod
    def _create_vlm_client(self) -> HttpVlmClient:
        """Create a new VLM client instance.

        Subclasses must implement this to create the backend-specific client
        with appropriate configuration (model name, prompts, sampling params, etc.).

        Returns:
            Configured HttpVlmClient instance.
        """
        pass

    @abstractmethod
    def _get_cache(self) -> dict:
        """Return the module-level cache dict for client instances.

        Subclasses must implement this to return their module-level cache.
        This enables singleton pattern for VLM clients.

        Returns:
            Module-level dictionary for caching client instances.
        """
        pass

    @property
    def vlm_client(self) -> HttpVlmClient:
        """Get the VLM client instance, using lazy initialization and caching."""
        if self._vlm_client is None:
            key = (self.server_url, "http-client")
            cache = self._get_cache()

            if key not in cache:
                start_time = time.time()
                cache[key] = self._create_vlm_client()
                elapsed = round(time.time() - start_time, 2)
                logger.info(f"Created {self.backend_name} http-client: {elapsed}s")

            self._vlm_client = cache[key]
        return self._vlm_client

    def _get_client_params(self) -> dict:
        """Get resolved client parameters for VLM client creation."""
        return {
            "server_url": self.server_url,
            "max_concurrency": self.max_concurrency,
            "http_timeout": self.http_timeout,
            "max_retries": self.max_retries,
            "debug": self.debug,
        }
