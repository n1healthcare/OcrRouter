"""VLM (Vision Language Model) client for HTTP-based inference.

This module provides:
- SamplingParams: Configuration for model generation parameters
- VlmClient: Abstract base class for VLM clients
- HttpVlmClient: HTTP-based implementation using OpenAI-compatible API
- new_vlm_client: Factory function to create VLM clients
"""

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterable, Literal, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from loguru import logger

from ocrrouter.observability import get_langfuse_handler
from .api_retry import api_retry
from .image_utils import aio_load_resource, get_image_data_url, get_png_bytes
from .debug_storage import save_failed_request, cleanup_old_debug_files


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "What is the text in the illustrate?"


def _inject_langfuse_callback(kwargs: dict) -> dict:
    """Inject Langfuse callback handler into kwargs if available.

    Args:
        kwargs: The kwargs dict to inject the callback into

    Returns:
        Updated kwargs dict with Langfuse callback if available
    """
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        config = kwargs.get("config", {})
        callbacks = config.get("callbacks", [])

        if langfuse_handler not in callbacks:
            callbacks = [*callbacks, langfuse_handler]

        kwargs = {**kwargs, "config": {**config, "callbacks": callbacks}}

    return kwargs


class UnsupportedError(NotImplementedError):
    """Raised when an unsupported operation is requested."""

    pass


class RequestError(ValueError):
    """Raised when there's an error with the request parameters."""

    pass


class ServerError(RuntimeError):
    """Raised when there's a server-side error."""

    pass


@dataclass
class SamplingParams:
    """Parameters for controlling model text generation.

    Attributes:
        temperature: Controls randomness in generation. Lower = more deterministic.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        presence_penalty: Penalty for token presence.
        frequency_penalty: Penalty for token frequency.
        repetition_penalty: Penalty for repetition (vLLM-specific).
        no_repeat_ngram_size: Prevent n-gram repetition (vLLM-specific).
        max_new_tokens: Maximum tokens to generate.
        skip_special_tokens: Whether to skip special tokens in output (vLLM-specific).
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    max_new_tokens: int | None = None
    skip_special_tokens: bool | None = None


class VlmClient:
    """Abstract base class for Vision Language Model clients.

    Provides the interface for single and batch image prediction.
    """

    def __init__(
        self,
        *,
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
    ) -> None:
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.text_before_image = text_before_image
        self.allow_truncated_content = allow_truncated_content

    def build_sampling_params(
        self,
        sampling_params: SamplingParams | None,
    ) -> SamplingParams:
        """Merge default and override sampling parameters."""
        if self.sampling_params:
            temperature = self.sampling_params.temperature
            top_p = self.sampling_params.top_p
            top_k = self.sampling_params.top_k
            presence_penalty = self.sampling_params.presence_penalty
            frequency_penalty = self.sampling_params.frequency_penalty
            repetition_penalty = self.sampling_params.repetition_penalty
            no_repeat_ngram_size = self.sampling_params.no_repeat_ngram_size
            max_new_tokens = self.sampling_params.max_new_tokens
            skip_special_tokens = self.sampling_params.skip_special_tokens
        else:
            temperature = None
            top_p = None
            top_k = None
            presence_penalty = None
            frequency_penalty = None
            repetition_penalty = None
            no_repeat_ngram_size = None
            max_new_tokens = None
            skip_special_tokens = None

        if sampling_params:
            if sampling_params.temperature is not None:
                temperature = sampling_params.temperature
            if sampling_params.top_p is not None:
                top_p = sampling_params.top_p
            if sampling_params.top_k is not None:
                top_k = sampling_params.top_k
            if sampling_params.presence_penalty is not None:
                presence_penalty = sampling_params.presence_penalty
            if sampling_params.frequency_penalty is not None:
                frequency_penalty = sampling_params.frequency_penalty
            if sampling_params.repetition_penalty is not None:
                repetition_penalty = sampling_params.repetition_penalty
            if sampling_params.no_repeat_ngram_size is not None:
                no_repeat_ngram_size = sampling_params.no_repeat_ngram_size
            if sampling_params.max_new_tokens is not None:
                max_new_tokens = sampling_params.max_new_tokens
            if sampling_params.skip_special_tokens is not None:
                skip_special_tokens = sampling_params.skip_special_tokens

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=skip_special_tokens,
        )

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """Predict on a single image asynchronously."""
        raise NotImplementedError()

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        """Predict on a batch of images asynchronously."""
        raise NotImplementedError()


class HttpVlmClient(VlmClient):
    """HTTP-based VLM client using OpenAI-compatible API.

    Uses LangChain's ChatOpenAI for communication with vLLM or other
    OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        server_url: str,
        api_key: str,
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 100,
        http_timeout: int = 600,
        debug: bool = False,
        debug_dir: str | None = None,
        max_retries: int = 3,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        self.max_concurrency = max_concurrency
        self.http_timeout = http_timeout
        self.max_retries = max_retries
        self.debug = debug
        self.debug_dir = debug_dir

        if not server_url:
            raise ValueError(
                "Server URL must be provided either as parameter or via OPENAI_BASE_URL environment variable"
            )

        if api_key is None:
            raise ValueError(
                "API key must be provided either as parameter or via OPENAI_API_KEY environment variable"
            )

        if model_name is None:
            raise ValueError(
                "VLM model name must be provided either as parameter or via VLM_MODEL_NAME environment variable"
            )

        # Initialize ChatOpenAI with retries disabled (we use api_retry instead)
        self.chat_model = ChatOpenAI(
            model=model_name,
            base_url=server_url,
            api_key=api_key,
            timeout=http_timeout,
            max_retries=0,  # Disable LangChain's retry, use api_retry instead
        )

        self.model_name = model_name
        self.server_url = server_url
        self.logger = logger
        self.custom_extra_body = extra_body or {}

        # Create retry-wrapped invoke methods
        @api_retry(max_retries=self.max_retries, logger=self.logger)
        async def _ainvoke_with_retry(messages, **kwargs):
            kwargs = _inject_langfuse_callback(kwargs)
            return await self.chat_model.ainvoke(messages, **kwargs)

        @api_retry(max_retries=self.max_retries, logger=self.logger)
        def _invoke_with_retry(messages, **kwargs):
            kwargs = _inject_langfuse_callback(kwargs)
            return self.chat_model.invoke(messages, **kwargs)

        self._ainvoke_with_retry = _ainvoke_with_retry
        self._invoke_with_retry = _invoke_with_retry

    def _build_messages(
        self,
        system_prompt: str,
        image: bytes,
        prompt: str,
        image_format: str | None,
    ) -> list:
        """Build message list for ChatOpenAI."""
        image_url = get_image_data_url(image, image_format)
        prompt = prompt or self.prompt

        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # Build user message with image
        if "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_content = [
                *([{"type": "text", "text": prompt_1}] if prompt_1.strip() else []),
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                },
                *([{"type": "text", "text": prompt_2}] if prompt_2.strip() else []),
            ]
        elif self.text_before_image:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:  # image before text, which is the default behavior.
            user_content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]

        messages.append(HumanMessage(content=user_content))
        return messages

    def _build_invoke_kwargs(
        self,
        sampling_params: SamplingParams | None,
        priority: int | None,
    ) -> dict:
        """Build kwargs for ChatOpenAI invoke, including extra_body for custom params."""
        sp = self.build_sampling_params(sampling_params)

        # Build extra_body for non-standard OpenAI parameters
        extra_body = {}

        if priority is not None:
            extra_body["priority"] = priority

        # Add vllm-specific parameters
        if sp.top_k is not None:
            extra_body["top_k"] = sp.top_k
        if sp.repetition_penalty is not None:
            extra_body["repetition_penalty"] = sp.repetition_penalty
        if sp.no_repeat_ngram_size is not None:
            extra_body["vllm_xargs"] = {
                "no_repeat_ngram_size": sp.no_repeat_ngram_size,
                "debug": self.debug,
            }
        if sp.skip_special_tokens is not None:
            extra_body["skip_special_tokens"] = sp.skip_special_tokens

        # Merge custom extra_body (e.g., DeepSeek vllm_xargs)
        if self.custom_extra_body:
            for key, value in self.custom_extra_body.items():
                if key == "vllm_xargs" and "vllm_xargs" in extra_body:
                    # Merge vllm_xargs dicts
                    extra_body["vllm_xargs"].update(value)
                else:
                    extra_body[key] = value

        # Remove parameters not supported by GPT models
        if self.model_name.lower().startswith("gpt"):
            extra_body.pop("top_k", None)
            extra_body.pop("repetition_penalty", None)
            extra_body.pop("skip_special_tokens", None)

        # Build main kwargs
        kwargs = {}

        if sp.temperature is not None:
            kwargs["temperature"] = sp.temperature
        if sp.top_p is not None:
            kwargs["top_p"] = sp.top_p
        if sp.presence_penalty is not None:
            kwargs["presence_penalty"] = sp.presence_penalty
        if sp.frequency_penalty is not None:
            kwargs["frequency_penalty"] = sp.frequency_penalty
        if sp.max_new_tokens is not None:
            kwargs["max_tokens"] = sp.max_new_tokens

        # Add extra_body if not empty
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    def _wrap_exception(
        self,
        e: Exception,
        context: dict | None = None,
    ) -> None:
        """Wrap LangChain exceptions to maintain same error interface.

        Args:
            e: The exception that occurred
            context: Optional context for debugging, containing:
                - image: The PIL Image or bytes that failed
                - prompt: The text prompt used
                - messages: The full message payload
                - kwargs: The invoke kwargs (sampling params, etc.)
        """
        self.logger.error(
            "VLM client error",
            exc_info=True,
            extra={"error_type": type(e).__name__, "model": self.model_name},
        )

        error_msg = str(e).lower()
        error_type = "unknown"

        # Determine error type for better categorization
        if "authentication" in error_msg or "unauthorized" in error_msg:
            error_type = "authentication"
            translated = ServerError(
                f"Failed to connect to server {self.server_url}. Please check if the server is running."
            )
        elif "connection" in error_msg or "timeout" in error_msg:
            error_type = "timeout" if "timeout" in error_msg else "connection"
            translated = ServerError(
                f"Failed to connect to server {self.server_url}. Please check if the server is running."
            )
        elif "rate limit" in error_msg or "too many requests" in error_msg:
            error_type = "rate_limit"
            translated = ServerError(
                f"Failed to get model name from {self.server_url}. Status code: 429, response body: {e}"
            )
        elif "length" in error_msg or "truncated" in error_msg:
            error_type = "truncated"
            if not self.allow_truncated_content:
                translated = RequestError(
                    "The response was truncated due to length limit."
                )
            else:
                self.logger.warning("The response was truncated due to length limit.")
                return
        elif "model" in error_msg and (
            "not found" in error_msg or "does not exist" in error_msg
        ):
            error_type = "model_not_found"
            translated = RequestError(
                f"Model '{self.model_name}' not found in the response from {self.server_url}/v1/models. Please check if the model is available on the server."
            )
        elif "finish_reason" in error_msg or "finish reason" in error_msg:
            error_type = "finish_reason"
            translated = RequestError(f"Unexpected finish reason: {e}")
        else:
            error_type = "server_error"
            translated = ServerError(
                f"Unexpected status code: [500], response body: {e}"
            )

        # Save failed request if debug mode is enabled
        should_save = self.debug and error_type in [
            "timeout",
            "connection",
            "server_error",
        ]

        if should_save and context is not None:
            try:
                save_failed_request(
                    error=e,
                    image=context.get("image"),
                    prompt=context.get("prompt"),
                    messages=context.get("messages"),
                    kwargs=context.get("kwargs"),
                    model_name=self.model_name,
                    server_url=self.server_url,
                    debug_dir=self.debug_dir,
                    error_type=error_type,
                )
            except Exception as debug_error:
                self.logger.warning(f"Failed to save debug info: {debug_error}")

        raise translated from e

    def _sanitize_messages_for_debug(self, messages: list) -> list:
        """Create a debug-friendly version of messages with truncated base64 image data."""
        import copy
        import re

        sanitized = copy.deepcopy(messages)

        def truncate_base64_in_dict(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if (
                        key == "url"
                        and isinstance(value, str)
                        and value.startswith("data:image")
                    ):
                        match = re.match(r"(data:image/[^;]+;base64,)(.+)", value)
                        if match:
                            prefix = match.group(1)
                            base64_data = match.group(2)
                            truncated = f"{prefix}{base64_data[:50]}...<truncated {len(base64_data)} chars>"
                            obj[key] = truncated
                    else:
                        truncate_base64_in_dict(value)
            elif isinstance(obj, list):
                for item in obj:
                    truncate_base64_in_dict(item)

        for message in sanitized:
            if hasattr(message, "content"):
                truncate_base64_in_dict(message.content)

        return sanitized

    def _extract_content(self, response) -> str:
        """Extract content from ChatOpenAI response."""
        try:
            content = response.content
            if content is None:
                content = ""
            return content
        except Exception as e:
            raise ServerError(
                f"Failed to parse response JSON: {e}, response body: {response}"
            )

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
        async_client: None = None,
    ) -> str:
        # Store original image for debug context
        original_image = image

        image_format = None
        if isinstance(image, str):
            image = await aio_load_resource(image)
        if isinstance(image, Image.Image):
            image = get_png_bytes(image)
            image_format = "png"

        messages = self._build_messages(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            image_format=image_format,
        )

        kwargs = self._build_invoke_kwargs(sampling_params, priority)

        if self.debug:
            sanitized_messages = self._sanitize_messages_for_debug(messages)
            self.logger.debug(f"Async messages: {sanitized_messages}")
            self.logger.debug(f"Kwargs: {json.dumps(kwargs, indent=2, default=str)}")

        try:
            response = await self._ainvoke_with_retry(messages, **kwargs)
            if self.debug:
                self.logger.debug(f"Async response: {response}")
            return self._extract_content(response)
        except Exception as e:
            # Pass context for debug storage
            context = {
                "image": original_image,
                "prompt": prompt or self.prompt,
                "messages": messages,
                "kwargs": kwargs,
            }
            self._wrap_exception(e, context)

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        """Async batch prediction with per-request parameter support."""
        if not images:
            return []

        # Normalize inputs to sequences
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), (
            "Length of sampling_params and images must match."
        )
        assert len(priority) == len(images), "Length of priority and images must match."

        # Load and prepare all images
        processed_images = []
        for image in images:
            image_format = None
            if isinstance(image, str):
                image = await aio_load_resource(image)
            if isinstance(image, Image.Image):
                image = get_png_bytes(image)
                image_format = "png"
            processed_images.append((image, image_format))

        # Build all messages with per-request kwargs
        tasks = []
        for (img, img_format), prompt, sp, p in zip(
            processed_images, prompts, sampling_params, priority
        ):
            messages = self._build_messages(
                system_prompt=self.system_prompt,
                image=img,
                prompt=prompt,
                image_format=img_format,
            )
            kwargs = self._build_invoke_kwargs(sp, p)
            tasks.append(self._ainvoke_with_retry(messages, **kwargs))

        if self.debug:
            self.logger.debug(f"Batch size: {len(tasks)}")
            self.logger.debug(f"Max concurrency: {self.max_concurrency}")

        try:
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _execute_with_semaphore(task):
                async with semaphore:
                    return await task

            responses = await asyncio.gather(
                *[_execute_with_semaphore(task) for task in tasks]
            )

            results = [self._extract_content(response) for response in responses]

            if self.debug:
                self.logger.debug(f"Batch completed: {len(results)} results")

            return results
        except Exception as e:
            self._wrap_exception(e)

    async def aio_batch_predict_as_iter(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AsyncIterable[tuple[int, str]]:
        """Async batch prediction as iterator, yielding results as they complete."""
        if not images:
            return

        # Normalize inputs to sequences
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), (
            "Length of sampling_params and images must match."
        )
        assert len(priority) == len(images), "Length of priority and images must match."

        # Load and prepare all images
        processed_images = []
        for image in images:
            image_format = None
            if isinstance(image, str):
                image = await aio_load_resource(image)
            if isinstance(image, Image.Image):
                image = get_png_bytes(image)
                image_format = "png"
            processed_images.append((image, image_format))

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_one(idx, img_data, prompt, sp, p):
            img, img_format = img_data
            messages = self._build_messages(
                system_prompt=self.system_prompt,
                image=img,
                prompt=prompt,
                image_format=img_format,
            )
            kwargs = self._build_invoke_kwargs(sp, p)
            async with semaphore:
                response = await self._ainvoke_with_retry(messages, **kwargs)
                return (idx, self._extract_content(response))

        if self.debug:
            self.logger.debug(f"Batch as iter size: {len(images)}")

        try:
            tasks = [
                _process_one(idx, img_data, prompt, sp, p)
                for idx, (img_data, prompt, sp, p) in enumerate(
                    zip(processed_images, prompts, sampling_params, priority)
                )
            ]

            for coro in asyncio.as_completed(tasks):
                yield await coro

        except Exception as e:
            self._wrap_exception(e)


def new_vlm_client(
    backend: Literal["http-client"] = "http-client",
    model_name: str | None = None,
    server_url: str | None = None,
    api_key: str | None = None,
    prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    sampling_params: SamplingParams | None = None,
    text_before_image: bool = False,
    allow_truncated_content: bool = False,
    max_concurrency: int = 100,
    http_timeout: int = 600,
    debug: bool = False,
    debug_dir: str | None = None,
    max_retries: int = 3,
    extra_body: dict | None = None,
) -> VlmClient:
    """Factory function to create a VLM client.

    Args:
        backend: Currently only 'http-client' is supported.
        model_name: Name of the model to use.
        server_url: Base URL of the VLM server.
        api_key: API key for authentication.
        prompt: Default user prompt.
        system_prompt: Default system prompt.
        sampling_params: Default sampling parameters.
        text_before_image: Whether to place text before image in prompts.
        allow_truncated_content: Whether to allow truncated responses.
        max_concurrency: Maximum concurrent requests.
        http_timeout: HTTP timeout in seconds.
        debug: Enable debug logging.
        debug_dir: Directory for debug output (default: ./debug).
        max_retries: Maximum retry attempts.
        extra_body: Extra parameters to pass in request body.

    Returns:
        A configured VlmClient instance.
    """
    if backend != "http-client":
        raise ValueError(
            f"Unsupported backend: {backend}. Only 'http-client' is supported."
        )

    return HttpVlmClient(
        model_name=model_name,
        server_url=server_url,
        api_key=api_key,
        prompt=prompt,
        system_prompt=system_prompt,
        sampling_params=sampling_params,
        text_before_image=text_before_image,
        allow_truncated_content=allow_truncated_content,
        max_concurrency=max_concurrency,
        http_timeout=http_timeout,
        debug=debug,
        debug_dir=debug_dir,
        max_retries=max_retries,
        extra_body=extra_body,
    )
