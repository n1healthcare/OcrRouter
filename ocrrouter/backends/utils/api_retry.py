"""
API retry logic using tenacity with exponential backoff and full jitter.

This module provides a configurable retry decorator for API calls with:
- Exponential backoff with full jitter (both min and max scale)
- Configurable retry/no-retry status codes
- Optional logging
- Optional reraise behavior

Uses exponential backoff with full jitter where both min and max scale
exponentially to prevent thundering herd:
- Retry 1: random(min_base, max_base)
- Retry 2: random(min_base*2, max_base*2)
- Retry 3: random(min_base*4, max_base*4)

Example:
    from ocrrouter.backends.utils.api_retry import api_retry
    import logging

    # Basic usage
    @api_retry()
    def call_api(...):
        ...

    # With logging and reraise
    logger = logging.getLogger(__name__)

    @api_retry(logger=logger, reraise=True)
    def call_api(...):
        ...

    # Custom retry parameters
    @api_retry(max_retries=5, min_base=5, max_base=20)
    def call_api(...):
        ...
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Callable, Optional, Set, TypeVar, Union
import loguru

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
)
from tenacity.wait import wait_base

T = TypeVar("T")

# Default status codes
DEFAULT_NO_RETRY_STATUS_CODES: Set[int] = {400, 401, 403, 422}
DEFAULT_RETRY_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}

# Default retry parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_BASE = 10  # seconds
DEFAULT_MAX_BASE = 30  # seconds

# Fallback exponent used when exponential calculation overflows
_OVERFLOW_FALLBACK_EXPONENT = 10


class wait_exponential_full_jitter(wait_base):
    """Wait strategy with exponentially scaling random window (Full Jitter).

    Both the minimum and maximum wait times scale exponentially with each retry,
    providing increasing jitter windows that help prevent thundering herd while
    ensuring progressively longer waits.

    Formula: random(min_base * 2^attempt, max_base * 2^attempt)

    Example with min_base=10, max_base=30:
        - Retry 1: random(10, 30) seconds
        - Retry 2: random(20, 60) seconds
        - Retry 3: random(40, 120) seconds

    Args:
        min_base: Base minimum wait time in seconds (default: 10)
        max_base: Base maximum wait time in seconds (default: 30)
        exp_base: Exponential base (default: 2)
        cap: Optional maximum cap for wait time (default: None)
    """

    def __init__(
        self,
        min_base: Union[int, float] = DEFAULT_MIN_BASE,
        max_base: Union[int, float] = DEFAULT_MAX_BASE,
        exp_base: Union[int, float] = 2,
        cap: Optional[Union[int, float]] = None,
    ) -> None:
        self.min_base = min_base
        self.max_base = max_base
        self.exp_base = exp_base
        self.cap = cap

    def __call__(self, retry_state) -> float:
        attempt = retry_state.attempt_number - 1  # 0-indexed for exponent
        try:
            exp = self.exp_base**attempt
            min_wait = self.min_base * exp
            max_wait = self.max_base * exp
        except OverflowError:
            if self.cap:
                return self.cap
            return self.max_base * (self.exp_base**_OVERFLOW_FALLBACK_EXPONENT)

        if self.cap:
            max_wait = min(max_wait, self.cap)
            min_wait = min(min_wait, self.cap)

        return random.uniform(min_wait, max_wait)


def _create_should_retry_exception(
    no_retry_status_codes: Set[int],
    retry_status_codes: Set[int],
) -> Callable[[BaseException], bool]:
    """
    Create a retry exception checker with configurable status codes.

    Logic priority:
    1. If not an Exception (e.g., KeyboardInterrupt) -> Don't retry
    2. If status code in no_retry_status_codes -> Don't retry (blocklist)
    3. If status code in retry_status_codes -> Retry (allowlist)
    4. For other HTTP errors -> Retry by default
    5. For non-HTTP exceptions (connection errors, timeouts) -> Retry

    Args:
        no_retry_status_codes: Status codes that should NOT be retried
        retry_status_codes: Status codes that SHOULD be retried

    Returns:
        A function that checks if an exception should be retried
    """

    def _should_retry_exception(exception: BaseException) -> bool:
        # Prevent exceptions like KeyboardInterrupt from being retried
        if not isinstance(exception, Exception):
            return False

        # Check for HTTP status code (works with httpx, requests, etc.)
        if hasattr(exception, "response") and hasattr(
            exception.response, "status_code"
        ):
            status_code = exception.response.status_code

            # Blocklist takes precedence
            if status_code in no_retry_status_codes:
                return False

            # Check allowlist
            if status_code in retry_status_codes:
                return True

            # For other HTTP errors, retry by default (conservative approach)
            return True

        # For non-HTTP exceptions (connection errors, timeouts, etc.), retry
        return True

    return _should_retry_exception


def api_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_base: float = DEFAULT_MIN_BASE,
    max_base: float = DEFAULT_MAX_BASE,
    no_retry_status_codes: Optional[Set[int]] = None,
    retry_status_codes: Optional[Set[int]] = None,
    cap: Optional[float] = None,
    logger: Optional[Union[logging.Logger, loguru.logger]] = None,
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory for API retry with exponential backoff and full jitter.

    Supports both sync and async functions - automatically detects the function
    type and applies the appropriate wrapper.

    Uses exponential backoff with full jitter where both min and max scale:
    - Retry 1: random(min_base, max_base) seconds
    - Retry 2: random(min_base*2, max_base*2) seconds
    - Retry 3: random(min_base*4, max_base*4) seconds

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        min_base: Base minimum wait time in seconds (default: 10)
        max_base: Base maximum wait time in seconds (default: 30)
        no_retry_status_codes: Status codes that should NOT be retried
            (default: {400, 401, 403, 422})
        retry_status_codes: Status codes that SHOULD be retried
            (default: {429, 500, 502, 503, 504})
        cap: Optional maximum cap for wait time in seconds (default: None)
        logger: Optional logger for retry logging. If provided, logs before
            sleep (WARNING level) and after retry (DEBUG level)
        reraise: If True, re-raises the exception after all retries exhausted
            (default: False)

    Returns:
        Decorator that wraps functions with retry logic

    Example:
        # Sync function
        @api_retry()
        def call_api(...):
            return client.get("/endpoint")

        # Async function
        @api_retry()
        async def call_api_async(...):
            return await client.get("/endpoint")

        @api_retry(logger=logging.getLogger(__name__), reraise=True)
        def call_api(...):
            return client.post("/endpoint", data=data)
    """
    # Use defaults if not provided
    if no_retry_status_codes is None:
        no_retry_status_codes = DEFAULT_NO_RETRY_STATUS_CODES
    if retry_status_codes is None:
        retry_status_codes = DEFAULT_RETRY_STATUS_CODES

    # Create the retry checker with configured status codes
    should_retry = _create_should_retry_exception(
        no_retry_status_codes=no_retry_status_codes,
        retry_status_codes=retry_status_codes,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Build retry kwargs
        retry_kwargs = {
            "stop": stop_after_attempt(
                max_retries + 1
            ),  # +1 because initial attempt counts
            "wait": wait_exponential_full_jitter(
                min_base=min_base,
                max_base=max_base,
                cap=cap,
            ),
            "retry": retry_if_exception(should_retry),
            "reraise": reraise,
        }

        # Add logging if logger is provided
        if logger is not None:
            retry_kwargs["before_sleep"] = before_sleep_log(logger, logging.WARNING)
            retry_kwargs["after"] = after_log(logger, logging.DEBUG)

        # Auto-detect async vs sync function
        if asyncio.iscoroutinefunction(func):

            @retry(**retry_kwargs)
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @retry(**retry_kwargs)
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator
