import asyncio
import threading
from queue import Queue
from typing import Any, AsyncIterable, Coroutine, Iterable, TypeVar

T = TypeVar("T")


def run_async(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Args:
        coroutine: The coroutine to run.

    Returns:
        The result of the coroutine.

    Raises:
        ValueError: If the argument is not a coroutine.
    """
    if not asyncio.iscoroutine(coroutine):
        raise ValueError("a coroutine was expected, got {!r}".format(coroutine))

    try:
        loop = asyncio.get_running_loop()
        # If there's already a running loop, use nest_asyncio pattern
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coroutine)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coroutine)


def iter_async(iterable: AsyncIterable[T]) -> Iterable[T]:
    if not isinstance(iterable, AsyncIterable):
        raise ValueError("an async iterable was expected, got {!r}".format(iterable))

    queue = Queue()

    async def async_helper():
        try:
            async for chunk in iterable:
                queue.put(chunk)
            queue.put(None)
        except Exception as e:
            queue.put(e)

    def helper():
        run_async(async_helper())

    thread = threading.Thread(target=helper, daemon=True)
    thread.start()

    while True:
        chunk = queue.get()
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        yield chunk

    thread.join()
