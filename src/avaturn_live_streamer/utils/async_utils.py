# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
import contextvars
from asyncio import CancelledError, Task
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import (
    AsyncContextManager,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    ParamSpec,
    TypeVar,
)

from avaturn_live_streamer.core.logs import get_logger

_LOGGER = get_logger()

P = ParamSpec("P")
R = TypeVar("R")


def aiotime() -> float:
    return asyncio.get_event_loop().time()


def run_in_thread(fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[R]":
    """Run ``fn`` in the default executor; return an ``asyncio.Future``.

    Like ``asyncio.to_thread(fn, *args)``
    except the thread is submitted synchronously on this line, so it is
    already running by the time the function returns. Useful when you
    want to launch a sync worker and concurrently do other async work,
    then ``await`` its result later -- typical pattern for a sync
    receive loop that pushes results into an asyncio queue.

    Propagates the current ``contextvars.Context`` into the thread, the
    same way ``asyncio.to_thread`` does.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return loop.run_in_executor(None, lambda: ctx.run(fn, *args, **kwargs))


async def collect_async_iterator[T](gen: AsyncIterable[T]) -> list[T]:
    result = list[T]()

    async for i in gen:
        result.append(i)

    return result


async def as_async_iterator[T](sync: Iterable[T]) -> AsyncIterator[T]:
    for i in sync:
        yield i


async def wait_for_cancellation_completion(awaitable: Awaitable):
    try:
        await awaitable
    except CancelledError:
        pass
    else:
        raise RuntimeError("Cancellation expected, but function completed normally")


async def cancel_and_wait_completion(task: Task, reason: str | None = None):
    task.cancel(reason)
    await wait_for_cancellation_completion(task)
    _LOGGER.debug("Task %s manual cancellation with reason %s succeeded", task, reason)


def typedasynccontextmanager[**P, R](
    func: Callable[P, AsyncGenerator[R, None] | AsyncIterator[R]],
) -> Callable[P, AsyncContextManager[R]]:
    """Same as asynccontextmanager, but correctly sets type annotations in object."""
    helper = asynccontextmanager(func)
    helper.__annotations__["return"] = AsyncContextManager[R]
    return helper
