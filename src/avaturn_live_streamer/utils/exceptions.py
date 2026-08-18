# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from asyncio import CancelledError
from collections.abc import Callable, Coroutine
from functools import wraps

from avaturn_live_streamer.core.logs import get_logger

_LOGGER = get_logger()


def async_supress_exception[**P, R](
    func: Callable[P, Coroutine[None, None, R]],
    exception_type: type[Exception] = Exception,
) -> Callable[P, Coroutine[None, None, R | None]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return await func(*args, **kwargs)
        except exception_type:
            _LOGGER.exception(
                "Call to %s failed with exception, but it was suppressed", func.__name__
            )
            return None

    return wrapper


def async_log_entry_exit[**P, R](
    func: Callable[P, Coroutine[None, None, R]],
) -> Callable[P, Coroutine[None, None, R]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            _LOGGER.debug("Call to %s started", func.__qualname__)
            res = await func(*args, **kwargs)
            _LOGGER.debug("Call to %s succeeded", func.__qualname__)
            return res
        except CancelledError as e:
            message = e.args[0] if e.args else ""
            _LOGGER.debug("Call to %s cancelled with message %s", func.__qualname__, message)
            raise
        except BaseException as e:
            _LOGGER.debug("Call to %s failed with exception", func.__qualname__, exc_info=True)
            raise e

    return wrapper
