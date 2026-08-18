# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

import asyncio
import contextvars
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def run_in_thread(fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[R]":
    """Run ``fn`` in the default executor; return an ``asyncio.Future``.

    Submits the thread synchronously so it is already running by the time
    the function returns. Propagates the current ``contextvars.Context``.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return loop.run_in_executor(None, lambda: ctx.run(fn, *args, **kwargs))
