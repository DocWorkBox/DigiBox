# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Minimal shim for avaturn_live_streamer.core.logs used by the localrtc streamer slice.

The upstream `get_logger` returns a structlog BoundLogger; downstream code calls
`.info(..., key=value)` etc. on it. `setup_logging(cfg)` does light stdlib +
structlog wiring at the requested level; nothing fancy — no Sentry, no
correlation IDs, no OTel.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from avaturn_live_streamer.core.logging_config import LoggingConfig

_IS_SETUP = False


def get_logger(*args: Any, **kwargs: Any) -> Any:
    return structlog.get_logger(*args, **kwargs)


def setup_logging(config: LoggingConfig) -> None:
    global _IS_SETUP
    if _IS_SETUP:
        return
    _IS_SETUP = True

    level = getattr(logging, config.level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
