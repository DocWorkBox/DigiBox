# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Minimal shim for avaturn_live_streamer.core.logging_config used by the localrtc streamer slice.

The upstream module exposes a much richer config; we keep just the level field
so `cfg.logging.level = ...; setup_logging(cfg.logging)` works in local_stream_cli.
"""

from pydantic_settings import BaseSettings


class LoggingConfig(BaseSettings):
    level: str = "INFO"
