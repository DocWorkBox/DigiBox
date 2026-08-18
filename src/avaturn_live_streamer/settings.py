# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Slim Pydantic-settings config for the localrtc streamer slice.

Replaces the heavy upstream `persona_api.config.Config` which pulled in db /
AWS Secrets / Stripe / etc. Here we keep only the fields the kept files
actually read.

Env vars (read from the process environment, with optional ``.env`` overlay):
    OPENAI__API_KEY                                      -> openai_api_key
    RENDERERS__RENDERERS__<NAME>__MODE                   -> renderers.renderers[<name>].mode
    RENDERERS__RENDERERS__<NAME>__LB_OR_INSTANCE_URL     -> renderers.renderers[<name>].lb_or_instance_url

The double-`RENDERERS__` segment matches upstream and is a consequence of
`Config.renderers: RenderersConfig` being itself a settings model with a
`renderers` field. The orchestrator sets these in the streamer subprocess env.

A ``.env`` file at the repo root is loaded if present (gitignored). Process
env takes precedence so the orchestrator's explicit injections still win.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from avaturn_live_streamer.core.logging_config import LoggingConfig
from avaturn_live_streamer.renderer import RenderersConfig

# Repo root is two levels up from this file: src/avaturn_live_streamer/settings.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=_REPO_ROOT / ".env",
        extra="ignore",
    )

    renderers: RenderersConfig = Field(default_factory=RenderersConfig)
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI__API_KEY", "openai_api_key"),
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@lru_cache()
def get_config() -> Config:
    return Config()
