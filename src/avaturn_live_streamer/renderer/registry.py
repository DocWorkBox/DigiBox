# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import attrs
from pydantic import field_validator
from pydantic_settings import BaseSettings

from avaturn_live_streamer.renderer.interface import AbstractRendererClient
from avaturn_live_streamer.renderer.models import ModelName

from .config import RendererClientConfig, configure_renderer_client

_ENV_KEY_ALIASES = {"avtrn_1": "avtrn-1"}


class RenderersConfig(BaseSettings):
    renderers: dict[str, RendererClientConfig] = {
        "avtrn-1": RendererClientConfig(
            mode="single",
            lb_or_instance_url="http://localhost:8000",
        ),
    }

    @field_validator("renderers", mode="before")
    @classmethod
    def _normalize_keys(cls, value: object) -> object:
        # Env var segments can't contain hyphens (k8s envFrom requires C-identifier
        # names), so we accept the underscore form and map it back to the canonical
        # hyphenated model name.
        if isinstance(value, dict):
            return {
                _ENV_KEY_ALIASES.get(k.lower(), k.lower()) if isinstance(k, str) else k: v
                for k, v in value.items()
            }
        return value


@attrs.define
class RendererClientRegistry:
    _renderers: dict[str, AbstractRendererClient]

    def get_renderer(self, model: ModelName) -> AbstractRendererClient:
        try:
            return self._renderers[model]
        except KeyError:
            raise ValueError(f"Unknown renderer: {model}")


def create_renderer_client_registry(config: RenderersConfig) -> RendererClientRegistry:
    return RendererClientRegistry(
        {n: configure_renderer_client(r, n) for n, r in config.renderers.items()}  # pyright: ignore[reportArgumentType]
    )
