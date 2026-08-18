# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Literal

from pydantic_settings import BaseSettings

from avaturn_live_streamer.renderer.models import ModelName

from .interface import AbstractRendererClient

RendererLoadBalancerMode = Literal["mock", "load-balanced", "single"]


class RendererClientConfig(BaseSettings):
    mode: RendererLoadBalancerMode = "load-balanced"
    lb_or_instance_url: str | None = None


def configure_renderer_client(
    config: RendererClientConfig, model: ModelName
) -> AbstractRendererClient:
    from .client import RendererClient
    from .mock import MockRendererClient

    if config.mode == "mock":
        return MockRendererClient()

    if config.mode == "single":
        return RendererClient(model=model, instance_url=config.lb_or_instance_url, lb_url=None)
    elif config.mode == "load-balanced":
        return RendererClient(model=model, lb_url=config.lb_or_instance_url)
    else:
        raise ValueError(f"Unknown renderer mode: {config.mode}")
