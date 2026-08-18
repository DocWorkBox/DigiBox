# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Awaitable, Literal

from avaturn_live_streamer.types import PixelFormat

RendererParamValue = str | int | float | bool


@dataclass(slots=True, unsafe_hash=True)
class RenderConfig:
    avatar_id: str
    background_id: str | Literal["transparent"]
    pixel_format: PixelFormat
    width: int
    height: int

    extra_params: dict[str, RendererParamValue] = field(default_factory=dict, hash=False)


@dataclass(slots=True)
class RendererRequest:
    current_chunk: bytes
    future_chunk: bytes
    current_chunk_listen: bytes
    future_chunk_listen: bytes
    timestamp_global: float
    state: bytes | None
    config: RenderConfig


@dataclass(slots=True)
class RenderResponse:
    frame_generator: AsyncGenerator[bytes, None]
    num_frames: int
    state: Awaitable[bytes | None]


class AbstractRendererClient(ABC):
    @abstractmethod
    def generate(self, request: RendererRequest) -> AbstractAsyncContextManager[RenderResponse]:
        pass
