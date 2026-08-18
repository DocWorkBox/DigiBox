# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Literal

from pydantic import BaseModel, Field

from avaturn_live_streamer.constant import VIDEO_RESOLUTION
from avaturn_live_streamer.conversation_engines.configs import ConversationEngineConfig
from avaturn_live_streamer.renderer.interface import RendererParamValue
from avaturn_live_streamer.renderer.models import ModelName
from avaturn_live_streamer.types import BackgroundId, PixelFormat, RendererAvatarId


class RendererConfig(BaseModel):
    avatar_id: RendererAvatarId
    background_id: BackgroundId | Literal["transparent"]
    pixel_format: PixelFormat = PixelFormat.YUV_I420
    model: ModelName = "avtrn-1"
    height: int = Field(default=VIDEO_RESOLUTION[0], gt=0, multiple_of=2)
    width: int = Field(default=VIDEO_RESOLUTION[1], gt=0, multiple_of=2)

    extra_params: dict[str, RendererParamValue] = Field(default_factory=dict)


class DailyTransportConfig(BaseModel):
    """Daily.co WebRTC transport."""

    type: Literal["daily"] = "daily"
    room_name: str


TransportConfig = DailyTransportConfig


class StreamConfig(BaseModel):
    renderer: RendererConfig
    conversation_engine: ConversationEngineConfig
    transport: TransportConfig

    # Session timeouts (always enabled)
    user_absent_timeout: int = 60  # seconds, default 1 minute
    max_duration: int = 3600  # seconds, default 1 hour
