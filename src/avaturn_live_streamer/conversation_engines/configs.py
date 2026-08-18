# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Literal

from pydantic import BaseModel, ConfigDict

OpenaiRealtimeApiVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]


class OpenAIRealtimeAPIConversationEngineConfig(BaseModel):
    type: Literal["openai-realtime"] = "openai-realtime"
    client_secret: str | None = None

    model_config = ConfigDict(title="OpenAIRealtimeAPICEConfig")


class CartesiaConversationEngineConfig(BaseModel):
    type: Literal["cartesia"] = "cartesia"
    access_token: str
    agent_id: str

    model_config = ConfigDict(title="CartesiaCEConfig")


class CodexRealtimeConversationEngineConfig(BaseModel):
    type: Literal["codex-realtime"] = "codex-realtime"

    model_config = ConfigDict(title="CodexRealtimeCEConfig")


class CustomAPIConversationEngineConfig(BaseModel):
    type: Literal["custom-api"] = "custom-api"

    model_config = ConfigDict(title="CustomAPICEConfig")


ConversationEngineConfig = (
    OpenAIRealtimeAPIConversationEngineConfig
    | CartesiaConversationEngineConfig
    | CodexRealtimeConversationEngineConfig
    | CustomAPIConversationEngineConfig
)
