# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Any, Literal, Union

from pydantic import BaseModel


class AvatarStartedSpeaking(BaseModel):
    type: Literal["avatar_started_speaking"] = "avatar_started_speaking"
    phrase_id: str


class AvatarStartedSpeakingPhrase(BaseModel):
    type: Literal["avatar_started_speaking_phrase"] = "avatar_started_speaking_phrase"
    phrase_id: str


class AvatarEndedSpeakingPhrase(BaseModel):
    type: Literal["avatar_ended_speaking_phrase"] = "avatar_ended_speaking_phrase"
    phrase_id: str


class AvatarEndedSpeaking(BaseModel):
    type: Literal["avatar_ended_speaking"] = "avatar_ended_speaking"


class ConversationEngineSendMessage(BaseModel):
    type: Literal["conversation_engine.external.message.send"] = (
        "conversation_engine.external.message.send"
    )
    data: dict[str, Any]


class ConversationEngineReceiveMessage(BaseModel):
    type: Literal["conversation_engine.external.message.received"] = (
        "conversation_engine.external.message.received"
    )
    data: dict[str, Any]


ConversationEngineMessage = Union[ConversationEngineSendMessage, ConversationEngineReceiveMessage]

AvatarEvent = Union[
    AvatarStartedSpeaking,
    AvatarStartedSpeakingPhrase,
    AvatarEndedSpeakingPhrase,
    AvatarEndedSpeaking,
]


class LifecycleError(BaseModel):
    type: Literal["session_lifecycle_error"] = "session_lifecycle_error"
    code: Literal["internal_error", "limits_exceeded", "openai-realtime-version-mismatch"]
    message: str
    session_id: str


class SDKErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    error: LifecycleError
