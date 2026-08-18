# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Event types for stream worker event bus communication.

Split into two categories:
1. Internal events (dataclasses) - only used within EventBus, never cross transport boundary
2. Transport events (Pydantic) - cross RabbitMQ transport, need serialization and type tags
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from pydantic import BaseModel

from avaturn_live_streamer.management.types import SegmentId, SegmentMetadata
from avaturn_live_streamer.sdk_events import ConversationEngineMessage
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.types import PhraseID, PixelFormat

# =============================================================================
# INTERNAL EVENTS (dataclasses) - never cross transport boundary
# =============================================================================


@dataclass(slots=True)
class StreamStarted:
    """Published when stream is ready (first frame sent to client)."""

    pass


@dataclass(slots=True)
class StreamEnded:
    """Published when stream ends (last frame sent to client)."""

    duration_seconds: int


@dataclass(slots=True)
class Frame:
    buffer: bytes
    timestamp: float
    audio: SpeechBuffer
    pixel_format: PixelFormat = PixelFormat.YUV_I420


@dataclass(slots=True)
class VideoFrameGenerated:
    """Published when a video frame is ready to be sent to RTC."""

    frame: Frame


@dataclass(slots=True)
class UserSpeechReceived:
    """Published when user audio is received from RTC."""

    buffer: SpeechBuffer


TurnLatencyPhase = Literal[
    "vad_complete",
    "asr_complete",
    "llm_first_token",
    "first_speakable_text",
    "tts_first_audio",
    "renderer_first_frame",
    "failed",
    "interrupted",
]


@dataclass(slots=True, frozen=True)
class TurnLatencyMilestone:
    """One credential-free timing marker for a custom-API audio turn."""

    turn_id: str
    phase: TurnLatencyPhase
    at_monotonic: float
    details: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(slots=True)
class CodexAssistantAudioReceived:
    """Assistant PCM captured from the browser-owned Codex WebRTC track."""

    buffer: SpeechBuffer


@dataclass(slots=True)
class CodexAssistantControlReceived:
    """Allow-listed turn/barging control forwarded by the Codex data channel."""

    type: Literal["speech_started", "output_audio_done"]
    item_id: str | None = None


@dataclass(slots=True)
class UserSpeechStreamStart:
    """Published when user audio data becomes available (first chunk received)."""

    pass


@dataclass(slots=True)
class UserSpeechStreamEnd:
    """Published when user audio data stops (timeout - no more chunks arriving)."""

    pass


@dataclass(slots=True)
class ParticipantJoined:
    """Published when a non-avatar participant joins the RTC room."""

    participant_id: str


@dataclass(slots=True)
class ParticipantLeft:
    """Published when a non-avatar participant leaves the RTC room."""

    participant_id: str


# Generation phase events (internal, with metadata for segment tracking)
@dataclass(slots=True)
class SegmentGenerationStarted:
    """TTS started generating a speech segment. CE sets metadata here."""

    segment_id: SegmentId
    metadata: SegmentMetadata = field(default_factory=dict)


@dataclass(slots=True)
class SegmentChunkGenerated:
    """TTS produced a chunk of audio for a segment."""

    segment_id: SegmentId
    buffer: SpeechBuffer


@dataclass(slots=True)
class SegmentGenerationCompleted:
    """TTS finished generating a speech segment."""

    segment_id: SegmentId


# Delayed event wrapper
@dataclass(slots=True)
class ScheduledEvent:
    """Wrapper for events that should be emitted at a specific clock time."""

    emit_at: Fraction
    event: Any


# SDK message events (internal)
@dataclass(slots=True)
class SdkMessageToClient:
    """Published when SDK needs to send message to client via RTC."""

    data: dict


@dataclass(slots=True)
class SdkMessageFromClient:
    """Published when client sends SDK message via RTC."""

    data: dict


@dataclass(slots=True)
class RTCMessageReceived:
    """Published when RTC receives a message from client."""

    message: ConversationEngineMessage


# Internal control events
ShutdownReason = Literal[
    "termination_command_received",
    "user_absent_timeout",
    "max_duration_reached",
    "agent_left",
    "conversation_engine_failed",
]


@dataclass(slots=True)
class Shutdown:
    """Published to shut down all workers."""

    reason: ShutdownReason = "termination_command_received"


@dataclass(slots=True)
class DiscardAvatarSpeechBuffer:
    """Emitted by CEs internally to discard speech buffers."""

    pass


# =============================================================================
# TRANSPORT COMMANDS (Pydantic) - cross RabbitMQ transport as StreamControlCommand payload
# =============================================================================


class TextEchoEnqueueText(BaseModel):
    """Request to convert text to speech. TextEcho CE command."""

    type: Literal["ce_commands.text_echo.text_task_enqueued"] = (
        "ce_commands.text_echo.text_task_enqueued"
    )
    phrase_id: PhraseID
    text: str


class InterruptAvatar(BaseModel):
    """Interrupt avatar speech. Common command for all CEs."""

    type: Literal["ce_commands.common.interrupt_avatar"] = "ce_commands.common.interrupt_avatar"


# Command union types
TextEchoCECommands = TextEchoEnqueueText
CommonCECommands = InterruptAvatar
CECommands = TextEchoCECommands | CommonCECommands


# =============================================================================
# TRANSPORT EVENTS (Pydantic) - cross RabbitMQ transport as StreamControlEvent payload
# =============================================================================


class SegmentPlaybackStarted(BaseModel):
    """Avatar started speaking a segment."""

    type: Literal["avatar.speech.playback.segment_started"] = (
        "avatar.speech.playback.segment_started"
    )
    segment_id: SegmentId
    metadata: SegmentMetadata = {}


class SegmentPlaybackCompleted(BaseModel):
    """Avatar finished speaking a segment."""

    type: Literal["avatar.speech.playback.segment_completed"] = (
        "avatar.speech.playback.segment_completed"
    )
    segment_id: SegmentId
    metadata: SegmentMetadata = {}


class SegmentPlaybackInterrupted(BaseModel):
    """Segment was interrupted mid-playback."""

    type: Literal["avatar.speech.playback.segment_interrupted"] = (
        "avatar.speech.playback.segment_interrupted"
    )
    segment_id: SegmentId
    played_duration: float
    metadata: SegmentMetadata = {}


class SegmentPlaybackCancelled(BaseModel):
    """Segment was cancelled before playback started."""

    type: Literal["avatar.speech.playback.segment_cancelled"] = (
        "avatar.speech.playback.segment_cancelled"
    )
    segment_id: SegmentId
    metadata: SegmentMetadata = {}


class InputTranscript(BaseModel):
    """User speech transcription. Realtime CE event."""

    type: Literal["ce_events.realtime.input_transcript"] = "ce_events.realtime.input_transcript"
    transcript: str
    timestamp: float


class ResponseTranscript(BaseModel):
    """AI response transcription. Realtime CE event."""

    type: Literal["ce_events.realtime.response_transcript"] = (
        "ce_events.realtime.response_transcript"
    )
    transcript: str
    timestamp: float


# Event union types
PlaybackEvents = (
    SegmentPlaybackStarted
    | SegmentPlaybackCompleted
    | SegmentPlaybackInterrupted
    | SegmentPlaybackCancelled
)
TranscriptEvents = InputTranscript | ResponseTranscript
StreamEvents = PlaybackEvents | TranscriptEvents
