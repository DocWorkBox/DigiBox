# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Adapter for MiniMax's historical native Realtime WebSocket API.

The endpoint is not listed in MiniMax's current public API directory. Accounts
must still have explicit Realtime permission; callers should surface that fact
when authentication or the WebSocket upgrade is rejected.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Protocol
from urllib.parse import urlencode

from websockets.asyncio.client import connect as websockets_connect

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.builders import (
    CustomAPIConnectionConfig,
    MiniMaxProvider,
)
from avaturn_live_streamer.conversation_engines.custom_api_client import (
    EnergyTurnDetector,
    TTSBackend,
)
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SegmentPlaybackCancelled,
    SegmentPlaybackCompleted,
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
    TextEchoEnqueueText,
    UserSpeechReceived,
)
from avaturn_live_streamer.management.types import SegmentId, make_segment_id
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.datetime import tzutcnow

_REALTIME_URL = "wss://api.minimaxi.com/ws/v1/realtime"
_SAMPLE_RATE = 24_000


class MiniMaxRealtimeError(RuntimeError):
    pass


class _WebSocket(Protocol):
    async def recv(self) -> str | bytes: ...

    async def send(self, message: str) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self) -> None: ...


type _Connector = Callable[..., Awaitable[_WebSocket]]


async def _websocket_connect(uri: str, **kwargs: object) -> _WebSocket:
    return await websockets_connect(uri, **kwargs)  # type: ignore[arg-type, return-value]


def _parse_event(raw: str | bytes) -> dict[str, object]:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiniMaxRealtimeError("MiniMax Realtime returned invalid JSON") from exc
    if not isinstance(event, dict):
        raise MiniMaxRealtimeError("MiniMax Realtime event must be an object")
    return event


def _error_message(event: dict[str, object]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("msg")
        code = error.get("code") or error.get("status_code")
    else:
        message = event.get("message") or event.get("msg")
        code = event.get("code") or event.get("status_code")
    if isinstance(message, str) and message:
        return f"{code}: {message}" if code is not None else message
    return f"status {code}" if code is not None else "unknown error"


class MiniMaxRealtimeClient:
    def __init__(
        self,
        provider: MiniMaxProvider,
        *,
        prompt: str,
        connector: _Connector | None = None,
        text_only: bool = False,
    ) -> None:
        self.provider = provider
        self.prompt = prompt
        self.text_only = text_only
        self._connector = connector
        self._websocket: _WebSocket | None = None
        self._send_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        if self._websocket is not None:
            return
        url = f"{_REALTIME_URL}?{urlencode({'model': self.provider.realtime_model})}"
        connector = self._connector or _websocket_connect
        try:
            websocket = await connector(
                url,
                additional_headers={
                    "Authorization": f"Bearer {self.provider.api_key.get_secret_value()}"
                },
                open_timeout=self.provider.timeout_seconds,
            )
            self._closed = False
            self._websocket = websocket
            await self._wait_for("session.created")
            session: dict[str, object] = {
                "modalities": ["text"] if self.text_only else ["text", "audio"],
                "instructions": self.prompt,
                "input_audio_format": "pcm16",
                "max_response_output_tokens": 1024,
            }
            if not self.text_only:
                session.update(
                    {
                        "voice": self.provider.voice,
                        "output_audio_format": "pcm16",
                    }
                )
            await self._send({"type": "session.update", "session": session})
            await self._wait_for("session.updated")
        except BaseException as exc:
            await self.close()
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            raise MiniMaxRealtimeError(
                "MiniMax 历史 Realtime 鉴权或账号权限失败；"
                f"请确认 API Key 与 Realtime 权限: {self._sanitise(exc)}"
            ) from exc

    def _sanitise(self, value: object) -> str:
        message = str(value)
        secret = self.provider.api_key.get_secret_value()
        if secret:
            message = message.replace(secret, "***")
        return message[:1000]

    async def _wait_for(self, expected_type: str) -> dict[str, object]:
        websocket = self._require_websocket()
        async with asyncio.timeout(self.provider.timeout_seconds):
            while True:
                event = _parse_event(await websocket.recv())
                event_type = event.get("type")
                if event_type == "error":
                    raise MiniMaxRealtimeError(
                        f"MiniMax Realtime error: {_error_message(event)}"
                    )
                if event_type == expected_type:
                    return event

    def _require_websocket(self) -> _WebSocket:
        if self._websocket is None:
            raise MiniMaxRealtimeError("MiniMax Realtime session is not connected")
        return self._websocket

    async def _send(self, message: dict[str, object]) -> None:
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        async with self._send_lock:
            await self._require_websocket().send(encoded)

    async def append_audio(self, audio: SpeechBuffer) -> None:
        pcm = audio.resample(_SAMPLE_RATE).to_bytes()
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create", "response": {}})

    async def append_text(self, text: str) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._send({"type": "response.create", "response": {}})

    async def notifications(self) -> AsyncIterator[dict[str, object]]:
        websocket = self._require_websocket()
        async for raw in websocket:
            event = _parse_event(raw)
            if event.get("type") == "error":
                raise MiniMaxRealtimeError(
                    "MiniMax 历史 Realtime 运行错误: "
                    f"{self._sanitise(_error_message(event))}"
                )
            yield event

    async def close(self) -> None:
        websocket = self._websocket
        if websocket is None or self._closed:
            return
        self._closed = True
        self._websocket = None
        with suppress(Exception):
            await websocket.close()


class MiniMaxRealtimeConversationEngine:
    def __init__(
        self,
        *,
        client: MiniMaxRealtimeClient,
        config: CustomAPIConnectionConfig,
        tts: TTSBackend | None = None,
    ) -> None:
        if not isinstance(config.provider, MiniMaxProvider):
            raise TypeError("MiniMax engine requires a minimax provider")
        self.client = client
        self.config = config
        self._tts = tts
        self._vad = EnergyTurnDetector(config.vad)
        self._active_segment: SegmentId | None = None
        self._tts_task: asyncio.Task[None] | None = None
        self._recent_response_ids: deque[str] = deque(maxlen=64)
        self._last_unidentified_response_text: str | None = None
        self._closed = False
        self._playback_segments: set[SegmentId] = set()

    async def _open_segment(self, bus: EventBus) -> SegmentId:
        segment_id = make_segment_id()
        self._active_segment = segment_id
        await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
        return segment_id

    async def _finish_segment(self, bus: EventBus) -> None:
        segment_id = self._active_segment
        if segment_id is None:
            return
        self._active_segment = None
        await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))

    async def _cancel_tts(self) -> None:
        task = self._tts_task
        self._tts_task = None
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _interrupt(self, bus: EventBus) -> None:
        await self._cancel_tts()
        await self._finish_segment(bus)
        await bus.publish(DiscardAvatarSpeechBuffer())

    async def _speak_local(self, bus: EventBus, text: str) -> None:
        tts = self._tts
        if tts is None:
            return
        try:
            async for audio in tts.stream_speech(text):
                segment_id = self._active_segment or await self._open_segment(bus)
                await bus.publish(
                    SegmentChunkGenerated(segment_id=segment_id, buffer=audio)
                )
        finally:
            await self._finish_segment(bus)

    def _start_local_tts(self, bus: EventBus, text: str) -> None:
        async def run() -> None:
            try:
                await self._speak_local(bus, text)
            finally:
                if self._tts_task is asyncio.current_task():
                    self._tts_task = None

        self._tts_task = asyncio.create_task(run(), name="MiniMaxRealtime.local_tts")

    @staticmethod
    def _response_text(event: dict[str, object]) -> str | None:
        event_type = event.get("type")
        field = "transcript" if event_type == "response.audio_transcript.done" else "text"
        value = event.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    def _is_duplicate_local_response(
        self,
        event: dict[str, object],
        text: str,
    ) -> bool:
        response_id = event.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            response = event.get("response")
            response_id = response.get("id") if isinstance(response, dict) else None
        if isinstance(response_id, str) and response_id:
            if response_id in self._recent_response_ids:
                return True
            self._recent_response_ids.append(response_id)
            return False
        if text == self._last_unidentified_response_text:
            return True
        self._last_unidentified_response_text = text
        return False

    def _begin_user_turn(self) -> None:
        # Events without response ids are deduplicated only within one assistant
        # turn so a later response may legitimately repeat the same sentence.
        self._last_unidentified_response_text = None

    async def _notification_loop(self, bus: EventBus) -> None:
        bus.ready()
        async for event in self.client.notifications():
            event_type = event.get("type")
            if event_type == "response.audio.delta" and self._tts is None:
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                try:
                    pcm = base64.b64decode(delta, validate=True)
                except ValueError as exc:
                    raise MiniMaxRealtimeError(
                        "MiniMax Realtime returned invalid audio Base64"
                    ) from exc
                if len(pcm) % 2:
                    raise MiniMaxRealtimeError(
                        "MiniMax Realtime returned an unaligned PCM16 audio chunk"
                    )
                segment_id = self._active_segment or await self._open_segment(bus)
                await bus.publish(
                    SegmentChunkGenerated(
                        segment_id=segment_id,
                        buffer=SpeechBuffer.from_bytes(pcm, _SAMPLE_RATE),
                    )
                )
            elif event_type == "response.audio.done" and self._tts is None:
                await self._finish_segment(bus)
            elif self._tts is not None and event_type in {
                "response.text.done",
                "response.output_text.done",
                "response.audio_transcript.done",
            }:
                transcript = self._response_text(event)
                if (
                    transcript is not None
                    and not self._is_duplicate_local_response(event, transcript)
                ):
                    await self._cancel_tts()
                    await self._finish_segment(bus)
                    await bus.publish(
                        ResponseTranscript(
                            transcript=transcript,
                            timestamp=tzutcnow().timestamp(),
                        )
                    )
                    self._start_local_tts(bus.clone(), transcript)
            elif event_type == "response.audio_transcript.done":
                transcript = event.get("transcript")
                if isinstance(transcript, str) and transcript.strip():
                    await bus.publish(
                        ResponseTranscript(
                            transcript=transcript.strip(),
                            timestamp=tzutcnow().timestamp(),
                        )
                    )
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript")
                if isinstance(transcript, str) and transcript.strip():
                    await bus.publish(
                        InputTranscript(
                            transcript=transcript.strip(),
                            timestamp=tzutcnow().timestamp(),
                        )
                    )
        await bus.publish(Shutdown(reason="agent_left"))

    async def _bus_loop(self, bus: EventBus) -> None:
        async with bus.subscribe(
            UserSpeechReceived,
            TextEchoEnqueueText,
            SegmentPlaybackStarted,
            SegmentPlaybackCompleted,
            SegmentPlaybackInterrupted,
            SegmentPlaybackCancelled,
            Shutdown,
        ) as sub:
            bus.ready()
            async for event in sub:
                match event:
                    case UserSpeechReceived(buffer=buffer):
                        was_confirmed = self._vad.is_speech_confirmed
                        utterance = self._vad.feed(buffer)
                        if not was_confirmed and self._vad.is_speech_confirmed:
                            self._begin_user_turn()
                            await self._interrupt(bus)
                        if utterance is not None:
                            await self.client.append_audio(utterance)
                    case TextEchoEnqueueText(text=text):
                        self._begin_user_turn()
                        await self._interrupt(bus)
                        if text.strip():
                            await self.client.append_text(text.strip())
                    case SegmentPlaybackStarted(segment_id=segment_id):
                        self._playback_segments.add(segment_id)
                        self._vad.set_assistant_playback(True)
                    case (
                        SegmentPlaybackCompleted(segment_id=segment_id)
                        | SegmentPlaybackInterrupted(segment_id=segment_id)
                        | SegmentPlaybackCancelled(segment_id=segment_id)
                    ):
                        self._playback_segments.discard(segment_id)
                        self._vad.set_assistant_playback(bool(self._playback_segments))
                    case Shutdown():
                        await self._cancel_tts()
                        await self._finish_segment(bus)
                        await self.close()
                        return

    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        _ = clocks
        try:
            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(
                    self._notification_loop(bus.clone()),
                    name="MiniMaxRealtime.notifications",
                )
                task_group.create_task(
                    self._bus_loop(bus),
                    name="MiniMaxRealtime.bus",
                )
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_tts()
        await self.client.close()
        if self._tts is not None:
            await self._tts.close()

    async def __call__(self, bus: EventBus, clocks: StreamClocks) -> None:
        await self.run(bus, clocks)
