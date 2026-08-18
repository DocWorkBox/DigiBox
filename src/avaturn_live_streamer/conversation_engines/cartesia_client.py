# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Cartesia Line "Calls API" client.

Protocol: https://docs.cartesia.ai/line/integrations/calls-api

Cartesia is a managed runtime — LLM agent, tools and behavior are deployed via
`cartesia deploy`. The WebSocket is a thin audio transport to an `agent_id`.

Segmentation
    Cartesia does not send turn boundaries. A segment opens lazily on the first
    `media_output` and closes when the buffered audio is expected to have drained
    (drain timer) or on `clear` (interruption). On `clear` we publish
    `SegmentGenerationCompleted` *before* `DiscardAvatarSpeechBuffer` so the
    playback scheduler releases the write-active segment before the next turn's
    chunks arrive.

Close codes
    1000 = graceful (e.g. `agent.end_call()`). Anything else raises
    `ConnectionError` so the runner doesn't misreport drops as success.
"""

import asyncio
import json
from base64 import b64decode, b64encode
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import attrs
import websockets
from websockets import State
from websockets.asyncio.client import ClientConnection

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer import constant
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.configs import CartesiaConversationEngineConfig
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TextEchoEnqueueText,
    UserSpeechReceived,
)
from avaturn_live_streamer.management.types import SegmentId
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.async_utils import run_in_thread
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()

_CARTESIA_URL = "wss://api.cartesia.ai/agents/stream"
_CARTESIA_VERSION = "2025-04-16"
_CARTESIA_INPUT_FORMAT = "pcm_24000"
_GRACEFUL_CLOSE_CODE = 1000

# Safety margin past the expected buffer-drain time.
DRAIN_GRACE_SEC: float = 0.06


@attrs.define
class CartesiaApiClient:
    _config: CartesiaConversationEngineConfig
    stream_id: str
    _turn_counter: int = 0
    _current_segment_id: SegmentId | None = None
    _drain_timer: asyncio.TimerHandle | None = None
    _segment_audio_seconds: float = 0.0
    _segment_start_loop_time: float = 0.0
    _media_output_count: int = 0
    _media_input_count: int = 0
    _clear_count: int = 0
    _drain_count: int = 0
    _tg: asyncio.TaskGroup = attrs.field(init=False)
    _bus: EventBus = attrs.field(init=False)

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[ClientConnection, None]:
        url = f"{_CARTESIA_URL}/{self._config.agent_id}"
        _LOGGER.info(
            "Connecting to Cartesia agent",
            agent_id=self._config.agent_id,
            stream_id=self.stream_id,
        )
        async with websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {self._config.access_token}",
                "Cartesia-Version": _CARTESIA_VERSION,
            },
            ping_interval=60,
            ping_timeout=20,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "stream_id": self.stream_id,
                        "config": {"input_format": _CARTESIA_INPUT_FORMAT},
                    }
                )
            )
            yield ws

    async def _send_speech(self, ws: ClientConnection, buffer: SpeechBuffer) -> None:
        msg = await run_in_thread(self._encode_media_input, buffer)
        try:
            await ws.send(msg)
        except websockets.exceptions.ConnectionClosed:
            _LOGGER.debug("Cartesia WS closed during media_input send, ignoring")
            return
        self._media_input_count += 1

    def _encode_media_input(self, buffer: SpeechBuffer) -> str:
        return json.dumps(
            {
                "event": "media_input",
                "stream_id": self.stream_id,
                "media": {"payload": b64encode(buffer.to_bytes()).decode()},
            }
        )

    @staticmethod
    def _decode_chunk(payload_base64: str) -> SpeechBuffer:
        return SpeechBuffer.from_bytes(
            b64decode(payload_base64), constant.NATIVE_SPEECH_SAMPLE_RATE
        )

    def _cancel_drain_timer(self) -> None:
        if self._drain_timer is not None:
            self._drain_timer.cancel()
            self._drain_timer = None

    def _schedule_drain_close(self, segment_id: SegmentId) -> None:
        self._cancel_drain_timer()
        loop = asyncio.get_running_loop()
        expected_drain_at = self._segment_start_loop_time + self._segment_audio_seconds
        delay = max(0.0, expected_drain_at + DRAIN_GRACE_SEC - loop.time())
        self._drain_timer = loop.call_later(delay, self._on_drain_deadline, segment_id)

    def _on_drain_deadline(self, segment_id: SegmentId) -> None:
        self._drain_timer = None
        if self._current_segment_id != segment_id:
            return
        self._tg.create_task(self._close_drained_segment(segment_id))

    async def _close_drained_segment(self, segment_id: SegmentId) -> None:
        if self._current_segment_id != segment_id:
            return
        self._current_segment_id = None
        self._drain_count += 1
        _LOGGER.debug("Cartesia drain-timer fired, closing segment", segment_id=segment_id)
        await self._bus.publish(SegmentGenerationCompleted(segment_id=segment_id))

    async def _open_segment(self) -> SegmentId:
        self._turn_counter += 1
        segment_id = SegmentId(f"cartesia-{self.stream_id}-turn-{self._turn_counter}")
        self._current_segment_id = segment_id
        self._segment_audio_seconds = 0.0
        self._segment_start_loop_time = asyncio.get_running_loop().time()
        await self._bus.publish(SegmentGenerationStarted(segment_id=segment_id))
        return segment_id

    async def _close_segment(self) -> None:
        if self._current_segment_id is None:
            return
        await self._bus.publish(SegmentGenerationCompleted(segment_id=self._current_segment_id))
        self._current_segment_id = None

    async def _listener(self, bus: EventBus, ws: ClientConnection) -> None:
        bus.ready()
        try:
            async for data in ws:
                if isinstance(data, bytes):
                    _LOGGER.warning("Unexpected binary frame from Cartesia, ignoring")
                    continue
                msg = await run_in_thread(json.loads, data)
                await self._dispatch_server_event(msg)
        finally:
            self._cancel_drain_timer()
            _LOGGER.info(
                "Cartesia WebSocket closed",
                stream_id=self.stream_id,
                close_code=ws.close_code,
                close_reason=ws.close_reason,
                media_output_count=self._media_output_count,
                media_input_count=self._media_input_count,
                clear_count=self._clear_count,
                drain_count=self._drain_count,
                turn_count=self._turn_counter,
            )

    async def _dispatch_server_event(self, msg: dict[str, Any]) -> None:
        match msg.get("event"):
            case "ack":
                _LOGGER.info("Cartesia ack", config=msg.get("config"))
            case "media_output":
                chunk = await run_in_thread(self._decode_chunk, msg["media"]["payload"])
                self._media_output_count += 1
                segment_id = self._current_segment_id or await self._open_segment()
                self._segment_audio_seconds += float(chunk.duration)
                await self._bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=chunk))
                self._schedule_drain_close(segment_id)
            case "clear":
                self._clear_count += 1
                _LOGGER.info("Cartesia clear (interruption)")
                self._cancel_drain_timer()
                # Completed must precede Discard so the scheduler releases the segment.
                await self._close_segment()
                await self._bus.publish(DiscardAvatarSpeechBuffer())
            case "transfer_call":
                _LOGGER.warning(
                    "Cartesia transfer_call not supported, ignoring",
                    transfer=msg.get("transfer"),
                )
            case _:
                _LOGGER.debug(
                    "Unhandled Cartesia event",
                    message={k: v for k, v in msg.items() if k != "media"},
                )

    async def _handle_bus_events(self, bus: EventBus, ws: ClientConnection) -> None:
        async with bus.subscribe(UserSpeechReceived, TextEchoEnqueueText, Shutdown) as sub:
            bus.ready()
            async for event in sub:
                match event:
                    case UserSpeechReceived(buffer=buf):
                        if ws.state != State.OPEN:
                            if ws.close_code == _GRACEFUL_CLOSE_CODE:
                                return
                            raise ConnectionError(
                                f"Cartesia WS closed unexpectedly (code={ws.close_code})"
                            )
                        await self._send_speech(ws, buf)
                    case TextEchoEnqueueText():
                        _LOGGER.warning("TextEchoEnqueueText not supported by Cartesia CE")
                    case Shutdown():
                        await ws.close()
                        return

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        self._bus = bus
        async with self._connect() as ws, asyncio.TaskGroup() as tg:
            self._tg = tg
            tg.create_task(self._listener(bus.clone(), ws))
            tg.create_task(self._handle_bus_events(bus.clone(), ws))
            bus.ready()
