# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Incremental text-to-local-TTS bridge shared by realtime engines."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol

from avaturn_live_streamer import constant
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TurnLatencyMilestone,
    TurnLatencyPhase,
)
from avaturn_live_streamer.management.types import SegmentId, make_segment_id
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer

_LOGGER = get_logger()
_QUEUE_END = object()
_QUEUE_CAPACITY = 4
_HARD_SENTENCE_BOUNDARIES = frozenset(".!?;\n\u3002\uff01\uff1f\uff1b")
_SOFT_SENTENCE_BOUNDARIES = frozenset(",:\u3001\uff0c\uff1a")


class StreamingTTSBackend(Protocol):
    def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]: ...

    async def close(self) -> None: ...


class IncrementalTextTTSTurn(Protocol):
    async def send_text(self, text: str) -> None: ...

    async def finish_text(self) -> None: ...

    def stream_audio(self) -> AsyncIterator[SpeechBuffer]: ...

    async def cancel(self) -> None: ...


class IncrementalTextTTSBackend(Protocol):
    async def open_text_stream(self) -> IncrementalTextTTSTurn: ...


class SentenceChunker:
    """Turn model deltas into speakable pieces without waiting for full output."""

    def __init__(
        self,
        max_chars: int = 24,
        soft_min_chars: int = 12,
        *,
        first_max_chars: int | None = None,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if not 1 <= soft_min_chars <= max_chars:
            raise ValueError("soft_min_chars must be between 1 and max_chars")
        if first_max_chars is not None and not 1 <= first_max_chars <= max_chars:
            raise ValueError("first_max_chars must be between 1 and max_chars")
        self._pending = ""
        self._max_chars = max_chars
        self._soft_min_chars = soft_min_chars
        self._first_max_chars = first_max_chars
        self._emitted = False

    @property
    def has_pending(self) -> bool:
        return bool(self._pending.strip())

    @staticmethod
    def _can_split_at(text: str, boundary: int) -> bool:
        if boundary >= len(text):
            return True
        before = text[boundary - 1]
        after = text[boundary]
        if before.isspace() or after.isspace():
            return True
        # CJK text has no word separators, so a character boundary is safe.
        return not (before.isascii() and after.isascii() and before.isalnum() and after.isalnum())

    def feed(self, delta: str) -> list[str]:
        self._pending += delta
        ready: list[str] = []
        while self._pending:
            current_max = (
                self._first_max_chars
                if not self._emitted and self._first_max_chars is not None
                else self._max_chars
            )
            visible = self._pending[: self._max_chars]
            candidates = [
                index + 1
                for index, char in enumerate(visible)
                if char in _HARD_SENTENCE_BOUNDARIES
                or (
                    char in _SOFT_SENTENCE_BOUNDARIES
                    and index + 1 >= self._soft_min_chars
                )
            ]
            if candidates:
                within_target = [value for value in candidates if value <= current_max]
                boundary = min(within_target or candidates)
            elif len(self._pending) >= current_max and self._can_split_at(
                self._pending,
                current_max,
            ):
                boundary = current_max
            else:
                break
            value = self._pending[:boundary].strip()
            self._pending = self._pending[boundary:]
            if value:
                ready.append(value)
                self._emitted = True
        return ready

    def flush_pending(self) -> list[str]:
        value = self._pending.strip()
        self._pending = ""
        if not value:
            return []
        self._emitted = True
        return [value]

    def finish(self) -> list[str]:
        return self.flush_pending()


@dataclass(slots=True)
class _ResponseState:
    response_id: str
    bus: EventBus
    chunker: SentenceChunker = field(
        default_factory=lambda: SentenceChunker(
            first_max_chars=8,
            max_chars=24,
            soft_min_chars=4,
        )
    )
    queue: asyncio.Queue[str | object] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_QUEUE_CAPACITY)
    )
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    enqueue_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None
    segment_id: SegmentId | None = None
    received_text: bool = False
    finishing: bool = False
    flush_task: asyncio.Task[None] | None = None
    milestones: set[TurnLatencyPhase] = field(default_factory=set)


class StreamingLocalTTSBridge:
    """Stream sentence-sized model text pieces through one local avatar segment."""

    def __init__(
        self,
        backend: StreamingTTSBackend,
        *,
        task_name: str = "Realtime.local_tts",
        idle_flush_seconds: float = 0.12,
    ) -> None:
        if idle_flush_seconds <= 0:
            raise ValueError("idle_flush_seconds must be positive")
        self._backend = backend
        self._task_name = task_name
        self._idle_flush_seconds = idle_flush_seconds
        self._state: _ResponseState | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self._backend_closed = False

    @property
    def is_active(self) -> bool:
        """Whether a response still owns the local streaming pipeline."""

        return self._state is not None

    async def _open_segment(self, state: _ResponseState) -> SegmentId:
        segment_id = make_segment_id()
        state.segment_id = segment_id
        await state.bus.publish(
            SegmentGenerationStarted(
                segment_id=segment_id,
                metadata={"turn_id": state.response_id},
            )
        )
        return segment_id

    @staticmethod
    async def _publish_milestone(
        state: _ResponseState,
        phase: TurnLatencyPhase,
    ) -> None:
        if phase in state.milestones:
            return
        state.milestones.add(phase)
        await state.bus.publish(
            TurnLatencyMilestone(
                turn_id=state.response_id,
                phase=phase,
                at_monotonic=time.perf_counter(),
            )
        )

    @staticmethod
    async def _close_segment(state: _ResponseState) -> None:
        segment_id = state.segment_id
        if segment_id is None:
            return
        state.segment_id = None
        await state.bus.publish(SegmentGenerationCompleted(segment_id=segment_id))

    @staticmethod
    def _drain_queue(state: _ResponseState) -> None:
        while True:
            try:
                state.queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    @staticmethod
    async def _put_or_cancel(state: _ResponseState, value: str | object) -> bool:
        if state.cancelled.is_set():
            return False
        put_task = asyncio.create_task(state.queue.put(value))
        cancelled_task = asyncio.create_task(state.cancelled.wait())
        try:
            done, _ = await asyncio.wait(
                (put_task, cancelled_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancelled_task in done:
                return False
            await put_task
            return not state.cancelled.is_set()
        finally:
            for task in (put_task, cancelled_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put_task, cancelled_task, return_exceptions=True)

    def _start_response(self, bus: EventBus, response_id: str) -> _ResponseState:
        state = _ResponseState(response_id=response_id, bus=bus)
        self._state = state
        self._idle.clear()
        state.task = asyncio.create_task(
            self._run_response(state),
            name=self._task_name,
        )
        return state

    async def _publish_audio(
        self,
        state: _ResponseState,
        audio: SpeechBuffer,
    ) -> None:
        if self._state is not state:
            return
        await self._publish_milestone(state, "tts_first_audio")
        segment_id = state.segment_id or await self._open_segment(state)
        await state.bus.publish(
            SegmentChunkGenerated(
                segment_id=segment_id,
                buffer=audio.resample(constant.NATIVE_SPEECH_SAMPLE_RATE),
            )
        )

    async def _run_incremental_response(
        self,
        state: _ResponseState,
        open_text_stream,
    ) -> None:
        turn: IncrementalTextTTSTurn = await open_text_stream()
        completed = False

        async def send_text() -> None:
            while True:
                value = await state.queue.get()
                if value is _QUEUE_END:
                    await turn.finish_text()
                    return
                assert isinstance(value, str)
                await turn.send_text(value)

        async def receive_audio() -> None:
            async for audio in turn.stream_audio():
                await self._publish_audio(state, audio)

        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(send_text())
                tasks.create_task(receive_audio())
            completed = True
        finally:
            if not completed:
                with suppress(Exception):
                    await turn.cancel()

    async def _flush_after_idle(self, state: _ResponseState) -> None:
        try:
            await asyncio.sleep(self._idle_flush_seconds)
            async with state.enqueue_lock:
                if (
                    self._state is not state
                    or state.cancelled.is_set()
                    or state.finishing
                ):
                    return
                pieces = state.chunker.flush_pending()
                if pieces:
                    await self._publish_milestone(state, "first_speakable_text")
                for piece in pieces:
                    if not await self._put_or_cancel(state, piece):
                        return
        except asyncio.CancelledError:
            raise
        finally:
            if state.flush_task is asyncio.current_task():
                state.flush_task = None

    def _schedule_idle_flush(self, state: _ResponseState) -> None:
        flush_task = state.flush_task
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
        if not state.chunker.has_pending:
            state.flush_task = None
            return
        state.flush_task = asyncio.create_task(
            self._flush_after_idle(state),
            name=f"{self._task_name}.idle_flush",
        )

    @staticmethod
    async def _cancel_idle_flush(state: _ResponseState) -> None:
        task = state.flush_task
        state.flush_task = None
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _run_response(self, state: _ResponseState) -> None:
        failure: BaseException | None = None
        try:
            open_text_stream = getattr(self._backend, "open_text_stream", None)
            if callable(open_text_stream):
                await self._run_incremental_response(state, open_text_stream)
            else:
                while True:
                    value = await state.queue.get()
                    if value is _QUEUE_END:
                        break
                    assert isinstance(value, str)
                    async for audio in self._backend.stream_speech(value):
                        await self._publish_audio(state, audio)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            failure = exc
        finally:
            state.cancelled.set()
            await self._cancel_idle_flush(state)
            async with state.enqueue_lock:
                self._drain_queue(state)
            if self._state is state:
                self._state = None
                await self._close_segment(state)
                self._idle.set()
        if failure is not None:
            _LOGGER.error("local streaming TTS failed", error=str(failure))
            await self._publish_milestone(state, "failed")
            await state.bus.publish(DiscardAvatarSpeechBuffer())
            await state.bus.publish(Shutdown(reason="conversation_engine_failed"))

    async def _cancel_state(self, bus: EventBus | None, *, discard: bool) -> None:
        state = self._state
        self._state = None
        if state is not None:
            state.cancelled.set()
            await self._cancel_idle_flush(state)
            task = state.task
            if task is not None and task is not asyncio.current_task():
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if bus is not None:
                # Use the caller's bus so cancellation remains ordered with the
                # speech-started/control event that triggered it.
                state.bus = bus
            await self._close_segment(state)
            async with state.enqueue_lock:
                self._drain_queue(state)
        self._idle.set()
        if discard and bus is not None:
            await bus.publish(DiscardAvatarSpeechBuffer())

    async def feed_text(self, bus: EventBus, response_id: str, delta: str) -> None:
        if self._closed:
            raise RuntimeError("local TTS bridge is closed")
        if not delta:
            return
        state = self._state
        if state is None or state.response_id != response_id:
            if state is not None:
                await self._cancel_state(bus, discard=False)
            state = self._start_response(bus, response_id)
        async with state.enqueue_lock:
            if (
                state.cancelled.is_set()
                or self._state is not state
                or state.finishing
            ):
                return
            state.received_text = True
            if delta.strip():
                await self._publish_milestone(state, "llm_first_token")
            pieces = state.chunker.feed(delta)
            if pieces:
                await self._publish_milestone(state, "first_speakable_text")
            for piece in pieces:
                if not await self._put_or_cancel(state, piece):
                    return
            self._schedule_idle_flush(state)

    async def finish_text(
        self,
        bus: EventBus,
        response_id: str,
        *,
        fallback_text: str | None = None,
    ) -> None:
        if self._closed:
            return
        state = self._state
        if state is None or state.response_id != response_id:
            if state is not None:
                await self._cancel_state(bus, discard=False)
            state = self._start_response(bus, response_id)
        await self._cancel_idle_flush(state)
        async with state.enqueue_lock:
            if state.cancelled.is_set() or self._state is not state or state.finishing:
                return
            pieces: list[str] = []
            if not state.received_text and fallback_text:
                state.received_text = True
                pieces.extend(state.chunker.feed(fallback_text))
            pieces.extend(state.chunker.finish())
            state.finishing = True
            if pieces:
                await self._publish_milestone(state, "first_speakable_text")
            for piece in pieces:
                if not await self._put_or_cancel(state, piece):
                    return
            await self._put_or_cancel(state, _QUEUE_END)

    async def interrupt(self, bus: EventBus) -> None:
        await self._cancel_state(bus, discard=True)

    async def wait_idle(self, *, timeout: float | None = None) -> None:
        waiter = self._idle.wait()
        if timeout is None:
            await waiter
        else:
            await asyncio.wait_for(waiter, timeout=timeout)

    async def close(self, bus: EventBus | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_state(bus, discard=False)
        if not self._backend_closed:
            self._backend_closed = True
            await self._backend.close()
