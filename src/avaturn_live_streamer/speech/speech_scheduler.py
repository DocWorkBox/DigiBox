# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
from collections import deque
from fractions import Fraction

from attr import dataclass

from avaturn_live_streamer.core.logs import get_logger

from .duration import Duration
from .speech_buffer import SpeechBuffer

_LOGGER = get_logger()


@dataclass(slots=True, frozen=True)
class SegmentStarted:
    id: str


@dataclass(slots=True, frozen=True)
class SegmentEnded:
    id: str


@dataclass(slots=True, frozen=True)
class SegmentInterrupted:
    id: str
    played_duration: Fraction


@dataclass(slots=True, frozen=True)
class SegmentCancelled:
    id: str


SegmentEvent = SegmentStarted | SegmentEnded | SegmentInterrupted | SegmentCancelled


@dataclass(slots=True, frozen=True)
class TimestampedEvent:
    event: SegmentEvent
    timestamp: Fraction


class SpeechScheduler:
    @dataclass(slots=True, frozen=True)
    class _StepResult:
        present: SpeechBuffer
        future: SpeechBuffer
        events: list[TimestampedEvent]

    def __init__(
        self,
        sample_rate: int,
        present_duration: Fraction,
        future_duration: Fraction,
        startup_coalesce_timeout: float = 0.1,
    ):
        if startup_coalesce_timeout < 0:
            raise ValueError("startup_coalesce_timeout must be non-negative")
        if startup_coalesce_timeout > 0.12:
            raise ValueError("startup_coalesce_timeout must be at most 0.12 seconds")
        self.sample_rate = sample_rate
        self.present_duration = present_duration
        self.future_duration = future_duration
        self.startup_coalesce_timeout = startup_coalesce_timeout

        self._write_active_segment: str | None = None
        self._queue = deque[SegmentEvent | SpeechBuffer]()

        self._max_padding_buffer = SpeechBuffer.silence(
            future_duration + present_duration, sample_rate
        )
        self._buffer = SpeechBuffer.empty()

        self._event_queue = deque[TimestampedEvent]()
        self._next_present_timestamp = Fraction(denominator=sample_rate)
        self._active_segment: str | None = None
        self._active_segment_start: Fraction = Fraction()
        self._coalescing_segment: str | None = None
        self._cancelled_coalescing_segment: str | None = None

        self._new_arrival = asyncio.Condition()

        self._autoclosed_segments: set[str] = set[str]()
        self._stop: bool = False

    @property
    def _next_event_timestamp(self) -> Fraction:
        return self._next_present_timestamp + self._buffer.duration

    async def interrupt(self):
        self._buffer = self._buffer.slice(
            0, self.future_duration
        )  # Drop overage as it will not be played back

        new_queue = deque[SpeechBuffer | SegmentEvent]()
        coalescing_segment = self._coalescing_segment
        pending_scheduled_start = (
            self._active_segment
            if self._active_segment is not None
            and any(
                isinstance(item.event, SegmentStarted) and item.event.id == self._active_segment
                for item in self._event_queue
            )
            else None
        )

        if coalescing_segment is not None:
            self._cancelled_coalescing_segment = coalescing_segment
            self._active_segment = None
            new_queue.append(SegmentCancelled(coalescing_segment))
        elif pending_scheduled_start is not None:
            self._event_queue = deque(
                item
                for item in self._event_queue
                if not (
                    isinstance(item.event, SegmentStarted)
                    and item.event.id == pending_scheduled_start
                )
            )
            if not self._buffer.is_empty:
                self._buffer = SpeechBuffer.silence(
                    self._buffer.duration,
                    self.sample_rate,
                )
            self._active_segment = None
            new_queue.append(SegmentCancelled(pending_scheduled_start))
        elif self._active_segment is not None:
            new_queue.append(SegmentEnded(self._active_segment))
            new_queue.append(
                SegmentInterrupted(
                    self._active_segment,
                    played_duration=self._next_event_timestamp - self._active_segment_start,
                )
            )
        if self._write_active_segment is not None:
            if self._write_active_segment != self._active_segment:
                self._autoclosed_segments.add(self._write_active_segment)
            self._write_active_segment = None

        for c in self._queue:
            if isinstance(c, SegmentStarted):
                new_queue.append(SegmentCancelled(c.id))

        self._queue = new_queue

        async with self._new_arrival:
            self._new_arrival.notify()

    async def stop(self) -> None:
        if self._write_active_segment is not None:
            await self.end_segment(self._write_active_segment)
        self._stop = True

    def is_stopped(self) -> bool:
        return self._stop and len(self._queue) == 0 and len(self._event_queue) == 0

    @property
    def unconsumed_duration(self) -> Duration:
        return self._buffer.duration + sum(
            [item.duration for item in self._queue if isinstance(item, SpeechBuffer)]
        )

    async def _append_queue(self, item: SegmentEvent | SpeechBuffer):
        self._queue.append(item)
        async with self._new_arrival:
            self._new_arrival.notify()

    async def append(self, chunk: SpeechBuffer, segment_id: str):
        if self._stop:
            return
        if self._write_active_segment != segment_id:
            _LOGGER.warning(
                "Current write-active segment id mismatch, write_active_segment_id=%s, segment_id=%s, chunk ignored",
                self._write_active_segment,
                segment_id,
            )
            return
        assert chunk.sample_rate == self.sample_rate, "Sample rates must match"
        await self._append_queue(chunk)

    async def start_segment(self, segment_id: str):
        if self._stop:
            return
        if self._write_active_segment is not None:
            _LOGGER.warning(
                "Double segment start, write_active_segment_id=%s, start_segment_id=%s, start ignored",
                self._write_active_segment,
                segment_id,
            )
            return
        self._write_active_segment = segment_id
        await self._append_queue(SegmentStarted(segment_id))

    async def end_segment(self, segment_id: str):
        if self._write_active_segment is None or self._write_active_segment != segment_id:
            _LOGGER.warning(
                "Unexpected segment end, write_active_segment_id=%s, end_segment_id=%s, end ignored",
                self._write_active_segment,
                segment_id,
                stack_info=True,
            )
            return
        self._write_active_segment = None
        await self._append_queue(SegmentEnded(segment_id))

    async def do_step(self) -> _StepResult:
        bufs = list[SpeechBuffer]()
        new_events = list[TimestampedEvent]()
        is_segment_start = False
        startup_deadline: float | None = None

        # Buffer can already contain enough because of last chunk size
        duration_to_consume = max(
            0, (self.present_duration + self.future_duration) - self._buffer.duration
        )
        consumed_duration = 0

        # Consume queue until we have enough duration or until we hit
        while duration_to_consume > 0:
            if is_segment_start and self._cancelled_coalescing_segment == self._coalescing_segment:
                break
            if len(self._queue) == 0:
                if is_segment_start and startup_deadline is not None:
                    remaining = startup_deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        async with self._new_arrival:
                            await asyncio.wait_for(
                                self._new_arrival.wait_for(
                                    lambda: bool(self._queue)
                                    or self._cancelled_coalescing_segment
                                    == self._coalescing_segment
                                ),
                                timeout=remaining,
                            )
                    except TimeoutError:
                        break
                    continue
                if self._active_segment is None:
                    # Stop as we can pad it to the left or to the right
                    break

                # Block and wait for new arrivals to the queue if we are midsegment
                async with self._new_arrival:
                    await self._new_arrival.wait()

            item = self._queue.popleft()
            if isinstance(item, SegmentStarted):
                timestamp = self._next_event_timestamp + consumed_duration
                new_events.append(TimestampedEvent(item, timestamp))
                self._active_segment = item.id
                is_segment_start = True
                self._coalescing_segment = item.id
                startup_deadline = asyncio.get_running_loop().time() + self.startup_coalesce_timeout
                self._active_segment_start = timestamp
            elif isinstance(item, SegmentEnded):
                new_events.append(
                    TimestampedEvent(item, self._next_event_timestamp + consumed_duration)
                )
                self._active_segment = None
                # We don't want to end one segment and start another in the same step because we
                # will not be able to pad left or right if there is not enough data
                break
            elif isinstance(item, SpeechBuffer):
                bufs.append(item)
                duration_to_consume -= item.duration
                consumed_duration += item.duration
            else:
                self._event_queue.append(
                    TimestampedEvent(item, self._next_event_timestamp + consumed_duration)
                )

        # Now we are in one of three situations:
        # 1. We have too little duration in started segment , we need to add silence either on the left if it's a segment start or on the right if it's a segment end
        # 2. We have too much duration because the last chunk was too long (it's ok, we keep it in present+future buffer)
        # 3. We have the exact duration we need
        # PS If duration is too little but midsegment -- we already blocked and waited above

        startup_cancelled = (
            is_segment_start
            and self._coalescing_segment is not None
            and self._cancelled_coalescing_segment == self._coalescing_segment
        )
        if startup_cancelled:
            bufs.clear()
            new_events.clear()
            duration_to_consume = max(
                0,
                (self.present_duration + self.future_duration) - self._buffer.duration,
            )
            pad = self._max_padding_buffer.slice(0, duration_to_consume)
            pad_left, pad_right = [], [pad]
        elif duration_to_consume > 0:
            pad = self._max_padding_buffer.slice(0, duration_to_consume)
            if self._active_segment is None:
                # We either don't have current active segment and pad it right
                pad_left, pad_right = [], [pad]
            elif is_segment_start:
                # Or we just started segment and didn't have enough of it so we pad it left
                pad_left, pad_right = [pad], []

                # We need to shift event timestamps right
                new_events = [
                    TimestampedEvent(i.event, i.timestamp + pad.duration) for i in new_events
                ]
            else:
                # In other cases we
                raise Exception("Should never get here")
        else:  # duration_to_consume == 0 or duration_to_consume < 0
            # If it's negative -- we keep overage in self._buffer, but we don't send it as future
            pad_left, pad_right = [], []

        self._event_queue.extend(new_events)
        self._buffer = SpeechBuffer.concat([self._buffer, *pad_left, *bufs, *pad_right])

        self._next_present_timestamp += self.present_duration
        present, self._buffer = self._buffer.split_in_two_parts_with_duration(self.present_duration)
        future, _ = self._buffer.split_in_two_parts_with_duration(self.future_duration)

        actual_events: list[TimestampedEvent] = []
        while len(self._event_queue) > 0:
            if self._event_queue[0].timestamp >= self._next_present_timestamp:
                break
            actual_events.append(self._event_queue.popleft())
        if is_segment_start:
            self._coalescing_segment = None
            self._cancelled_coalescing_segment = None
        return SpeechScheduler._StepResult(present, future, actual_events)
