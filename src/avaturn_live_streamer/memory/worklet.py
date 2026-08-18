from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Protocol

from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import InputTranscript, Shutdown
from avaturn_live_streamer.memory.extractor import HeuristicMemoryExtractor
from avaturn_live_streamer.memory.models import CandidateBatch


_LOGGER = logging.getLogger(__name__)


class CandidateSink(Protocol):
    def try_submit(self, batch: CandidateBatch) -> object: ...


class MemoryWorklet:
    """Move final user transcripts off the EventBus path for local extraction."""

    def __init__(
        self,
        *,
        service: CandidateSink,
        extractor: HeuristicMemoryExtractor,
        session_id: str,
        engine_kind: str,
        queue_size: int = 64,
        drain_timeout_seconds: float = 0.25,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if drain_timeout_seconds < 0:
            raise ValueError("drain_timeout_seconds must not be negative")
        self._service = service
        self._extractor = extractor
        self._session_id = session_id
        self._engine_kind = engine_kind
        self._queue_size = queue_size
        self._drain_timeout_seconds = drain_timeout_seconds
        self._pending: asyncio.Queue[InputTranscript] | None = None
        self._dropped = 0
        self._seen_turns: dict[str, None] = {}

    @property
    def dropped_transcripts(self) -> int:
        return self._dropped

    @property
    def pending_transcripts(self) -> int:
        pending = self._pending
        return pending.qsize() if pending is not None else 0

    async def run(self, bus: EventBus, clocks: object) -> None:
        _ = clocks
        pending: asyncio.Queue[InputTranscript] = asyncio.Queue(
            maxsize=self._queue_size
        )
        self._pending = pending
        worker = asyncio.create_task(
            self._consume(pending),
            name="MemoryWorklet.extractor",
        )
        try:
            async with bus.subscribe(InputTranscript, Shutdown, buffer_size=256) as subscription:
                bus.ready()
                async for event in subscription:
                    if isinstance(event, Shutdown):
                        break
                    if not event.transcript.strip():
                        continue
                    turn_digest = self._turn_digest(event)
                    if turn_digest in self._seen_turns:
                        continue
                    try:
                        pending.put_nowait(event)
                    except asyncio.QueueFull:
                        self._dropped += 1
                        _LOGGER.warning("local memory transcript queue is full; dropping input")
                    else:
                        self._seen_turns[turn_digest] = None
                        if len(self._seen_turns) > 512:
                            oldest = next(iter(self._seen_turns))
                            self._seen_turns.pop(oldest, None)
        finally:
            try:
                await asyncio.wait_for(
                    pending.join(),
                    timeout=self._drain_timeout_seconds,
                )
            except TimeoutError:
                pass
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            self._clear_pending(pending)
            self._pending = None

    async def _consume(self, pending: asyncio.Queue[InputTranscript]) -> None:
        while True:
            event = await pending.get()
            try:
                transcript = event.transcript.strip()
                turn_digest = self._turn_digest(event)
                batch = await asyncio.to_thread(
                    self._extractor.extract_user_transcript,
                    transcript,
                    session_id=self._session_id,
                    turn_id=f"input-{turn_digest[:32]}",
                    engine_kind=self._engine_kind,
                    observed_at=datetime.fromtimestamp(event.timestamp, tz=timezone.utc),
                )
                if (
                    (batch.owner_name is not None and batch.owner_name.strip())
                    or batch.people
                    or batch.relationships
                    or batch.events
                ):
                    self._service.try_submit(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._dropped += 1
                _LOGGER.exception("local memory extraction failed; dropping input")
            finally:
                pending.task_done()

    def _turn_digest(self, event: InputTranscript) -> str:
        return hashlib.sha256(
            (
                f"{self._session_id}\0{event.timestamp:.6f}\0"
                f"{event.transcript.strip()}"
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _clear_pending(pending: asyncio.Queue[InputTranscript]) -> None:
        while True:
            try:
                pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                pending.task_done()
