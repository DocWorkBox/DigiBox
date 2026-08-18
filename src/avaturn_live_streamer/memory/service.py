from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol, TypeVar

from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    PurgeReport,
    RecallQuery,
    RecallResult,
    SessionMemoryContext,
    SubmissionReason,
    SubmissionResult,
)


_T = TypeVar("_T")


class MemoryStore(Protocol):
    def initialize(self) -> None: ...

    def ingest(self, batch: CandidateBatch) -> object: ...

    def recall(self, query: RecallQuery) -> RecallResult: ...

    def build_session_profile(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> SessionMemoryContext: ...

    def claim_follow_up_before(
        self,
        event_id: str,
        *,
        expected_revision: int,
        asked_at: datetime,
        deadline_monotonic: float,
    ) -> bool: ...

    def purge_expired(self, *, now: datetime) -> PurgeReport: ...


class MemoryService:
    """Async, fail-open boundary around the synchronous local memory store."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        queue_size: int = 64,
        purge_interval_seconds: float = 24 * 60 * 60,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if purge_interval_seconds <= 0:
            raise ValueError("purge_interval_seconds must be positive")
        self._store = store
        self._queue: asyncio.Queue[CandidateBatch] = asyncio.Queue(maxsize=queue_size)
        self._writer_task: asyncio.Task[None] | None = None
        self._retention_task: asyncio.Task[None] | None = None
        self._recall_task: asyncio.Task[RecallResult] | None = None
        self._profile_task: asyncio.Task[SessionMemoryContext] | None = None
        self._detached_mutations: set[asyncio.Task[object]] = set()
        self._detached_claims: set[asyncio.Task[bool]] = set()
        self._mutation_lock = threading.Lock()
        self._purge_interval_seconds = purge_interval_seconds
        self._clear_lock = asyncio.Lock()
        self._clear_in_progress = False
        self._ingest_cutoff: datetime | None = None
        self._started = False
        self._closed = False
        self._available = False
        self._degraded_reason: str | None = None
        self._dropped_submissions = 0

    @property
    def available(self) -> bool:
        return self._available

    @property
    def degraded_reason(self) -> str | None:
        return self._degraded_reason

    @property
    def pending_writes(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_submissions(self) -> int:
        return self._dropped_submissions

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            await asyncio.to_thread(self._store.initialize)
        except Exception as exc:
            self._available = False
            self._degraded_reason = f"{type(exc).__name__}: {exc}"[:500]
            return
        retention_warning = None
        purge_expired = getattr(self._store, "purge_expired", None)
        if callable(purge_expired):
            try:
                await asyncio.to_thread(
                    purge_expired,
                    now=datetime.now(timezone.utc),
                )
            except Exception as exc:
                retention_warning = (
                    f"retention cleanup failed: {type(exc).__name__}: {exc}"
                )[:500]
        self._available = True
        self._degraded_reason = retention_warning
        self._writer_task = asyncio.create_task(
            self._writer_loop(),
            name="MemoryService.writer",
        )
        if callable(purge_expired):
            self._retention_task = asyncio.create_task(
                self._retention_loop(),
                name="MemoryService.retention",
            )

    async def close(self, *, drain_timeout_seconds: float = 2.0) -> None:
        self._closed = True
        retention = self._retention_task
        if retention is not None:
            retention.cancel()
            await asyncio.gather(retention, return_exceptions=True)
            self._retention_task = None
        writer = self._writer_task
        if writer is not None:
            try:
                await asyncio.wait_for(
                    self._queue.join(),
                    timeout=max(drain_timeout_seconds, 0),
                )
            except TimeoutError:
                pass
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            self._writer_task = None
        self._discard_queued_batches()
        for task in (self._recall_task, self._profile_task):
            if task is not None and not task.done():
                task.cancel()
        self._available = False

    async def _retention_loop(self) -> None:
        purge_expired = getattr(self._store, "purge_expired", None)
        if not callable(purge_expired):
            return
        while True:
            await asyncio.sleep(self._purge_interval_seconds)
            try:
                await asyncio.to_thread(
                    purge_expired,
                    now=datetime.now(timezone.utc),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._degraded_reason = (
                    f"retention cleanup failed: {type(exc).__name__}: {exc}"
                )[:500]

    def try_submit(self, batch: CandidateBatch) -> SubmissionResult:
        if self._closed:
            return SubmissionResult(False, SubmissionReason.CLOSED, self._queue.qsize())
        if not self._started:
            return SubmissionResult(
                False,
                SubmissionReason.NOT_STARTED,
                self._queue.qsize(),
            )
        if self._clear_in_progress:
            self._dropped_submissions += 1
            return SubmissionResult(
                False,
                "clear_in_progress",
                self._queue.qsize(),
            )
        if (
            self._ingest_cutoff is not None
            and batch.observed_at.astimezone(timezone.utc) <= self._ingest_cutoff
        ):
            self._dropped_submissions += 1
            return SubmissionResult(
                False,
                "before_clear_cutoff",
                self._queue.qsize(),
            )
        if not self._available:
            return SubmissionResult(
                False,
                SubmissionReason.DEGRADED,
                self._queue.qsize(),
            )
        try:
            self._queue.put_nowait(batch)
        except asyncio.QueueFull:
            self._dropped_submissions += 1
            return SubmissionResult(
                False,
                SubmissionReason.QUEUE_FULL,
                self._queue.qsize(),
            )
        return SubmissionResult(
            True,
            SubmissionReason.ACCEPTED,
            self._queue.qsize(),
        )

    async def _writer_loop(self) -> None:
        while True:
            batch = await self._queue.get()
            ingest_task = asyncio.create_task(
                asyncio.to_thread(
                    self._run_mutation,
                    lambda batch=batch: self._store.ingest(batch),
                ),
                name="MemoryService.ingest",
            )
            try:
                await asyncio.shield(ingest_task)
            except asyncio.CancelledError:
                self._track_detached_mutation(ingest_task)
                raise
            except Exception as exc:
                self._degraded_reason = f"{type(exc).__name__}: {exc}"[:500]
                self._dropped_submissions += 1
            else:
                self._available = True
                self._degraded_reason = None
            finally:
                self._queue.task_done()

    def _run_mutation(self, operation: Callable[[], _T]) -> _T:
        with self._mutation_lock:
            return operation()

    def _track_detached_mutation(self, task: asyncio.Task[object]) -> None:
        self._detached_mutations.add(task)
        task.add_done_callback(self._detached_mutation_finished)

    def _detached_mutation_finished(self, task: asyncio.Task[object]) -> None:
        self._detached_mutations.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None and not self._closed:
            self._degraded_reason = f"{type(error).__name__}: {error}"[:500]
            self._dropped_submissions += 1

    async def run_clear_barrier(
        self,
        operation: Callable[[], _T],
        *,
        cutoff: datetime,
    ) -> _T:
        """Run a destructive clear after isolating all accepted writer work."""

        if cutoff.utcoffset() is None:
            raise ValueError("clear cutoff must be timezone-aware")
        cutoff_utc = cutoff.astimezone(timezone.utc)
        async with self._clear_lock:
            if self._closed:
                raise RuntimeError("memory service is closed")
            self._clear_in_progress = True
            barrier_task = asyncio.create_task(
                self._execute_clear_barrier(operation, cutoff=cutoff_utc),
                name="MemoryService.clear",
            )
            try:
                return await asyncio.shield(barrier_task)
            except asyncio.CancelledError:
                await asyncio.gather(barrier_task, return_exceptions=True)
                raise

    async def _execute_clear_barrier(
        self,
        operation: Callable[[], _T],
        *,
        cutoff: datetime,
    ) -> _T:
        writer = self._writer_task
        writer_was_running = writer is not None
        if writer is not None:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            self._writer_task = None
        queued = self._take_queued_batches()
        try:
            result = await asyncio.to_thread(
                self._run_mutation,
                operation,
            )
        except BaseException:
            self._restore_queued_batches(queued)
            raise
        else:
            self._dropped_submissions += len(queued)
            if self._ingest_cutoff is None or cutoff > self._ingest_cutoff:
                self._ingest_cutoff = cutoff
            return result
        finally:
            self._clear_in_progress = False
            if writer_was_running and not self._closed:
                self._writer_task = asyncio.create_task(
                    self._writer_loop(),
                    name="MemoryService.writer",
                )

    def _take_queued_batches(self) -> list[CandidateBatch]:
        batches: list[CandidateBatch] = []
        while True:
            try:
                batches.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return batches
            else:
                self._queue.task_done()

    def _restore_queued_batches(self, batches: list[CandidateBatch]) -> None:
        for batch in batches:
            self._queue.put_nowait(batch)

    def _discard_queued_batches(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._queue.task_done()
                self._dropped_submissions += 1

    async def recall(
        self,
        query: RecallQuery,
        *,
        timeout_ms: int = 25,
    ) -> RecallResult:
        if not self._available:
            return RecallResult(
                items=(),
                database_revision=None,
                degraded_reason=self._degraded_reason or "memory service is unavailable",
            )
        inflight = self._recall_task
        if inflight is not None and not inflight.done():
            return RecallResult(
                items=(),
                database_revision=None,
                timed_out=True,
            )
        task = asyncio.create_task(
            asyncio.to_thread(self._store.recall, query),
            name="MemoryService.recall",
        )
        self._recall_task = task
        task.add_done_callback(self._recall_finished)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(timeout_ms, 0) / 1000,
            )
            self._available = True
            self._degraded_reason = None
            return result
        except TimeoutError:
            return RecallResult(
                items=(),
                database_revision=None,
                timed_out=True,
            )
        except Exception as exc:
            self._degraded_reason = f"{type(exc).__name__}: {exc}"[:500]
            return RecallResult(
                items=(),
                database_revision=None,
                degraded_reason=self._degraded_reason,
            )

    def _recall_finished(self, task: asyncio.Task[RecallResult]) -> None:
        if self._recall_task is task:
            self._recall_task = None
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def build_session_profile(
        self,
        *,
        limit: int = 8,
        timeout_ms: int = 50,
        now: datetime | None = None,
    ) -> SessionMemoryContext:
        if not self._available:
            return SessionMemoryContext(
                prompt="",
                item_ids=(),
                follow_up_id=None,
                database_revision=None,
                degraded_reason=self._degraded_reason or "memory service is unavailable",
            )
        profile_at = now or datetime.now(timezone.utc)
        inflight = self._profile_task
        if inflight is not None and not inflight.done():
            return SessionMemoryContext(
                prompt="",
                item_ids=(),
                follow_up_id=None,
                database_revision=None,
                timed_out=True,
            )
        task = asyncio.create_task(
            asyncio.to_thread(
                self._store.build_session_profile,
                now=profile_at,
                limit=limit,
            ),
            name="MemoryService.profile",
        )
        self._profile_task = task
        task.add_done_callback(self._profile_finished)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(timeout_ms, 0) / 1000,
            )
            self._available = True
            self._degraded_reason = None
            return result
        except TimeoutError:
            return SessionMemoryContext(
                prompt="",
                item_ids=(),
                follow_up_id=None,
                database_revision=None,
                timed_out=True,
            )
        except Exception as exc:
            self._degraded_reason = f"{type(exc).__name__}: {exc}"[:500]
            return SessionMemoryContext(
                prompt="",
                item_ids=(),
                follow_up_id=None,
                database_revision=None,
                degraded_reason=self._degraded_reason,
            )

    def _profile_finished(
        self,
        task: asyncio.Task[SessionMemoryContext],
    ) -> None:
        if self._profile_task is task:
            self._profile_task = None
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def claim_follow_up(
        self,
        event_id: str,
        *,
        expected_revision: int,
        asked_at: datetime | None = None,
        timeout_ms: int = 25,
    ) -> bool:
        if not self._available:
            return False
        timeout_seconds = max(timeout_ms, 0) / 1000
        if timeout_seconds <= 0:
            return False
        claim_before = getattr(self._store, "claim_follow_up_before", None)
        if not callable(claim_before):
            return False
        claim_at = asked_at or datetime.now(timezone.utc)
        deadline = time.monotonic() + timeout_seconds
        task = asyncio.create_task(
            asyncio.to_thread(
                claim_before,
                event_id,
                expected_revision=expected_revision,
                asked_at=claim_at,
                deadline_monotonic=deadline,
            ),
            name="MemoryService.claim_follow_up",
        )
        try:
            return bool(
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=timeout_seconds,
                )
            )
        except TimeoutError:
            if task.done():
                try:
                    return bool(task.result())
                except Exception:
                    return False
            self._track_detached_claim(task)
            return False
        except asyncio.CancelledError:
            self._track_detached_claim(task)
            raise
        except Exception:
            return False

    def _track_detached_claim(self, task: asyncio.Task[bool]) -> None:
        self._detached_claims.add(task)
        task.add_done_callback(self._detached_claim_finished)

    def _detached_claim_finished(self, task: asyncio.Task[bool]) -> None:
        self._detached_claims.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
