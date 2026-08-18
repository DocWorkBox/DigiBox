from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from avaturn_live_streamer.memory.service import MemoryService
from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    RecallQuery,
    RecallResult,
    SessionMemoryContext,
    SubmissionReason,
)


class _ThreadRecordingStore:
    def __init__(self) -> None:
        self.initialize_thread_id: int | None = None

    def initialize(self) -> None:
        self.initialize_thread_id = threading.get_ident()


def _batch(
    turn_id: str,
    *,
    observed_at: datetime | None = None,
) -> CandidateBatch:
    return CandidateBatch(
        session_id="session-1",
        turn_id=turn_id,
        engine_kind="custom_api",
        observed_at=observed_at or datetime.now(timezone.utc),
        transcript_sha256=hashlib.sha256(turn_id.encode()).hexdigest(),
    )


def test_start_initializes_store_off_event_loop_thread() -> None:
    async def exercise() -> tuple[int, int | None]:
        event_loop_thread_id = threading.get_ident()
        store = _ThreadRecordingStore()
        service = MemoryService(store)
        await service.start()
        await service.close()
        return event_loop_thread_id, store.initialize_thread_id

    event_loop_thread_id, initialize_thread_id = asyncio.run(exercise())

    assert initialize_thread_id is not None
    assert initialize_thread_id != event_loop_thread_id


def test_start_purges_expired_candidates_off_event_loop() -> None:
    class RetentionStore:
        def __init__(self) -> None:
            self.purge_thread_id: int | None = None
            self.purged_at: datetime | None = None

        def initialize(self) -> None:
            pass

        def purge_expired(self, *, now: datetime) -> object:
            self.purge_thread_id = threading.get_ident()
            self.purged_at = now
            return object()

    async def exercise() -> tuple[RetentionStore, int]:
        loop_thread_id = threading.get_ident()
        store = RetentionStore()
        service = MemoryService(store)
        await service.start()
        assert service.available is True
        await service.close()
        return store, loop_thread_id

    store, loop_thread_id = asyncio.run(exercise())

    assert store.purge_thread_id is not None
    assert store.purge_thread_id != loop_thread_id
    assert store.purged_at is not None
    assert store.purged_at.utcoffset() == timezone.utc.utcoffset(store.purged_at)


def test_start_retention_failure_warns_but_keeps_memory_service_available() -> None:
    class FailingRetentionStore:
        def initialize(self) -> None:
            pass

        def purge_expired(self, *, now: datetime) -> object:
            assert now.utcoffset() is not None
            raise sqlite3.OperationalError("retention cleanup locked")

        def recall(self, query: RecallQuery) -> RecallResult:
            _ = query
            return RecallResult(items=(), database_revision=4)

    async def exercise() -> tuple[bool, str | None, RecallResult]:
        service = MemoryService(FailingRetentionStore())
        await service.start()
        available = service.available
        warning = service.degraded_reason
        result = await service.recall(
            RecallQuery(text="张三", now=datetime.now(timezone.utc)),
            timeout_ms=250,
        )
        await service.close()
        return available, warning, result

    available, warning, result = asyncio.run(exercise())

    assert available is True
    assert warning is not None and "retention cleanup" in warning
    assert result.database_revision == 4


def test_retention_cleanup_repeats_at_injected_interval_and_close_cancels_it() -> None:
    class PeriodicRetentionStore:
        def __init__(self) -> None:
            self.purge_calls: list[datetime] = []
            self.second_purge = threading.Event()

        def initialize(self) -> None:
            pass

        def purge_expired(self, *, now: datetime) -> object:
            self.purge_calls.append(now)
            if len(self.purge_calls) >= 2:
                self.second_purge.set()
            return object()

    async def exercise() -> tuple[int, int]:
        store = PeriodicRetentionStore()
        service = MemoryService(store, purge_interval_seconds=0.01)
        await service.start()
        assert await asyncio.wait_for(
            asyncio.to_thread(store.second_purge.wait),
            timeout=0.5,
        )
        await service.close()
        calls_at_close = len(store.purge_calls)
        await asyncio.sleep(0.04)
        return calls_at_close, len(store.purge_calls)

    calls_at_close, calls_after_wait = asyncio.run(exercise())

    assert calls_at_close >= 2
    assert calls_after_wait == calls_at_close


@pytest.mark.parametrize(
    "failure",
    [sqlite3.DatabaseError("database disk image is malformed"), sqlite3.OperationalError("locked")],
)
def test_start_degrades_instead_of_raising_on_database_failure(
    failure: sqlite3.Error,
) -> None:
    class FailingStore:
        def initialize(self) -> None:
            raise failure

    async def exercise() -> MemoryService:
        service = MemoryService(FailingStore())
        await service.start()
        return service

    service = asyncio.run(exercise())

    assert service.available is False
    assert service.degraded_reason


def test_recall_timeout_returns_empty_fail_open_result() -> None:
    class SlowStore:
        def initialize(self) -> None:
            pass

        def recall(self, query: RecallQuery) -> RecallResult:
            _ = query
            time.sleep(0.1)
            return RecallResult(items=(), database_revision=1)

    async def exercise() -> tuple[RecallResult, float]:
        service = MemoryService(SlowStore())
        await service.start()
        started_at = asyncio.get_running_loop().time()
        result = await service.recall(
            RecallQuery(text="张三", now=datetime.now(timezone.utc)),
            timeout_ms=10,
        )
        elapsed = asyncio.get_running_loop().time() - started_at
        await service.close()
        return result, elapsed

    result, elapsed = asyncio.run(exercise())

    assert result.items == ()
    assert result.timed_out is True
    assert result.degraded_reason is None
    assert elapsed < 0.08


def test_recall_timeout_keeps_only_one_background_call_until_it_finishes() -> None:
    class BlockingRecallStore:
        def __init__(self) -> None:
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()
            self.first_finished = threading.Event()

        def initialize(self) -> None:
            pass

        def recall(self, query: RecallQuery) -> RecallResult:
            _ = query
            self.calls += 1
            call_number = self.calls
            if call_number == 1:
                self.started.set()
                self.release.wait(timeout=1.0)
                self.first_finished.set()
            return RecallResult(items=(), database_revision=call_number)

    async def exercise() -> tuple[RecallResult, RecallResult, RecallResult, float, int]:
        store = BlockingRecallStore()
        service = MemoryService(store)
        await service.start()
        query = RecallQuery(text="张三", now=datetime.now(timezone.utc))
        first = await service.recall(query, timeout_ms=50)
        assert store.started.is_set()

        started_at = asyncio.get_running_loop().time()
        second = await service.recall(query, timeout_ms=200)
        second_elapsed = asyncio.get_running_loop().time() - started_at

        store.release.set()
        assert await asyncio.wait_for(
            asyncio.to_thread(store.first_finished.wait),
            timeout=0.5,
        )
        await asyncio.sleep(0)
        third = await service.recall(query, timeout_ms=200)
        await service.close()
        return first, second, third, second_elapsed, store.calls

    first, second, third, second_elapsed, calls = asyncio.run(exercise())

    assert first.timed_out is True
    assert second.timed_out is True
    assert second_elapsed < 0.03
    assert third.database_revision == 2
    assert calls == 2


def test_build_session_profile_delegates_with_a_hard_timeout_boundary() -> None:
    class ProfileStore:
        def initialize(self) -> None:
            pass

        def build_session_profile(
            self,
            *,
            now: datetime,
            limit: int,
        ) -> SessionMemoryContext:
            assert now.utcoffset() is not None
            assert limit == 4
            return SessionMemoryContext(
                prompt="主人叫小雨",
                item_ids=("person-1",),
                follow_up_id=None,
                database_revision=2,
            )

    async def exercise() -> SessionMemoryContext:
        service = MemoryService(ProfileStore())
        await service.start()
        result = await service.build_session_profile(limit=4, timeout_ms=50)
        await service.close()
        return result

    result = asyncio.run(exercise())

    assert result.prompt == "主人叫小雨"
    assert result.item_ids == ("person-1",)
    assert result.timed_out is False


def test_claim_follow_up_uses_the_cooperative_deadline_protocol_off_loop() -> None:
    class ClaimStore:
        def __init__(self) -> None:
            self.thread_id: int | None = None
            self.call: tuple[str, int, datetime, float] | None = None

        def initialize(self) -> None:
            pass

        def claim_follow_up_before(
            self,
            event_id: str,
            *,
            expected_revision: int,
            asked_at: datetime,
            deadline_monotonic: float,
        ) -> bool:
            self.thread_id = threading.get_ident()
            self.call = (
                event_id,
                expected_revision,
                asked_at,
                deadline_monotonic,
            )
            return True

    async def exercise() -> tuple[bool, ClaimStore, int]:
        loop_thread_id = threading.get_ident()
        store = ClaimStore()
        service = MemoryService(store)
        await service.start()
        asked_at = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
        claimed = await service.claim_follow_up(
            "event-1",
            expected_revision=7,
            asked_at=asked_at,
            timeout_ms=50,
        )
        assert service.available is True
        await service.close()
        return claimed, store, loop_thread_id

    claimed, store, loop_thread_id = asyncio.run(exercise())

    assert claimed is True
    assert store.call is not None
    assert store.call[:2] == ("event-1", 7)
    assert store.call[3] > 0
    assert store.thread_id is not None and store.thread_id != loop_thread_id


def test_slow_cooperative_claim_times_out_without_a_late_commit() -> None:
    class SlowCooperativeClaimStore:
        def __init__(self) -> None:
            self.calls = 0
            self.finished = threading.Event()
            self.committed = False

        def initialize(self) -> None:
            pass

        def claim_follow_up_before(
            self,
            event_id: str,
            *,
            expected_revision: int,
            asked_at: datetime,
            deadline_monotonic: float,
        ) -> bool:
            _ = (event_id, expected_revision, asked_at)
            self.calls += 1
            time.sleep(0.05)
            if time.monotonic() >= deadline_monotonic:
                self.finished.set()
                return False
            self.committed = True
            self.finished.set()
            return True

    async def exercise() -> tuple[bool, SlowCooperativeClaimStore, float]:
        store = SlowCooperativeClaimStore()
        service = MemoryService(store)
        await service.start()
        started_at = asyncio.get_running_loop().time()
        claimed = await service.claim_follow_up(
            "event-1",
            expected_revision=1,
            timeout_ms=1,
        )
        elapsed = asyncio.get_running_loop().time() - started_at
        assert await asyncio.wait_for(
            asyncio.to_thread(store.finished.wait, 1.0),
            timeout=0.5,
        )
        await service.close()
        return claimed, store, elapsed

    claimed, store, elapsed = asyncio.run(exercise())

    assert claimed is False
    assert store.calls == 1
    assert store.committed is False
    assert elapsed < 0.03


def test_legacy_slow_claim_is_not_started_because_it_cannot_be_cancelled() -> None:
    class LegacySlowClaimStore:
        def __init__(self) -> None:
            self.calls = 0
            self.committed = False

        def initialize(self) -> None:
            pass

        def claim_follow_up(
            self,
            event_id: str,
            *,
            expected_revision: int,
            asked_at: datetime,
        ) -> bool:
            _ = (event_id, expected_revision, asked_at)
            self.calls += 1
            time.sleep(0.05)
            self.committed = True
            return True

    async def exercise() -> tuple[bool, LegacySlowClaimStore, float]:
        store = LegacySlowClaimStore()
        service = MemoryService(store)
        await service.start()
        started_at = asyncio.get_running_loop().time()
        claimed = await service.claim_follow_up(
            "event-1",
            expected_revision=1,
            timeout_ms=1,
        )
        elapsed = asyncio.get_running_loop().time() - started_at
        await asyncio.sleep(0.08)
        await service.close()
        return claimed, store, elapsed

    claimed, store, elapsed = asyncio.run(exercise())

    assert claimed is False
    assert store.calls == 0
    assert store.committed is False
    assert elapsed < 0.03


def test_recall_transient_error_does_not_permanently_disable_memory() -> None:
    class FlakyRecallStore:
        def __init__(self) -> None:
            self.calls = 0

        def initialize(self) -> None:
            pass

        def recall(self, query: RecallQuery) -> RecallResult:
            _ = query
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("temporarily locked")
            return RecallResult(items=(), database_revision=9)

    async def exercise() -> tuple[RecallResult, RecallResult, bool, str | None, int]:
        store = FlakyRecallStore()
        service = MemoryService(store)
        await service.start()
        query = RecallQuery(text="张三", now=datetime.now(timezone.utc))
        first = await service.recall(query, timeout_ms=100)
        available_after_error = service.available
        second = await service.recall(query, timeout_ms=100)
        degraded_after_recovery = service.degraded_reason
        await service.close()
        return first, second, available_after_error, degraded_after_recovery, store.calls

    first, second, available, degraded_reason, calls = asyncio.run(exercise())

    assert first.degraded_reason is not None
    assert available is True
    assert second.database_revision == 9
    assert degraded_reason is None
    assert calls == 2


def test_profile_transient_error_does_not_permanently_disable_memory() -> None:
    class FlakyProfileStore:
        def __init__(self) -> None:
            self.calls = 0

        def initialize(self) -> None:
            pass

        def build_session_profile(
            self,
            *,
            now: datetime,
            limit: int,
        ) -> SessionMemoryContext:
            _ = now, limit
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("temporarily locked")
            return SessionMemoryContext(
                prompt="主人喜欢喝茶",
                item_ids=("person-1",),
                follow_up_id=None,
                database_revision=12,
            )

    async def exercise() -> tuple[SessionMemoryContext, SessionMemoryContext, bool, str | None, int]:
        store = FlakyProfileStore()
        service = MemoryService(store)
        await service.start()
        first = await service.build_session_profile(timeout_ms=100)
        available_after_error = service.available
        second = await service.build_session_profile(timeout_ms=100)
        degraded_after_recovery = service.degraded_reason
        await service.close()
        return first, second, available_after_error, degraded_after_recovery, store.calls

    first, second, available, degraded_reason, calls = asyncio.run(exercise())

    assert first.degraded_reason is not None
    assert available is True
    assert second.prompt == "主人喜欢喝茶"
    assert second.database_revision == 12
    assert degraded_reason is None
    assert calls == 2


def test_profile_timeout_keeps_one_background_call_until_it_finishes() -> None:
    class BlockingProfileStore:
        def __init__(self) -> None:
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()
            self.first_finished = threading.Event()

        def initialize(self) -> None:
            pass

        def build_session_profile(
            self,
            *,
            now: datetime,
            limit: int,
        ) -> SessionMemoryContext:
            _ = now, limit
            self.calls += 1
            call_number = self.calls
            if call_number == 1:
                self.started.set()
                self.release.wait(timeout=1.0)
                self.first_finished.set()
            return SessionMemoryContext(
                prompt=f"profile-{call_number}",
                item_ids=(),
                follow_up_id=None,
                database_revision=call_number,
            )

    async def exercise() -> tuple[
        SessionMemoryContext,
        SessionMemoryContext,
        SessionMemoryContext,
        float,
        int,
    ]:
        store = BlockingProfileStore()
        service = MemoryService(store)
        await service.start()
        first = await service.build_session_profile(timeout_ms=20)
        assert store.started.is_set()

        started_at = asyncio.get_running_loop().time()
        second = await service.build_session_profile(timeout_ms=200)
        second_elapsed = asyncio.get_running_loop().time() - started_at

        store.release.set()
        assert await asyncio.wait_for(
            asyncio.to_thread(store.first_finished.wait),
            timeout=0.5,
        )
        await asyncio.sleep(0)
        third = await service.build_session_profile(timeout_ms=200)
        await service.close()
        return first, second, third, second_elapsed, store.calls

    first, second, third, second_elapsed, calls = asyncio.run(exercise())

    assert first.timed_out is True
    assert second.timed_out is True
    assert second_elapsed < 0.03
    assert third.prompt == "profile-2"
    assert calls == 2


def test_claim_follow_up_error_is_fail_open_without_degrading_service() -> None:
    class FailingClaimStore:
        def initialize(self) -> None:
            pass

        def claim_follow_up_before(
            self,
            event_id: str,
            *,
            expected_revision: int,
            asked_at: datetime,
            deadline_monotonic: float,
        ) -> bool:
            _ = (event_id, expected_revision, asked_at, deadline_monotonic)
            raise sqlite3.OperationalError("locked")

    async def exercise() -> tuple[bool, bool, str | None]:
        service = MemoryService(FailingClaimStore())
        await service.start()
        claimed = await service.claim_follow_up(
            "event-1",
            expected_revision=1,
        )
        state = (claimed, service.available, service.degraded_reason)
        await service.close()
        return state

    claimed, available, degraded_reason = asyncio.run(exercise())

    assert claimed is False
    assert available is True
    assert degraded_reason is None


def test_try_submit_is_immediate_and_reports_a_full_bounded_queue() -> None:
    class BlockingStore:
        def __init__(self) -> None:
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            _ = batch
            self.ingest_started.set()
            self.release_ingest.wait(timeout=1.0)
            return object()

    async def exercise() -> tuple[object, object, object, float]:
        store = BlockingStore()
        service = MemoryService(store, queue_size=1)
        await service.start()

        started_at = asyncio.get_running_loop().time()
        first = service.try_submit(_batch("turn-1"))
        elapsed = asyncio.get_running_loop().time() - started_at
        await asyncio.wait_for(asyncio.to_thread(store.ingest_started.wait), timeout=0.5)
        second = service.try_submit(_batch("turn-2"))
        third = service.try_submit(_batch("turn-3"))

        store.release_ingest.set()
        await service.close()
        return first, second, third, elapsed

    first, second, third, elapsed = asyncio.run(exercise())

    assert first.accepted is True
    assert first.reason is SubmissionReason.ACCEPTED
    assert second.accepted is True
    assert third.accepted is False
    assert third.reason is SubmissionReason.QUEUE_FULL
    assert elapsed < 0.01


def test_close_drains_submissions_through_one_serial_off_loop_writer() -> None:
    class SerialStore:
        def __init__(self) -> None:
            self.turn_ids: list[str] = []
            self.thread_ids: list[int] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self.thread_ids.append(threading.get_ident())
                time.sleep(0.01)
                self.turn_ids.append(batch.turn_id)
                return object()
            finally:
                with self.lock:
                    self.active -= 1

    async def exercise() -> tuple[SerialStore, int]:
        event_loop_thread_id = threading.get_ident()
        store = SerialStore()
        service = MemoryService(store, queue_size=3)
        await service.start()
        for turn_id in ("turn-1", "turn-2", "turn-3"):
            assert service.try_submit(_batch(turn_id)).accepted
        await service.close(drain_timeout_seconds=1.0)
        return store, event_loop_thread_id

    store, event_loop_thread_id = asyncio.run(exercise())

    assert store.turn_ids == ["turn-1", "turn-2", "turn-3"]
    assert store.max_active == 1
    assert all(thread_id != event_loop_thread_id for thread_id in store.thread_ids)


def test_close_drain_timeout_is_bounded_while_ingest_thread_is_still_blocked() -> None:
    class SlowStore:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            _ = batch
            self.started.set()
            self.release.wait(timeout=1.0)
            self.finished.set()
            return object()

    async def exercise() -> tuple[bool, bool, float]:
        store = SlowStore()
        service = MemoryService(store)
        await service.start()
        assert service.try_submit(_batch("slow-turn")).accepted
        assert await asyncio.wait_for(
            asyncio.to_thread(store.started.wait, 1.0),
            timeout=0.5,
        )

        started_at = asyncio.get_running_loop().time()
        close_task = asyncio.create_task(
            service.close(drain_timeout_seconds=0.01)
        )
        await asyncio.sleep(0.08)
        finished_before_release = close_task.done()
        ingest_finished_at_close = store.finished.is_set()
        elapsed = asyncio.get_running_loop().time() - started_at

        store.release.set()
        await asyncio.wait_for(close_task, timeout=0.5)
        return finished_before_release, ingest_finished_at_close, elapsed

    finished, ingest_finished, elapsed = asyncio.run(exercise())

    assert finished is True
    assert ingest_finished is False
    assert elapsed < 0.15


def test_writer_drops_only_the_failed_batch_and_recovers_on_the_next_success() -> None:
    class FlakyStore:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.persisted: list[str] = []
            self.recovered = threading.Event()

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            self.calls.append(batch.turn_id)
            if batch.turn_id == "turn-fails":
                raise sqlite3.OperationalError("database temporarily locked")
            self.persisted.append(batch.turn_id)
            self.recovered.set()
            return object()

    async def exercise() -> tuple[FlakyStore, bool, str | None, int]:
        store = FlakyStore()
        service = MemoryService(store)
        await service.start()
        assert service.try_submit(_batch("turn-fails")).accepted
        assert service.try_submit(_batch("turn-recovers")).accepted
        assert await asyncio.wait_for(
            asyncio.to_thread(store.recovered.wait, 1.0),
            timeout=0.5,
        )
        for _ in range(100):
            if service.degraded_reason is None:
                break
            await asyncio.sleep(0.005)
        state = (
            store,
            service.available,
            service.degraded_reason,
            service.dropped_submissions,
        )
        await service.close()
        return state

    store, available, degraded_reason, dropped = asyncio.run(exercise())

    assert store.calls == ["turn-fails", "turn-recovers"]
    assert store.persisted == ["turn-recovers"]
    assert available is True
    assert degraded_reason is None
    assert dropped == 1


def test_successful_clear_barrier_drops_queued_and_late_pre_clear_batches() -> None:
    cutoff = datetime(2026, 8, 17, 15, 0, tzinfo=timezone.utc)

    class BarrierStore:
        def __init__(self) -> None:
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()
            self.new_ingested = threading.Event()
            self.persisted: list[str] = []

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            if batch.turn_id == "current":
                self.ingest_started.set()
                self.release_ingest.wait(timeout=1.0)
            self.persisted.append(batch.turn_id)
            if batch.turn_id == "after-clear":
                self.new_ingested.set()
            return object()

    async def exercise() -> tuple[BarrierStore, object, object, object, int]:
        store = BarrierStore()
        service = MemoryService(store)
        await service.start()
        assert service.try_submit(
            _batch("current", observed_at=cutoff - timedelta(minutes=3))
        ).accepted
        assert await asyncio.wait_for(
            asyncio.to_thread(store.ingest_started.wait, 1.0),
            timeout=0.5,
        )
        assert service.try_submit(
            _batch("queued-before-clear", observed_at=cutoff - timedelta(minutes=2))
        ).accepted

        def clear_operation() -> str:
            assert store.persisted == ["current"]
            return "cleared"

        clear_task = asyncio.create_task(
            service.run_clear_barrier(clear_operation, cutoff=cutoff)
        )
        await asyncio.sleep(0)
        during = service.try_submit(
            _batch("during-clear", observed_at=cutoff + timedelta(minutes=1))
        )
        store.release_ingest.set()
        await clear_task

        late_old = service.try_submit(
            _batch("late-old", observed_at=cutoff - timedelta(seconds=1))
        )
        after = service.try_submit(
            _batch("after-clear", observed_at=cutoff + timedelta(seconds=1))
        )
        assert await asyncio.wait_for(
            asyncio.to_thread(store.new_ingested.wait, 1.0),
            timeout=0.5,
        )
        dropped = service.dropped_submissions
        await service.close()
        return store, during, late_old, after, dropped

    store, during, late_old, after, dropped = asyncio.run(exercise())

    assert store.persisted == ["current", "after-clear"]
    assert during.accepted is False
    assert during.reason == "clear_in_progress"
    assert late_old.accepted is False
    assert late_old.reason == "before_clear_cutoff"
    assert after.accepted is True
    assert dropped == 3


def test_clear_waits_on_the_same_sync_mutation_lock_before_deleting_old_ingest() -> None:
    cutoff = datetime(2026, 8, 17, 15, 30, tzinfo=timezone.utc)

    class LockedStore:
        def __init__(self) -> None:
            self.mutation_lock = threading.Lock()
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()
            self.clear_started = threading.Event()
            self.persisted: list[str] = []

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            with self.mutation_lock:
                self.ingest_started.set()
                self.release_ingest.wait(timeout=1.0)
                self.persisted.append(batch.turn_id)
            return object()

        def clear(self) -> None:
            with self.mutation_lock:
                self.clear_started.set()
                self.persisted.clear()

    async def exercise() -> tuple[bool, list[str]]:
        store = LockedStore()
        service = MemoryService(store)
        await service.start()
        assert service.try_submit(
            _batch("old-turn", observed_at=cutoff - timedelta(minutes=1))
        ).accepted
        assert await asyncio.wait_for(
            asyncio.to_thread(store.ingest_started.wait, 1.0),
            timeout=0.5,
        )

        clear_task = asyncio.create_task(
            service.run_clear_barrier(store.clear, cutoff=cutoff)
        )
        await asyncio.sleep(0.03)
        clear_started_before_release = store.clear_started.is_set()
        store.release_ingest.set()
        await asyncio.wait_for(clear_task, timeout=0.5)
        persisted = list(store.persisted)
        await service.close()
        return clear_started_before_release, persisted

    clear_started_early, persisted = asyncio.run(exercise())

    assert clear_started_early is False
    assert persisted == []


def test_failed_clear_barrier_restores_queued_batches_in_order() -> None:
    cutoff = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)

    class BarrierStore:
        def __init__(self) -> None:
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()
            self.all_ingested = threading.Event()
            self.persisted: list[str] = []

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            if batch.turn_id == "current":
                self.ingest_started.set()
                self.release_ingest.wait(timeout=1.0)
            self.persisted.append(batch.turn_id)
            if len(self.persisted) >= 4:
                self.all_ingested.set()
            return object()

    async def exercise() -> tuple[BarrierStore, object, object]:
        store = BarrierStore()
        service = MemoryService(store)
        await service.start()
        assert service.try_submit(_batch("current")).accepted
        assert await asyncio.wait_for(
            asyncio.to_thread(store.ingest_started.wait, 1.0),
            timeout=0.5,
        )
        assert service.try_submit(_batch("queued-1")).accepted
        assert service.try_submit(_batch("queued-2")).accepted

        def failing_clear() -> None:
            assert store.persisted == ["current"]
            raise sqlite3.OperationalError("clear transaction rolled back")

        clear_task = asyncio.create_task(
            service.run_clear_barrier(failing_clear, cutoff=cutoff)
        )
        await asyncio.sleep(0)
        during = service.try_submit(_batch("during-clear"))
        store.release_ingest.set()
        with pytest.raises(sqlite3.OperationalError, match="rolled back"):
            await clear_task
        after_failure = service.try_submit(_batch("after-failure"))
        assert await asyncio.wait_for(
            asyncio.to_thread(store.all_ingested.wait, 1.0),
            timeout=0.5,
        )
        await service.close()
        return store, during, after_failure

    store, during, after_failure = asyncio.run(exercise())

    assert during.accepted is False
    assert during.reason == "clear_in_progress"
    assert after_failure.accepted is True
    assert store.persisted == ["current", "queued-1", "queued-2", "after-failure"]


def test_cancelling_clear_waits_for_the_clear_outcome_before_reopening_writes() -> None:
    cutoff = datetime(2026, 8, 17, 17, 0, tzinfo=timezone.utc)

    class BarrierStore:
        def __init__(self) -> None:
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()
            self.persisted: list[str] = []

        def initialize(self) -> None:
            pass

        def ingest(self, batch: CandidateBatch) -> object:
            if batch.turn_id == "current":
                self.ingest_started.set()
                self.release_ingest.wait(timeout=1.0)
            self.persisted.append(batch.turn_id)
            return object()

    async def exercise() -> tuple[BarrierStore, bool, object]:
        store = BarrierStore()
        service = MemoryService(store)
        await service.start()
        assert service.try_submit(
            _batch("current", observed_at=cutoff - timedelta(minutes=2))
        ).accepted
        assert await asyncio.wait_for(
            asyncio.to_thread(store.ingest_started.wait, 1.0),
            timeout=0.5,
        )
        assert service.try_submit(
            _batch("queued-old", observed_at=cutoff - timedelta(minutes=1))
        ).accepted
        clear_started = threading.Event()
        release_clear = threading.Event()

        def successful_clear() -> None:
            clear_started.set()
            release_clear.wait(timeout=1.0)

        task = asyncio.create_task(
            service.run_clear_barrier(successful_clear, cutoff=cutoff)
        )
        store.release_ingest.set()
        assert await asyncio.wait_for(
            asyncio.to_thread(clear_started.wait, 1.0),
            timeout=0.5,
        )
        task.cancel()
        await asyncio.sleep(0.02)
        finished_while_clear_running = task.done()
        release_clear.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        late_old = service.try_submit(
            _batch("late-old", observed_at=cutoff - timedelta(seconds=1))
        )
        await service.close()
        return store, finished_while_clear_running, late_old

    store, finished_while_clear_running, late_old = asyncio.run(exercise())

    assert finished_while_clear_running is False
    assert store.persisted == ["current"]
    assert late_old.accepted is False
    assert late_old.reason == "before_clear_cutoff"
