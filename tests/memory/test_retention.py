from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    EventCandidate,
    EventParticipantCandidate,
    EventStatus,
    MemoryState,
    PersonCandidate,
    RetentionClass,
)
from avaturn_live_streamer.memory.sqlite_store import SQLiteMemoryStore


UTC = timezone.utc
NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    store.initialize(owner_uuid="owner-test")
    return store


def _person_batch(
    *,
    turn: int,
    observed_at: datetime,
    confidence: float,
    state: MemoryState = MemoryState.CANDIDATE,
    name: str = "阿青",
) -> CandidateBatch:
    return CandidateBatch(
        session_id="session-1",
        turn_id=f"turn-{turn}",
        engine_kind="custom_api",
        observed_at=observed_at,
        transcript_sha256=f"{turn:x}" * 64,
        people=(
            PersonCandidate(
                name=name,
                confidence=confidence,
                state=state,
                evidence_excerpt=f"{name}可能要来",
            ),
        ),
    )


def test_low_confidence_candidate_expires_exactly_thirty_days_after_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    result = store.ingest(
        _person_batch(turn=1, observed_at=NOW, confidence=0.59)
    )

    record = store.get(result.person_ids[0])

    assert record is not None
    assert record.retention_class is RetentionClass.TEMPORARY_30D
    assert record.expires_at == NOW + timedelta(days=30)


def test_remention_refreshes_low_candidate_expiry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.ingest(_person_batch(turn=1, observed_at=NOW, confidence=0.4))
    refreshed_at = NOW + timedelta(days=20)

    second = store.ingest(
        _person_batch(turn=2, observed_at=refreshed_at, confidence=0.4)
    )

    assert second.person_ids == first.person_ids
    record = store.get(first.person_ids[0])
    assert record is not None
    assert record.last_seen_at == refreshed_at
    assert record.expires_at == refreshed_at + timedelta(days=30)


def test_purge_uses_inclusive_expiry_boundary(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.ingest(_person_batch(turn=1, observed_at=NOW, confidence=0.4))
    expiry = NOW + timedelta(days=30)

    before = store.purge_expired(now=expiry - timedelta(microseconds=1))
    at_boundary = store.purge_expired(now=expiry)

    assert before.deleted_ids == ()
    assert at_boundary.deleted_ids == result.person_ids
    assert store.get(result.person_ids[0]) is None


def test_high_candidate_and_confirmed_memory_are_never_auto_deleted(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    high = store.ingest(
        _person_batch(turn=1, observed_at=NOW, confidence=0.60, name="高可信候选")
    )
    confirmed = store.ingest(
        _person_batch(
            turn=2,
            observed_at=NOW,
            confidence=1.0,
            state=MemoryState.CONFIRMED,
            name="正式记忆",
        )
    )

    purged = store.purge_expired(now=NOW + timedelta(days=3650))

    assert purged.deleted_ids == ()
    assert store.get(high.person_ids[0]) is not None
    assert store.get(confirmed.person_ids[0]) is not None


def test_expired_person_referenced_by_confirmed_event_is_retained(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.ingest(
        CandidateBatch(
            session_id="session-1",
            turn_id="turn-1",
            engine_kind="custom_api",
            observed_at=NOW,
            transcript_sha256="a" * 64,
            people=(
                PersonCandidate(
                    name="阿青",
                    confidence=0.4,
                    state=MemoryState.CANDIDATE,
                ),
            ),
            events=(
                EventCandidate(
                    title="和阿青见面",
                    starts_at=NOW + timedelta(days=1),
                    status=EventStatus.PLANNED,
                    participants=(EventParticipantCandidate(name="阿青"),),
                    confidence=1.0,
                    state=MemoryState.CONFIRMED,
                ),
            ),
        )
    )

    purged = store.purge_expired(now=NOW + timedelta(days=31))

    assert purged.deleted_ids == ()
    assert purged.retained_due_to_reference == result.person_ids
    assert store.get(result.person_ids[0]) is not None
    assert store.get(result.event_ids[0]) is not None
