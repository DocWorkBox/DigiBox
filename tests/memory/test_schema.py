from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    EventCandidate,
    EventParticipantCandidate,
    EventStatus,
    MemoryState,
    PersonCandidate,
    RelationshipCandidate,
    SubmissionResult,
)
from avaturn_live_streamer.memory.schema import (
    SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    initialize_database,
    open_database,
)


def test_candidate_batch_is_immutable_and_requires_aware_time() -> None:
    person = PersonCandidate(
        name="张三",
        aliases=("老张",),
        confidence=0.9,
        state=MemoryState.CONFIRMED,
        evidence_excerpt="张三是我的同事",
    )
    relationship = RelationshipCandidate(
        source_name=None,
        target_name="张三",
        relation_type="同事",
        confidence=0.9,
        state=MemoryState.CONFIRMED,
        evidence_excerpt="张三是我的同事",
    )
    event = EventCandidate(
        title="和张三开会",
        starts_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        status=EventStatus.PLANNED,
        participants=(EventParticipantCandidate(name="张三", role="参与者"),),
        confidence=0.9,
        state=MemoryState.CONFIRMED,
        evidence_excerpt="明天和张三开会",
    )
    batch = CandidateBatch(
        session_id="session-1",
        turn_id="turn-1",
        engine_kind="custom_api",
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        transcript_sha256="a" * 64,
        owner_name="小雨",
        people=(person,),
        relationships=(relationship,),
        events=(event,),
    )

    assert batch.people == (person,)
    assert batch.relationships == (relationship,)
    assert batch.events == (event,)
    with pytest.raises(FrozenInstanceError):
        batch.owner_name = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError, match="timezone-aware"):
        CandidateBatch(
            session_id="session-1",
            turn_id="turn-2",
            engine_kind="custom_api",
            observed_at=datetime(2026, 8, 17),
            transcript_sha256="b" * 64,
        )


def test_submission_result_reports_non_blocking_queue_outcome() -> None:
    result = SubmissionResult(
        accepted=False,
        reason="queue_full",
        pending_count=128,
    )

    assert result.accepted is False
    assert result.reason == "queue_full"
    assert result.pending_count == 128


def test_memory_package_exports_core_store_contract() -> None:
    import avaturn_live_streamer.memory as memory

    assert memory.SQLiteMemoryStore.__name__ == "SQLiteMemoryStore"
    assert memory.RecallQuery.__name__ == "RecallQuery"
    assert memory.RecallResult.__name__ == "RecallResult"
    assert memory.SessionMemoryContext.__name__ == "SessionMemoryContext"
    assert memory.SubmissionResult.__name__ == "SubmissionResult"


def _memory_values(
    *,
    memory_id: str = "memory-1",
    state: str = "candidate",
    retention: str = "persistent",
    expires_at: str | None = None,
) -> tuple[object, ...]:
    return (
        memory_id,
        "event",
        state,
        0.8,
        retention,
        "event:key",
        "a" * 64,
        1,
        "2026-08-17T00:00:00+00:00",
        "2026-08-17T00:00:00+00:00",
        "2026-08-17T00:00:00+00:00",
        expires_at,
    )


_INSERT_MEMORY = """
INSERT INTO memories (
    id, kind, state, confidence, retention_class, canonical_key,
    content_fingerprint, revision, created_at, updated_at, last_seen_at, expires_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def test_initialize_creates_schema_idempotently(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "memory.sqlite3"

    first = initialize_database(database, owner_uuid="owner-test")
    second = initialize_database(database, owner_uuid="ignored-on-reopen")

    assert first.schema_version == SCHEMA_VERSION
    assert first.owner_uuid == "owner-test"
    assert second.owner_uuid == "owner-test"
    with open_database(database, read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "meta",
            "owner_profile",
            "turn_sources",
            "memories",
            "people",
            "person_aliases",
            "relationships",
            "events",
            "event_participants",
            "evidence",
            "revisions",
            "tombstones",
        } <= tables


def test_confirmed_memory_cannot_have_expiry(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    initialize_database(database)

    with open_database(database) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            _INSERT_MEMORY,
            _memory_values(
                state="confirmed",
                retention="persistent",
                expires_at="2026-09-16T00:00:00+00:00",
            ),
        )


@pytest.mark.parametrize(
    ("state", "expires_at"),
    [
        ("confirmed", "2026-09-16T00:00:00+00:00"),
        ("candidate", None),
    ],
)
def test_temporary_retention_requires_candidate_and_expiry(
    tmp_path: Path,
    state: str,
    expires_at: str | None,
) -> None:
    database = tmp_path / "memory.sqlite3"
    initialize_database(database)

    with open_database(database) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            _INSERT_MEMORY,
            _memory_values(
                state=state,
                retention="temporary_30d",
                expires_at=expires_at,
            ),
        )


def test_foreign_keys_are_enabled_on_every_connection(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    initialize_database(database)

    with open_database(database) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO people (memory_id, display_name, normalized_name, notes)
                VALUES ('missing-memory', 'Alice', 'alice', NULL)
                """
            )


def test_open_database_context_closes_the_connection_on_exit(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    initialize_database(database)

    with open_database(database, read_only=True) as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_newer_schema_is_refused_without_overwriting_data(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        connection.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel VALUES ('keep-me')")

    with pytest.raises(UnsupportedSchemaVersion):
        initialize_database(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM sentinel").fetchone()[0] == "keep-me"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 1
