from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    EventCandidate,
    EventParticipantCandidate,
    EventStatus,
    FollowUpState,
    MemoryKind,
    MemoryState,
    PersonCandidate,
    RelationshipCandidate,
)
from avaturn_live_streamer.memory.extractor import HeuristicMemoryExtractor
from avaturn_live_streamer.memory.schema import open_database
from avaturn_live_streamer.memory.sqlite_store import (
    SQLiteMemoryStore,
    SourceTurnConflict,
    StaleRevisionError,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


def _batch(
    *,
    turn_id: str = "turn-1",
    observed_at: datetime = NOW,
    digest: str = "a" * 64,
    owner_name: str | None = None,
    people: tuple[PersonCandidate, ...] = (),
    relationships: tuple[RelationshipCandidate, ...] = (),
    events: tuple[EventCandidate, ...] = (),
) -> CandidateBatch:
    return CandidateBatch(
        session_id="session-1",
        turn_id=turn_id,
        engine_kind="custom_api",
        observed_at=observed_at,
        transcript_sha256=digest,
        owner_name=owner_name,
        people=people,
        relationships=relationships,
        events=events,
    )


def _store(tmp_path: Path) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    store.initialize(owner_uuid="owner-test")
    return store


def test_ingests_owner_person_relationship_event_and_evidence_atomically(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    result = store.ingest(
        _batch(
            owner_name="小雨",
            people=(
                PersonCandidate(
                    name="张三",
                    aliases=("老张",),
                    evidence_excerpt="张三是我的同事",
                ),
            ),
            relationships=(
                RelationshipCandidate(
                    source_name=None,
                    target_name="张三",
                    relation_type="同事",
                    evidence_excerpt="张三是我的同事",
                ),
            ),
            events=(
                EventCandidate(
                    title="和张三开会",
                    starts_at=NOW + timedelta(days=1),
                    status=EventStatus.PLANNED,
                    participants=(EventParticipantCandidate(name="张三"),),
                    evidence_excerpt="明天和张三开会",
                ),
            ),
        )
    )

    assert result.replayed is False
    assert len(result.person_ids) == 1
    assert len(result.relationship_ids) == 1
    assert len(result.event_ids) == 1
    assert result.database_revision == 1
    assert store.owner_profile().display_name == "小雨"
    assert store.count_records() == 3
    assert store.count_evidence() == 3


def test_invalid_relationship_rolls_back_entire_batch(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="unknown or ambiguous person"):
        store.ingest(
            _batch(
                owner_name="must-roll-back",
                people=(PersonCandidate(name="张三"),),
                relationships=(
                    RelationshipCandidate(
                        source_name=None,
                        target_name="不存在的人",
                        relation_type="同事",
                    ),
                ),
            )
        )

    assert store.count_records() == 0
    assert store.owner_profile().display_name is None
    assert store.database_revision() == 0


def test_replaying_same_source_turn_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = _batch(people=(PersonCandidate(name="张三"),))

    first = store.ingest(batch)
    second = store.ingest(batch)

    assert second.replayed is True
    assert second.person_ids == first.person_ids
    assert store.count_records() == 1
    assert store.count_evidence() == 0
    assert store.database_revision() == 1


def test_reusing_source_turn_with_different_transcript_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ingest(_batch(people=(PersonCandidate(name="张三"),)))

    with pytest.raises(SourceTurnConflict):
        store.ingest(
            _batch(
                digest="b" * 64,
                people=(PersonCandidate(name="李四"),),
            )
        )

    assert store.count_records() == 1


def test_empty_candidate_batch_does_not_record_source_or_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)

    result = store.ingest(_batch(turn_id="empty", digest="0" * 64))

    with open_database(store.database, read_only=True) as connection:
        source_count = int(
            connection.execute("SELECT COUNT(*) FROM turn_sources").fetchone()[0]
        )
    assert result.replayed is False
    assert result.person_ids == ()
    assert result.relationship_ids == ()
    assert result.event_ids == ()
    assert source_count == 0
    assert store.database_revision() == 0


def test_uncertain_owner_wording_does_not_update_owner_profile(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "我叫小雨，但也许我记错了。",
        session_id="session-1",
        turn_id="uncertain-owner",
        engine_kind="custom_api",
        observed_at=NOW,
    )

    store.ingest(batch)

    assert store.owner_profile().display_name is None


def test_relation_context_keeps_same_named_people_distinct_across_turns(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    extractor = HeuristicMemoryExtractor()

    colleague = extractor.extract_user_transcript(
        "我的同事叫张三。",
        session_id="session-1",
        turn_id="colleague",
        engine_kind="custom_api",
        observed_at=NOW,
    )
    neighbor = extractor.extract_user_transcript(
        "我的邻居叫张三。",
        session_id="session-1",
        turn_id="neighbor",
        engine_kind="custom_api",
        observed_at=NOW + timedelta(minutes=1),
    )

    first = store.ingest(colleague)
    second = store.ingest(neighbor)
    replayed_first = store.ingest(colleague)
    replayed_second = store.ingest(neighbor)

    assert first.person_ids != second.person_ids
    assert replayed_first.person_ids == first.person_ids
    assert replayed_first.relationship_ids == first.relationship_ids
    assert replayed_second.person_ids == second.person_ids
    assert replayed_second.relationship_ids == second.relationship_ids
    assert store.count_records(kind=MemoryKind.PERSON) == 2
    assert store.count_records(kind=MemoryKind.RELATIONSHIP) == 2


def test_first_named_event_participant_is_created_and_ingested_atomically(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "明天和张三开会。",
        session_id="session-1",
        turn_id="first-meeting",
        engine_kind="custom_api",
        observed_at=NOW,
    )

    result = store.ingest(batch)
    replayed = store.ingest(batch)

    assert len(result.person_ids) == 1
    assert len(result.event_ids) == 1
    assert replayed.replayed is True
    assert replayed.person_ids == result.person_ids
    assert replayed.event_ids == result.event_ids
    assert store.count_records(kind=MemoryKind.PERSON) == 1
    assert store.count_records(kind=MemoryKind.EVENT) == 1


def test_contextless_event_reuses_the_only_existing_same_named_person(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    extractor = HeuristicMemoryExtractor()
    relationship = extractor.extract_user_transcript(
        "张三是我的同事。",
        session_id="session-1",
        turn_id="known-colleague",
        engine_kind="custom_api",
        observed_at=NOW,
    )
    event = extractor.extract_user_transcript(
        "明天和张三开会。",
        session_id="session-1",
        turn_id="meeting-known-colleague",
        engine_kind="custom_api",
        observed_at=NOW + timedelta(minutes=1),
    )

    known = store.ingest(relationship)
    meeting = store.ingest(event)

    with open_database(store.database, read_only=True) as connection:
        linked_people = tuple(
            str(row[0])
            for row in connection.execute(
                """
                SELECT person_memory_id FROM event_participants
                WHERE event_memory_id = ? AND is_owner = 0
                """,
                (meeting.event_ids[0],),
            )
        )
    assert meeting.person_ids == known.person_ids
    assert store.count_records(kind=MemoryKind.PERSON) == 1
    assert linked_people == known.person_ids


def test_explicit_relation_enriches_the_only_contextless_same_named_person(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    extractor = HeuristicMemoryExtractor()
    event = extractor.extract_user_transcript(
        "明天和张三开会。",
        session_id="session-1",
        turn_id="meeting-first",
        engine_kind="custom_api",
        observed_at=NOW,
    )
    relationship = extractor.extract_user_transcript(
        "张三是我的同事。",
        session_id="session-1",
        turn_id="relation-second",
        engine_kind="custom_api",
        observed_at=NOW + timedelta(minutes=1),
    )

    meeting = store.ingest(event)
    related = store.ingest(relationship)
    replayed_meeting = store.ingest(event)
    replayed_relation = store.ingest(relationship)

    with open_database(store.database, read_only=True) as connection:
        person_row = connection.execute(
            """
            SELECT p.notes, r.target_person_id
            FROM people AS p
            JOIN relationships AS r ON r.target_person_id = p.memory_id
            WHERE p.memory_id = ?
            """,
            (meeting.person_ids[0],),
        ).fetchone()
    assert related.person_ids == meeting.person_ids
    assert replayed_meeting.person_ids == meeting.person_ids
    assert replayed_relation.person_ids == meeting.person_ids
    assert store.count_records(kind=MemoryKind.PERSON) == 1
    assert person_row is not None
    assert person_row["notes"] == "同事"
    assert person_row["target_person_id"] == meeting.person_ids[0]


def test_contextless_event_does_not_bind_or_create_when_same_name_is_ambiguous(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    extractor = HeuristicMemoryExtractor()
    for turn_id, text in (
        ("colleague", "我的同事叫张三。"),
        ("neighbor", "我的邻居叫张三。"),
    ):
        store.ingest(
            extractor.extract_user_transcript(
                text,
                session_id="session-1",
                turn_id=turn_id,
                engine_kind="custom_api",
                observed_at=NOW,
            )
        )
    event = extractor.extract_user_transcript(
        "明天和张三开会。",
        session_id="session-1",
        turn_id="ambiguous-meeting",
        engine_kind="custom_api",
        observed_at=NOW + timedelta(minutes=1),
    )

    meeting = store.ingest(event)

    with open_database(store.database, read_only=True) as connection:
        non_owner_links = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM event_participants
                WHERE event_memory_id = ? AND is_owner = 0
                """,
                (meeting.event_ids[0],),
            ).fetchone()[0]
        )
    assert meeting.person_ids == ()
    assert store.count_records(kind=MemoryKind.PERSON) == 2
    assert len(meeting.event_ids) == 1
    assert non_owner_links == 0


def test_normalization_does_not_force_same_name_people_to_merge(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.ingest(
        _batch(
            turn_id="turn-1",
            digest="1" * 64,
            people=(PersonCandidate(name="Ａlice  Smith", notes="同事"),),
        )
    )
    second = store.ingest(
        _batch(
            turn_id="turn-2",
            digest="2" * 64,
            people=(PersonCandidate(name="alice smith", notes="邻居"),),
        )
    )

    assert first.person_ids != second.person_ids
    assert store.count_records(kind=MemoryKind.PERSON) == 2


def test_confirm_uses_optimistic_revision_and_clears_expiry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inserted = store.ingest(
        _batch(
            people=(
                PersonCandidate(
                    name="可能叫阿青",
                    confidence=0.4,
                    state=MemoryState.CANDIDATE,
                ),
            )
        )
    )
    memory_id = inserted.person_ids[0]
    before = store.get(memory_id)
    assert before is not None and before.expires_at is not None

    confirmed = store.confirm(memory_id, expected_revision=before.revision, now=NOW)

    assert confirmed.state is MemoryState.CONFIRMED
    assert confirmed.expires_at is None
    assert confirmed.revision == before.revision + 1
    with pytest.raises(StaleRevisionError):
        store.confirm(memory_id, expected_revision=before.revision, now=NOW)


def test_forget_person_removes_dependent_relationships_and_detaches_events(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    inserted = store.ingest(
        _batch(
            people=(PersonCandidate(name="张三"),),
            relationships=(
                RelationshipCandidate(
                    source_name=None,
                    target_name="张三",
                    relation_type="同事",
                ),
            ),
            events=(
                EventCandidate(
                    title="和张三开会",
                    participants=(EventParticipantCandidate(name="张三"),),
                ),
            ),
        )
    )
    person_id = inserted.person_ids[0]
    relationship_id = inserted.relationship_ids[0]
    event_id = inserted.event_ids[0]
    person = store.get(person_id)
    assert person is not None

    result = store.forget(
        person_id,
        expected_revision=person.revision,
        reason="user_deleted_in_settings",
        now=NOW,
    )

    assert result.deleted_ids == (relationship_id, person_id)
    assert result.database_revision == 2
    assert store.get(person_id) is None
    assert store.get(relationship_id) is None
    assert store.get(event_id) is not None
    with open_database(store.database, read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM event_participants WHERE event_memory_id = ?",
            (event_id,),
        ).fetchone()[0] == 0
        tombstones = tuple(
            connection.execute(
                "SELECT kind, reason_code FROM tombstones ORDER BY kind"
            )
        )
    assert [tuple(row) for row in tombstones] == [
        ("person", "user_deleted_in_settings"),
        ("relationship", "user_deleted_in_settings"),
    ]


def test_forgotten_person_relationship_and_event_are_not_resurrected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = store.ingest(
        _batch(
            turn_id="turn-original",
            digest="1" * 64,
            people=(PersonCandidate(name="张三"),),
            relationships=(
                RelationshipCandidate(
                    source_name=None,
                    target_name="张三",
                    relation_type="同事",
                ),
            ),
            events=(EventCandidate(title="不再记住的活动"),),
        )
    )
    relationship = store.get(original.relationship_ids[0])
    event = store.get(original.event_ids[0])
    assert relationship is not None and event is not None
    store.forget(
        relationship.memory_id,
        expected_revision=relationship.revision,
        reason="user_deleted_in_settings",
        now=NOW,
    )
    store.forget(
        event.memory_id,
        expected_revision=event.revision,
        reason="user_deleted_in_settings",
        now=NOW,
    )

    replay_relations_and_events = store.ingest(
        _batch(
            turn_id="turn-reextracted",
            digest="2" * 64,
            relationships=(
                RelationshipCandidate(
                    source_name=None,
                    target_name="张三",
                    relation_type="同事",
                ),
            ),
            events=(EventCandidate(title="不再记住的活动"),),
        )
    )

    assert replay_relations_and_events.relationship_ids == ()
    assert replay_relations_and_events.event_ids == ()
    person = store.get(original.person_ids[0])
    assert person is not None
    store.forget(
        person.memory_id,
        expected_revision=person.revision,
        reason="user_deleted_in_settings",
        now=NOW,
    )

    replay_person = store.ingest(
        _batch(
            turn_id="turn-person-reextracted",
            digest="3" * 64,
            people=(PersonCandidate(name="张三"),),
        )
    )

    assert replay_person.person_ids == ()
    assert store.count_records() == 0


def test_follow_up_claim_is_one_shot_and_optimistic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.ingest(
        _batch(
            events=(
                EventCandidate(
                    title="面试",
                    starts_at=NOW - timedelta(hours=1),
                    status=EventStatus.PLANNED,
                    follow_up_after=NOW,
                ),
            )
        )
    )
    event_id = result.event_ids[0]
    before = store.get(event_id)
    assert before is not None
    assert before.follow_up_state is FollowUpState.ELIGIBLE

    assert store.claim_follow_up(
        event_id,
        expected_revision=before.revision,
        asked_at=NOW,
    )
    assert not store.claim_follow_up(
        event_id,
        expected_revision=before.revision,
        asked_at=NOW,
    )
    assert store.get(event_id).follow_up_state is FollowUpState.ASKED  # type: ignore[union-attr]


def test_deadline_claim_fails_immediately_when_store_mutation_lock_is_busy(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    result = store.ingest(
        _batch(
            events=(
                EventCandidate(
                    title="锁占用事件",
                    starts_at=NOW - timedelta(hours=1),
                    status=EventStatus.PLANNED,
                    follow_up_after=NOW,
                ),
            )
        )
    )
    event_id = result.event_ids[0]
    before = store.get(event_id)
    assert before is not None

    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_mutation_lock() -> None:
        with store._mutation_lock:  # type: ignore[attr-defined]
            lock_held.set()
            release_lock.wait(timeout=1.0)

    holder = threading.Thread(target=hold_mutation_lock)
    holder.start()
    assert lock_held.wait(timeout=0.5)
    try:
        started_at = time.monotonic()
        claimed = store.claim_follow_up_before(
            event_id,
            expected_revision=before.revision,
            asked_at=NOW,
            deadline_monotonic=started_at + 0.1,
        )
        elapsed = time.monotonic() - started_at
    finally:
        release_lock.set()
        holder.join(timeout=0.5)

    assert claimed is False
    assert elapsed < 0.03
    assert store.get(event_id).follow_up_state is FollowUpState.ELIGIBLE  # type: ignore[union-attr]


def test_deadline_claim_commits_once_when_budget_and_locks_are_available(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    event_id = store.ingest(
        _batch(
            events=(
                EventCandidate(
                    title="可跟进事件",
                    starts_at=NOW - timedelta(hours=1),
                    status=EventStatus.PLANNED,
                    follow_up_after=NOW,
                ),
            )
        )
    ).event_ids[0]
    before = store.get(event_id)
    assert before is not None

    claimed = store.claim_follow_up_before(
        event_id,
        expected_revision=before.revision,
        asked_at=NOW,
        deadline_monotonic=time.monotonic() + 0.2,
    )
    replayed = store.claim_follow_up_before(
        event_id,
        expected_revision=before.revision,
        asked_at=NOW,
        deadline_monotonic=time.monotonic() + 0.2,
    )

    assert claimed is True
    assert replayed is False
    assert store.get(event_id).follow_up_state is FollowUpState.ASKED  # type: ignore[union-attr]


def test_deadline_claim_fails_without_late_commit_when_sqlite_writer_is_busy(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    event_id = store.ingest(
        _batch(
            events=(
                EventCandidate(
                    title="数据库锁事件",
                    starts_at=NOW - timedelta(hours=1),
                    status=EventStatus.PLANNED,
                    follow_up_after=NOW,
                ),
            )
        )
    ).event_ids[0]
    before = store.get(event_id)
    assert before is not None

    with open_database(store.database) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        started_at = time.monotonic()
        claimed = store.claim_follow_up_before(
            event_id,
            expected_revision=before.revision,
            asked_at=NOW,
            deadline_monotonic=started_at + 0.02,
        )
        elapsed = time.monotonic() - started_at
        assert claimed is False
        assert elapsed < 0.03
        assert store.get(event_id).follow_up_state is FollowUpState.ELIGIBLE  # type: ignore[union-attr]
    assert store.get(event_id).follow_up_state is FollowUpState.ELIGIBLE  # type: ignore[union-attr]


def test_reobserved_person_updates_display_and_adds_aliases(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.ingest(
        _batch(
            turn_id="turn-person-first",
            digest="4" * 64,
            people=(PersonCandidate(name="Ａlice"),),
        )
    )

    second = store.ingest(
        _batch(
            turn_id="turn-person-second",
            digest="5" * 64,
            people=(PersonCandidate(name="Alice", aliases=("小爱",)),),
        )
    )

    assert second.person_ids == first.person_ids
    record = store.get(first.person_ids[0])
    assert record is not None
    assert record.person_name == "Alice"
    with open_database(store.database, read_only=True) as connection:
        aliases = tuple(
            row[0]
            for row in connection.execute(
                "SELECT alias FROM person_aliases WHERE person_memory_id = ?",
                (record.memory_id,),
            )
        )
    assert aliases == ("小爱",)


def test_person_fingerprint_uses_final_alias_set_and_blocks_tombstone_replay(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = store.ingest(
        _batch(
            turn_id="turn-alias-first",
            digest="a" * 64,
            people=(
                PersonCandidate(name="张三", aliases=(" 老张 ", "", "老张")),
            ),
        )
    )
    second = store.ingest(
        _batch(
            turn_id="turn-alias-second",
            digest="b" * 64,
            people=(
                PersonCandidate(name="张三", aliases=(" 三哥 ", "三哥", "   ")),
            ),
        )
    )
    assert second.person_ids == first.person_ids
    person_id = first.person_ids[0]
    with open_database(store.database, read_only=True) as connection:
        aliases = tuple(
            row[0]
            for row in connection.execute(
                """
                SELECT normalized_alias FROM person_aliases
                WHERE person_memory_id = ? ORDER BY normalized_alias
                """,
                (person_id,),
            )
        )
    assert aliases == ("三哥", "老张")
    record = store.get(person_id)
    assert record is not None
    store.forget(
        person_id,
        expected_revision=record.revision,
        reason="user_deleted_in_settings",
        now=NOW,
    )

    replay = store.ingest(
        _batch(
            turn_id="turn-alias-replay",
            digest="c" * 64,
            people=(
                PersonCandidate(name="张三", aliases=("老张", "三哥", "老张")),
            ),
        )
    )

    assert replay.person_ids == ()
    assert store.count_records(kind=MemoryKind.PERSON) == 0


def test_reobserved_completed_event_updates_child_and_dismisses_eligible_followup(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    starts_at = NOW + timedelta(days=1)
    first = store.ingest(
        _batch(
            turn_id="turn-event-planned",
            digest="6" * 64,
            events=(
                EventCandidate(
                    title="项目发布",
                    summary="准备中",
                    starts_at=starts_at,
                    status=EventStatus.PLANNED,
                    follow_up_after=NOW,
                ),
            ),
        )
    )

    second = store.ingest(
        _batch(
            turn_id="turn-event-completed",
            digest="7" * 64,
            events=(
                EventCandidate(
                    title="项目发布",
                    summary="已经完成",
                    starts_at=starts_at,
                    status=EventStatus.COMPLETED,
                ),
            ),
        )
    )

    assert second.event_ids == first.event_ids
    record = store.get(first.event_ids[0])
    assert record is not None
    assert "已经完成" in record.summary
    assert record.event_status is EventStatus.COMPLETED
    assert record.follow_up_state is FollowUpState.DISMISSED


def test_reobserved_event_preserves_asked_followup_audit_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    starts_at = NOW + timedelta(days=1)
    event_id = store.ingest(
        _batch(
            turn_id="turn-event-asked",
            digest="8" * 64,
            events=(
                EventCandidate(
                    title="面试",
                    starts_at=starts_at,
                    status=EventStatus.PLANNED,
                    follow_up_after=NOW,
                ),
            ),
        )
    ).event_ids[0]
    before = store.get(event_id)
    assert before is not None
    assert store.claim_follow_up(
        event_id,
        expected_revision=before.revision,
        asked_at=NOW,
    )

    store.ingest(
        _batch(
            turn_id="turn-event-after-asked",
            digest="9" * 64,
            events=(
                EventCandidate(
                    title="面试",
                    starts_at=starts_at,
                    status=EventStatus.COMPLETED,
                ),
            ),
        )
    )

    record = store.get(event_id)
    assert record is not None
    assert record.event_status is EventStatus.COMPLETED
    assert record.follow_up_state is FollowUpState.ASKED


def test_clear_all_is_atomic_and_guarded_by_database_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ingest(_batch(people=(PersonCandidate(name="张三"),)))

    with pytest.raises(StaleRevisionError):
        store.clear_all(expected_database_revision=0, now=NOW)
    assert store.count_records() == 1

    result = store.clear_all(expected_database_revision=1, now=NOW)

    assert result.deleted_count == 1
    assert result.database_revision == 2
    assert store.count_records() == 0
    assert store.count_evidence() == 0
    assert store.owner_profile().display_name is None
