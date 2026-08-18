from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from avaturn_live_streamer.memory.admin import MemoryAdminService
from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    EventCandidate,
    EventStatus,
    ForgetResult,
    MemoryKind,
    MemoryState,
    PersonCandidate,
)
from avaturn_live_streamer.memory.schema import open_database
from avaturn_live_streamer.memory.sqlite_store import (
    SQLiteMemoryStore,
    StaleRevisionError,
)
from avaturn_live_streamer.memory.transfer import MemoryTransfer
from avaturn_live_streamer.memory.transfer import StaleImportPlanError


UTC = timezone.utc
NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


def _batch(
    label: str,
    *,
    people: tuple[PersonCandidate, ...] = (),
    events: tuple[EventCandidate, ...] = (),
) -> CandidateBatch:
    return CandidateBatch(
        session_id=f"session-{label}",
        turn_id=f"turn-{label}",
        engine_kind="custom_api",
        observed_at=NOW,
        transcript_sha256=hashlib.sha256(label.encode()).hexdigest(),
        people=tuple(
            person
            if person.evidence_excerpt
            else replace(person, evidence_excerpt=label)
            for person in people
        ),
        events=tuple(
            event
            if event.evidence_excerpt
            else replace(event, evidence_excerpt=label)
            for event in events
        ),
    )


def _components(
    root: Path,
    name: str,
) -> tuple[SQLiteMemoryStore, MemoryTransfer, MemoryAdminService]:
    database = root / name / "memory.sqlite3"
    store = SQLiteMemoryStore(database)
    store.initialize(owner_uuid=f"owner-{name}")
    transfer = MemoryTransfer(database, root / name / "backups", clock=lambda: NOW)
    admin = MemoryAdminService(store, transfer, clock=lambda: NOW)
    return store, transfer, admin


def test_stats_are_directly_json_serializable_and_match_management_counts(
    tmp_path: Path,
) -> None:
    store, _, admin = _components(tmp_path, "stats")
    store.ingest(
        _batch(
            "stats",
            people=(
                PersonCandidate(name="张三"),
                PersonCandidate(
                    name="可能叫小青",
                    confidence=0.4,
                    state=MemoryState.CANDIDATE,
                ),
            ),
            events=(
                EventCandidate(
                    title="明天复诊",
                    starts_at=NOW + timedelta(days=1),
                    follow_up_after=NOW,
                    status=EventStatus.PLANNED,
                ),
            ),
        )
    )

    result = asyncio.run(admin.stats())

    assert result["db_revision"] == 1
    assert result["counts"] == {
        "total": 3,
        "confirmed": 2,
        "candidates": 1,
        "low_confidence_candidates": 1,
        "people": 2,
        "relationships": 0,
        "events": 1,
        "pending_followups": 1,
    }
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    assert admin.enabled is True
    assert admin.available is True
    assert admin.degraded_reason is None


def test_list_records_filters_searches_and_uses_an_opaque_cursor(
    tmp_path: Path,
) -> None:
    store, _, admin = _components(tmp_path, "list")
    inserted = store.ingest(
        _batch(
            "list",
            people=(
                PersonCandidate(name="张三", notes="同事"),
                PersonCandidate(name="张四", notes="邻居"),
                PersonCandidate(name="李五", notes="同学"),
            ),
        )
    )

    first = asyncio.run(
        admin.list_records(
            kind="person",
            state=MemoryState.CONFIRMED,
            q="张",
            cursor=None,
            limit=1,
        )
    )
    second = asyncio.run(
        admin.list_records(
            kind=MemoryKind.PERSON,
            state="confirmed",
            q="张",
            cursor=first["next_cursor"],
            limit=1,
        )
    )

    ids = {first["items"][0]["id"], second["items"][0]["id"]}
    assert ids <= set(inserted.person_ids)
    assert len(ids) == 2
    assert first["next_cursor"]
    assert first["next_cursor"] not in ids
    assert second["next_cursor"] is None
    assert all(item["kind"] == "person" for item in first["items"] + second["items"])
    assert all("张" in item["summary"] for item in first["items"] + second["items"])
    assert json.dumps(first, ensure_ascii=False)

    with pytest.raises(ValueError, match="limit"):
        asyncio.run(admin.list_records(limit=101))
    with pytest.raises(ValueError, match="cursor"):
        asyncio.run(admin.list_records(cursor="not-a-valid-cursor"))


def test_get_record_returns_a_serializable_detail_or_none(tmp_path: Path) -> None:
    store, _, admin = _components(tmp_path, "detail")
    memory_id = store.ingest(
        _batch("detail", people=(PersonCandidate(name="王老师"),))
    ).person_ids[0]

    record = asyncio.run(admin.get_record(memory_id))

    assert record is not None
    assert record["id"] == memory_id
    assert record["person_name"] == "王老师"
    assert record["content"] == record["summary"]
    assert record["created_at"].endswith("+00:00")
    assert json.dumps(record, ensure_ascii=False)
    assert asyncio.run(admin.get_record("missing")) is None


def test_forget_is_optimistic_and_writes_a_tombstone(tmp_path: Path) -> None:
    store, _, admin = _components(tmp_path, "forget")
    memory_id = store.ingest(
        _batch("forget", people=(PersonCandidate(name="不再记住"),))
    ).person_ids[0]
    before = store.get(memory_id)
    assert before is not None
    with open_database(store.database, read_only=True) as connection:
        fingerprint = connection.execute(
            "SELECT content_fingerprint FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]

    result = asyncio.run(
        admin.forget(
            memory_id,
            expected_revision=before.revision,
            reason="user_deleted_in_settings",
        )
    )

    assert result == {
        "id": memory_id,
        "kind": "person",
        "db_revision": 2,
        "deleted_ids": [memory_id],
    }
    assert store.get(memory_id) is None
    with open_database(store.database, read_only=True) as connection:
        tombstone = connection.execute(
            "SELECT kind, content_fingerprint, reason_code FROM tombstones"
        ).fetchone()
    assert tuple(tombstone) == (
        "person",
        fingerprint,
        "user_deleted_in_settings",
    )

    other_id = store.ingest(
        _batch("forget-stale", people=(PersonCandidate(name="保留"),))
    ).person_ids[0]
    with pytest.raises(StaleRevisionError):
        asyncio.run(
            admin.forget(
                other_id,
                expected_revision=99,
                reason="user_deleted_in_settings",
            )
        )
    assert store.get(other_id) is not None


def test_clear_all_creates_a_pre_clear_backup_by_default(tmp_path: Path) -> None:
    store, _, admin = _components(tmp_path, "clear")
    store.ingest(_batch("clear", people=(PersonCandidate(name="备份中的人物"),)))

    result = asyncio.run(
        admin.clear_all(expected_db_revision=store.database_revision())
    )

    backup_path = Path(result["backup_path"])
    assert result["deleted"] == 1
    assert result["db_revision"] == 2
    assert backup_path.is_file()
    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert store.count_records() == 0

    store.ingest(_batch("clear-no-backup", people=(PersonCandidate(name="临时"),)))
    without_backup = asyncio.run(
        admin.clear_all(
            expected_db_revision=store.database_revision(),
            backup=False,
        )
    )
    assert without_backup["backup_path"] is None


def test_clear_all_uses_the_live_memory_service_clear_barrier(tmp_path: Path) -> None:
    class BarrierProvider:
        enabled = True
        available = True
        degraded_reason: str | None = None

        def __init__(self) -> None:
            self.cutoffs: list[datetime] = []

        async def run_clear_barrier(self, operation: object, *, cutoff: datetime) -> object:
            self.cutoffs.append(cutoff)
            return await asyncio.to_thread(operation)

    store, transfer, _ = _components(tmp_path, "clear-barrier-admin")
    store.ingest(_batch("clear-barrier", people=(PersonCandidate(name="待清空"),)))
    provider = BarrierProvider()
    admin = MemoryAdminService(
        store,
        transfer,
        clock=lambda: NOW,
        status_provider=provider,
    )

    result = asyncio.run(
        admin.clear_all(expected_db_revision=store.database_revision())
    )

    assert result["deleted"] == 1
    assert provider.cutoffs == [NOW]


def test_export_preview_and_apply_round_trip_through_serializable_admin_dicts(
    tmp_path: Path,
) -> None:
    source, _, source_admin = _components(tmp_path, "source")
    source.ingest(_batch("transfer", people=(PersonCandidate(name="导入人物"),)))
    bundle = tmp_path / "temporary" / "memory.json"

    exported = asyncio.run(source_admin.export_json(bundle))

    assert exported["destination"] == str(bundle.resolve())
    assert exported["records"] == 1
    assert json.dumps(exported)

    target, _, target_admin = _components(tmp_path, "target")
    preview = asyncio.run(target_admin.preview_import(bundle))

    assert preview["db_revision"] == 0
    assert preview["counts"] == {
        "insertable": 1,
        "duplicates": 0,
        "conflicts": 0,
        "invalid": 0,
    }
    assert target.count_records() == 0
    assert json.dumps(preview)
    bundle.unlink()

    applied = asyncio.run(
        target_admin.apply_import(
            {
                "preview_token": preview["preview_token"],
                "expected_db_revision": preview["db_revision"],
            }
        )
    )

    assert applied["inserted"] == 1
    assert applied["db_revision"] == 1
    assert Path(applied["backup_path"]).is_file()
    assert target.count_records() == 1
    assert json.dumps(applied)
    assert not tuple((tmp_path / "target" / "pending-imports").glob("*"))


def test_first_preview_only_cleans_expired_strictly_named_orphans(
    tmp_path: Path,
) -> None:
    source, _, source_admin = _components(tmp_path, "orphan-source")
    source.ingest(_batch("orphan", people=(PersonCandidate(name="导入人物"),)))
    bundle = tmp_path / "orphan-bundle.json"
    asyncio.run(source_admin.export_json(bundle))
    _, _, target_admin = _components(tmp_path, "orphan-target")
    pending = tmp_path / "orphan-target" / "pending-imports"
    pending.mkdir(parents=True)
    expired = pending / f"preview-{'a' * 32}.json"
    recent = pending / f"preview-{'b' * 32}.json"
    invalid_name = pending / "preview-not-a-managed-name.json"
    for path in (expired, recent, invalid_name):
        path.write_text("private import payload", encoding="utf-8")
    old_timestamp = (NOW - timedelta(minutes=11)).timestamp()
    os.utime(expired, (old_timestamp, old_timestamp))
    os.utime(invalid_name, (old_timestamp, old_timestamp))
    recent_timestamp = (NOW - timedelta(minutes=1)).timestamp()
    os.utime(recent, (recent_timestamp, recent_timestamp))

    asyncio.run(target_admin.preview_import(bundle))

    assert not expired.exists()
    assert recent.exists()
    assert invalid_name.exists()


def test_start_removes_only_managed_leftovers_once_and_close_cleans_them(
    tmp_path: Path,
) -> None:
    _, _, admin = _components(tmp_path, "startup-cleanup")
    pending = tmp_path / "startup-cleanup" / "pending-imports"
    pending.mkdir(parents=True)
    staged = pending / f"preview-{'a' * 32}.json"
    temporary = pending / f".preview-{'b' * 32}.json.{'c' * 32}.tmp"
    unmanaged = pending / "preview-not-managed.json"
    managed_directory = pending / f"preview-{'d' * 32}.json"
    for path in (staged, temporary, unmanaged):
        path.write_text("private import payload", encoding="utf-8")
    managed_directory.mkdir()

    async def exercise() -> None:
        await admin.start()
        assert not staged.exists()
        assert not temporary.exists()
        assert unmanaged.exists()
        assert managed_directory.is_dir()

        staged.write_text("current process payload", encoding="utf-8")
        await admin.start()
        assert staged.exists()

        await admin.close()
        assert not staged.exists()
        assert unmanaged.exists()
        assert managed_directory.is_dir()

    asyncio.run(exercise())


def test_start_and_close_orphan_cleanup_errors_are_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, admin = _components(tmp_path, "startup-cleanup-error")
    pending = tmp_path / "startup-cleanup-error" / "pending-imports"
    pending.mkdir(parents=True)
    staged = pending / f"preview-{'e' * 32}.json"
    staged.write_text("temporarily locked", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_managed_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == staged:
            raise OSError("antivirus temporarily holds staged file")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_managed_unlink)

    async def exercise() -> None:
        await admin.start()
        assert staged.exists()
        await admin.close()
        assert staged.exists()

    asyncio.run(exercise())


def test_owner_only_import_is_actionable_and_reports_profile_fields(
    tmp_path: Path,
) -> None:
    source, _, source_admin = _components(tmp_path, "owner-admin-source")
    with open_database(source.database) as connection:
        connection.execute(
            "UPDATE owner_profile SET display_name = ?, profile = ? WHERE id = 1",
            ("小雨", "住在上海"),
        )
    bundle = tmp_path / "owner-only.json"
    asyncio.run(source_admin.export_json(bundle))
    target, _, target_admin = _components(tmp_path, "owner-admin-target")

    preview = asyncio.run(target_admin.preview_import(bundle))

    assert preview["counts"]["insertable"] == 2
    assert preview["owner_profile"]["updates"] == ["display_name", "profile"]
    applied = asyncio.run(
        target_admin.apply_import(
            {
                "preview_token": preview["preview_token"],
                "expected_db_revision": preview["db_revision"],
            }
        )
    )
    assert applied["inserted"] == 2
    assert applied["inserted_memories"] == 0
    assert applied["owner_profile_updated"] == ["display_name", "profile"]
    assert target.owner_profile().display_name == "小雨"


def test_pending_imports_are_bounded_expire_on_stats_and_close_cleans_all(
    tmp_path: Path,
) -> None:
    source, _, source_admin = _components(tmp_path, "pending-source")
    source.ingest(_batch("pending", people=(PersonCandidate(name="待导入"),)))
    bundle = tmp_path / "pending.json"
    asyncio.run(source_admin.export_json(bundle))
    target = SQLiteMemoryStore(tmp_path / "pending-target" / "memory.sqlite3")
    target.initialize(owner_uuid="owner-pending-target")
    now = [NOW]
    transfer = MemoryTransfer(
        target.database,
        tmp_path / "pending-target" / "backups",
        clock=lambda: now[0],
    )
    admin = MemoryAdminService(
        target,
        transfer,
        clock=lambda: now[0],
        max_pending_imports=2,
    )

    async def exercise() -> tuple[str, str, str]:
        first = await admin.preview_import(bundle)
        second = await admin.preview_import(bundle)
        third = await admin.preview_import(bundle)
        with pytest.raises(StaleImportPlanError):
            await admin.apply_import(
                {
                    "preview_token": first["preview_token"],
                    "expected_db_revision": first["db_revision"],
                }
            )
        pending = tmp_path / "pending-target" / "pending-imports"
        assert len(tuple(pending.glob("preview-*.json"))) == 2
        now[0] = NOW + timedelta(minutes=11)
        await admin.stats()
        assert not tuple(pending.glob("preview-*.json"))
        with pytest.raises(StaleImportPlanError):
            await admin.apply_import(
                {
                    "preview_token": second["preview_token"],
                    "expected_db_revision": second["db_revision"],
                }
            )
        fresh = await admin.preview_import(bundle)
        await admin.close()
        assert not tuple(pending.glob("preview-*.json"))
        with pytest.raises(RuntimeError, match="closed"):
            await admin.preview_import(bundle)
        return (
            first["preview_token"],
            third["preview_token"],
            fresh["preview_token"],
        )

    tokens = asyncio.run(exercise())

    assert len(set(tokens)) == 3


def test_successful_import_is_not_masked_when_staged_file_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, source_admin = _components(tmp_path, "unlink-source")
    source.ingest(_batch("unlink", people=(PersonCandidate(name="仍应导入"),)))
    bundle = tmp_path / "unlink.json"
    asyncio.run(source_admin.export_json(bundle))
    target, _, target_admin = _components(tmp_path, "unlink-target")
    preview = asyncio.run(target_admin.preview_import(bundle))
    original_unlink = Path.unlink

    def fail_staged_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith("preview-"):
            raise OSError("antivirus temporarily holds staged file")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_staged_unlink)

    result = asyncio.run(
        target_admin.apply_import(
            {
                "preview_token": preview["preview_token"],
                "expected_db_revision": preview["db_revision"],
            }
        )
    )

    assert result["inserted"] == 1
    assert target.count_records() == 1


def test_status_properties_follow_the_live_memory_service_provider(
    tmp_path: Path,
) -> None:
    class StatusProvider:
        enabled = True
        available = True
        degraded_reason: str | None = None

    store, transfer, _ = _components(tmp_path, "status-provider")
    status = StatusProvider()
    admin = MemoryAdminService(store, transfer, status_provider=status)

    assert admin.enabled is True
    assert admin.available is True
    assert admin.degraded_reason is None

    status.available = False
    status.degraded_reason = "database writer failed"

    assert admin.enabled is True
    assert admin.available is False
    assert admin.degraded_reason == "database writer failed"


def test_sqlite_mutations_run_off_loop_and_share_one_async_lock(tmp_path: Path) -> None:
    class ProbeStore:
        def __init__(self) -> None:
            self.database = tmp_path / "probe.sqlite3"
            self.thread_ids: list[int] = []
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def forget(
            self,
            memory_id: str,
            *,
            expected_revision: int,
            reason: str,
            now: datetime,
        ) -> ForgetResult:
            assert reason and now.utcoffset() is not None
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self.thread_ids.append(threading.get_ident())
                time.sleep(0.02)
                return ForgetResult(
                    memory_id=memory_id,
                    kind=MemoryKind.PERSON,
                    database_revision=expected_revision + 1,
                )
            finally:
                with self.guard:
                    self.active -= 1

    class ProbeTransfer:
        backups = tmp_path / "backups"

    async def exercise() -> tuple[ProbeStore, int]:
        loop_thread_id = threading.get_ident()
        store = ProbeStore()
        admin = MemoryAdminService(store, ProbeTransfer(), clock=lambda: NOW)
        await asyncio.gather(
            admin.forget("one", expected_revision=1, reason="user"),
            admin.forget("two", expected_revision=2, reason="user"),
        )
        return store, loop_thread_id

    probe, loop_thread_id = asyncio.run(exercise())

    assert probe.max_active == 1
    assert probe.thread_ids
    assert all(thread_id != loop_thread_id for thread_id in probe.thread_ids)
