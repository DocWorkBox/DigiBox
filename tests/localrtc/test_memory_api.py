from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from avaturn_live_streamer.memory.admin import MemoryAdminService
from avaturn_live_streamer.memory.models import CandidateBatch, PersonCandidate
from avaturn_live_streamer.memory.sqlite_store import (
    SQLiteMemoryStore,
    StaleRevisionError,
)
from avaturn_live_streamer.memory.transfer import (
    InvalidTransferError,
    MemoryTransfer,
    StaleImportPlanError,
)


class FakeMemoryAdmin:
    def __init__(self) -> None:
        self.enabled = True
        self.available = True
        self.degraded_reason: str | None = None
        self.calls: list[tuple[str, object]] = []
        self.errors: dict[str, Exception] = {}
        self.stats_payload: object = {
            "counts": {"confirmed": 3, "candidates": 2, "pending_followups": 1},
            "db_revision": 7,
        }
        self.records_payload: object = {
            "items": [{"id": "memory-1", "revision": 4, "kind": "event"}],
            "next_cursor": "cursor-2",
        }
        self.record_payload: object | None = {
            "id": "memory-1",
            "revision": 4,
            "kind": "event",
        }
        self.delete_payload: object | None = {
            "deleted": True,
            "id": "memory-1",
            "db_revision": 8,
        }
        self.clear_payload: object = {"deleted": 5, "db_revision": 8}
        self.export_payload = b'{"format":"digibox-person-event-memory"}'
        self.preview_payload: object = {
            "preview_token": "preview-1",
            "db_revision": 7,
            "counts": {
                "insertable": 2,
                "duplicates": 1,
                "conflicts": 1,
                "invalid": 0,
            },
        }
        self.apply_payload: object = {
            "inserted": 2,
            "duplicates": 1,
            "conflicts": 1,
            "db_revision": 8,
        }
        self.export_thread_id: int | None = None
        self.preview_thread_id: int | None = None

    def _raise_if_configured(self, method: str) -> None:
        error = self.errors.get(method)
        if error is not None:
            raise error

    async def stats(self) -> object:
        self.calls.append(("stats", None))
        self._raise_if_configured("stats")
        return self.stats_payload

    async def list_records(
        self,
        *,
        kind: str | None,
        state: str | None,
        q: str | None,
        cursor: str | None,
        limit: int,
    ) -> object:
        arguments = {
            "kind": kind,
            "state": state,
            "q": q,
            "cursor": cursor,
            "limit": limit,
        }
        self.calls.append(("list_records", arguments))
        self._raise_if_configured("list_records")
        return self.records_payload

    async def get_record(self, memory_id: str) -> object | None:
        self.calls.append(("get_record", memory_id))
        self._raise_if_configured("get_record")
        return self.record_payload

    async def forget(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> object | None:
        self.calls.append(
            (
                "forget",
                {
                    "memory_id": memory_id,
                    "expected_revision": expected_revision,
                    "reason": reason,
                },
            )
        )
        self._raise_if_configured("forget")
        return self.delete_payload

    async def clear_all(
        self,
        *,
        expected_db_revision: int,
        backup: bool = True,
    ) -> object:
        self.calls.append(
            (
                "clear_all",
                {"expected_db_revision": expected_db_revision, "backup": backup},
            )
        )
        self._raise_if_configured("clear_all")
        return self.clear_payload

    async def export_json(self, destination: Path) -> object:
        self.export_thread_id = threading.get_ident()
        self.calls.append(("export_json", destination.name))
        self._raise_if_configured("export_json")
        destination.write_bytes(self.export_payload)
        return {"filename": destination.name}

    async def preview_import(
        self,
        source: Path,
    ) -> object:
        self.preview_thread_id = threading.get_ident()
        self.calls.append(
            (
                "preview_import",
                {"payload": source.read_bytes(), "filename": source.name},
            )
        )
        self._raise_if_configured("preview_import")
        return self.preview_payload

    async def apply_import(
        self,
        plan: dict[str, object] | str,
    ) -> object:
        self.calls.append(("apply_import", plan))
        self._raise_if_configured("apply_import")
        return self.apply_payload


def _client(
    admin: Any,
    *,
    loopback: bool = True,
) -> tuple[TestClient, list[str]]:
    from avaturn_live_streamer.memory.api import create_memory_router

    checks: list[str] = []

    def require_loopback(request: Request) -> None:
        checks.append(request.url.path)
        if not loopback:
            raise HTTPException(status_code=403, detail="loopback only")

    app = FastAPI()
    app.include_router(create_memory_router(admin, require_loopback))
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ), checks


def test_stats_reports_availability_counts_and_revision() -> None:
    admin = FakeMemoryAdmin()
    client, checks = _client(admin)

    response = client.get("/memory/stats")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": True,
        "available": True,
        "degraded_reason": None,
        "counts": {"confirmed": 3, "candidates": 2, "pending_followups": 1},
        "db_revision": 7,
    }
    assert response.headers["cache-control"] == "no-store"
    assert checks == ["/memory/stats"]
    assert admin.calls == [("stats", None)]


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "memory.attacker.test"},
        {"Host": "localhost.evil"},
        {"Host": "127.0.0.1", "Origin": "https://memory.attacker.test"},
        {"Host": "127.0.0.1", "Origin": "null"},
    ],
)
def test_memory_routes_reject_untrusted_host_or_origin_before_loopback(
    headers: dict[str, str],
) -> None:
    admin = FakeMemoryAdmin()
    client, checks = _client(admin)

    response = client.get("/memory/stats", headers=headers)

    assert response.status_code == 403
    assert checks == []
    assert admin.calls == []


@pytest.mark.parametrize(
    ("host", "origin"),
    [
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("LOCALHOST:5173", "tauri://localhost"),
        ("[::1]:8000", "http://[::1]:8000"),
    ],
)
def test_memory_routes_allow_loopback_host_and_origin(
    host: str,
    origin: str,
) -> None:
    admin = FakeMemoryAdmin()
    client, checks = _client(admin)

    response = client.get(
        "/memory/stats",
        headers={"Host": host, "Origin": origin},
    )

    assert response.status_code == 200
    assert checks == ["/memory/stats"]
    assert admin.calls == [("stats", None)]


@pytest.mark.parametrize("mode", ["unavailable", "stats-error"])
def test_stats_remains_200_when_memory_is_degraded(mode: str) -> None:
    admin = FakeMemoryAdmin()
    if mode == "unavailable":
        admin.available = False
        admin.degraded_reason = "database unavailable"
    else:
        admin.errors["stats"] = RuntimeError("database locked")
    client, _checks = _client(admin)

    response = client.get("/memory/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["available"] is False
    assert payload["counts"] == {}
    assert payload["db_revision"] is None
    assert payload["degraded_reason"]
    if mode == "unavailable":
        assert admin.calls == []


def test_records_list_forwards_frozen_filters_and_cursor() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.get(
        "/memory/records",
        params={
            "kind": "event",
            "state": "candidate",
            "q": "Shanghai trip",
            "cursor": "cursor-1",
            "limit": 12,
        },
    )

    assert response.status_code == 200
    assert response.json() == admin.records_payload
    assert admin.calls == [
        (
            "list_records",
            {
                "kind": "event",
                "state": "candidate",
                "q": "Shanghai trip",
                "cursor": "cursor-1",
                "limit": 12,
            },
        )
    ]


def test_records_list_maps_a_tampered_cursor_to_bad_request() -> None:
    admin = FakeMemoryAdmin()
    admin.errors["list_records"] = ValueError("invalid cursor")
    client, _checks = _client(admin)

    response = client.get("/memory/records", params={"cursor": "tampered"})

    assert response.status_code == 400
    assert "cursor" in response.json()["detail"]


def test_record_detail_and_missing_mapping() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    found = client.get("/memory/records/memory-1")
    admin.record_payload = None
    missing = client.get("/memory/records/missing")

    assert found.status_code == 200
    assert found.json() == {"id": "memory-1", "revision": 4, "kind": "event"}
    assert missing.status_code == 404


def test_delete_uses_expected_revision_query_and_maps_stale_revision() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    deleted = client.delete(
        "/memory/records/memory-1",
        params={"expected_revision": 4},
    )
    admin.errors["forget"] = StaleRevisionError("memory revision changed")
    stale = client.delete(
        "/memory/records/memory-1",
        params={"expected_revision": 4},
    )

    assert deleted.status_code == 200
    assert deleted.json() == admin.delete_payload
    assert admin.calls[0] == (
        "forget",
        {
            "memory_id": "memory-1",
            "expected_revision": 4,
            "reason": "user_deleted",
        },
    )
    assert stale.status_code == 409
    assert "revision" in stale.json()["detail"]


def test_clear_requires_exact_confirmation_and_database_revision() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    rejected = client.post(
        "/memory/clear",
        json={"confirmation": "clear", "expected_db_revision": 7},
    )
    cleared = client.post(
        "/memory/clear",
        json={
            "confirmation": "清空全部本地记忆",
            "expected_db_revision": 7,
        },
    )

    assert rejected.status_code == 400
    assert cleared.status_code == 200
    assert cleared.json() == admin.clear_payload
    assert admin.calls == [
        ("clear_all", {"expected_db_revision": 7, "backup": True})
    ]


def test_export_is_a_json_attachment() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.get("/memory/export")

    assert response.status_code == 200
    assert response.content == admin.export_payload
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"] == (
        'attachment; filename="digibox-memory.json"'
    )
    assert response.headers["cache-control"] == "no-store"
    assert admin.calls == [("export_json", "digibox-memory.json")]


def test_export_reads_the_temporary_file_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)
    original_read_bytes = Path.read_bytes
    read_thread_ids: list[int] = []

    def observed_read_bytes(path: Path) -> bytes:
        if path.name == "digibox-memory.json":
            read_thread_ids.append(threading.get_ident())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observed_read_bytes)

    response = client.get("/memory/export")

    assert response.status_code == 200
    assert read_thread_ids
    assert admin.export_thread_id is not None
    assert all(thread_id != admin.export_thread_id for thread_id in read_thread_ids)


def test_import_preview_accepts_only_raw_json_and_forwards_optional_filename() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)
    payload = b'{"format":"digibox-person-event-memory","records":[]}'

    response = client.post(
        "/memory/import/preview",
        content=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-DigiBox-Filename": "my-memory.json",
        },
    )

    assert response.status_code == 200
    assert response.json() == admin.preview_payload
    assert admin.calls == [
        (
            "preview_import",
            {"payload": payload, "filename": "my-memory.json"},
        )
    ]


def test_import_writes_the_bounded_temporary_file_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)
    original_write_bytes = Path.write_bytes
    write_thread_ids: list[int] = []

    def observed_write_bytes(path: Path, payload: bytes) -> int:
        if path.name == "my-memory.json":
            write_thread_ids.append(threading.get_ident())
        return original_write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", observed_write_bytes)

    response = client.post(
        "/memory/import/preview",
        content=b'{"format":"digibox-person-event-memory","records":[]}',
        headers={
            "Content-Type": "application/json",
            "X-DigiBox-Filename": "my-memory.json",
        },
    )

    assert response.status_code == 200
    assert write_thread_ids
    assert admin.preview_thread_id is not None
    assert all(thread_id != admin.preview_thread_id for thread_id in write_thread_ids)


def test_import_preview_rejects_wrong_media_type_before_admin() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.post(
        "/memory/import/preview",
        content=b"{}",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    assert admin.calls == []


@pytest.mark.parametrize("content", [b"not-json", b"[]"])
def test_import_preview_delegates_json_validation_to_transfer(content: bytes) -> None:
    admin = FakeMemoryAdmin()
    admin.errors["preview_import"] = InvalidTransferError("invalid JSON document")
    client, _checks = _client(admin)

    response = client.post(
        "/memory/import/preview",
        content=content,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert admin.calls == [
        (
            "preview_import",
            {"payload": content, "filename": "digibox-memory-import.json"},
        )
    ]


@pytest.mark.parametrize("filename", ["bad:name.json", "CON.json"])
def test_import_preview_rejects_windows_unsafe_filenames(filename: str) -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.post(
        "/memory/import/preview",
        content=b"{}",
        headers={
            "Content-Type": "application/json",
            "X-DigiBox-Filename": filename,
        },
    )

    assert response.status_code == 400
    assert admin.calls == []


def test_import_preview_enforces_sixteen_mibibyte_limit_before_admin() -> None:
    from avaturn_live_streamer.memory.api import MAX_IMPORT_BYTES

    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.post(
        "/memory/import/preview",
        content=b'{"value":"' + (b"x" * MAX_IMPORT_BYTES) + b'"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert admin.calls == []


def test_import_errors_and_stale_apply_map_to_400_and_409() -> None:
    admin = FakeMemoryAdmin()
    admin.errors["preview_import"] = InvalidTransferError("unsupported format")
    client, _checks = _client(admin)

    invalid = client.post(
        "/memory/import/preview",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    admin.errors["apply_import"] = StaleImportPlanError("preview expired")
    stale = client.post(
        "/memory/import/apply",
        json={"preview_token": "preview-1", "expected_db_revision": 7},
    )

    assert invalid.status_code == 400
    assert stale.status_code == 409


def test_import_apply_forwards_preview_token_and_expected_database_revision() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.post(
        "/memory/import/apply",
        json={"preview_token": "preview-1", "expected_db_revision": 7},
    )

    assert response.status_code == 200
    assert response.json() == admin.apply_payload
    assert admin.calls == [
        (
            "apply_import",
            {"preview_token": "preview-1", "expected_db_revision": 7},
        )
    ]


def test_import_preview_survives_http_temp_cleanup_until_real_apply(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    source_root = tmp_path / "source"
    source_database = source_root / "memory.sqlite3"
    source_store = SQLiteMemoryStore(source_database)
    source_store.initialize(owner_uuid="source-owner")
    source_store.ingest(
        CandidateBatch(
            session_id="session-http-import",
            turn_id="turn-http-import",
            engine_kind="custom_api",
            observed_at=now,
            transcript_sha256=hashlib.sha256(b"http-import").hexdigest(),
            people=(
                PersonCandidate(
                    name="跨请求导入人物",
                    evidence_excerpt="跨请求导入人物",
                ),
            ),
        )
    )
    source_transfer = MemoryTransfer(
        source_database,
        source_root / "backups",
        clock=lambda: now,
    )
    source_admin = MemoryAdminService(source_store, source_transfer, clock=lambda: now)
    bundle = tmp_path / "memory-export.json"
    asyncio.run(source_admin.export_json(bundle))

    target_root = tmp_path / "target"
    target_database = target_root / "memory.sqlite3"
    target_store = SQLiteMemoryStore(target_database)
    target_store.initialize(owner_uuid="target-owner")
    target_transfer = MemoryTransfer(
        target_database,
        target_root / "backups",
        clock=lambda: now,
    )
    target_admin = MemoryAdminService(target_store, target_transfer, clock=lambda: now)
    client, _checks = _client(target_admin)

    preview = client.post(
        "/memory/import/preview",
        content=bundle.read_bytes(),
        headers={
            "Content-Type": "application/json",
            "X-DigiBox-Filename": "round-trip.json",
        },
    )
    assert preview.status_code == 200
    assert preview.json()["counts"]["insertable"] == 1

    applied = client.post(
        "/memory/import/apply",
        json={
            "preview_token": preview.json()["preview_token"],
            "expected_db_revision": preview.json()["db_revision"],
        },
    )

    assert applied.status_code == 200
    assert applied.json()["inserted"] == 1
    assert target_store.count_records() == 1
    assert list((target_root / "pending-imports").glob("*.json")) == []


RouteCase = tuple[str, str, dict[str, Any]]
ROUTE_CASES: tuple[RouteCase, ...] = (
    ("GET", "/memory/stats", {}),
    ("GET", "/memory/records", {}),
    ("GET", "/memory/records/memory-1", {}),
    ("DELETE", "/memory/records/memory-1?expected_revision=4", {}),
    (
        "POST",
        "/memory/clear",
        {
            "json": {
                "confirmation": "清空全部本地记忆",
                "expected_db_revision": 7,
            }
        },
    ),
    ("GET", "/memory/export", {}),
    (
        "POST",
        "/memory/import/preview",
        {"content": b"{}", "headers": {"Content-Type": "application/json"}},
    ),
    (
        "POST",
        "/memory/import/apply",
        {"json": {"preview_token": "preview-1", "expected_db_revision": 7}},
    ),
)


@pytest.mark.parametrize(("method", "path", "kwargs"), ROUTE_CASES)
def test_every_memory_route_requires_loopback(
    method: str,
    path: str,
    kwargs: dict[str, Any],
) -> None:
    admin = FakeMemoryAdmin()
    client, checks = _client(admin, loopback=False)

    response = client.request(method, path, **kwargs)

    assert response.status_code == 403
    assert checks == [path.split("?")[0]]
    assert admin.calls == []


@pytest.mark.parametrize(
    "path",
    ["/memory/clear", "/memory/import/apply"],
)
def test_loopback_guard_runs_before_json_body_parsing(path: str) -> None:
    admin = FakeMemoryAdmin()
    client, checks = _client(admin, loopback=False)

    response = client.post(
        path,
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 403
    assert checks == [path]
    assert admin.calls == []


@pytest.mark.parametrize(("method", "path", "kwargs"), ROUTE_CASES[1:])
def test_non_stats_routes_return_503_when_memory_is_unavailable(
    method: str,
    path: str,
    kwargs: dict[str, Any],
) -> None:
    admin = FakeMemoryAdmin()
    admin.available = False
    admin.degraded_reason = "memory database unavailable"
    client, _checks = _client(admin)

    response = client.request(method, path, **kwargs)

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    assert admin.calls == []


def test_request_validation_rejects_missing_or_invalid_revisions() -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    missing_delete_revision = client.delete("/memory/records/memory-1")
    invalid_clear_revision = client.post(
        "/memory/clear",
        json={
            "confirmation": "清空全部本地记忆",
            "expected_db_revision": -1,
        },
    )
    missing_apply_revision = client.post(
        "/memory/import/apply",
        json={"preview_token": "preview-1"},
    )

    assert missing_delete_revision.status_code == 422
    assert invalid_clear_revision.status_code == 422
    assert missing_apply_revision.status_code == 422
    assert admin.calls == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/memory/clear",
            {
                "confirmation": "清空全部本地记忆",
                "expected_db_revision": True,
            },
        ),
        (
            "/memory/import/apply",
            {"preview_token": "preview-1", "expected_db_revision": False},
        ),
    ],
)
def test_json_revision_fields_reject_booleans(
    path: str,
    payload: dict[str, object],
) -> None:
    admin = FakeMemoryAdmin()
    client, _checks = _client(admin)

    response = client.post(path, json=payload)

    assert response.status_code == 422
    assert admin.calls == []


def test_router_does_not_introduce_multipart_upload_dependency() -> None:
    api_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "avaturn_live_streamer"
        / "memory"
        / "api.py"
    )
    source = api_path.read_text(encoding="utf-8")

    assert "UploadFile" not in source
    assert "python-multipart" not in source
    assert "multipart/form-data" not in source
