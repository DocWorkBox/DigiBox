from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from avaturn_live_streamer.memory.models import (
    ClearResult,
    ForgetResult,
    MemoryKind,
    MemoryRecord,
    MemoryRecordPage,
    MemoryState,
    MemoryStats,
)
from avaturn_live_streamer.memory.sqlite_store import SQLiteMemoryStore
from avaturn_live_streamer.memory.transfer import (
    ExportResult,
    ImportPlan,
    ImportResult,
    InvalidTransferError,
    MemoryTransfer,
    StaleImportPlanError,
)


_MANAGED_IMPORT_NAME = re.compile(r"preview-[0-9a-f]{32}\.json\Z")
_MANAGED_IMPORT_TEMPORARY_NAME = re.compile(
    r"\.preview-[0-9a-f]{32}\.json\.[0-9a-f]{32}\.tmp\Z"
)


@dataclass(frozen=True, slots=True)
class _CachedImport:
    plan: ImportPlan
    source: Path


def _aware_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("admin clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _datetime_value(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _record_dict(record: MemoryRecord) -> dict[str, object]:
    return {
        "id": record.memory_id,
        "kind": record.kind.value,
        "state": record.state.value,
        "confidence": record.confidence,
        "retention_class": record.retention_class.value,
        "canonical_key": record.canonical_key,
        "revision": record.revision,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "last_seen_at": record.last_seen_at.isoformat(),
        "expires_at": _datetime_value(record.expires_at),
        "summary": record.summary,
        "content": record.summary,
        "starts_at": _datetime_value(record.starts_at),
        "event_status": (
            record.event_status.value if record.event_status is not None else None
        ),
        "follow_up_state": (
            record.follow_up_state.value
            if record.follow_up_state is not None
            else None
        ),
        "person_name": record.person_name,
        "title": record.title,
    }


class MemoryAdminService:
    """Async, JSON-safe facade for loopback memory management routes."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        transfer: MemoryTransfer,
        *,
        clock: Callable[[], datetime] | None = None,
        enabled: bool = True,
        available: bool = True,
        degraded_reason: str | None = None,
        status_provider: object | None = None,
        max_pending_imports: int = 8,
    ) -> None:
        if isinstance(max_pending_imports, bool) or max_pending_imports <= 0:
            raise ValueError("max_pending_imports must be positive")
        self._store = store
        self._transfer = transfer
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._enabled = bool(enabled)
        self._available = bool(available)
        self._degraded_reason = degraded_reason
        self._status_provider = status_provider
        self._max_pending_imports = max_pending_imports
        self._mutation_lock = asyncio.Lock()
        self._import_lock = asyncio.Lock()
        self._imports: dict[str, _CachedImport] = {}
        self._started = False
        self._closed = False
        self._pending_imports = (
            Path(self._transfer.backups).resolve().parent / "pending-imports"
        )

    @property
    def enabled(self) -> bool:
        provider_enabled = getattr(self._status_provider, "enabled", True)
        return self._enabled and bool(provider_enabled)

    @property
    def available(self) -> bool:
        provider_available = getattr(self._status_provider, "available", True)
        return self.enabled and self._available and bool(provider_available)

    @property
    def degraded_reason(self) -> str | None:
        provider_reason = getattr(self._status_provider, "degraded_reason", None)
        return str(provider_reason) if provider_reason else self._degraded_reason

    def _now(self) -> datetime:
        return _aware_utc(self._clock())

    async def start(self) -> None:
        """Discard managed import files that cannot belong to this process."""
        async with self._import_lock:
            if self._closed:
                raise RuntimeError("memory admin service is closed")
            if self._started:
                return
            await asyncio.to_thread(
                self._cleanup_orphaned_imports,
                remove_all=True,
            )
            self._started = True

    async def stats(self) -> dict[str, object]:
        async with self._import_lock:
            await self._discard_expired_imports()
        result = await asyncio.to_thread(self._store.stats)
        return self._stats_dict(result)

    @staticmethod
    def _stats_dict(result: MemoryStats) -> dict[str, object]:
        return {
            "db_revision": result.database_revision,
            "counts": {
                "total": result.total,
                "confirmed": result.confirmed,
                "candidates": result.candidates,
                "low_confidence_candidates": result.low_confidence_candidates,
                "people": result.people,
                "relationships": result.relationships,
                "events": result.events,
                "pending_followups": result.pending_followups,
            },
        }

    async def list_records(
        self,
        *,
        kind: MemoryKind | str | None = None,
        state: MemoryState | str | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 30,
    ) -> dict[str, object]:
        page = await asyncio.to_thread(
            self._store.list_records,
            kind=kind,
            state=state,
            q=q,
            cursor=cursor,
            limit=limit,
        )
        return self._page_dict(page)

    @staticmethod
    def _page_dict(page: MemoryRecordPage) -> dict[str, object]:
        return {
            "items": [_record_dict(record) for record in page.items],
            "next_cursor": page.next_cursor,
            "db_revision": page.database_revision,
        }

    async def get_record(self, memory_id: str) -> dict[str, object] | None:
        record = await asyncio.to_thread(self._store.get, memory_id)
        return _record_dict(record) if record is not None else None

    async def forget(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, object]:
        async with self._mutation_lock:
            result = await asyncio.to_thread(
                self._store.forget,
                memory_id,
                expected_revision=expected_revision,
                reason=reason,
                now=self._now(),
            )
        return self._forget_dict(result)

    @staticmethod
    def _forget_dict(result: ForgetResult) -> dict[str, object]:
        return {
            "id": result.memory_id,
            "kind": result.kind.value,
            "db_revision": result.database_revision,
            "deleted_ids": list(result.deleted_ids),
        }

    async def clear_all(
        self,
        *,
        expected_db_revision: int,
        backup: bool = True,
    ) -> dict[str, object]:
        clear_at = self._now()
        backup_path = None
        if backup:
            stamp = clear_at.strftime("%Y%m%dT%H%M%SZ")
            backup_path = Path(self._transfer.backups).resolve() / (
                f"memory-before-clear-{stamp}-{uuid4().hex}.sqlite3"
            )
        operation = partial(
            self._store.clear_all,
            expected_database_revision=expected_db_revision,
            now=clear_at,
            backup_path=backup_path,
        )
        async with self._mutation_lock:
            barrier = getattr(self._status_provider, "run_clear_barrier", None)
            if callable(barrier):
                result = await barrier(operation, cutoff=clear_at)
            else:
                result = await asyncio.to_thread(operation)
        return self._clear_dict(result)

    @staticmethod
    def _clear_dict(result: ClearResult) -> dict[str, object]:
        return {
            "deleted": result.deleted_count,
            "db_revision": result.database_revision,
            "backup_path": (
                str(result.backup_path) if result.backup_path is not None else None
            ),
        }

    async def export_json(self, destination: Path) -> dict[str, object]:
        result = await asyncio.to_thread(self._transfer.export_json, Path(destination))
        return self._export_dict(result)

    @staticmethod
    def _export_dict(result: ExportResult) -> dict[str, object]:
        return {
            "destination": str(result.destination),
            "payload_sha256": result.payload_sha256,
            "records": result.record_count,
        }

    def _stage_and_preview(self, source: Path) -> _CachedImport:
        source_path = Path(source).resolve()
        limit = self._transfer.limits.max_bytes
        self._pending_imports.mkdir(parents=True, exist_ok=True)
        self._cleanup_orphaned_imports()
        staged = self._pending_imports / f"preview-{uuid4().hex}.json"
        temporary = self._pending_imports / f".{staged.name}.{uuid4().hex}.tmp"
        try:
            total = 0
            try:
                with source_path.open("rb") as input_file, temporary.open("xb") as output:
                    while chunk := input_file.read(1024 * 1024):
                        total += len(chunk)
                        if total > limit:
                            raise InvalidTransferError(
                                "transfer size exceeds configured 16 MiB limit"
                            )
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except InvalidTransferError:
                raise
            except OSError as exc:
                raise InvalidTransferError(
                    f"cannot stage transfer file: {exc}"
                ) from exc
            os.replace(temporary, staged)
            plan = self._transfer.preview_import(staged)
            return _CachedImport(plan=plan, source=staged)
        except BaseException:
            for path in (temporary, staged):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _cleanup_orphaned_imports(self, *, remove_all: bool = False) -> None:
        pending = self._pending_imports
        try:
            if pending.is_symlink() or not pending.is_dir():
                return
            paths = tuple(pending.iterdir())
        except OSError:
            return
        cutoff = None
        if not remove_all:
            cutoff = (
                self._now() - self._transfer.limits.plan_ttl
            ).timestamp()
        for path in paths:
            if path.parent != pending or not (
                _MANAGED_IMPORT_NAME.fullmatch(path.name)
                or _MANAGED_IMPORT_TEMPORARY_NAME.fullmatch(path.name)
            ):
                continue
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                if remove_all or (
                    cutoff is not None and path.stat().st_mtime < cutoff
                ):
                    path.unlink()
            except OSError:
                continue

    async def _discard_expired_imports(self) -> None:
        now = self._now()
        expired = [
            cached
            for cached in self._imports.values()
            if now > cached.plan.expires_at
        ]
        for cached in expired:
            self._imports.pop(cached.plan.token, None)
        if expired:
            await asyncio.to_thread(self._discard_cached_imports, expired)

    def _discard_cached_imports(self, imports: list[_CachedImport]) -> None:
        for cached in imports:
            try:
                self._transfer.discard_import(cached.plan)
            except Exception:
                pass
            try:
                cached.source.unlink(missing_ok=True)
            except OSError:
                pass

    async def preview_import(self, path: Path) -> dict[str, object]:
        async with self._import_lock:
            if self._closed:
                raise RuntimeError("memory admin service is closed")
            await self._discard_expired_imports()
            evicted: list[_CachedImport] = []
            while len(self._imports) >= self._max_pending_imports:
                oldest_token = next(iter(self._imports))
                evicted.append(self._imports.pop(oldest_token))
            if evicted:
                await asyncio.to_thread(self._discard_cached_imports, evicted)
            cached = await asyncio.to_thread(self._stage_and_preview, Path(path))
            self._imports[cached.plan.token] = cached
        return self._plan_dict(cached.plan)

    @staticmethod
    def _plan_dict(plan: ImportPlan) -> dict[str, object]:
        return {
            "preview_token": plan.token,
            "db_revision": plan.base_database_revision,
            "expires_at": plan.expires_at.isoformat(),
            "source_file_sha256": plan.source_file_sha256,
            "counts": {
                "insertable": (
                    plan.insertable_count + len(plan.owner_profile_updates)
                ),
                "duplicates": (
                    plan.identical_count + len(plan.owner_profile_duplicates)
                ),
                "conflicts": (
                    plan.conflict_count + len(plan.owner_profile_conflicts)
                ),
                "invalid": 0,
            },
            "owner_profile": {
                "updates": list(plan.owner_profile_updates),
                "duplicates": list(plan.owner_profile_duplicates),
                "conflicts": list(plan.owner_profile_conflicts),
            },
            "conflicts": [
                {"id": conflict.memory_id, "reason": conflict.reason}
                for conflict in plan.conflicts
            ],
        }

    @staticmethod
    def _plan_token(plan: Mapping[str, object] | ImportPlan | str) -> str:
        if isinstance(plan, ImportPlan):
            return plan.token
        if isinstance(plan, str):
            token = plan
        else:
            value = plan.get("preview_token", plan.get("token"))
            token = value if isinstance(value, str) else ""
        if not token:
            raise StaleImportPlanError("import preview token is missing")
        return token

    async def apply_import(
        self,
        plan: Mapping[str, object] | ImportPlan | str,
    ) -> dict[str, object]:
        token = self._plan_token(plan)
        async with self._mutation_lock:
            async with self._import_lock:
                if self._closed:
                    raise RuntimeError("memory admin service is closed")
                await self._discard_expired_imports()
                cached = self._imports.get(token)
                if cached is None:
                    raise StaleImportPlanError(
                        "import preview token is unknown or expired"
                    )
                if isinstance(plan, Mapping):
                    expected_revision = plan.get("expected_db_revision")
                    if (
                        isinstance(expected_revision, bool)
                        or not isinstance(expected_revision, int)
                        or expected_revision != cached.plan.base_database_revision
                    ):
                        raise StaleImportPlanError(
                            "import preview revision changed"
                        )
                try:
                    result = await asyncio.to_thread(
                        self._transfer.apply_import,
                        cached.plan,
                    )
                finally:
                    self._imports.pop(token, None)
                    await asyncio.to_thread(
                        self._discard_cached_imports,
                        [cached],
                    )
        return self._import_result_dict(result)

    async def close(self) -> None:
        async with self._mutation_lock:
            async with self._import_lock:
                if self._closed:
                    return
                self._closed = True
                cached = list(self._imports.values())
                self._imports.clear()
                if cached:
                    await asyncio.to_thread(
                        self._discard_cached_imports,
                        cached,
                    )
                await asyncio.to_thread(
                    self._cleanup_orphaned_imports,
                    remove_all=True,
                )

    @staticmethod
    def _import_result_dict(result: ImportResult) -> dict[str, Any]:
        return {
            "inserted": (
                result.inserted_count + len(result.owner_profile_updated)
            ),
            "inserted_memories": result.inserted_count,
            "inserted_ids": list(result.inserted_ids),
            "skipped_duplicates": result.skipped_identical,
            "skipped_conflicts": result.skipped_conflicts,
            "owner_profile_updated": list(result.owner_profile_updated),
            "owner_profile_skipped_conflicts": list(
                result.owner_profile_skipped_conflicts
            ),
            "db_revision": result.database_revision,
            "backup_path": (
                str(result.backup_path) if result.backup_path is not None else None
            ),
        }
