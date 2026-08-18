from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from avaturn_live_streamer.memory.extractor import contains_sensitive_credential
from avaturn_live_streamer.memory.schema import open_database


FORMAT_NAME = "digibox-person-event-memory"
FORMAT_VERSION = 1
SAFE_IMPORT_LOW_CONFIDENCE_CUTOFF = 0.60
SAFE_IMPORT_TEMPORARY_TTL = timedelta(days=30)
# A normal record contributes a source, memory, evidence, revision, and
# occasionally alias/participant nodes.  Keep the public 10k record meaning
# while bounding distributed short-node payloads well below the 16 MiB cap.
COLLECTION_NODE_BUDGET_PER_RECORD = 64


class InvalidTransferError(ValueError):
    pass


class StaleImportPlanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TransferLimits:
    max_bytes: int = 16 * 1024 * 1024
    max_records: int = 10_000
    max_string_length: int = 100_000
    max_depth: int = 16
    plan_ttl: timedelta = timedelta(minutes=10)

    def __post_init__(self) -> None:
        if min(
            self.max_bytes,
            self.max_records,
            self.max_string_length,
            self.max_depth,
        ) <= 0:
            raise ValueError("transfer limits must be positive")
        if self.plan_ttl <= timedelta(0):
            raise ValueError("plan_ttl must be positive")


@dataclass(frozen=True, slots=True)
class ExportResult:
    destination: Path
    payload_sha256: str
    record_count: int


@dataclass(frozen=True, slots=True)
class ImportConflict:
    memory_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ImportPlan:
    token: str
    insertable_count: int
    identical_count: int
    conflict_count: int
    conflicts: tuple[ImportConflict, ...]
    base_database_revision: int
    source_file_sha256: str
    expires_at: datetime
    owner_profile_updates: tuple[str, ...] = ()
    owner_profile_duplicates: tuple[str, ...] = ()
    owner_profile_conflicts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportResult:
    inserted_count: int
    inserted_ids: tuple[str, ...]
    skipped_identical: int
    skipped_conflicts: int
    database_revision: int
    backup_path: Path | None
    owner_profile_updated: tuple[str, ...] = ()
    owner_profile_skipped_conflicts: tuple[str, ...] = ()


@dataclass(slots=True)
class _PreparedPlan:
    public: ImportPlan
    source: Path
    document: dict[str, Any]
    insertable_ids: tuple[str, ...]
    insertable_tombstones: tuple[tuple[str, str], ...]
    owner_profile_updates: tuple[str, ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_SPACE = re.compile(r"\s+")
_HEX_64 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _normalize_text(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _derived_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256(encoded)


def _aware_datetime(value: object, field: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise InvalidTransferError(f"{field} must be an aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidTransferError(f"{field} must be an aware ISO timestamp") from exc
    if parsed.utcoffset() is None:
        raise InvalidTransferError(f"{field} must be an aware ISO timestamp")
    return parsed


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidTransferError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise InvalidTransferError(f"{field} must be a string or null")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidTransferError(f"{field} must be an integer >= {minimum}")
    return value


def _confidence(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTransferError(f"{field} must be a number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise InvalidTransferError(f"{field} must be a number between 0 and 1")
    return result


def _require_shape(
    value: dict[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    field: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise InvalidTransferError(
            f"{field} fields are invalid (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )


def _aware_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("transfer clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _database_revision(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM meta WHERE key = 'database_revision'"
    ).fetchone()
    if row is None:
        raise sqlite3.DatabaseError("memory database has no revision")
    return int(row[0])


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidTransferError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_sensitive_credential_material(value: object) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if contains_sensitive_credential(current):
                raise InvalidTransferError(
                    "transfer contains sensitive credential material"
                )
            if current.lstrip().startswith(("{", "[", '"')):
                try:
                    nested = json.loads(current)
                except (ValueError, RecursionError):
                    pass
                else:
                    pending.append(nested)
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


class MemoryTransfer:
    def __init__(
        self,
        database: Path,
        backups: Path,
        *,
        limits: TransferLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = Path(database).resolve()
        self.backups = Path(backups).resolve()
        self.limits = limits or TransferLimits()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._plans: dict[str, _PreparedPlan] = {}
        self._lock = threading.RLock()

    def _now(self) -> datetime:
        return _aware_utc(self._clock())

    @staticmethod
    def _snapshot(database: Path) -> dict[str, Any]:
        with open_database(database, read_only=True) as connection:
            connection.execute("BEGIN")
            revision = _database_revision(connection)
            owner = connection.execute(
                """
                SELECT owner_uuid, display_name, profile
                FROM owner_profile WHERE id = 1
                """
            ).fetchone()
            if owner is None:
                raise sqlite3.DatabaseError("memory database has no owner")
            source_rows = tuple(
                connection.execute(
                    """
                    SELECT id, session_id, turn_id, engine_kind,
                           transcript_sha256, observed_at
                    FROM turn_sources ORDER BY session_id, turn_id
                    """
                )
            )
            sources = [
                {
                    "session_id": row["session_id"],
                    "turn_id": row["turn_id"],
                    "engine_kind": row["engine_kind"],
                    "transcript_sha256": row["transcript_sha256"],
                    "observed_at": row["observed_at"],
                }
                for row in source_rows
            ]
            source_by_id = {
                int(row["id"]): (str(row["session_id"]), str(row["turn_id"]))
                for row in source_rows
            }
            memories: list[dict[str, Any]] = []
            for base_row in connection.execute("SELECT * FROM memories ORDER BY id"):
                memory_id = str(base_row["id"])
                kind = str(base_row["kind"])
                item: dict[str, Any] = {
                    "id": memory_id,
                    "kind": kind,
                    "base": dict(base_row),
                }
                if kind == "person":
                    person = connection.execute(
                        """
                        SELECT display_name, normalized_name, notes
                        FROM people WHERE memory_id = ?
                        """,
                        (memory_id,),
                    ).fetchone()
                    if person is None:
                        raise sqlite3.DatabaseError("person memory has no person row")
                    item["person"] = {
                        **dict(person),
                        "aliases": [
                            dict(row)
                            for row in connection.execute(
                                """
                                SELECT alias, normalized_alias FROM person_aliases
                                WHERE person_memory_id = ?
                                ORDER BY normalized_alias, alias
                                """,
                                (memory_id,),
                            )
                        ],
                    }
                elif kind == "relationship":
                    row = connection.execute(
                        """
                        SELECT source_person_id, target_person_id,
                               relation_type, description
                        FROM relationships WHERE memory_id = ?
                        """,
                        (memory_id,),
                    ).fetchone()
                    if row is None:
                        raise sqlite3.DatabaseError("relationship memory has no child row")
                    item["relationship"] = dict(row)
                elif kind == "event":
                    row = connection.execute(
                        "SELECT * FROM events WHERE memory_id = ?",
                        (memory_id,),
                    ).fetchone()
                    if row is None:
                        raise sqlite3.DatabaseError("event memory has no event row")
                    event = dict(row)
                    event.pop("memory_id", None)
                    event["participants"] = [
                        dict(participant)
                        for participant in connection.execute(
                            """
                            SELECT person_memory_id, is_owner, role
                            FROM event_participants WHERE event_memory_id = ?
                            ORDER BY is_owner DESC, person_memory_id, role
                            """,
                            (memory_id,),
                        )
                    ]
                    item["event"] = event
                item["evidence"] = [
                    {
                        "source_session_id": source_by_id[int(row["turn_source_id"])][0],
                        "source_turn_id": source_by_id[int(row["turn_source_id"])][1],
                        "excerpt": row["excerpt"],
                        "excerpt_sha256": row["excerpt_sha256"],
                        "confidence": row["confidence"],
                        "observed_at": row["observed_at"],
                    }
                    for row in connection.execute(
                        """
                        SELECT turn_source_id, excerpt, excerpt_sha256,
                               confidence, observed_at
                        FROM evidence WHERE memory_id = ?
                        ORDER BY observed_at, id
                        """,
                        (memory_id,),
                    )
                ]
                item["revisions"] = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT revision, action, before_json, after_json, created_at
                        FROM revisions WHERE memory_id = ? ORDER BY revision
                        """,
                        (memory_id,),
                    )
                ]
                memories.append(item)
            tombstones = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT kind, content_fingerprint, reason_code, created_at
                    FROM tombstones ORDER BY kind, content_fingerprint
                    """
                )
            ]
            connection.rollback()
        return {
            "owner_uuid": str(owner[0]),
            "owner_profile": {
                "display_name": owner["display_name"],
                "profile": owner["profile"],
            },
            "database_revision": revision,
            "sources": sources,
            "memories": memories,
            "tombstones": tombstones,
        }

    def export_json(self, destination: Path) -> ExportResult:
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._snapshot(self.database)
        document = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "exported_at": self._now().isoformat(),
            "payload_sha256": "",
            "payload": payload,
        }
        self._check_limits(document)
        self._validate_bundle(payload)
        digest = _sha256(_canonical_json(payload))
        document["payload_sha256"] = digest
        encoded = _canonical_json(document)
        if len(encoded) > self.limits.max_bytes:
            raise InvalidTransferError(
                "transfer size exceeds configured 16 MiB limit"
            )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return ExportResult(
            destination=target,
            payload_sha256=digest,
            record_count=len(payload["memories"]),
        )

    def _check_limits(
        self,
        value: object,
        *,
        depth: int = 0,
        collection_nodes: list[int] | None = None,
    ) -> None:
        if collection_nodes is None:
            collection_nodes = [0]
        if depth > self.limits.max_depth:
            raise InvalidTransferError("JSON nesting depth exceeds configured limit")
        if isinstance(value, str):
            if len(value) > self.limits.max_string_length:
                raise InvalidTransferError("JSON string exceeds configured limit")
            return
        if isinstance(value, list):
            if len(value) > self.limits.max_records:
                raise InvalidTransferError(
                    "JSON collection exceeds configured record limit"
                )
            collection_nodes[0] += len(value)
            aggregate_limit = (
                self.limits.max_records * COLLECTION_NODE_BUDGET_PER_RECORD
            )
            if collection_nodes[0] > aggregate_limit:
                raise InvalidTransferError(
                    "JSON aggregate collection node count exceeds configured limit"
                )
            for item in value:
                self._check_limits(
                    item,
                    depth=depth + 1,
                    collection_nodes=collection_nodes,
                )
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._check_limits(
                    key,
                    depth=depth + 1,
                    collection_nodes=collection_nodes,
                )
                self._check_limits(
                    item,
                    depth=depth + 1,
                    collection_nodes=collection_nodes,
                )

    def _read_bundle(self, source: Path) -> tuple[dict[str, Any], str]:
        path = Path(source).resolve()
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise InvalidTransferError(f"cannot read transfer file: {exc}") from exc
        if size > self.limits.max_bytes:
            raise InvalidTransferError("transfer size exceeds configured 16 MiB limit")
        encoded = path.read_bytes()
        file_digest = _sha256(encoded)
        try:
            document = json.loads(
                encoded,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except InvalidTransferError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidTransferError(f"invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise InvalidTransferError("transfer document must be a JSON object")
        self._check_limits(document)
        if document.get("format") != FORMAT_NAME:
            raise InvalidTransferError("unknown transfer format")
        if document.get("format_version") != FORMAT_VERSION:
            raise InvalidTransferError("unsupported format version")
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise InvalidTransferError("transfer payload must be an object")
        expected_digest = document.get("payload_sha256")
        if not isinstance(expected_digest, str) or expected_digest != _sha256(
            _canonical_json(payload)
        ):
            raise InvalidTransferError("payload SHA-256 mismatch")
        memories = payload.get("memories")
        sources = payload.get("sources")
        tombstones = payload.get("tombstones")
        if not isinstance(memories, list) or not isinstance(sources, list):
            raise InvalidTransferError("payload records must be arrays")
        if not isinstance(tombstones, list):
            raise InvalidTransferError("payload tombstones must be an array")
        if len(memories) > self.limits.max_records:
            raise InvalidTransferError("memory record count exceeds configured limit")
        _reject_sensitive_credential_material(payload)
        return document, file_digest

    @staticmethod
    def _validate_bundle(
        payload: dict[str, Any],
    ) -> None:
        _require_shape(
            payload,
            required={
                "owner_uuid",
                "database_revision",
                "sources",
                "memories",
                "tombstones",
            },
            allowed={
                "owner_uuid",
                "owner_profile",
                "database_revision",
                "sources",
                "memories",
                "tombstones",
            },
            field="payload",
        )
        _required_text(payload["owner_uuid"], "payload.owner_uuid")
        _integer(payload["database_revision"], "payload.database_revision")
        if "owner_profile" in payload:
            owner_profile = payload["owner_profile"]
            if not isinstance(owner_profile, dict):
                raise InvalidTransferError("owner_profile must be an object")
            _require_shape(
                owner_profile,
                required=set(),
                allowed={"display_name", "profile"},
                field="owner_profile",
            )
            for field in ("display_name", "profile"):
                _optional_text(owner_profile.get(field), f"owner_profile.{field}")

        memories = payload["memories"]
        sources = payload["sources"]
        tombstones = payload["tombstones"]
        if not isinstance(memories, list) or not isinstance(sources, list):
            raise InvalidTransferError("payload records must be arrays")
        if not isinstance(tombstones, list):
            raise InvalidTransferError("payload tombstones must be an array")

        source_keys: set[tuple[str, str]] = set()
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise InvalidTransferError("turn source must be an object")
            _require_shape(
                source,
                required={
                    "session_id",
                    "turn_id",
                    "engine_kind",
                    "transcript_sha256",
                    "observed_at",
                },
                allowed={
                    "session_id",
                    "turn_id",
                    "engine_kind",
                    "transcript_sha256",
                    "observed_at",
                },
                field=f"sources[{index}]",
            )
            session_id = _required_text(
                source["session_id"], f"sources[{index}].session_id"
            )
            turn_id = _required_text(
                source["turn_id"], f"sources[{index}].turn_id"
            )
            _required_text(source["engine_kind"], f"sources[{index}].engine_kind")
            digest = source["transcript_sha256"]
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                raise InvalidTransferError("invalid transcript SHA-256")
            _aware_datetime(source["observed_at"], f"sources[{index}].observed_at")
            source_key = (session_id, turn_id)
            if source_key in source_keys:
                raise InvalidTransferError("duplicate turn source key")
            source_keys.add(source_key)

        ids: set[str] = set()
        bundle_people: set[str] = set()
        for index, item in enumerate(memories):
            if not isinstance(item, dict):
                raise InvalidTransferError("memory record must be an object")
            memory_id = _required_text(item.get("id"), f"memories[{index}].id")
            kind = item.get("kind")
            base = item.get("base")
            if memory_id in ids:
                raise InvalidTransferError(f"duplicate memory id: {memory_id}")
            ids.add(memory_id)
            if kind not in {"person", "relationship", "event"}:
                raise InvalidTransferError(f"invalid memory kind: {kind}")
            if not isinstance(base, dict) or base.get("id") != memory_id:
                raise InvalidTransferError("memory base id does not match record id")
            if base.get("kind") != kind:
                raise InvalidTransferError("memory base kind does not match record kind")
            base_fields = {
                "id",
                "kind",
                "state",
                "confidence",
                "retention_class",
                "canonical_key",
                "content_fingerprint",
                "revision",
                "created_at",
                "updated_at",
                "last_seen_at",
                "expires_at",
            }
            _require_shape(
                base,
                required=base_fields,
                allowed=base_fields,
                field=f"memories[{index}].base",
            )
            if base["state"] not in {"candidate", "confirmed"}:
                raise InvalidTransferError("invalid memory state")
            base["confidence"] = _confidence(
                base["confidence"], f"memories[{index}].base.confidence"
            )
            retention = base["retention_class"]
            if retention not in {"temporary_30d", "persistent"}:
                raise InvalidTransferError("invalid memory retention class")
            _required_text(base["canonical_key"], "memory canonical_key")
            _required_text(base["content_fingerprint"], "memory fingerprint")
            base_revision = _integer(
                base["revision"],
                f"memories[{index}].base.revision",
                minimum=1,
            )
            _aware_datetime(base["created_at"], "memory created_at")
            _aware_datetime(base["updated_at"], "memory updated_at")
            last_seen_at = _aware_datetime(
                base["last_seen_at"],
                "memory last_seen_at",
            )
            assert last_seen_at is not None
            expires_at = _aware_datetime(
                base["expires_at"], "memory expires_at", optional=True
            )
            if base["state"] == "confirmed":
                if retention != "persistent" or expires_at is not None:
                    raise InvalidTransferError("invalid confirmed memory retention")
            elif base["confidence"] < SAFE_IMPORT_LOW_CONFIDENCE_CUTOFF:
                if retention != "temporary_30d":
                    raise InvalidTransferError(
                        "low-confidence candidate retention must be temporary_30d"
                    )
                expected_expiry = last_seen_at + SAFE_IMPORT_TEMPORARY_TTL
                if expires_at != expected_expiry:
                    raise InvalidTransferError(
                        "temporary memory expiry must equal last_seen_at plus 30 days"
                    )
            elif retention != "persistent" or expires_at is not None:
                raise InvalidTransferError(
                    "high-confidence candidate retention must be persistent"
                )

            common_fields = {"id", "kind", "base", "evidence", "revisions"}
            if kind == "person":
                bundle_people.add(memory_id)
                _require_shape(
                    item,
                    required=common_fields | {"person"},
                    allowed=common_fields | {"person"},
                    field=f"memories[{index}]",
                )
                person = item.get("person")
                if not isinstance(person, dict):
                    raise InvalidTransferError("person memory has no person payload")
                _require_shape(
                    person,
                    required={"display_name", "normalized_name", "notes", "aliases"},
                    allowed={"display_name", "normalized_name", "notes", "aliases"},
                    field=f"memories[{index}].person",
                )
                display_name = _required_text(
                    person["display_name"], "person.display_name"
                )
                notes = _optional_text(person["notes"], "person.notes")
                aliases = person["aliases"]
                if not isinstance(aliases, list):
                    raise InvalidTransferError("person.aliases must be an array")
                normalized_aliases: set[str] = set()
                for alias_index, alias in enumerate(aliases):
                    if not isinstance(alias, dict):
                        raise InvalidTransferError("person alias must be an object")
                    _require_shape(
                        alias,
                        required={"alias", "normalized_alias"},
                        allowed={"alias", "normalized_alias"},
                        field=f"person.aliases[{alias_index}]",
                    )
                    alias_text = _required_text(alias["alias"], "person alias")
                    normalized_alias = _normalize_text(alias_text)
                    if normalized_alias in normalized_aliases:
                        raise InvalidTransferError("duplicate normalized person alias")
                    normalized_aliases.add(normalized_alias)
                    alias["normalized_alias"] = normalized_alias
                person["normalized_name"] = _normalize_text(display_name)
                base["canonical_key"] = "\x1f".join(
                    ("person", person["normalized_name"], _normalize_text(notes or ""))
                )
                base["content_fingerprint"] = _derived_fingerprint(
                    {
                        "name": person["normalized_name"],
                        "aliases": sorted(normalized_aliases),
                        "notes": _normalize_text(notes or ""),
                    }
                )
            elif kind == "relationship":
                _require_shape(
                    item,
                    required=common_fields | {"relationship"},
                    allowed=common_fields | {"relationship"},
                    field=f"memories[{index}]",
                )
                relationship = item.get("relationship")
                if not isinstance(relationship, dict):
                    raise InvalidTransferError("relationship memory has no payload")
                relationship_fields = {
                    "source_person_id",
                    "target_person_id",
                    "relation_type",
                    "description",
                }
                _require_shape(
                    relationship,
                    required=relationship_fields,
                    allowed=relationship_fields,
                    field="relationship",
                )
                source_id = _optional_text(
                    relationship["source_person_id"], "relationship.source_person_id"
                )
                target_id = _optional_text(
                    relationship["target_person_id"], "relationship.target_person_id"
                )
                if source_id is None and target_id is None:
                    raise InvalidTransferError("relationship has no person")
                relation_type = _required_text(
                    relationship["relation_type"], "relationship.relation_type"
                )
                description = _optional_text(
                    relationship["description"], "relationship.description"
                )
                base["canonical_key"] = "\x1f".join(
                    (
                        "relationship",
                        source_id or "__owner__",
                        target_id or "__owner__",
                        _normalize_text(relation_type),
                        _normalize_text(description or ""),
                    )
                )
                base["content_fingerprint"] = _derived_fingerprint(
                    {
                        "source": source_id or "owner",
                        "target": target_id or "owner",
                        "type": _normalize_text(relation_type),
                        "description": _normalize_text(description or ""),
                    }
                )
            else:
                _require_shape(
                    item,
                    required=common_fields | {"event"},
                    allowed=common_fields | {"event"},
                    field=f"memories[{index}]",
                )
                event = item.get("event")
                if not isinstance(event, dict):
                    raise InvalidTransferError("event memory has no event payload")
                event_fields = {
                    "title",
                    "summary",
                    "starts_at",
                    "ends_at",
                    "time_precision",
                    "timezone",
                    "location",
                    "event_status",
                    "follow_up_state",
                    "follow_up_after",
                    "follow_up_asked_at",
                    "participants",
                }
                _require_shape(
                    event,
                    required=event_fields,
                    allowed=event_fields,
                    field="event",
                )
                title = _required_text(event["title"], "event.title")
                summary = event["summary"]
                if not isinstance(summary, str):
                    raise InvalidTransferError("event.summary must be a string")
                starts_at = _aware_datetime(
                    event["starts_at"], "event.starts_at", optional=True
                )
                ends_at = _aware_datetime(
                    event["ends_at"], "event.ends_at", optional=True
                )
                for optional_field in ("time_precision", "timezone", "location"):
                    _optional_text(event[optional_field], f"event.{optional_field}")
                if event["event_status"] not in {
                    "planned", "ongoing", "completed", "cancelled", "unknown"
                }:
                    raise InvalidTransferError("invalid event status")
                if event["follow_up_state"] not in {
                    "none", "eligible", "asked", "dismissed"
                }:
                    raise InvalidTransferError("invalid follow-up state")
                _aware_datetime(
                    event["follow_up_after"], "event.follow_up_after", optional=True
                )
                _aware_datetime(
                    event["follow_up_asked_at"],
                    "event.follow_up_asked_at",
                    optional=True,
                )
                participants = event["participants"]
                if not isinstance(participants, list):
                    raise InvalidTransferError("event participants must be an array")
                participant_keys: set[tuple[object, object, object]] = set()
                for participant_index, participant in enumerate(participants):
                    if not isinstance(participant, dict):
                        raise InvalidTransferError("event participant must be an object")
                    participant_fields = {"person_memory_id", "is_owner", "role"}
                    _require_shape(
                        participant,
                        required=participant_fields,
                        allowed=participant_fields,
                        field=f"event.participants[{participant_index}]",
                    )
                    person_id = _optional_text(
                        participant["person_memory_id"],
                        "event participant person_memory_id",
                    )
                    is_owner = participant["is_owner"]
                    if isinstance(is_owner, bool):
                        is_owner = int(is_owner)
                        participant["is_owner"] = is_owner
                    if is_owner not in (0, 1):
                        raise InvalidTransferError("invalid event participant owner flag")
                    if (is_owner == 1) == (person_id is not None):
                        raise InvalidTransferError("invalid event participant identity")
                    role = _required_text(participant["role"], "event participant role")
                    participant_key = (person_id, is_owner, role)
                    if participant_key in participant_keys:
                        raise InvalidTransferError("duplicate event participant")
                    participant_keys.add(participant_key)
                normalized_starts = (
                    starts_at.astimezone(timezone.utc).isoformat()
                    if starts_at is not None
                    else ""
                )
                base["canonical_key"] = "\x1f".join(
                    (
                        "event",
                        _normalize_text(title),
                        normalized_starts,
                        _normalize_text(event["location"] or ""),
                    )
                )
                base["content_fingerprint"] = _derived_fingerprint(
                    {
                        "title": _normalize_text(title),
                        "summary": _normalize_text(summary),
                        "starts_at": normalized_starts or None,
                        "ends_at": (
                            ends_at.astimezone(timezone.utc).isoformat()
                            if ends_at is not None
                            else None
                        ),
                        "location": _normalize_text(event["location"] or ""),
                        "status": event["event_status"],
                    }
                )

            evidence = item.get("evidence")
            revisions = item.get("revisions")
            if not isinstance(evidence, list) or not isinstance(revisions, list):
                raise InvalidTransferError("memory evidence and revisions must be arrays")
            if not evidence:
                raise InvalidTransferError(
                    "memory must include user transcript evidence"
                )
            evidence_keys: set[tuple[str, str, str]] = set()
            for evidence_index, entry in enumerate(evidence):
                if not isinstance(entry, dict):
                    raise InvalidTransferError("evidence entry must be an object")
                evidence_fields = {
                    "source_session_id",
                    "source_turn_id",
                    "excerpt",
                    "excerpt_sha256",
                    "confidence",
                    "observed_at",
                }
                _require_shape(
                    entry,
                    required=evidence_fields,
                    allowed=evidence_fields,
                    field=f"evidence[{evidence_index}]",
                )
                source_session = _required_text(
                    entry["source_session_id"], "evidence.source_session_id"
                )
                source_turn = _required_text(
                    entry["source_turn_id"], "evidence.source_turn_id"
                )
                excerpt = _required_text(entry["excerpt"], "evidence.excerpt")
                entry["excerpt_sha256"] = _sha256(excerpt.encode("utf-8"))
                entry["confidence"] = _confidence(
                    entry["confidence"], "evidence.confidence"
                )
                _aware_datetime(entry["observed_at"], "evidence.observed_at")
                evidence_key = (
                    source_session,
                    source_turn,
                    entry["excerpt_sha256"],
                )
                if evidence_key in evidence_keys:
                    raise InvalidTransferError("duplicate evidence entry")
                evidence_keys.add(evidence_key)
                if (source_session, source_turn) not in source_keys:
                    raise InvalidTransferError("dangling evidence source reference")
            revision_numbers: set[int] = set()
            for revision_index, revision in enumerate(revisions):
                if not isinstance(revision, dict):
                    raise InvalidTransferError("revision entry must be an object")
                revision_fields = {
                    "revision", "action", "before_json", "after_json", "created_at"
                }
                _require_shape(
                    revision,
                    required=revision_fields,
                    allowed=revision_fields,
                    field=f"revisions[{revision_index}]",
                )
                revision_number = _integer(
                    revision["revision"], "revision.revision", minimum=1
                )
                if revision_number > base_revision or revision_number in revision_numbers:
                    raise InvalidTransferError("invalid or duplicate memory revision")
                revision_numbers.add(revision_number)
                _required_text(revision["action"], "revision.action")
                _optional_text(revision["before_json"], "revision.before_json")
                _optional_text(revision["after_json"], "revision.after_json")
                _aware_datetime(revision["created_at"], "revision.created_at")
            if base_revision not in revision_numbers:
                raise InvalidTransferError(
                    "memory revision audit must include the current base revision"
                )

        for item in memories:
            if item["kind"] == "relationship":
                relationship = item["relationship"]
                for key in ("source_person_id", "target_person_id"):
                    person_id = relationship.get(key)
                    if person_id is not None and person_id not in bundle_people:
                        raise InvalidTransferError(
                            f"dangling bundle person reference: {person_id}"
                        )
            elif item["kind"] == "event":
                participants = item["event"].get("participants", [])
                if not isinstance(participants, list):
                    raise InvalidTransferError("event participants must be an array")
                for participant in participants:
                    if not isinstance(participant, dict):
                        raise InvalidTransferError("event participant must be an object")
                    person_id = participant.get("person_memory_id")
                    if person_id is not None and person_id not in bundle_people:
                        raise InvalidTransferError(
                            f"dangling bundle person reference: {person_id}"
                        )
        tombstone_keys: set[tuple[str, str]] = set()
        for index, tombstone in enumerate(tombstones):
            if not isinstance(tombstone, dict):
                raise InvalidTransferError("tombstone must be an object")
            tombstone_fields = {
                "kind", "content_fingerprint", "reason_code", "created_at"
            }
            _require_shape(
                tombstone,
                required=tombstone_fields,
                allowed=tombstone_fields,
                field=f"tombstones[{index}]",
            )
            if tombstone["kind"] not in {"person", "relationship", "event"}:
                raise InvalidTransferError("invalid tombstone kind")
            fingerprint = tombstone["content_fingerprint"]
            if not isinstance(fingerprint, str) or _HEX_64.fullmatch(fingerprint) is None:
                raise InvalidTransferError("invalid tombstone fingerprint")
            _required_text(tombstone["reason_code"], "tombstone.reason_code")
            _aware_datetime(tombstone["created_at"], "tombstone.created_at")
            tombstone_key = (str(tombstone["kind"]), fingerprint)
            if tombstone_key in tombstone_keys:
                raise InvalidTransferError("duplicate tombstone")
            tombstone_keys.add(tombstone_key)

    def preview_import(self, source: Path) -> ImportPlan:
        source_path = Path(source).resolve()
        document, file_digest = self._read_bundle(source_path)
        payload = document["payload"]
        with open_database(self.database, read_only=True) as connection:
            self._validate_bundle(payload)
            base_revision = _database_revision(connection)
            local_owner = connection.execute(
                "SELECT display_name, profile FROM owner_profile WHERE id = 1"
            ).fetchone()
            if local_owner is None:
                raise sqlite3.DatabaseError("memory database has no owner profile")
            local_by_id = {
                str(row["id"]): row
                for row in connection.execute(
                    "SELECT id, kind, canonical_key, content_fingerprint FROM memories"
                )
            }
            fingerprints = {
                (str(row["kind"]), str(row["content_fingerprint"])): str(row["id"])
                for row in local_by_id.values()
            }
            canonical_owners = {
                (str(row["kind"]), str(row["canonical_key"])): str(row["id"])
                for row in local_by_id.values()
            }
            locally_forgotten = {
                (str(row["kind"]), str(row["content_fingerprint"]))
                for row in connection.execute(
                    "SELECT kind, content_fingerprint FROM tombstones"
                )
            }
        owner_updates: list[str] = []
        owner_duplicates: list[str] = []
        owner_conflicts: list[str] = []
        incoming_owner = payload.get("owner_profile")
        if isinstance(incoming_owner, dict):
            for field in ("display_name", "profile"):
                incoming_value = incoming_owner.get(field)
                if not isinstance(incoming_value, str) or not incoming_value.strip():
                    continue
                local_value = local_owner[field]
                if local_value is None or not str(local_value).strip():
                    owner_updates.append(field)
                elif str(local_value) == incoming_value:
                    owner_duplicates.append(field)
                else:
                    owner_conflicts.append(field)
        insertable: list[str] = []
        identical = 0
        conflicts: list[ImportConflict] = []
        bundle_tombstones = {
            (str(item["kind"]), str(item["content_fingerprint"]))
            for item in payload["tombstones"]
        }
        importable_tombstones = tuple(
            sorted(
                bundle_tombstones
                - locally_forgotten
                - set(fingerprints)
            )
        )

        def classify(item: dict[str, Any]) -> str:
            memory_id = str(item["id"])
            base = item["base"]
            identity = (str(item["kind"]), str(base["content_fingerprint"]))
            if identity in bundle_tombstones:
                conflicts.append(ImportConflict(memory_id, "bundle_tombstoned"))
                return "blocked"
            existing = local_by_id.get(memory_id)
            if existing is not None:
                if existing["content_fingerprint"] == base["content_fingerprint"]:
                    return "identical"
                else:
                    conflicts.append(ImportConflict(memory_id, "id_conflict"))
                return "blocked"
            if identity in locally_forgotten:
                conflicts.append(ImportConflict(memory_id, "locally_forgotten"))
                return "blocked"
            fingerprint_owner = fingerprints.get(identity)
            if fingerprint_owner is not None:
                conflicts.append(ImportConflict(memory_id, "fingerprint_conflict"))
                return "blocked"
            canonical_owner = canonical_owners.get(
                (str(item["kind"]), str(base["canonical_key"]))
            )
            if canonical_owner is not None:
                conflicts.append(ImportConflict(memory_id, "canonical_conflict"))
                return "blocked"
            insertable.append(memory_id)
            fingerprints[identity] = memory_id
            canonical_owners[
                (str(item["kind"]), str(base["canonical_key"]))
            ] = memory_id
            return "insertable"

        bundled_people = {
            str(item["id"]): item
            for item in payload["memories"]
            if item["kind"] == "person"
        }
        blocked_people: set[str] = set()
        for person_id, item in bundled_people.items():
            classification = classify(item)
            if classification == "identical":
                identical += 1
            elif classification == "blocked":
                blocked_people.add(person_id)

        for item in payload["memories"]:
            if item["kind"] == "person":
                continue
            memory_id = str(item["id"])
            identity = (
                str(item["kind"]),
                str(item["base"]["content_fingerprint"]),
            )
            if identity in bundle_tombstones:
                conflicts.append(ImportConflict(memory_id, "bundle_tombstoned"))
                continue
            dependencies: set[str] = set()
            if item["kind"] == "relationship":
                relationship = item["relationship"]
                dependencies.update(
                    person_id
                    for person_id in (
                        relationship.get("source_person_id"),
                        relationship.get("target_person_id"),
                    )
                    if person_id is not None
                )
            else:
                dependencies.update(
                    participant["person_memory_id"]
                    for participant in item["event"]["participants"]
                    if participant["person_memory_id"] is not None
                )
            if dependencies & blocked_people:
                conflicts.append(ImportConflict(memory_id, "dependency_conflict"))
                continue
            classification = classify(item)
            if classification == "identical":
                identical += 1
        token = str(uuid4())
        public = ImportPlan(
            token=token,
            insertable_count=len(insertable),
            identical_count=identical,
            conflict_count=len(conflicts),
            conflicts=tuple(conflicts),
            base_database_revision=base_revision,
            source_file_sha256=file_digest,
            expires_at=self._now() + self.limits.plan_ttl,
            owner_profile_updates=tuple(owner_updates),
            owner_profile_duplicates=tuple(owner_duplicates),
            owner_profile_conflicts=tuple(owner_conflicts),
        )
        self._plans[token] = _PreparedPlan(
            public=public,
            source=source_path,
            document=document,
            insertable_ids=tuple(insertable),
            insertable_tombstones=importable_tombstones,
            owner_profile_updates=tuple(owner_updates),
        )
        return public

    def _backup(self) -> Path:
        self.backups.mkdir(parents=True, exist_ok=True)
        stamp = self._now().strftime("%Y%m%dT%H%M%SZ")
        destination = self.backups / f"memory-before-import-{stamp}-{uuid4().hex}.sqlite3"
        source = sqlite3.connect(self.database)
        backup: sqlite3.Connection | None = None
        try:
            backup = sqlite3.connect(destination)
            source.backup(backup)
        finally:
            if backup is not None:
                backup.close()
            source.close()
        return destination

    def discard_import(self, plan_or_token: ImportPlan | str) -> None:
        token = (
            plan_or_token.token
            if isinstance(plan_or_token, ImportPlan)
            else plan_or_token
        )
        with self._lock:
            self._plans.pop(token, None)

    @staticmethod
    def _insert_sources(
        connection: sqlite3.Connection,
        sources: list[dict[str, Any]],
    ) -> None:
        for source in sources:
            existing = connection.execute(
                """
                SELECT transcript_sha256 FROM turn_sources
                WHERE session_id = ? AND turn_id = ?
                """,
                (source["session_id"], source["turn_id"]),
            ).fetchone()
            if existing is not None:
                if existing[0] != source["transcript_sha256"]:
                    raise InvalidTransferError("turn source conflicts with local source")
                continue
            connection.execute(
                """
                INSERT INTO turn_sources(
                    session_id, turn_id, engine_kind, transcript_sha256, observed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source["session_id"],
                    source["turn_id"],
                    source["engine_kind"],
                    source["transcript_sha256"],
                    source["observed_at"],
                ),
            )

    @staticmethod
    def _insert_memories(
        connection: sqlite3.Connection,
        memories: list[dict[str, Any]],
    ) -> None:
        base_columns = (
            "id",
            "kind",
            "state",
            "confidence",
            "retention_class",
            "canonical_key",
            "content_fingerprint",
            "revision",
            "created_at",
            "updated_at",
            "last_seen_at",
            "expires_at",
        )
        for item in memories:
            base = item["base"]
            connection.execute(
                f"INSERT INTO memories({', '.join(base_columns)}) "
                f"VALUES ({', '.join('?' for _ in base_columns)})",
                tuple(base[column] for column in base_columns),
            )
        for item in memories:
            memory_id = item["id"]
            if item["kind"] == "person":
                person = item["person"]
                connection.execute(
                    """
                    INSERT INTO people(memory_id, display_name, normalized_name, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        person["display_name"],
                        person["normalized_name"],
                        person.get("notes"),
                    ),
                )
                for alias in person.get("aliases", []):
                    connection.execute(
                        """
                        INSERT INTO person_aliases(
                            person_memory_id, alias, normalized_alias
                        ) VALUES (?, ?, ?)
                        """,
                        (memory_id, alias["alias"], alias["normalized_alias"]),
                    )
        for item in memories:
            memory_id = item["id"]
            if item["kind"] == "relationship":
                relationship = item["relationship"]
                connection.execute(
                    """
                    INSERT INTO relationships(
                        memory_id, source_person_id, target_person_id,
                        relation_type, description
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        relationship.get("source_person_id"),
                        relationship.get("target_person_id"),
                        relationship["relation_type"],
                        relationship.get("description"),
                    ),
                )
            elif item["kind"] == "event":
                event = item["event"]
                columns = (
                    "title",
                    "summary",
                    "starts_at",
                    "ends_at",
                    "time_precision",
                    "timezone",
                    "location",
                    "event_status",
                    "follow_up_state",
                    "follow_up_after",
                    "follow_up_asked_at",
                )
                connection.execute(
                    f"INSERT INTO events(memory_id, {', '.join(columns)}) "
                    f"VALUES (?, {', '.join('?' for _ in columns)})",
                    (memory_id, *(event.get(column) for column in columns)),
                )
                for participant in event.get("participants", []):
                    connection.execute(
                        """
                        INSERT INTO event_participants(
                            event_memory_id, person_memory_id, is_owner, role
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            memory_id,
                            participant.get("person_memory_id"),
                            participant["is_owner"],
                            participant["role"],
                        ),
                    )
        for item in memories:
            memory_id = item["id"]
            for evidence in item.get("evidence", []):
                source = connection.execute(
                    """
                    SELECT id FROM turn_sources
                    WHERE session_id = ? AND turn_id = ?
                    """,
                    (
                        evidence["source_session_id"],
                        evidence["source_turn_id"],
                    ),
                ).fetchone()
                if source is None:
                    raise InvalidTransferError("dangling evidence source during apply")
                connection.execute(
                    """
                    INSERT INTO evidence(
                        memory_id, turn_source_id, excerpt, excerpt_sha256,
                        confidence, observed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        source[0],
                        evidence["excerpt"],
                        evidence["excerpt_sha256"],
                        evidence["confidence"],
                        evidence["observed_at"],
                    ),
                )
            for revision in item.get("revisions", []):
                connection.execute(
                    """
                    INSERT INTO revisions(
                        memory_id, revision, action, before_json, after_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        revision["revision"],
                        revision["action"],
                        revision.get("before_json"),
                        revision.get("after_json"),
                        revision["created_at"],
                    ),
                )

    @staticmethod
    def _merge_owner_profile(
        connection: sqlite3.Connection,
        incoming: dict[str, Any],
        planned_fields: tuple[str, ...],
        *,
        updated_at: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        row = connection.execute(
            "SELECT display_name, profile FROM owner_profile WHERE id = 1"
        ).fetchone()
        if row is None:
            raise sqlite3.DatabaseError("memory database has no owner profile")
        updated: list[str] = []
        conflicts: list[str] = []
        values: list[str] = []
        for field in planned_fields:
            incoming_value = incoming.get(field)
            if not isinstance(incoming_value, str) or not incoming_value.strip():
                continue
            local_value = row[field]
            if local_value is None or not str(local_value).strip():
                updated.append(field)
                values.append(incoming_value)
            elif str(local_value) != incoming_value:
                conflicts.append(field)
        if updated:
            assignments = ", ".join(f"{field} = ?" for field in updated)
            connection.execute(
                f"""
                UPDATE owner_profile
                SET {assignments}, revision = revision + 1, updated_at = ?
                WHERE id = 1
                """,
                (*values, updated_at),
            )
        return tuple(updated), tuple(conflicts)

    def apply_import(self, plan: ImportPlan) -> ImportResult:
        with self._lock:
            prepared = self._plans.get(plan.token)
            if prepared is None or prepared.public != plan:
                raise StaleImportPlanError("import preview token is unknown or changed")
            if self._now() > plan.expires_at:
                raise StaleImportPlanError("import preview token expired")
            try:
                current_file_hash = _sha256(prepared.source.read_bytes())
            except OSError as exc:
                raise StaleImportPlanError(f"source hash cannot be verified: {exc}") from exc
            if current_file_hash != plan.source_file_sha256:
                raise StaleImportPlanError("source hash changed after preview")
            with open_database(self.database, read_only=True) as check:
                current_revision = _database_revision(check)
            if current_revision != plan.base_database_revision:
                raise StaleImportPlanError("database revision changed after preview")
            _reject_sensitive_credential_material(prepared.document["payload"])
            if (
                not prepared.insertable_ids
                and not prepared.insertable_tombstones
                and not prepared.owner_profile_updates
            ):
                self._plans.pop(plan.token, None)
                return ImportResult(
                    inserted_count=0,
                    inserted_ids=(),
                    skipped_identical=plan.identical_count,
                    skipped_conflicts=plan.conflict_count,
                    database_revision=current_revision,
                    backup_path=None,
                    owner_profile_updated=(),
                    owner_profile_skipped_conflicts=plan.owner_profile_conflicts,
                )

            backup_path = self._backup()
            payload = prepared.document["payload"]
            insertable = set(prepared.insertable_ids)
            insertable_tombstones = set(prepared.insertable_tombstones)
            records = [
                item for item in payload["memories"] if item["id"] in insertable
            ]
            referenced_source_keys = {
                (
                    str(evidence["source_session_id"]),
                    str(evidence["source_turn_id"]),
                )
                for item in records
                for evidence in item.get("evidence", [])
            }
            referenced_sources = [
                source
                for source in payload["sources"]
                if (str(source["session_id"]), str(source["turn_id"]))
                in referenced_source_keys
            ]
            incoming_owner = payload.get("owner_profile")
            if not isinstance(incoming_owner, dict):
                incoming_owner = {}
            connection = open_database(self.database)
            try:
                connection.execute("BEGIN IMMEDIATE")
                if _database_revision(connection) != plan.base_database_revision:
                    raise StaleImportPlanError("database revision changed after backup")
                if records:
                    self._insert_sources(connection, referenced_sources)
                    self._insert_memories(connection, records)
                inserted_tombstone = False
                for tombstone in payload["tombstones"]:
                    identity = (
                        str(tombstone["kind"]),
                        str(tombstone["content_fingerprint"]),
                    )
                    if identity not in insertable_tombstones:
                        continue
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO tombstones(
                            kind, content_fingerprint, reason_code, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            tombstone["kind"],
                            tombstone["content_fingerprint"],
                            tombstone["reason_code"],
                            tombstone["created_at"],
                        ),
                    )
                    inserted_tombstone = inserted_tombstone or cursor.rowcount > 0
                owner_updated, late_owner_conflicts = self._merge_owner_profile(
                    connection,
                    incoming_owner,
                    prepared.owner_profile_updates,
                    updated_at=self._now().isoformat(),
                )
                changed = bool(records or owner_updated or inserted_tombstone)
                revision = plan.base_database_revision
                if changed:
                    revision += 1
                    connection.execute(
                        "UPDATE meta SET value = ? WHERE key = 'database_revision'",
                        (str(revision),),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
            self._plans.pop(plan.token, None)
            skipped_owner_conflicts = tuple(
                dict.fromkeys(
                    (*plan.owner_profile_conflicts, *late_owner_conflicts)
                )
            )
            return ImportResult(
                inserted_count=len(records),
                inserted_ids=tuple(item["id"] for item in records),
                skipped_identical=plan.identical_count,
                skipped_conflicts=plan.conflict_count,
                database_revision=revision,
                backup_path=backup_path,
                owner_profile_updated=owner_updated,
                owner_profile_skipped_conflicts=skipped_owner_conflicts,
            )
