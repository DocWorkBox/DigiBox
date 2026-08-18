from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


SCHEMA_VERSION = 1


class UnsupportedSchemaVersion(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SchemaInfo:
    schema_version: int
    database_revision: int
    owner_uuid: str


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS owner_profile (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        owner_uuid TEXT NOT NULL UNIQUE,
        display_name TEXT,
        profile TEXT,
        revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turn_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        engine_kind TEXT NOT NULL,
        transcript_sha256 TEXT NOT NULL CHECK (length(transcript_sha256) = 64),
        observed_at TEXT NOT NULL,
        UNIQUE (session_id, turn_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('person', 'relationship', 'event')),
        state TEXT NOT NULL CHECK (state IN ('candidate', 'confirmed')),
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        retention_class TEXT NOT NULL
            CHECK (retention_class IN ('temporary_30d', 'persistent')),
        canonical_key TEXT NOT NULL,
        content_fingerprint TEXT NOT NULL CHECK (length(content_fingerprint) = 64),
        revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        expires_at TEXT,
        CHECK (
            (retention_class = 'temporary_30d'
                AND state = 'candidate'
                AND expires_at IS NOT NULL)
            OR
            (retention_class = 'persistent' AND expires_at IS NULL)
        ),
        CHECK (
            state != 'confirmed'
            OR (retention_class = 'persistent' AND expires_at IS NULL)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS people (
        memory_id TEXT PRIMARY KEY
            REFERENCES memories(id) ON DELETE CASCADE,
        display_name TEXT NOT NULL,
        normalized_name TEXT NOT NULL,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS person_aliases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_memory_id TEXT NOT NULL
            REFERENCES people(memory_id) ON DELETE CASCADE,
        alias TEXT NOT NULL,
        normalized_alias TEXT NOT NULL,
        UNIQUE (person_memory_id, normalized_alias)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS relationships (
        memory_id TEXT PRIMARY KEY
            REFERENCES memories(id) ON DELETE CASCADE,
        source_person_id TEXT REFERENCES people(memory_id) ON DELETE RESTRICT,
        target_person_id TEXT REFERENCES people(memory_id) ON DELETE RESTRICT,
        relation_type TEXT NOT NULL,
        description TEXT,
        CHECK (source_person_id IS NOT NULL OR target_person_id IS NOT NULL)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        memory_id TEXT PRIMARY KEY
            REFERENCES memories(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        starts_at TEXT,
        ends_at TEXT,
        time_precision TEXT,
        timezone TEXT,
        location TEXT,
        event_status TEXT NOT NULL DEFAULT 'unknown'
            CHECK (event_status IN ('planned', 'ongoing', 'completed', 'cancelled', 'unknown')),
        follow_up_state TEXT NOT NULL DEFAULT 'none'
            CHECK (follow_up_state IN ('none', 'eligible', 'asked', 'dismissed')),
        follow_up_after TEXT,
        follow_up_asked_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_memory_id TEXT NOT NULL
            REFERENCES events(memory_id) ON DELETE CASCADE,
        person_memory_id TEXT REFERENCES people(memory_id) ON DELETE RESTRICT,
        is_owner INTEGER NOT NULL DEFAULT 0 CHECK (is_owner IN (0, 1)),
        role TEXT NOT NULL,
        CHECK (
            (is_owner = 1 AND person_memory_id IS NULL)
            OR (is_owner = 0 AND person_memory_id IS NOT NULL)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        turn_source_id INTEGER NOT NULL REFERENCES turn_sources(id) ON DELETE RESTRICT,
        excerpt TEXT NOT NULL,
        excerpt_sha256 TEXT NOT NULL CHECK (length(excerpt_sha256) = 64),
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        observed_at TEXT NOT NULL,
        UNIQUE (memory_id, turn_source_id, excerpt_sha256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        action TEXT NOT NULL,
        before_json TEXT,
        after_json TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (memory_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tombstones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL CHECK (kind IN ('person', 'relationship', 'event')),
        content_fingerprint TEXT NOT NULL CHECK (length(content_fingerprint) = 64),
        reason_code TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (kind, content_fingerprint)
    )
    """,
    "CREATE INDEX IF NOT EXISTS memories_lookup ON memories(kind, state, canonical_key)",
    "CREATE INDEX IF NOT EXISTS memories_expiry ON memories(retention_class, expires_at)",
    "CREATE INDEX IF NOT EXISTS people_name ON people(normalized_name)",
    "CREATE INDEX IF NOT EXISTS aliases_name ON person_aliases(normalized_alias)",
    "CREATE INDEX IF NOT EXISTS events_time ON events(starts_at, event_status)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS event_participant_identity
    ON event_participants(
        event_memory_id,
        COALESCE(person_memory_id, '__owner__'),
        role
    )
    """,
)


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


def open_database(database: Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(database).resolve()
    if read_only:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=0.02,
            factory=_ClosingConnection,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=1.0, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {20 if read_only else 1000}")
    if read_only:
        connection.execute("PRAGMA query_only = ON")
    else:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _schema_info(connection: sqlite3.Connection) -> SchemaInfo:
    values = dict(connection.execute("SELECT key, value FROM meta"))
    owner = connection.execute(
        "SELECT owner_uuid FROM owner_profile WHERE id = 1"
    ).fetchone()
    if owner is None:
        raise sqlite3.DatabaseError("memory database has no owner profile")
    return SchemaInfo(
        schema_version=int(values["schema_version"]),
        database_revision=int(values["database_revision"]),
        owner_uuid=str(owner[0]),
    )


def initialize_database(
    database: Path,
    *,
    owner_uuid: str | None = None,
) -> SchemaInfo:
    path = Path(database).resolve()
    if path.exists():
        with open_database(path, read_only=True) as probe:
            current_version = int(probe.execute("PRAGMA user_version").fetchone()[0])
        if current_version > SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"memory schema {current_version} is newer than supported {SCHEMA_VERSION}"
            )

    connection = open_database(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        created_at = "1970-01-01T00:00:00+00:00"
        selected_owner_uuid = owner_uuid or str(uuid4())
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('database_revision', '0')"
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO owner_profile(
                id, owner_uuid, display_name, profile, revision, created_at, updated_at
            ) VALUES (1, ?, NULL, NULL, 1, ?, ?)
            """,
            (selected_owner_uuid, created_at, created_at),
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
        return _schema_info(connection)
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
