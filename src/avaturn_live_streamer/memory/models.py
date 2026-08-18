from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path


class MemoryKind(StrEnum):
    PERSON = "person"
    RELATIONSHIP = "relationship"
    EVENT = "event"


class MemoryState(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


class RetentionClass(StrEnum):
    TEMPORARY_30D = "temporary_30d"
    PERSISTENT = "persistent"


class EventStatus(StrEnum):
    PLANNED = "planned"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class FollowUpState(StrEnum):
    NONE = "none"
    ELIGIBLE = "eligible"
    ASKED = "asked"
    DISMISSED = "dismissed"


class SubmissionReason(StrEnum):
    ACCEPTED = "accepted"
    NOT_STARTED = "not_started"
    DEGRADED = "degraded"
    QUEUE_FULL = "queue_full"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    accepted: bool
    reason: SubmissionReason | str
    pending_count: int

    def __post_init__(self) -> None:
        if self.pending_count < 0:
            raise ValueError("pending_count must not be negative")


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_confidence(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    low_confidence_cutoff: float = 0.60
    candidate_ttl: timedelta = timedelta(days=30)
    recall_limit: int = 5
    session_profile_limit: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.low_confidence_cutoff < 1.0:
            raise ValueError("low_confidence_cutoff must be between 0 and 1")
        if self.candidate_ttl <= timedelta(0):
            raise ValueError("candidate_ttl must be positive")
        if self.recall_limit <= 0 or self.session_profile_limit <= 0:
            raise ValueError("memory limits must be positive")


@dataclass(frozen=True, slots=True)
class PersonCandidate:
    name: str
    aliases: tuple[str, ...] = ()
    notes: str | None = None
    confidence: float = 1.0
    state: MemoryState = MemoryState.CONFIRMED
    evidence_excerpt: str = ""

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class RelationshipCandidate:
    source_name: str | None
    target_name: str | None
    relation_type: str
    description: str | None = None
    confidence: float = 1.0
    state: MemoryState = MemoryState.CONFIRMED
    evidence_excerpt: str = ""

    def __post_init__(self) -> None:
        if self.source_name is None and self.target_name is None:
            raise ValueError("a relationship must include at least one person")
        _require_text(self.relation_type, "relation_type")
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class EventParticipantCandidate:
    name: str | None = None
    role: str = "participant"
    is_owner: bool = False

    def __post_init__(self) -> None:
        if self.is_owner == (self.name is not None):
            raise ValueError("participant must identify either owner or one named person")
        _require_text(self.role, "role")


@dataclass(frozen=True, slots=True)
class EventCandidate:
    title: str
    summary: str = ""
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    time_precision: str | None = None
    timezone: str | None = None
    location: str | None = None
    status: EventStatus = EventStatus.UNKNOWN
    participants: tuple[EventParticipantCandidate, ...] = ()
    follow_up_after: datetime | None = None
    confidence: float = 1.0
    state: MemoryState = MemoryState.CONFIRMED
    evidence_excerpt: str = ""

    def __post_init__(self) -> None:
        _require_text(self.title, "title")
        _require_confidence(self.confidence)
        for field_name, value in (
            ("starts_at", self.starts_at),
            ("ends_at", self.ends_at),
            ("follow_up_after", self.follow_up_after),
        ):
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    session_id: str
    turn_id: str
    engine_kind: str
    observed_at: datetime
    transcript_sha256: str
    owner_name: str | None = None
    people: tuple[PersonCandidate, ...] = ()
    relationships: tuple[RelationshipCandidate, ...] = ()
    events: tuple[EventCandidate, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_text(self.turn_id, "turn_id")
        _require_text(self.engine_kind, "engine_kind")
        if self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if len(self.transcript_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in self.transcript_sha256
        ):
            raise ValueError("transcript_sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True, slots=True)
class OwnerProfile:
    owner_uuid: str
    display_name: str | None
    profile: str | None
    revision: int


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    kind: MemoryKind
    state: MemoryState
    confidence: float
    retention_class: RetentionClass
    canonical_key: str
    revision: int
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime
    expires_at: datetime | None
    summary: str
    starts_at: datetime | None = None
    event_status: EventStatus | None = None
    follow_up_state: FollowUpState | None = None
    person_name: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryStats:
    total: int
    confirmed: int
    candidates: int
    low_confidence_candidates: int
    people: int
    relationships: int
    events: int
    pending_followups: int
    database_revision: int


@dataclass(frozen=True, slots=True)
class MemoryRecordPage:
    items: tuple[MemoryRecord, ...]
    next_cursor: str | None
    database_revision: int


@dataclass(frozen=True, slots=True)
class ForgetResult:
    memory_id: str
    kind: MemoryKind
    database_revision: int
    deleted_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestResult:
    replayed: bool
    person_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    database_revision: int


@dataclass(frozen=True, slots=True)
class RecallQuery:
    text: str
    now: datetime
    limit: int = 5

    def __post_init__(self) -> None:
        if self.now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if self.limit <= 0:
            raise ValueError("limit must be positive")


@dataclass(frozen=True, slots=True)
class RecalledMemory:
    memory_id: str
    kind: MemoryKind
    summary: str
    score: float
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RecallResult:
    items: tuple[RecalledMemory, ...]
    database_revision: int | None
    timed_out: bool = False
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SessionMemoryContext:
    prompt: str
    item_ids: tuple[str, ...]
    follow_up_id: str | None
    database_revision: int | None
    follow_up_summary: str | None = None
    follow_up_revision: int | None = None
    timed_out: bool = False
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PurgeReport:
    deleted_ids: tuple[str, ...]
    retained_due_to_reference: tuple[str, ...]
    database_revision: int


@dataclass(frozen=True, slots=True)
class ClearResult:
    deleted_count: int
    database_revision: int
    backup_path: Path | None = None
