from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

from avaturn_live_streamer.memory.models import (
    CandidateBatch,
    EventCandidate,
    EventParticipantCandidate,
    EventStatus,
    MemoryState,
    PersonCandidate,
    RelationshipCandidate,
)


_OWNER_NAME = re.compile(
    r"(?:^|[，,。！？!?\s])(?:我叫|我的名字是)\s*"
    r"(?P<name>[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9·._-]{0,31})"
)
_RELATION = (
    r"同事|朋友|姐姐|妹妹|哥哥|弟弟|妈妈|母亲|爸爸|父亲|妻子|丈夫|"
    r"女儿|儿子|老师|学生|老板|上司|邻居|同学|伴侣"
)
_NAMED_IS_OWNER_RELATION = re.compile(
    rf"(?P<name>[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9·._-]{{0,31}})"
    rf"(?:是|就是)我的(?P<relation>{_RELATION})"
)
_OWNER_RELATION_CALLED = re.compile(
    rf"我的(?P<relation>{_RELATION})(?:叫|名叫|是)"
    rf"(?P<name>[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9·._-]{{0,31}})"
)
_OWNER_RELATION_COMPACT = re.compile(
    rf"(?:^|[，,。！？!?\s])(?:我的)?(?P<relation>{_RELATION})"
    rf"(?P<name>[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9·._-]{{0,15}})"
    r"(?=[，,。！？!?\s]|$)"
)
_REMEMBER_CUE = re.compile(r"^(?:请)?(?:记住|记一下|记得|别忘了)\s*[：:，,]?\s*")
_ABSOLUTE_DATE = re.compile(
    r"(?P<year>20\d{2})\s*年\s*(?P<month>\d{1,2})\s*月\s*"
    r"(?P<day>\d{1,2})\s*[日号]?"
)
_MONTH_DAY = re.compile(
    r"(?<!年)(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*[日号]?"
)
_RELATIVE_DATE = re.compile(r"(?P<relative>前天|昨天|今天|明天|后天)")
_LOCATION = re.compile(
    r"在(?P<location>[\u3400-\u9fff]{2,12}?)"
    r"(?=和|跟|与|见面|开会|面试|出差|体检|参加|进行|完成)"
)
_EVENT_COMPANION = re.compile(
    r"(?:和|跟|与)(?P<name>[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9·._-]{0,15})"
    r"(?=见面|开会|会面|吃饭|旅行|出差|参加)"
)
_PLANNED_WORDS = ("计划", "打算", "准备", "预约", "将要", "要去")
_COMPLETED_WORDS = ("完成", "已经", "结束", "办完", "参加了", "去了")
_UNCERTAINTY_CUE = re.compile(r"听说|可能|也许|好像|似乎|大概|据说")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----\s+[A-Za-z0-9+/=]{24,}",
    re.IGNORECASE,
)
_KNOWN_CREDENTIAL_SHAPES = (
    re.compile(r"\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}\b"
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
)
_LABELED_CREDENTIAL = re.compile(
    r"(?P<label>"
    r"api[\s_-]*(?:key|密钥|密鑰|秘钥)|"
    r"access[\s_-]*token|refresh[\s_-]*token|auth[\s_-]*token|"
    r"client[\s_-]*secret|private[\s_-]*key|"
    r"password|passwd|pwd|secret|token|"
    r"访问令牌|刷新令牌|认证令牌|令牌|密码|口令|私钥|密钥|密鑰|秘钥"
    r")\s*(?:[:：=]|是|为|\bis\b)\s*"
    r'(?:"(?P<double>[^"\r\n]+)"|'
    r"'(?P<single>[^'\r\n]+)'|"
    r"(?P<bare>[^\s,，。！？!?;；\"']+))",
    re.IGNORECASE,
)
_NON_CREDENTIAL_VALUES = {
    "a",
    "an",
    "empty",
    "forgot",
    "forgotten",
    "important",
    "management",
    "missing",
    "needed",
    "none",
    "not",
    "null",
    "policy",
    "required",
    "rotation",
    "secure",
    "strong",
    "the",
    "unknown",
    "unavailable",
    "unsafe",
    "unset",
    "very",
    "weak",
    "what",
    "where",
    "why",
}
_NON_CREDENTIAL_PREFIXES = (
    "啥",
    "哪",
    "什么",
    "哪里",
    "为什么",
    "怎么",
    "如何",
    "忘了",
    "忘记",
    "管理",
    "轮换",
    "策略",
    "规范",
    "要求",
    "未设置",
    "没有",
    "为空",
)


def contains_sensitive_credential(text: str) -> bool:
    if _PRIVATE_KEY_BLOCK.search(text) is not None:
        return True
    if any(pattern.search(text) is not None for pattern in _KNOWN_CREDENTIAL_SHAPES):
        return True
    for match in _LABELED_CREDENTIAL.finditer(text):
        value = next(
            group
            for group in (
                match.group("double"),
                match.group("single"),
                match.group("bare"),
            )
            if group is not None
        ).strip()
        normalized = value.casefold().rstrip(".")
        if normalized in _NON_CREDENTIAL_VALUES or value.startswith(
            _NON_CREDENTIAL_PREFIXES
        ):
            continue
        return True
    return False


def _event_date(text: str, observed_at: datetime) -> datetime | None:
    absolute = _ABSOLUTE_DATE.search(text)
    if absolute is not None:
        parts = (
            int(absolute.group("year")),
            int(absolute.group("month")),
            int(absolute.group("day")),
        )
    else:
        month_day = _MONTH_DAY.search(text)
        if month_day is not None:
            parts = (
                observed_at.year,
                int(month_day.group("month")),
                int(month_day.group("day")),
            )
        else:
            relative = _RELATIVE_DATE.search(text)
            if relative is None:
                return None
            offsets = {"前天": -2, "昨天": -1, "今天": 0, "明天": 1, "后天": 2}
            target = observed_at + timedelta(days=offsets[relative.group("relative")])
            parts = (target.year, target.month, target.day)
    try:
        return datetime(*parts, tzinfo=observed_at.tzinfo)
    except ValueError:
        return None


class HeuristicMemoryExtractor:
    """Extract local-memory candidates from one final user transcript."""

    def extract_user_transcript(
        self,
        text: str,
        *,
        session_id: str,
        turn_id: str,
        engine_kind: str,
        observed_at: datetime,
    ) -> CandidateBatch:
        cleaned = text.strip()
        transcript_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if contains_sensitive_credential(cleaned):
            return CandidateBatch(
                session_id=session_id,
                turn_id=turn_id,
                engine_kind=engine_kind,
                observed_at=observed_at,
                transcript_sha256=transcript_sha256,
            )
        uncertain = _UNCERTAINTY_CUE.search(cleaned) is not None
        explicit_remember = _REMEMBER_CUE.match(cleaned) is not None
        confidence = 0.45 if uncertain else (1.0 if explicit_remember else 0.9)
        state = MemoryState.CANDIDATE if uncertain else MemoryState.CONFIRMED
        match = None if uncertain else _OWNER_NAME.search(cleaned)
        owner_name = match.group("name") if match is not None else None
        people: list[PersonCandidate] = []
        relationships: list[RelationshipCandidate] = []
        events: list[EventCandidate] = []
        relationship_source = _UNCERTAINTY_CUE.sub("", cleaned)
        relationship_match = _NAMED_IS_OWNER_RELATION.search(relationship_source)
        if relationship_match is None:
            relationship_match = _OWNER_RELATION_CALLED.search(relationship_source)
        if relationship_match is None:
            relationship_match = _OWNER_RELATION_COMPACT.search(relationship_source)
        if relationship_match is not None:
            name = relationship_match.group("name")
            relation = relationship_match.group("relation")
            evidence = cleaned.strip(" ，,。！？!?")[:160]
            people.append(
                PersonCandidate(
                    name=name,
                    notes=relation,
                    confidence=confidence,
                    state=state,
                    evidence_excerpt=evidence,
                )
            )
            relationships.append(
                RelationshipCandidate(
                    source_name=None,
                    target_name=name,
                    relation_type=relation,
                    confidence=confidence,
                    state=state,
                    evidence_excerpt=evidence,
                )
            )
        starts_at = _event_date(cleaned, observed_at)
        if starts_at is not None:
            if any(word in cleaned for word in _COMPLETED_WORDS):
                status = EventStatus.COMPLETED
            elif (
                any(word in cleaned for word in _PLANNED_WORDS)
                or starts_at.date() > observed_at.date()
            ):
                status = EventStatus.PLANNED
            elif starts_at.date() < observed_at.date():
                status = EventStatus.COMPLETED
            else:
                status = EventStatus.UNKNOWN
            participants: list[EventParticipantCandidate] = []
            if "我" in cleaned:
                participants.append(EventParticipantCandidate(is_owner=True))
            companion_match = _EVENT_COMPANION.search(cleaned)
            if companion_match is not None:
                companion_name = companion_match.group("name")
                participants.append(
                    EventParticipantCandidate(name=companion_name)
                )
            location_match = _LOCATION.search(cleaned)
            title = cleaned.strip(" ，,。！？!?")
            events.append(
                EventCandidate(
                    title=title,
                    summary=title,
                    starts_at=starts_at,
                    time_precision="day",
                    timezone=str(observed_at.tzinfo),
                    location=(
                        location_match.group("location")
                        if location_match is not None
                        else None
                    ),
                    status=status,
                    participants=tuple(participants),
                    follow_up_after=(
                        starts_at if status is EventStatus.PLANNED else None
                    ),
                    confidence=confidence,
                    state=state,
                    evidence_excerpt=title,
                )
            )
        if not events and explicit_remember:
            remembered = _REMEMBER_CUE.sub("", cleaned, count=1).strip(" ，,。！？!?")
            if remembered:
                events.append(
                    EventCandidate(
                        title=remembered,
                        summary=remembered,
                        confidence=confidence,
                        state=state,
                        evidence_excerpt=remembered,
                    )
                )
        return CandidateBatch(
            session_id=session_id,
            turn_id=turn_id,
            engine_kind=engine_kind,
            observed_at=observed_at,
            transcript_sha256=transcript_sha256,
            owner_name=owner_name,
            people=tuple(people),
            relationships=tuple(relationships),
            events=tuple(events),
        )
