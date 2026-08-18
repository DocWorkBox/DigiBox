from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from avaturn_live_streamer.memory.extractor import HeuristicMemoryExtractor
from avaturn_live_streamer.memory.models import EventStatus, MemoryState


SHANGHAI = timezone(timedelta(hours=8))
OBSERVED_AT = datetime(2026, 8, 17, 10, 30, tzinfo=SHANGHAI)


@pytest.mark.parametrize(
    "transcript",
    [
        "我叫小雨。请记住我的 API Key 是 sk-test-secret。",
        "张三是我的同事，密码是 123456。",
        "明天讨论部署，access token: abcDEF123456。",
        "请记住，password: hunter2。",
        '请记住，client_secret="abcDEF123456"。',
        (
            "请记住这个私钥：-----BEGIN "
            "PRIVATE KEY----- "
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC -----END PRIVATE KEY-----"
        ),
        "明天轮换 " "ghp_" "0123456789abcdefghijklmnopqrst。",
        "明天更换 " "AKIA" "ABCDEFGHIJKLMNOP。",
        (
            "明天刷新 eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "abcdefghijklmnopqrstuvwx。"
        ),
        "明天停用 Bearer abcDEF1234567890。",
        "请记住，API Key 是 abc$def!ghi。",
        "明天轮换 access token: abc:def@ghi。",
        "请记住，secret is p@ssw0rd!。",
    ],
)
def test_sensitive_credential_rejects_the_entire_candidate_batch(
    transcript: str,
) -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        transcript,
        session_id="session-sensitive",
        turn_id="turn-sensitive",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert batch.owner_name is None
    assert batch.people == ()
    assert batch.relationships == ()
    assert batch.events == ()


@pytest.mark.parametrize(
    "transcript, expected_title",
    [
        ("请记住，下周讨论 API Key 管理流程。", "下周讨论 API Key 管理流程"),
        ("我忘了密码，明天和张三开会。", "我忘了密码，明天和张三开会"),
        (
            "请记住，API Key 是什么需要下周确认。",
            "API Key 是什么需要下周确认",
        ),
        (
            "请记住，password policy must require 12 characters。",
            "password policy must require 12 characters",
        ),
        (
            "请记住，字符串 -----BEGIN " "PRIVATE KEY----- 只是文件头示例。",
            "字符串 -----BEGIN " "PRIVATE KEY----- 只是文件头示例",
        ),
        (
            "请记住，password is required for login。",
            "password is required for login",
        ),
    ],
)
def test_credential_discussion_without_a_value_is_still_extractable(
    transcript: str,
    expected_title: str,
) -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        transcript,
        session_id="session-safe-discussion",
        turn_id="turn-safe-discussion",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert len(batch.events) == 1
    assert batch.events[0].title == expected_title


def test_extracts_explicit_owner_name() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "我叫小雨。",
        session_id="session-1",
        turn_id="turn-1",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert batch.owner_name == "小雨"
    assert batch.session_id == "session-1"
    assert batch.turn_id == "turn-1"
    assert batch.observed_at == OBSERVED_AT
    assert len(batch.transcript_sha256) == 64


def test_uncertain_owner_name_is_not_promoted_to_the_owner_profile() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "我叫小雨，但也许我记错了。",
        session_id="session-1",
        turn_id="turn-owner-uncertain",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert batch.owner_name is None


def test_extracts_named_person_is_owner_relationship() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "张三是我的同事。",
        session_id="session-1",
        turn_id="turn-2",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert [(person.name, person.state) for person in batch.people] == [
        ("张三", MemoryState.CONFIRMED)
    ]
    assert len(batch.relationships) == 1
    relationship = batch.relationships[0]
    assert relationship.source_name is None
    assert relationship.target_name == "张三"
    assert relationship.relation_type == "同事"


def test_extracts_owner_relationship_called_named_person() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "我的姐姐叫李四。",
        session_id="session-1",
        turn_id="turn-3",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert [person.name for person in batch.people] == ["李四"]
    assert len(batch.relationships) == 1
    relationship = batch.relationships[0]
    assert relationship.source_name is None
    assert relationship.target_name == "李四"
    assert relationship.relation_type == "姐姐"


def test_extracts_compact_relation_context_for_same_named_people() -> None:
    extractor = HeuristicMemoryExtractor()

    colleague = extractor.extract_user_transcript(
        "同事张三。",
        session_id="session-1",
        turn_id="compact-colleague",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )
    neighbor = extractor.extract_user_transcript(
        "邻居张三。",
        session_id="session-1",
        turn_id="compact-neighbor",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert [(item.name, item.notes) for item in colleague.people] == [
        ("张三", "同事")
    ]
    assert [(item.name, item.notes) for item in neighbor.people] == [("张三", "邻居")]


def test_explicit_remember_cue_keeps_unstructured_fact_as_confirmed_event() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "请记住，小雨喜欢喝拿铁。",
        session_id="session-1",
        turn_id="turn-4",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert len(batch.events) == 1
    event = batch.events[0]
    assert event.title == "小雨喜欢喝拿铁"
    assert event.state is MemoryState.CONFIRMED
    assert event.confidence == 1.0
    assert event.evidence_excerpt == "小雨喜欢喝拿铁"


def test_extracts_dated_planned_event_with_location_and_participants() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "我计划2026年9月3日在上海和张三见面。",
        session_id="session-1",
        turn_id="turn-5",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert len(batch.events) == 1
    event = batch.events[0]
    assert event.status is EventStatus.PLANNED
    assert event.starts_at == datetime(2026, 9, 3, tzinfo=SHANGHAI)
    assert event.time_precision == "day"
    assert event.location == "上海"
    assert event.follow_up_after == event.starts_at
    assert [(item.name, item.is_owner) for item in event.participants] == [
        (None, True),
        ("张三", False),
    ]


def test_extracts_dated_completed_event() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "我在2026年8月16日完成了体检。",
        session_id="session-1",
        turn_id="turn-6",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert len(batch.events) == 1
    event = batch.events[0]
    assert event.status is EventStatus.COMPLETED
    assert event.starts_at == datetime(2026, 8, 16, tzinfo=SHANGHAI)
    assert event.follow_up_after is None


def test_uncertain_relationship_becomes_low_confidence_candidate() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "听说张三可能是我的同事。",
        session_id="session-1",
        turn_id="turn-7",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert [person.name for person in batch.people] == ["张三"]
    assert batch.people[0].state is MemoryState.CANDIDATE
    assert batch.people[0].confidence < 0.60
    assert len(batch.relationships) == 1
    assert batch.relationships[0].state is MemoryState.CANDIDATE
    assert batch.relationships[0].confidence < 0.60
    assert batch.relationships[0].evidence_excerpt == "听说张三可能是我的同事"


def test_extracts_tomorrow_event_with_named_person() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "明天我要和张三开会。",
        session_id="session-1",
        turn_id="turn-8",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert len(batch.events) == 1
    event = batch.events[0]
    assert event.starts_at == datetime(2026, 8, 18, tzinfo=SHANGHAI)
    assert event.status is EventStatus.PLANNED
    assert event.follow_up_after == event.starts_at
    assert [(item.name, item.is_owner) for item in event.participants] == [
        (None, True),
        ("张三", False),
    ]


def test_extracts_yearless_month_day_using_observed_year() -> None:
    batch = HeuristicMemoryExtractor().extract_user_transcript(
        "8月20日我在上海和王强见面。",
        session_id="session-1",
        turn_id="turn-9",
        engine_kind="custom_api",
        observed_at=OBSERVED_AT,
    )

    assert len(batch.events) == 1
    event = batch.events[0]
    assert event.starts_at == datetime(2026, 8, 20, tzinfo=SHANGHAI)
    assert event.location == "上海"
    assert [item.name for item in event.participants if not item.is_owner] == ["王强"]
