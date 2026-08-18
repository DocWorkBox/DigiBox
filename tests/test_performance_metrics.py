from __future__ import annotations

import asyncio
from importlib import import_module
from types import ModuleType

import pytest

from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import SegmentPlaybackStarted, Shutdown
from avaturn_live_streamer.management.types import SegmentId


def _subject() -> ModuleType:
    try:
        return import_module("avaturn_live_streamer.performance_metrics")
    except ModuleNotFoundError:
        pytest.fail("performance_metrics is not implemented", pytrace=False)


def _milestone(subject: ModuleType, phase: str, at: float, **details: int | str):
    event_type = getattr(subject, "TurnLatencyMilestone", None)
    if event_type is None:
        pytest.fail("TurnLatencyMilestone is not implemented", pytrace=False)
    return event_type(
        turn_id="turn-1",
        phase=phase,
        at_monotonic=at,
        details=details,
    )


def test_turn_latency_store_reports_stage_and_end_to_frame_durations() -> None:
    subject = _subject()
    store = subject.TurnLatencyStore(max_turns=4)
    store.begin_session("session-1")

    store.record(_milestone(subject, "vad_complete", 10.00, vad_wait_ms=320))
    store.record(_milestone(subject, "asr_complete", 10.45))
    store.record(_milestone(subject, "llm_first_token", 10.70))
    store.record(_milestone(subject, "first_speakable_text", 10.80))
    store.record(_milestone(subject, "tts_first_audio", 10.95))
    store.record(_milestone(subject, "renderer_first_frame", 11.10))

    snapshot = store.snapshot()
    assert snapshot["session_id"] == "session-1"
    assert snapshot["active"] is True
    assert snapshot["turns"] == [
        {
            "turn_id": "turn-1",
            "status": "complete",
            "slo_eligible": True,
            "slo_exclusion_reason": None,
            "phase_ms": {
                "vad_complete": 0,
                "asr_complete": 450,
                "llm_first_token": 700,
                "first_speakable_text": 800,
                "tts_first_audio": 950,
                "renderer_first_frame": 1100,
            },
            "durations_ms": {
                "vad_wait": 320,
                "asr": 450,
                "llm_ttft": 250,
                "chunk_wait": 100,
                "tts_ttfa": 150,
                "renderer": 150,
                "post_vad_to_frame": 1100,
                "speech_end_to_frame": 1420,
            },
        }
    ]
    # A renderer frame alone is not enough to claim the audible-response SLO.
    assert snapshot["summary"]["fast_slo_complete_turns"] == 0
    assert snapshot["summary"]["stages_ms"]["speech_end_to_frame"] == {
        "samples": 1,
        "p50": 1420,
        "p95": 1420,
    }


def test_turn_latency_store_is_bounded_and_resets_between_sessions() -> None:
    subject = _subject()
    store = subject.TurnLatencyStore(max_turns=2)
    store.begin_session("first")
    for index in range(3):
        store.record(
            subject.TurnLatencyMilestone(
                turn_id=f"turn-{index}",
                phase="vad_complete",
                at_monotonic=float(index),
                details={"vad_wait_ms": 320},
            )
        )

    assert [item["turn_id"] for item in store.snapshot()["turns"]] == [
        "turn-1",
        "turn-2",
    ]

    store.end_session()
    assert store.snapshot()["active"] is False
    store.begin_session("second")
    assert store.snapshot() == {
        "session_id": "second",
        "active": True,
        "turns": [],
        "summary": {"fast_slo_complete_turns": 0, "stages_ms": {}},
    }


def test_collector_does_not_treat_scheduled_playback_as_a_rendered_frame() -> None:
    subject = _subject()

    async def exercise() -> dict[str, object]:
        store = subject.TurnLatencyStore()
        store.begin_session("session-1")
        collector = subject.TurnLatencyCollector(store)
        bus = EventBus()
        task = asyncio.create_task(collector.run(bus.clone(), object()))
        bus.ready()

        await bus.publish(_milestone(subject, "tts_first_audio", 4.0))
        await bus.publish(
            SegmentPlaybackStarted(
                segment_id=SegmentId("segment-1"),
                metadata={"turn_id": "turn-1"},
            )
        )
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return store.snapshot()

    snapshot = asyncio.run(exercise())
    turn = snapshot["turns"][0]
    assert "renderer_first_frame" not in turn["phase_ms"]
    assert turn["status"] == "active"


def test_browser_first_audible_is_attached_to_latest_speaking_turn() -> None:
    subject = _subject()
    store = subject.TurnLatencyStore(max_turns=4)
    store.begin_session("session-1")

    store.record(_milestone(subject, "vad_complete", 10.00, vad_wait_ms=220))
    store.record(_milestone(subject, "tts_first_audio", 10.70))
    store.record(_milestone(subject, "renderer_first_frame", 10.80))

    assert store.record_browser_first_audible(10.95) == "turn-1"
    assert store.record_browser_first_audible(11.00) is None

    turn = store.snapshot()["turns"][0]
    assert turn["phase_ms"]["renderer_first_frame"] == 800
    assert turn["phase_ms"]["browser_first_audible"] == 950
    assert turn["durations_ms"]["speech_end_to_frame"] == 1020
    assert turn["durations_ms"]["speech_end_to_audible"] == 1170


def test_latency_summary_excludes_deep_thinking_turns_from_fast_slo() -> None:
    subject = _subject()
    store = subject.TurnLatencyStore(max_turns=8)
    store.begin_session("session-1")

    for index, response_ms in enumerate((1_000, 2_000, 3_000)):
        turn_id = f"fast-{index}"
        store.record(
            subject.TurnLatencyMilestone(
                turn_id=turn_id,
                phase="vad_complete",
                at_monotonic=10.0 + index * 10,
                details={"vad_wait_ms": 220, "thinking_mode": "fast"},
            )
        )
        store.record(
            subject.TurnLatencyMilestone(
                turn_id=turn_id,
                phase="renderer_first_frame",
                at_monotonic=10.5 + index * 10,
            )
        )
        store.record(
            subject.TurnLatencyMilestone(
                turn_id=turn_id,
                phase="browser_first_audible",
                at_monotonic=10.0 + index * 10 + (response_ms - 220) / 1_000,
            )
        )

    store.record(
        subject.TurnLatencyMilestone(
            turn_id="deep-1",
            phase="vad_complete",
            at_monotonic=50.0,
            details={"vad_wait_ms": 220, "thinking_mode": "deep"},
        )
    )
    store.record(
        subject.TurnLatencyMilestone(
            turn_id="deep-1",
            phase="renderer_first_frame",
            at_monotonic=58.0,
        )
    )
    store.record(
        subject.TurnLatencyMilestone(
            turn_id="deep-1",
            phase="browser_first_audible",
            at_monotonic=58.78,
        )
    )

    snapshot = store.snapshot()
    deep = next(turn for turn in snapshot["turns"] if turn["turn_id"] == "deep-1")
    assert deep["slo_eligible"] is False
    assert deep["slo_exclusion_reason"] == "deep_thinking"
    assert snapshot["summary"]["fast_slo_complete_turns"] == 3
    assert snapshot["summary"]["stages_ms"]["speech_end_to_audible"] == {
        "samples": 3,
        "p50": 2_000,
        "p95": 3_000,
    }
