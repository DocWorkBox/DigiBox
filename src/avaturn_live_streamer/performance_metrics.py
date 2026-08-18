# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Bounded, credential-free per-turn latency metrics for the local UI."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    Shutdown,
    TurnLatencyMilestone,
)

_PHASES = (
    "vad_complete",
    "asr_complete",
    "llm_first_token",
    "first_speakable_text",
    "tts_first_audio",
    "renderer_first_frame",
    "browser_first_audible",
)

_SUMMARY_STAGE_NAMES = (
    "speech_end_to_audible",
    "speech_end_to_frame",
    "asr",
    "llm_ttft",
    "chunk_wait",
    "tts_ttfa",
    "renderer",
)


@dataclass(slots=True)
class _TurnRecord:
    turn_id: str
    phases: dict[str, float] = field(default_factory=dict)
    details: dict[str, int | float | str] = field(default_factory=dict)
    status: str = "active"


class TurnLatencyStore:
    """Keep only the latest local-session timing records.

    Methods are deliberately synchronous: all callers run on the FastAPI event
    loop and none of them await while mutating a record, so snapshots cannot
    observe a partially-applied milestone.
    """

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._max_turns = max_turns
        self._session_id: str | None = None
        self._active = False
        self._turns: OrderedDict[str, _TurnRecord] = OrderedDict()

    def begin_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._active = True
        self._turns.clear()

    def end_session(self) -> None:
        self._active = False

    def record(self, event: TurnLatencyMilestone) -> None:
        record = self._turns.get(event.turn_id)
        if record is None:
            record = _TurnRecord(turn_id=event.turn_id)
            self._turns[event.turn_id] = record
            while len(self._turns) > self._max_turns:
                self._turns.popitem(last=False)
        record.phases.setdefault(event.phase, event.at_monotonic)
        record.details.update(event.details)
        if event.phase == "renderer_first_frame":
            record.status = "complete"
        elif event.phase in ("failed", "interrupted"):
            record.status = event.phase

    def record_browser_first_audible(self, at_monotonic: float) -> str | None:
        """Attach one browser playout observation to the latest speaking turn."""

        for turn_id, record in reversed(self._turns.items()):
            if record.status in {"failed", "interrupted"}:
                continue
            if "tts_first_audio" not in record.phases:
                continue
            if "browser_first_audible" in record.phases:
                continue
            record.phases["browser_first_audible"] = at_monotonic
            return turn_id
        return None

    @staticmethod
    def _duration_ms(record: _TurnRecord, start: str, end: str) -> int | None:
        if start not in record.phases or end not in record.phases:
            return None
        return max(0, round((record.phases[end] - record.phases[start]) * 1000))

    def _serialise_turn(self, record: _TurnRecord) -> dict[str, object]:
        origin = record.phases.get("vad_complete")
        if origin is None and record.phases:
            origin = min(record.phases.values())
        phase_ms = {
            phase: max(0, round((record.phases[phase] - origin) * 1000))
            for phase in _PHASES
            if origin is not None and phase in record.phases
        }

        stage_pairs = (
            ("asr", "vad_complete", "asr_complete"),
            ("llm_ttft", "asr_complete", "llm_first_token"),
            ("chunk_wait", "llm_first_token", "first_speakable_text"),
            ("tts_ttfa", "first_speakable_text", "tts_first_audio"),
            ("renderer", "tts_first_audio", "renderer_first_frame"),
            ("post_vad_to_frame", "vad_complete", "renderer_first_frame"),
            ("post_vad_to_audible", "vad_complete", "browser_first_audible"),
            ("tts_to_audible", "tts_first_audio", "browser_first_audible"),
        )
        durations: dict[str, int] = {}
        vad_wait = record.details.get("vad_wait_ms")
        if isinstance(vad_wait, (int, float)):
            durations["vad_wait"] = max(0, round(vad_wait))
        for name, start, end in stage_pairs:
            value = self._duration_ms(record, start, end)
            if value is not None:
                durations[name] = value
        if "vad_wait" in durations and "post_vad_to_frame" in durations:
            durations["speech_end_to_frame"] = (
                durations["vad_wait"] + durations["post_vad_to_frame"]
            )
        if "vad_wait" in durations and "post_vad_to_audible" in durations:
            durations["speech_end_to_audible"] = (
                durations["vad_wait"] + durations["post_vad_to_audible"]
            )

        thinking_mode = str(record.details.get("thinking_mode", "fast")).lower()
        slo_eligible = thinking_mode != "deep"

        return {
            "turn_id": record.turn_id,
            "status": record.status,
            "slo_eligible": slo_eligible,
            "slo_exclusion_reason": None if slo_eligible else "deep_thinking",
            "phase_ms": phase_ms,
            "durations_ms": durations,
        }

    @staticmethod
    def _nearest_rank(values: list[int], percentile: float) -> int:
        ordered = sorted(values)
        index = max(0, math.ceil(len(ordered) * percentile) - 1)
        return ordered[index]

    def _summary(self, turns: list[dict[str, object]]) -> dict[str, object]:
        fast_complete = [
            turn for turn in turns if turn["status"] == "complete" and turn["slo_eligible"] is True
        ]
        fast_slo_complete_turns = sum(
            1
            for turn in fast_complete
            if isinstance(turn["durations_ms"], dict)
            and isinstance(turn["durations_ms"].get("speech_end_to_audible"), int)
        )
        stages: dict[str, dict[str, int]] = {}
        for name in _SUMMARY_STAGE_NAMES:
            values = [
                int(durations[name])
                for turn in fast_complete
                if isinstance((durations := turn["durations_ms"]), dict)
                and isinstance(durations.get(name), int)
            ]
            if values:
                stages[name] = {
                    "samples": len(values),
                    "p50": self._nearest_rank(values, 0.50),
                    "p95": self._nearest_rank(values, 0.95),
                }
        return {
            "fast_slo_complete_turns": fast_slo_complete_turns,
            "stages_ms": stages,
        }

    def snapshot(self) -> dict[str, object]:
        turns = [self._serialise_turn(record) for record in self._turns.values()]
        return {
            "session_id": self._session_id,
            "active": self._active,
            "turns": turns,
            "summary": self._summary(turns),
        }


class TurnLatencyCollector:
    """Collect milestones measured by the component that owns each phase."""

    def __init__(self, store: TurnLatencyStore) -> None:
        self._store = store

    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        _ = clocks
        async with bus.subscribe(
            TurnLatencyMilestone,
            Shutdown,
        ) as subscription:
            bus.ready()
            async for event in subscription:
                if isinstance(event, Shutdown):
                    return
                if isinstance(event, TurnLatencyMilestone):
                    self._store.record(event)
