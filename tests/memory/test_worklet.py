from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from types import ModuleType
from uuid import uuid4


if importlib.util.find_spec("typeid") is None:
    typeid_stub = ModuleType("typeid")

    class _TypeID:
        @classmethod
        def from_uuid(cls, value, prefix):
            _ = prefix
            return value

        @classmethod
        def from_string(cls, value):
            return cls(value)

    typeid_stub.TypeID = _TypeID  # type: ignore[attr-defined]
    sys.modules["typeid"] = typeid_stub

if importlib.util.find_spec("uuid6") is None:
    uuid6_stub = ModuleType("uuid6")
    uuid6_stub.uuid7 = uuid4  # type: ignore[attr-defined]
    sys.modules["uuid6"] = uuid6_stub

from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import InputTranscript, ResponseTranscript, Shutdown
from avaturn_live_streamer.memory.extractor import HeuristicMemoryExtractor
from avaturn_live_streamer.memory.models import CandidateBatch
from avaturn_live_streamer.memory.worklet import MemoryWorklet


class _RecordingService:
    def __init__(self) -> None:
        self.batches: list[CandidateBatch] = []
        self.submitted = asyncio.Event()

    def try_submit(self, batch: CandidateBatch) -> object:
        self.batches.append(batch)
        self.submitted.set()
        return object()


def test_worklet_extracts_only_final_user_input_transcript() -> None:
    async def exercise() -> list[CandidateBatch]:
        service = _RecordingService()
        worklet = MemoryWorklet(
            service=service,
            extractor=HeuristicMemoryExtractor(),
            session_id="session-1",
            engine_kind="custom_api",
        )
        bus = EventBus()
        task = asyncio.create_task(worklet.run(bus.clone(), object()))
        bus.ready()

        await bus.publish(
            ResponseTranscript(transcript="用户的姐姐叫李四。", timestamp=1.0)
        )
        await bus.publish(InputTranscript(transcript="我的姐姐叫李四。", timestamp=2.0))
        await asyncio.wait_for(service.submitted.wait(), timeout=0.5)
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return service.batches

    batches = asyncio.run(exercise())

    assert len(batches) == 1
    assert [person.name for person in batches[0].people] == ["李四"]
    assert batches[0].session_id == "session-1"


def test_worklet_deduplicates_replayed_input_transcript() -> None:
    async def exercise() -> list[CandidateBatch]:
        service = _RecordingService()
        worklet = MemoryWorklet(
            service=service,
            extractor=HeuristicMemoryExtractor(),
            session_id="session-1",
            engine_kind="openai",
        )
        bus = EventBus()
        task = asyncio.create_task(worklet.run(bus.clone(), object()))
        bus.ready()
        event = InputTranscript(transcript="我叫小雨。", timestamp=3.0)

        await bus.publish(event)
        await bus.publish(event)
        await asyncio.wait_for(service.submitted.wait(), timeout=0.5)
        for _ in range(20):
            await asyncio.sleep(0)
        await bus.publish(Shutdown())
        await task
        return service.batches

    batches = asyncio.run(exercise())

    assert len(batches) == 1


def test_worklet_does_not_submit_a_transcript_without_extractable_facts() -> None:
    async def exercise() -> list[CandidateBatch]:
        service = _RecordingService()
        worklet = MemoryWorklet(
            service=service,
            extractor=HeuristicMemoryExtractor(),
            session_id="session-empty",
            engine_kind="custom_api",
        )
        bus = EventBus()
        task = asyncio.create_task(worklet.run(bus.clone(), object()))
        bus.ready()

        await bus.publish(
            InputTranscript(transcript="你好呀。", timestamp=4.0)
        )
        for _ in range(20):
            await asyncio.sleep(0)
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return service.batches

    assert asyncio.run(exercise()) == []


def test_full_internal_queue_never_blocks_event_bus_and_shutdown_clears_it() -> None:
    async def exercise() -> tuple[MemoryWorklet, float]:
        service = _RecordingService()
        worklet = MemoryWorklet(
            service=service,
            extractor=HeuristicMemoryExtractor(),
            session_id="session-1",
            engine_kind="openai",
            queue_size=1,
        )
        bus = EventBus()
        task = asyncio.create_task(worklet.run(bus.clone(), object()))
        bus.ready()

        started_at = asyncio.get_running_loop().time()
        for index in range(20):
            await bus.publish(
                InputTranscript(transcript=f"我叫小雨{index}。", timestamp=10.0 + index)
            )
        elapsed = asyncio.get_running_loop().time() - started_at
        await bus.publish(Shutdown())
        await task
        return worklet, elapsed

    worklet, elapsed = asyncio.run(exercise())

    assert elapsed < 0.05
    assert worklet.dropped_transcripts > 0
    assert worklet.pending_transcripts == 0


def test_extraction_runs_off_the_realtime_event_loop_thread() -> None:
    class ThreadRecordingExtractor(HeuristicMemoryExtractor):
        def __init__(self) -> None:
            self.thread_id: int | None = None

        def extract_user_transcript(self, *args, **kwargs):
            self.thread_id = threading.get_ident()
            return super().extract_user_transcript(*args, **kwargs)

    async def exercise() -> tuple[int, int | None]:
        event_loop_thread = threading.get_ident()
        extractor = ThreadRecordingExtractor()
        service = _RecordingService()
        worklet = MemoryWorklet(
            service=service,
            extractor=extractor,
            session_id="session-thread",
            engine_kind="custom_api",
        )
        bus = EventBus()
        task = asyncio.create_task(worklet.run(bus.clone(), object()))
        bus.ready()
        await bus.publish(InputTranscript(transcript="我叫小雨。", timestamp=20.0))
        await asyncio.wait_for(service.submitted.wait(), timeout=0.5)
        await bus.publish(Shutdown())
        await task
        return event_loop_thread, extractor.thread_id

    event_loop_thread, extractor_thread = asyncio.run(exercise())

    assert extractor_thread is not None
    assert extractor_thread != event_loop_thread


def test_shutdown_briefly_drains_an_already_accepted_transcript() -> None:
    class SlowExtractor(HeuristicMemoryExtractor):
        def extract_user_transcript(self, *args, **kwargs):
            time.sleep(0.03)
            return super().extract_user_transcript(*args, **kwargs)

    async def exercise() -> list[CandidateBatch]:
        service = _RecordingService()
        worklet = MemoryWorklet(
            service=service,
            extractor=SlowExtractor(),
            session_id="session-drain",
            engine_kind="openai",
        )
        bus = EventBus()
        task = asyncio.create_task(worklet.run(bus.clone(), object()))
        bus.ready()
        await bus.publish(InputTranscript(transcript="我的姐姐叫李四。", timestamp=21.0))
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return service.batches

    batches = asyncio.run(exercise())

    assert len(batches) == 1
