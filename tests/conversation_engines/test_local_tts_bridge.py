from __future__ import annotations

import asyncio

from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    TurnLatencyMilestone,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


def _buffer(value: int) -> SpeechBuffer:
    return SpeechBuffer.from_bytes(value.to_bytes(2, "little", signed=True) * 2, 24_000)


def test_sentence_chunker_uses_shared_low_latency_boundaries() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        SentenceChunker,
    )

    chunker = SentenceChunker()

    comma_piece = "甲" * 11 + "\uff0c"
    assert chunker.feed(comma_piece) == [comma_piece]
    assert chunker.feed("乙" * 25) == ["乙" * 24]
    assert chunker.finish() == ["乙"]

    assert chunker.feed("短句。") == ["短句。"]
    assert chunker.finish() == []


def test_sentence_chunker_releases_a_short_cjk_first_piece_without_splitting_words() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        SentenceChunker,
    )

    chinese = SentenceChunker(first_max_chars=8, max_chars=24, soft_min_chars=4)
    assert chinese.feed("当然可以\uff0c我") == ["当然可以\uff0c"]
    assert chinese.feed("现在回答你的问题") == []
    assert chinese.finish() == ["我现在回答你的问题"]

    english = SentenceChunker(first_max_chars=8, max_chars=24, soft_min_chars=4)
    assert english.feed("Absolutely") == []
    assert english.feed(" yes, I can help") == ["Absolutely yes,"]


def test_local_tts_idle_flush_starts_a_short_answer_before_response_done() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class RecordingTTS:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.texts: list[str] = []

        async def stream_speech(self, text: str):
            self.texts.append(text)
            self.started.set()
            yield _buffer(1)

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        tts = RecordingTTS()
        bridge = StreamingLocalTTSBridge(
            tts,
            task_name="test.local_tts_idle_flush",
            idle_flush_seconds=0.02,
        )
        bus = EventBus()
        bus.ready()
        await bridge.feed_text(bus, "response-1", "我知道了")
        await asyncio.wait_for(tts.started.wait(), timeout=0.2)
        await bridge.finish_text(bus, "response-1")
        await bridge.wait_idle(timeout=0.5)
        await bridge.close()
        return tts.texts

    assert asyncio.run(exercise()) == ["我知道了"]


def test_local_tts_uses_one_incremental_session_and_preserves_clone_voice() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class FakeTurn:
        def __init__(self, voice: str) -> None:
            self.voice = voice
            self.texts: list[str] = []
            self.audio: asyncio.Queue[SpeechBuffer | None] = asyncio.Queue()
            self.finished = False
            self.cancelled = False

        async def send_text(self, text: str) -> None:
            self.texts.append(text)
            await self.audio.put(_buffer(len(self.texts)))

        async def finish_text(self) -> None:
            self.finished = True
            await self.audio.put(None)

        async def stream_audio(self):
            while True:
                item = await self.audio.get()
                if item is None:
                    return
                yield item

        async def cancel(self) -> None:
            self.cancelled = True
            await self.audio.put(None)

    class IncrementalCloneTTS:
        def __init__(self) -> None:
            self.voice = "voice_local_clone"
            self.turns: list[FakeTurn] = []

        async def open_text_stream(self) -> FakeTurn:
            turn = FakeTurn(self.voice)
            self.turns.append(turn)
            return turn

        async def stream_speech(self, text: str):
            raise AssertionError(f"incremental backend must not use per-piece HTTP: {text}")
            yield _buffer(0)

        async def close(self) -> None:
            return None

    async def exercise() -> IncrementalCloneTTS:
        tts = IncrementalCloneTTS()
        bridge = StreamingLocalTTSBridge(tts, task_name="test.incremental_clone")
        bus = EventBus()
        bus.ready()
        await bridge.feed_text(bus, "response-1", "第一句。第二句。")
        await bridge.finish_text(bus, "response-1")
        await bridge.wait_idle(timeout=0.5)
        await bridge.close()
        return tts

    tts = asyncio.run(exercise())
    assert len(tts.turns) == 1
    assert tts.turns[0].voice == "voice_local_clone"
    assert tts.turns[0].texts == ["第一句。", "第二句。"]
    assert tts.turns[0].finished is True
    assert tts.turns[0].cancelled is False


def test_bounded_local_tts_queue_applies_async_backpressure_and_keeps_order() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class FirstPieceBlockingTTS:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.texts: list[str] = []

        async def stream_speech(self, text: str):
            self.texts.append(text)
            if len(self.texts) == 1:
                self.started.set()
                await self.release.wait()
            yield _buffer(len(self.texts))

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[list[str], int, int]:
        tts = FirstPieceBlockingTTS()
        bridge = StreamingLocalTTSBridge(tts, task_name="test.bounded_local_tts")
        bus = EventBus()
        bus.ready()
        pieces = [f"piece-{index}." for index in range(6)]
        producer = asyncio.create_task(
            bridge.feed_text(bus, "response-1", "".join(pieces))
        )
        await asyncio.wait_for(tts.started.wait(), timeout=0.5)
        state = bridge._state
        assert state is not None
        async with asyncio.timeout(0.5):
            while state.queue.qsize() < 4:
                await asyncio.sleep(0)
        assert state.queue.maxsize == 4
        assert state.queue.qsize() == 4
        assert not producer.done()

        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.5)

        tts.release.set()
        await asyncio.wait_for(producer, timeout=0.5)
        await bridge.finish_text(bus, "response-1")
        await bridge.wait_idle(timeout=0.5)
        remaining = state.queue.qsize()
        await bridge.close()
        return tts.texts, state.queue.maxsize, remaining

    texts, maxsize, remaining = asyncio.run(exercise())

    assert texts == [f"piece-{index}." for index in range(6)]
    assert maxsize == 4
    assert remaining == 0


def test_interrupt_clears_full_local_tts_queue_and_unblocks_producer() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class BlockingTTS:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.finalized = asyncio.Event()
            self.texts: list[str] = []

        async def stream_speech(self, text: str):
            self.texts.append(text)
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.finalized.set()
            if False:
                yield _buffer(1)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[list[str], int, bool]:
        tts = BlockingTTS()
        bridge = StreamingLocalTTSBridge(tts, task_name="test.cancel_bounded_tts")
        bus = EventBus()
        bus.ready()
        producer = asyncio.create_task(
            bridge.feed_text(
                bus,
                "response-1",
                "".join(f"piece-{index}." for index in range(6)),
            )
        )
        await asyncio.wait_for(tts.started.wait(), timeout=0.5)
        state = bridge._state
        assert state is not None
        async with asyncio.timeout(0.5):
            while state.queue.qsize() < 4:
                await asyncio.sleep(0)
        assert state.queue.qsize() == 4
        assert not producer.done()

        await bridge.interrupt(bus)
        await asyncio.wait_for(producer, timeout=0.5)
        await asyncio.wait_for(tts.finalized.wait(), timeout=0.5)
        await asyncio.sleep(0)
        queue_size = state.queue.qsize()
        active = bridge.is_active
        await bridge.close()
        return tts.texts, queue_size, active

    texts, queue_size, active = asyncio.run(exercise())

    assert texts == ["piece-0."]
    assert queue_size == 0
    assert active is False


def test_text_delta_starts_local_streaming_before_response_done() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class FakeTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.closed = 0

        async def stream_speech(self, text: str):
            self.texts.append(text)
            yield _buffer(len(self.texts))

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> tuple[list[object], FakeTTS]:
        tts = FakeTTS()
        bridge = StreamingLocalTTSBridge(tts, task_name="test.local_tts")
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
        ) as subscription:
            bus.ready()
            await bridge.feed_text(bus, "response-1", "第一句话。")
            # The first sentence must reach CosyVoice before the model emits done.
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            assert tts.texts == ["第一句话。"]

            await bridge.feed_text(bus, "response-1", "第二句话")
            await bridge.finish_text(bus, "response-1", fallback_text="")
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await bridge.wait_idle(timeout=0.5)
            await bridge.close()
        return events, tts

    events, tts = asyncio.run(exercise())

    assert [type(event) for event in events] == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
    ]
    assert tts.texts == ["第一句话。", "第二句话"]
    assert tts.closed == 1


def test_streamed_local_tts_publishes_only_observable_latency_milestones() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class FakeTTS:
        async def stream_speech(self, text: str):
            assert text == "Ready."
            yield _buffer(7)

        async def close(self) -> None:
            return None

    async def exercise() -> list[object]:
        bridge = StreamingLocalTTSBridge(FakeTTS(), task_name="test.metrics")
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            TurnLatencyMilestone,
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
        ) as subscription:
            bus.ready()
            await bridge.feed_text(bus, "response-1", "Ready.")
            await bridge.finish_text(bus, "response-1")
            await bridge.wait_idle(timeout=0.5)
            while True:
                try:
                    event = await subscription.get_next(timeout=0.01)
                except TimeoutError:
                    break
                assert event is not None
                events.append(event)
            await bridge.close()
        return events

    events = asyncio.run(exercise())
    milestones = [event for event in events if isinstance(event, TurnLatencyMilestone)]
    assert [event.phase for event in milestones] == [
        "llm_first_token",
        "first_speakable_text",
        "tts_first_audio",
    ]
    assert {event.turn_id for event in milestones} == {"response-1"}
    segment_started = next(
        event for event in events if isinstance(event, SegmentGenerationStarted)
    )
    assert segment_started.metadata == {"turn_id": "response-1"}


def test_interrupt_cancels_inflight_local_tts_and_discards_avatar_audio() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class BlockingTTS:
        def __init__(self) -> None:
            self.waiting = asyncio.Event()
            self.release = asyncio.Event()
            self.finalized = asyncio.Event()
            self.closed = 0

        async def stream_speech(self, text: str):
            _ = text
            try:
                yield _buffer(7)
                self.waiting.set()
                await self.release.wait()
                yield _buffer(8)
            finally:
                self.finalized.set()

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> tuple[list[object], BlockingTTS]:
        tts = BlockingTTS()
        bridge = StreamingLocalTTSBridge(tts, task_name="test.local_tts")
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            DiscardAvatarSpeechBuffer,
        ) as subscription:
            bus.ready()
            await bridge.feed_text(bus, "response-1", "正在输出。")
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await asyncio.wait_for(tts.waiting.wait(), timeout=0.5)

            await bridge.interrupt(bus)
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await asyncio.wait_for(tts.finalized.wait(), timeout=0.5)
            await bridge.close()
        return events, tts

    events, tts = asyncio.run(exercise())

    assert [type(event) for event in events] == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
        DiscardAvatarSpeechBuffer,
    ]
    assert tts.closed == 1


def test_closing_active_local_tts_without_explicit_bus_completes_open_segment() -> None:
    from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
        StreamingLocalTTSBridge,
    )

    class BlockingTTS:
        def __init__(self) -> None:
            self.waiting = asyncio.Event()

        async def stream_speech(self, _text: str):
            yield _buffer(1)
            self.waiting.set()
            await asyncio.Event().wait()

        async def close(self) -> None:
            return None

    async def exercise() -> list[type[object]]:
        backend = BlockingTTS()
        bridge = StreamingLocalTTSBridge(backend, task_name="test.close_segment")
        bus = EventBus()
        observed: list[type[object]] = []
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
        ) as subscription:
            bus.ready()
            await bridge.feed_text(bus, "response-1", "正在回答。")
            observed.append(type(await subscription.get_next(timeout=0.5)))
            observed.append(type(await subscription.get_next(timeout=0.5)))
            await asyncio.wait_for(backend.waiting.wait(), timeout=0.5)
            await bridge.close()
            observed.append(type(await subscription.get_next(timeout=0.5)))
        return observed

    assert asyncio.run(exercise()) == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
    ]
