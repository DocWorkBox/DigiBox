from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from avaturn_live_streamer.conversation_engines import codex_realtime_client as subject
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    CodexAssistantAudioReceived,
    CodexAssistantControlReceived,
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


class _FakeCodexClient:
    def __init__(self) -> None:
        self.events: asyncio.Queue[object | None] = asyncio.Queue()
        self.appended_text: list[str] = []
        self.closed = False

    async def notifications(self) -> AsyncIterator[object]:
        while True:
            event = await self.events.get()
            if event is None:
                return
            yield event

    async def append_text(self, text: str) -> None:
        self.appended_text.append(text)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.events.put(None)


def _buffer(*samples: int) -> SpeechBuffer:
    return SpeechBuffer(np.array(samples, dtype=np.int16), 24_000)


def test_codex_audio_bridge_becomes_one_renderable_segment() -> None:
    assert hasattr(subject, "CodexConversationEngine"), (
        "Codex conversation engine worklet is not implemented"
    )

    async def exercise() -> None:
        client = _FakeCodexClient()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="v=0\r\ns=answer\r\n",
            stream_id="local",
        )
        bus = EventBus()
        first = _buffer(1, 2)
        second = _buffer(3, 4)

        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await bus.publish(CodexAssistantAudioReceived(buffer=first))
            await bus.publish(CodexAssistantAudioReceived(buffer=second))
            await bus.publish(
                CodexAssistantControlReceived(type="output_audio_done", item_id="item-1")
            )

            started = await subscription.get_next(timeout=0.2)
            chunk_one = await subscription.get_next(timeout=0.2)
            chunk_two = await subscription.get_next(timeout=0.2)
            completed = await subscription.get_next(timeout=0.2)

            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)

        assert isinstance(started, SegmentGenerationStarted)
        assert isinstance(chunk_one, SegmentChunkGenerated)
        assert isinstance(chunk_two, SegmentChunkGenerated)
        assert isinstance(completed, SegmentGenerationCompleted)
        assert chunk_one.segment_id == started.segment_id == completed.segment_id
        assert chunk_two.segment_id == started.segment_id
        assert chunk_one.buffer is first
        assert chunk_two.buffer is second
        assert client.closed
        assert engine.answer_sdp == "v=0\r\ns=answer\r\n"

    asyncio.run(exercise())


def test_codex_user_speech_interrupts_the_active_avatar_segment() -> None:
    assert hasattr(subject, "CodexConversationEngine")

    async def exercise() -> None:
        client = _FakeCodexClient()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
        )
        bus = EventBus()

        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            DiscardAvatarSpeechBuffer,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await bus.publish(CodexAssistantAudioReceived(buffer=_buffer(5, 6)))
            await subscription.get_next(timeout=0.2)
            await subscription.get_next(timeout=0.2)

            await bus.publish(CodexAssistantControlReceived(type="speech_started"))
            completed = await subscription.get_next(timeout=0.2)
            discarded = await subscription.get_next(timeout=0.2)

            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)

        assert isinstance(completed, SegmentGenerationCompleted)
        assert isinstance(discarded, DiscardAvatarSpeechBuffer)

    asyncio.run(exercise())


def test_codex_final_transcripts_are_forwarded_to_the_stream_bus() -> None:
    assert hasattr(subject, "CodexConversationEngine")

    async def exercise() -> None:
        client = _FakeCodexClient()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
        )
        bus = EventBus()

        async with bus.subscribe(InputTranscript, ResponseTranscript) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test", role="user", text="Hello"
                )
            )
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test", role="assistant", text="Hi there"
                )
            )

            user = await subscription.get_next(timeout=0.2)
            assistant = await subscription.get_next(timeout=0.2)
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)

        assert isinstance(user, InputTranscript)
        assert user.transcript == "Hello"
        assert isinstance(assistant, ResponseTranscript)
        assert assistant.transcript == "Hi there"

    asyncio.run(exercise())


def test_codex_engine_stops_when_the_app_server_notification_stream_ends() -> None:
    async def exercise() -> None:
        client = _FakeCodexClient()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
        )
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus, object()))  # type: ignore[arg-type]

        await client.events.put(None)
        await asyncio.wait_for(task, timeout=0.5)

        assert client.closed

    asyncio.run(exercise())


def test_codex_hybrid_streams_assistant_text_and_ignores_cloud_audio() -> None:
    class FakeTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.closed = 0

        async def stream_speech(self, text: str):
            self.texts.append(text)
            yield _buffer(41, 42)

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> tuple[list[object], FakeTTS]:
        client = _FakeCodexClient()
        tts = FakeTTS()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
            tts=tts,
        )
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            ResponseTranscript,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await bus.publish(CodexAssistantAudioReceived(buffer=_buffer(91, 92)))
            await client.events.put(
                subject.CodexTranscriptDelta(
                    thread_id="thread-test",
                    role="assistant",
                    delta="第一句话。",
                )
            )
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            assert tts.texts == ["第一句话。"]
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test",
                    role="assistant",
                    text="第一句话。",
                )
            )
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        return events, tts

    events, tts = asyncio.run(exercise())

    assert tts.texts == ["第一句话。"]
    assert tts.closed == 1
    chunks = [event for event in events if isinstance(event, SegmentChunkGenerated)]
    assert len(chunks) == 1
    assert chunks[0].buffer.to_bytes() == _buffer(41, 42).to_bytes()
    assert sum(isinstance(event, ResponseTranscript) for event in events) == 1


@pytest.mark.parametrize("interrupt_source", ["user_transcript", "speech_started"])
def test_codex_hybrid_user_speech_cancels_inflight_local_tts(
    interrupt_source: str,
) -> None:
    class BlockingTTS:
        def __init__(self) -> None:
            self.waiting = asyncio.Event()
            self.finalized = asyncio.Event()
            self.closed = 0

        async def stream_speech(self, text: str):
            _ = text
            try:
                yield _buffer(51, 52)
                self.waiting.set()
                await asyncio.Event().wait()
            finally:
                self.finalized.set()

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> tuple[list[object], BlockingTTS]:
        client = _FakeCodexClient()
        tts = BlockingTTS()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
            tts=tts,
        )
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            DiscardAvatarSpeechBuffer,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await client.events.put(
                subject.CodexTranscriptDelta(
                    thread_id="thread-test",
                    role="assistant",
                    delta="正在输出。",
                )
            )
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await asyncio.wait_for(tts.waiting.wait(), timeout=0.5)
            if interrupt_source == "user_transcript":
                await client.events.put(
                    subject.CodexTranscriptDelta(
                        thread_id="thread-test",
                        role="user",
                        delta="打断",
                    )
                )
            else:
                await bus.publish(CodexAssistantControlReceived(type="speech_started"))
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await asyncio.wait_for(tts.finalized.wait(), timeout=0.5)
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test",
                    role="assistant",
                    text="这段被打断的旧回复不应重新播放。",
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await subscription.get_next(timeout=0.05)
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        return events, tts

    events, tts = asyncio.run(exercise())

    assert [type(event) for event in events] == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
        DiscardAvatarSpeechBuffer,
    ]
    assert tts.closed == 1


def test_codex_hybrid_done_only_reply_survives_an_idle_user_turn() -> None:
    class FakeTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []

        async def stream_speech(self, text: str):
            self.texts.append(text)
            yield _buffer(61, 62)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[list[object], FakeTTS]:
        client = _FakeCodexClient()
        tts = FakeTTS()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
            tts=tts,
        )
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            ResponseTranscript,
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await client.events.put(
                subject.CodexTranscriptDelta(
                    thread_id="thread-test",
                    role="user",
                    delta="question",
                )
            )
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test",
                    role="user",
                    text="question",
                )
            )
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test",
                    role="assistant",
                    text="done-only answer",
                )
            )
            for _ in range(4):
                events.append(await subscription.get_next(timeout=0.5))
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        return events, tts

    events, tts = asyncio.run(exercise())
    assert tts.texts == ["done-only answer"]
    assert sum(isinstance(event, ResponseTranscript) for event in events) == 1
    assert sum(isinstance(event, SegmentChunkGenerated) for event in events) == 1


@pytest.mark.parametrize("old_done_timing", ["after_user", "while_user"])
def test_codex_hybrid_ignores_late_old_delta_until_old_done_boundary(
    old_done_timing: str,
) -> None:
    class FirstCallBlocksTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.first_waiting = asyncio.Event()
            self.first_finalized = asyncio.Event()

        async def stream_speech(self, text: str):
            self.texts.append(text)
            if len(self.texts) == 1:
                try:
                    yield _buffer(71, 72)
                    self.first_waiting.set()
                    await asyncio.Event().wait()
                finally:
                    self.first_finalized.set()
            else:
                yield _buffer(73, 74)

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        client = _FakeCodexClient()
        tts = FirstCallBlocksTTS()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
            tts=tts,
        )
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        await client.events.put(
            subject.CodexTranscriptDelta(
                thread_id="thread-test",
                role="assistant",
                delta="first.",
            )
        )
        await asyncio.wait_for(tts.first_waiting.wait(), timeout=0.5)
        await bus.publish(CodexAssistantControlReceived(type="speech_started"))
        await asyncio.wait_for(tts.first_finalized.wait(), timeout=0.5)
        await bus.publish(CodexAssistantControlReceived(type="output_audio_done"))
        if old_done_timing == "while_user":
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test",
                    role="assistant",
                    text="cancelled old.",
                )
            )
        await client.events.put(
            subject.CodexTranscriptDone(
                thread_id="thread-test",
                role="user",
                text="interrupt",
            )
        )
        if old_done_timing == "after_user":
            await client.events.put(
                subject.CodexTranscriptDelta(
                    thread_id="thread-test",
                    role="assistant",
                    delta="late old.",
                )
            )
            await client.events.put(
                subject.CodexTranscriptDone(
                    thread_id="thread-test",
                    role="assistant",
                    text="late old.",
                )
            )
            await asyncio.sleep(0.05)
            assert "late old." not in tts.texts
        await client.events.put(
            subject.CodexTranscriptDelta(
                thread_id="thread-test",
                role="assistant",
                delta="new.",
            )
        )
        await client.events.put(
            subject.CodexTranscriptDone(
                thread_id="thread-test",
                role="assistant",
                text="new.",
            )
        )
        async with asyncio.timeout(0.5):
            while "new." not in tts.texts:
                await asyncio.sleep(0)
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return tts.texts

    assert asyncio.run(exercise()) == ["first.", "new."]


def test_codex_hybrid_barge_in_after_transcript_done_keeps_next_reply() -> None:
    class FirstCallBlocksTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.first_waiting = asyncio.Event()
            self.first_finalized = asyncio.Event()

        async def stream_speech(self, text: str):
            self.texts.append(text)
            if len(self.texts) == 1:
                try:
                    yield _buffer(81, 82)
                    self.first_waiting.set()
                    await asyncio.Event().wait()
                finally:
                    self.first_finalized.set()
            else:
                yield _buffer(83, 84)

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        client = _FakeCodexClient()
        tts = FirstCallBlocksTTS()
        engine = subject.CodexConversationEngine(
            client=client,
            answer_sdp="answer",
            stream_id="local",
            tts=tts,
        )
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        await client.events.put(
            subject.CodexTranscriptDelta(
                thread_id="thread-test",
                role="assistant",
                delta="first.",
            )
        )
        await client.events.put(
            subject.CodexTranscriptDone(
                thread_id="thread-test",
                role="assistant",
                text="first.",
            )
        )
        await asyncio.wait_for(tts.first_waiting.wait(), timeout=0.5)

        # Upstream text is already final, but its queued local audio is still
        # playing when the user starts the next turn.
        await bus.publish(CodexAssistantControlReceived(type="speech_started"))
        await asyncio.wait_for(tts.first_finalized.wait(), timeout=0.5)
        await client.events.put(
            subject.CodexTranscriptDone(
                thread_id="thread-test",
                role="user",
                text="next question",
            )
        )
        await client.events.put(
            subject.CodexTranscriptDelta(
                thread_id="thread-test",
                role="assistant",
                delta="second.",
            )
        )
        await client.events.put(
            subject.CodexTranscriptDone(
                thread_id="thread-test",
                role="assistant",
                text="second.",
            )
        )
        async with asyncio.timeout(0.5):
            while "second." not in tts.texts:
                await asyncio.sleep(0)
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return tts.texts

    assert asyncio.run(exercise()) == ["first.", "second."]
