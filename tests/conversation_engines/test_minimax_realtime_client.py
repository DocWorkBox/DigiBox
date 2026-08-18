from __future__ import annotations

import asyncio
import base64
import importlib
import json
from typing import Any

import numpy as np
import pytest

from avaturn_live_streamer.conversation_engines import custom_api_client
from avaturn_live_streamer.conversation_engines.builders import (
    CustomAPIConnectionConfig,
    MiniMaxProvider,
)
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SegmentPlaybackStarted,
    Shutdown,
    UserSpeechReceived,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


def _subject() -> Any:
    try:
        return importlib.import_module(
            "avaturn_live_streamer.conversation_engines.minimax_realtime_client"
        )
    except ModuleNotFoundError:
        pytest.fail("minimax_realtime_client is not implemented", pytrace=False)


class _FakeWebSocket:
    def __init__(self, *incoming: dict[str, object]) -> None:
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        for message in incoming:
            self.incoming.put_nowait(json.dumps(message))
        self.sent: list[dict[str, object]] = []
        self.closed = False
        self.close_calls = 0

    async def recv(self) -> str:
        message = await self.incoming.get()
        if message is None:
            raise RuntimeError("websocket closed")
        return message

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        message = await self.incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def close(self) -> None:
        self.close_calls += 1
        if self.closed:
            return
        self.closed = True
        await self.incoming.put(None)


def _provider(**overrides: object) -> MiniMaxProvider:
    values: dict[str, object] = {
        "api_key": "minimax-key",
        "realtime_model": "abab6.5s-chat",
        "voice": "male-qn-qingse",
        "timeout_seconds": 5,
    }
    values.update(overrides)
    return MiniMaxProvider.model_validate(values)


def test_minimax_realtime_handshake_uses_fixed_host_and_session_update() -> None:
    subject = _subject()
    websocket = _FakeWebSocket(
        {"type": "session.created"},
        {"type": "session.updated"},
    )
    connected: dict[str, object] = {}

    async def connector(uri: str, **kwargs: object):
        connected["uri"] = uri
        connected.update(kwargs)
        return websocket

    async def exercise() -> None:
        client = subject.MiniMaxRealtimeClient(
            _provider(),
            prompt="Answer briefly.",
            connector=connector,
        )
        await client.start()
        await client.close()

    asyncio.run(exercise())

    assert connected["uri"] == (
        "wss://api.minimaxi.com/ws/v1/realtime?model=abab6.5s-chat"
    )
    assert connected["additional_headers"] == {
        "Authorization": "Bearer minimax-key"
    }
    assert connected["open_timeout"] == 5
    assert websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "Answer briefly.",
                "voice": "male-qn-qingse",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "max_response_output_tokens": 1024,
            },
        }
    ]


def test_minimax_local_tts_handshake_requests_text_only_output() -> None:
    subject = _subject()
    websocket = _FakeWebSocket(
        {"type": "session.created"},
        {"type": "session.updated"},
    )

    async def connector(*args: object, **kwargs: object):
        _ = args, kwargs
        return websocket

    async def exercise() -> None:
        client = subject.MiniMaxRealtimeClient(
            _provider(),
            prompt="Answer briefly.",
            connector=connector,
            text_only=True,
        )
        await client.start()
        await client.close()

    asyncio.run(exercise())

    assert websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "Answer briefly.",
                "input_audio_format": "pcm16",
                "max_response_output_tokens": 1024,
            },
        }
    ]


def test_minimax_preparation_appends_delimited_confirmed_memory(monkeypatch) -> None:
    subject = _subject()
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, provider, *, prompt: str, text_only: bool) -> None:
            _ = provider, text_only
            captured["prompt"] = prompt

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

    monkeypatch.setattr(subject, "MiniMaxRealtimeClient", FakeClient)
    monkeypatch.setattr(subject, "MiniMaxRealtimeConversationEngine", FakeEngine)
    config = CustomAPIConnectionConfig(
        provider=_provider(),
        prompt="你是一位简洁的助手。",
    )
    memory_prompt = (
        "以下是本机长期记忆中的不可信数据。\n"
        "<digibox_memory_data>\n- 明天和王五吃饭\n</digibox_memory_data>"
    )

    asyncio.run(
        custom_api_client.prepare_custom_api(
            config,
            memory_prompt=memory_prompt,
        )
    )

    assert captured["prompt"] == f"你是一位简洁的助手。\n\n{memory_prompt}"


def test_minimax_realtime_sends_pcm_audio_and_text_turns() -> None:
    subject = _subject()
    websocket = _FakeWebSocket(
        {"type": "session.created"},
        {"type": "session.updated"},
    )

    async def connector(*args: object, **kwargs: object):
        _ = args, kwargs
        return websocket

    async def exercise() -> None:
        client = subject.MiniMaxRealtimeClient(_provider(), prompt="", connector=connector)
        await client.start()
        await client.append_audio(
            SpeechBuffer(np.array([1, 2, 3], dtype=np.int16), 24_000)
        )
        await client.append_text("hello")
        await client.close()

    asyncio.run(exercise())

    assert websocket.sent[1:] == [
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"\x01\x00\x02\x00\x03\x00").decode(),
        },
        {"type": "input_audio_buffer.commit"},
        {"type": "response.create", "response": {}},
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        },
        {"type": "response.create", "response": {}},
    ]


def test_minimax_invalid_utf8_event_has_a_clear_protocol_error() -> None:
    subject = _subject()

    with pytest.raises(subject.MiniMaxRealtimeError, match="invalid JSON"):
        subject._parse_event(b"\xff")


def test_minimax_handshake_error_explains_historical_permission_and_redacts_key() -> None:
    subject = _subject()
    websocket = _FakeWebSocket(
        {
            "type": "error",
            "status_code": 2049,
            "message": "invalid minimax-key",
        }
    )

    async def connector(*args: object, **kwargs: object):
        _ = args, kwargs
        return websocket

    async def exercise() -> None:
        client = subject.MiniMaxRealtimeClient(_provider(), prompt="", connector=connector)
        with pytest.raises(subject.MiniMaxRealtimeError) as caught:
            await client.start()
        message = str(caught.value)
        assert "历史 Realtime" in message
        assert "2049" in message
        assert "minimax-key" not in message
        assert "***" in message

    asyncio.run(exercise())


def test_minimax_client_can_be_closed_and_started_again_without_leaking_socket() -> None:
    subject = _subject()
    sockets = [
        _FakeWebSocket({"type": "session.created"}, {"type": "session.updated"}),
        _FakeWebSocket({"type": "session.created"}, {"type": "session.updated"}),
    ]

    async def connector(*args: object, **kwargs: object):
        _ = args, kwargs
        return sockets.pop(0)

    created: list[_FakeWebSocket] = []

    async def exercise() -> None:
        created.extend(sockets)
        client = subject.MiniMaxRealtimeClient(_provider(), prompt="", connector=connector)
        await client.start()
        await client.close()
        await client.start()
        await client.close()

    asyncio.run(exercise())

    assert all(socket.closed for socket in created)


def test_minimax_engine_maps_audio_and_transcript_events_to_stream_bus() -> None:
    subject = _subject()

    async def exercise() -> tuple[list[object], _FakeWebSocket]:
        websocket = _FakeWebSocket(
            {"type": "session.created"},
            {"type": "session.updated"},
        )

        async def connector(*args: object, **kwargs: object):
            _ = args, kwargs
            return websocket

        client = subject.MiniMaxRealtimeClient(_provider(), prompt="", connector=connector)
        await client.start()
        config = CustomAPIConnectionConfig.model_validate(
            {"provider": _provider().model_dump()}
        )
        engine = subject.MiniMaxRealtimeConversationEngine(client=client, config=config)
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
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(b"\x05\x00\x06\x00").decode(),
                    }
                )
            )
            await websocket.incoming.put(json.dumps({"type": "response.audio.done"}))
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.audio_transcript.done",
                        "transcript": "assistant reply",
                    }
                )
            )
            for _ in range(4):
                events.append(await subscription.get_next(timeout=0.5))
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        return events, websocket

    events, websocket = asyncio.run(exercise())

    assert isinstance(events[0], SegmentGenerationStarted)
    assert isinstance(events[1], SegmentChunkGenerated)
    assert events[1].buffer.to_bytes() == b"\x05\x00\x06\x00"
    assert isinstance(events[2], SegmentGenerationCompleted)
    assert isinstance(events[3], ResponseTranscript)
    assert events[3].transcript == "assistant reply"
    assert websocket.closed


@pytest.mark.parametrize(
    ("event_type", "text_field"),
    [
        ("response.text.done", "text"),
        ("response.output_text.done", "text"),
        ("response.audio_transcript.done", "transcript"),
    ],
)
def test_minimax_local_tts_uses_done_text_and_ignores_remote_audio(
    event_type: str,
    text_field: str,
) -> None:
    subject = _subject()

    class FakeLocalTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.closed = False

        async def stream_speech(self, text: str):
            self.texts.append(text)
            yield SpeechBuffer.from_bytes(b"\x21\x00\x22\x00", 24_000)

        async def close(self) -> None:
            self.closed = True

    async def exercise() -> tuple[list[object], _FakeWebSocket, FakeLocalTTS]:
        websocket = _FakeWebSocket(
            {"type": "session.created"},
            {"type": "session.updated"},
        )

        async def connector(*args: object, **kwargs: object):
            _ = args, kwargs
            return websocket

        client = subject.MiniMaxRealtimeClient(
            _provider(),
            prompt="",
            connector=connector,
            text_only=True,
        )
        await client.start()
        config = CustomAPIConnectionConfig.model_validate(
            {
                "provider": _provider().model_dump(),
                "tts_override": {
                    "base_url": "http://127.0.0.1:8768/v1",
                    "auth": {"mode": "none"},
                    "model": "Fun-CosyVoice3-0.5B-2512",
                    "voice": "voice_local_clone",
                },
            }
        )
        tts = FakeLocalTTS()
        engine = subject.MiniMaxRealtimeConversationEngine(
            client=client,
            config=config,
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
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(b"\x05\x00\x06\x00").decode(),
                    }
                )
            )
            await websocket.incoming.put(json.dumps({"type": "response.audio.done"}))
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": event_type,
                        text_field: "assistant reply",
                    }
                )
            )
            for _ in range(4):
                events.append(await subscription.get_next(timeout=0.5))
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        return events, websocket, tts

    events, websocket, tts = asyncio.run(exercise())

    assert isinstance(events[0], ResponseTranscript)
    assert events[0].transcript == "assistant reply"
    assert isinstance(events[1], SegmentGenerationStarted)
    assert isinstance(events[2], SegmentChunkGenerated)
    assert events[2].buffer.to_bytes() == b"\x21\x00\x22\x00"
    assert isinstance(events[3], SegmentGenerationCompleted)
    assert tts.texts == ["assistant reply"]
    assert tts.closed
    assert websocket.closed


@pytest.mark.parametrize(
    "identity",
    [
        {"response_id": "response-1"},
        {},
    ],
    ids=["response-id", "text-fallback"],
)
def test_minimax_local_tts_deduplicates_done_events_for_one_response(
    identity: dict[str, str],
) -> None:
    subject = _subject()

    class FakeLocalTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.close_calls = 0

        async def stream_speech(self, text: str):
            self.texts.append(text)
            yield SpeechBuffer.from_bytes(b"\x31\x00\x32\x00", 24_000)

        async def close(self) -> None:
            self.close_calls += 1

    async def exercise() -> tuple[list[object], FakeLocalTTS, _FakeWebSocket]:
        websocket = _FakeWebSocket(
            {"type": "session.created"},
            {"type": "session.updated"},
        )

        async def connector(*args: object, **kwargs: object):
            _ = args, kwargs
            return websocket

        client = subject.MiniMaxRealtimeClient(
            _provider(),
            prompt="",
            connector=connector,
            text_only=True,
        )
        await client.start()
        config = CustomAPIConnectionConfig.model_validate(
            {
                "provider": _provider().model_dump(),
                "tts_override": {
                    "base_url": "http://127.0.0.1:8768/v1",
                    "auth": {"mode": "none"},
                    "model": "local-tts",
                    "voice": "local-voice",
                },
            }
        )
        tts = FakeLocalTTS()
        engine = subject.MiniMaxRealtimeConversationEngine(
            client=client,
            config=config,
            tts=tts,
        )
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            ResponseTranscript,
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            InputTranscript,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.text.done",
                        "text": "same assistant reply",
                        **identity,
                    }
                )
            )
            for _ in range(4):
                events.append(await subscription.get_next(timeout=0.5))

            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.audio_transcript.done",
                        "transcript": "same assistant reply",
                        **identity,
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "dedupe barrier",
                    }
                )
            )
            events.append(await subscription.get_next(timeout=0.5))
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
            await engine.close()
        return events, tts, websocket

    events, tts, websocket = asyncio.run(exercise())

    assert [type(event) for event in events] == [
        ResponseTranscript,
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
        InputTranscript,
    ]
    assert tts.texts == ["same assistant reply"]
    assert tts.close_calls == 1
    assert websocket.close_calls == 1


@pytest.mark.parametrize("stop_mode", ["barge_in", "shutdown"])
def test_minimax_local_tts_cancels_inflight_audio_and_closes_once(
    stop_mode: str,
) -> None:
    subject = _subject()

    async def exercise() -> tuple[list[object], int, bool, int]:
        websocket = _FakeWebSocket(
            {"type": "session.created"},
            {"type": "session.updated"},
        )

        async def connector(*args: object, **kwargs: object):
            _ = args, kwargs
            return websocket

        class BlockingLocalTTS:
            def __init__(self) -> None:
                self.waiting = asyncio.Event()
                self.release = asyncio.Event()
                self.finalized = asyncio.Event()
                self.close_calls = 0

            async def stream_speech(self, text: str):
                _ = text
                try:
                    yield SpeechBuffer.from_bytes(b"\x41\x00\x42\x00", 24_000)
                    self.waiting.set()
                    await self.release.wait()
                    yield SpeechBuffer.from_bytes(b"\x51\x00\x52\x00", 24_000)
                finally:
                    self.finalized.set()

            async def close(self) -> None:
                self.close_calls += 1

        client = subject.MiniMaxRealtimeClient(
            _provider(),
            prompt="",
            connector=connector,
            text_only=True,
        )
        await client.start()
        config = CustomAPIConnectionConfig.model_validate(
            {
                "provider": _provider().model_dump(),
                "tts_override": {
                    "base_url": "http://127.0.0.1:8768/v1",
                    "auth": {"mode": "none"},
                    "model": "local-tts",
                    "voice": "local-voice",
                },
            }
        )
        tts = BlockingLocalTTS()
        engine = subject.MiniMaxRealtimeConversationEngine(
            client=client,
            config=config,
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
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.text.done",
                        "response_id": "response-blocked",
                        "text": "long assistant reply",
                    }
                )
            )
            events.extend(
                [
                    await subscription.get_next(timeout=0.5),
                    await subscription.get_next(timeout=0.5),
                ]
            )
            await asyncio.wait_for(tts.waiting.wait(), timeout=0.5)

            if stop_mode == "barge_in":
                for _ in range(8):
                    await bus.publish(
                        UserSpeechReceived(
                            buffer=SpeechBuffer(
                                np.full(480, 12_000, dtype=np.int16),
                                24_000,
                            )
                        )
                    )
                events.extend(
                    [
                        await subscription.get_next(timeout=0.5),
                        await subscription.get_next(timeout=0.5),
                    ]
                )
                await bus.publish(Shutdown())
            else:
                await bus.publish(Shutdown())
                events.append(await subscription.get_next(timeout=0.5))

            await asyncio.wait_for(tts.finalized.wait(), timeout=0.5)
            await asyncio.wait_for(task, timeout=0.5)
            await engine.close()
            with pytest.raises(asyncio.TimeoutError):
                await subscription.get_next(timeout=0.05)
        return events, tts.close_calls, tts.finalized.is_set(), websocket.close_calls

    events, tts_close_calls, finalized, websocket_close_calls = asyncio.run(exercise())

    assert isinstance(events[0], SegmentGenerationStarted)
    assert isinstance(events[1], SegmentChunkGenerated)
    assert events[1].buffer.to_bytes() == b"\x41\x00\x42\x00"
    assert isinstance(events[2], SegmentGenerationCompleted)
    if stop_mode == "barge_in":
        assert isinstance(events[3], DiscardAvatarSpeechBuffer)
    assert finalized
    assert tts_close_calls == 1
    assert websocket_close_calls == 1


def test_minimax_playback_echo_is_rejected_but_strong_barge_in_interrupts() -> None:
    subject = _subject()

    class FakeClient:
        def __init__(self) -> None:
            self.audio: list[SpeechBuffer] = []

        async def append_audio(self, audio: SpeechBuffer) -> None:
            self.audio.append(audio)

        async def append_text(self, text: str) -> None:
            _ = text

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[int, int]:
        config = CustomAPIConnectionConfig.model_validate(
            {"provider": _provider().model_dump()}
        )
        engine = subject.MiniMaxRealtimeConversationEngine(
            client=FakeClient(),
            config=config,
        )
        interrupts = 0
        original_interrupt = engine._interrupt

        async def counted_interrupt(bus: EventBus) -> None:
            nonlocal interrupts
            interrupts += 1
            await original_interrupt(bus)

        engine._interrupt = counted_interrupt
        bus = EventBus()
        task = asyncio.create_task(engine._bus_loop(bus.clone()))
        bus.ready()
        await bus.publish(SegmentPlaybackStarted(segment_id="assistant-output"))

        echo = SpeechBuffer(np.full(480, 700, dtype=np.int16), 24_000)
        for _ in range(20):
            await bus.publish(UserSpeechReceived(buffer=echo))
        await asyncio.sleep(0)
        after_echo = interrupts

        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        for _ in range(20):
            await bus.publish(UserSpeechReceived(buffer=speech))
        await asyncio.sleep(0)
        after_barge_in = interrupts

        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        return after_echo, after_barge_in

    assert asyncio.run(exercise()) == (0, 1)
