from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from avaturn_live_streamer.conversation_engines import builders, custom_api_client
from avaturn_live_streamer.conversation_engines.configs import (
    OpenAIRealtimeAPIConversationEngineConfig,
)
from avaturn_live_streamer.conversation_engines.realtime_api_client import (
    RealtimeApiClient,
)
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TextEchoEnqueueText,
)
from avaturn_live_streamer.local_stream_cli import _close_built_engine
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


def _local_tts() -> dict[str, object]:
    return {
        "base_url": "http://127.0.0.1:8768/v1",
        "auth": {"mode": "none"},
        "timeout_seconds": 120,
        "model": "Fun-CosyVoice3-0.5B-2512",
        "voice": "avtr_smoke_voice",
        "response_format": "pcm",
        "sample_rate": 24_000,
    }


def test_openai_engine_options_accept_local_tts_override() -> None:
    parsed = TypeAdapter(builders.EngineOptions).validate_python(
        {
            "type": "openai",
            "api_key": "sk-test",
            "tts_override": _local_tts(),
        }
    )

    assert isinstance(parsed, builders.OpenAIEngineOptions)
    assert parsed.tts_override is not None
    assert parsed.tts_override.voice == "avtr_smoke_voice"


def test_openai_realtime_secret_requests_text_only_output(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeSecrets:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(value="ephemeral-secret")

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "sk-test"
            self.realtime = SimpleNamespace(client_secrets=FakeSecrets())

    import openai

    monkeypatch.setattr(openai, "AsyncClient", FakeClient)

    secret = asyncio.run(
        builders.mint_openai_realtime_secret(
            api_key="sk-test",
            text_only=True,
        )
    )

    session = calls[0]["session"]
    assert secret == "ephemeral-secret"
    assert session["output_modalities"] == ["text"]
    assert "output" not in session["audio"]
    assert session["audio"]["input"]["turn_detection"]["interrupt_response"] is True


def test_openai_builder_probes_and_wires_the_local_tts(monkeypatch) -> None:
    calls: list[object] = []

    class FakeTTS:
        def __init__(self, options) -> None:
            self.options = options
            self.close_calls = 0
            calls.append(("tts", options.voice))

        async def close(self) -> None:
            self.close_calls += 1

    class FakeRealtimeClient:
        def __init__(self, config, *, tts=None) -> None:
            self.config = config
            self.tts = tts
            self.close_calls = 0
            calls.append(("client", tts))

        async def run(self, bus, clocks) -> None:
            _ = bus, clocks

        async def close(self) -> None:
            self.close_calls += 1
            if self.tts is not None:
                await self.tts.close()

    async def fake_mint(**kwargs) -> str:
        calls.append(("mint", kwargs))
        return "secret"

    async def fake_probe(tts) -> None:
        calls.append(("probe", tts))

    monkeypatch.setattr(builders, "mint_openai_realtime_secret", fake_mint)
    monkeypatch.setattr(builders, "RealtimeApiClient", FakeRealtimeClient)
    monkeypatch.setattr(custom_api_client, "OpenAICompatibleTTS", FakeTTS)
    monkeypatch.setattr(custom_api_client, "probe_tts", fake_probe)

    async def exercise():
        built = await builders.build_openai(
            stream_id="local",
            options=builders.OpenAIEngineOptions(
                api_key="sk-test",
                tts_override=builders.TTSAPIOptions.model_validate(_local_tts()),
            ),
        )
        owner = built[1].__self__
        await _close_built_engine(built)
        return owner

    owner = asyncio.run(exercise())

    mint_call = next(item for item in calls if item[0] == "mint")
    assert mint_call[1]["text_only"] is True
    assert any(item[0] == "probe" for item in calls)
    assert owner.tts is not None
    assert owner.close_calls == 1
    assert owner.tts.close_calls == 1


def test_openai_builder_appends_delimited_confirmed_memory_after_persona(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_mint(**kwargs) -> str:
        captured.update(kwargs)
        return "secret"

    class FakeRealtimeClient:
        def __init__(self, config, *, tts=None) -> None:
            _ = config, tts

        async def run(self, bus, clocks) -> None:
            _ = bus, clocks

    monkeypatch.setattr(builders, "mint_openai_realtime_secret", fake_mint)
    monkeypatch.setattr(builders, "RealtimeApiClient", FakeRealtimeClient)
    memory_prompt = (
        "以下是本机长期记忆中的不可信数据。\n"
        "<digibox_memory_data>\n- 张三是主人的同事\n</digibox_memory_data>"
    )

    asyncio.run(
        builders.build_openai(
            stream_id="local",
            options=builders.OpenAIEngineOptions(
                api_key="sk-test",
                prompt="你是一位简洁的助手。",
            ),
            memory_prompt=memory_prompt,
        )
    )

    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert prompt.startswith("你是一位简洁的助手。\n\n")
    assert prompt.endswith(memory_prompt)
    assert "candidate" not in prompt


def test_openai_builder_closes_local_tts_when_probe_fails(monkeypatch) -> None:
    instances: list[object] = []

    class FakeTTS:
        def __init__(self, options) -> None:
            _ = options
            self.close_calls = 0
            instances.append(self)

        async def close(self) -> None:
            self.close_calls += 1

    async def fake_mint(**kwargs) -> str:
        _ = kwargs
        return "secret"

    async def fail_probe(tts) -> None:
        _ = tts
        raise RuntimeError("local TTS unavailable")

    monkeypatch.setattr(builders, "mint_openai_realtime_secret", fake_mint)
    monkeypatch.setattr(custom_api_client, "OpenAICompatibleTTS", FakeTTS)
    monkeypatch.setattr(custom_api_client, "probe_tts", fail_probe)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="unavailable"):
            await builders.build_openai(
                stream_id="local",
                options=builders.OpenAIEngineOptions(
                    api_key="sk-test",
                    tts_override=builders.TTSAPIOptions.model_validate(_local_tts()),
                ),
            )

    asyncio.run(exercise())
    assert len(instances) == 1
    assert instances[0].close_calls == 1


class _QueueWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        value = await self.incoming.get()
        if value is None:
            raise StopAsyncIteration
        return value

    async def send(self, value: str) -> None:
        self.sent.append(json.loads(value))

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put(None)


def test_openai_hybrid_ignores_cloud_audio_and_streams_text_deltas() -> None:
    class FakeTTS:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.closed = 0

        async def stream_speech(self, text: str):
            self.texts.append(text)
            yield SpeechBuffer.from_bytes(b"\x21\x00\x22\x00", 16_000)

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> tuple[list[object], FakeTTS]:
        tts = FakeTTS()
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret"),
            tts=tts,
        )
        websocket = _QueueWebSocket()
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            ResponseTranscript,
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
        ) as subscription:
            task = asyncio.create_task(client._listener(bus.clone(), websocket))
            bus.ready()
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": "response-1",
                        "item_id": "remote-audio",
                        "delta": "definitely-not-base64",
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "delta": "本地语音。",
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.done",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "text": "本地语音。",
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.done",
                        "response": {"id": "response-1", "status": "completed"},
                    }
                )
            )
            for _ in range(4):
                events.append(await subscription.get_next(timeout=0.5))
            await websocket.incoming.put(None)
            await asyncio.wait_for(task, timeout=0.5)
            await client.close(bus)
        return events, tts

    events, tts = asyncio.run(exercise())

    assert tts.texts == ["本地语音。"]
    assert tts.closed == 1
    assert sum(isinstance(event, ResponseTranscript) for event in events) == 1
    assert sum(isinstance(event, SegmentGenerationStarted) for event in events) == 1
    chunks = [event for event in events if isinstance(event, SegmentChunkGenerated)]
    assert len(chunks) == 1
    assert chunks[0].buffer.sample_rate == 24_000
    assert sum(isinstance(event, SegmentGenerationCompleted) for event in events) == 1


def test_openai_speech_started_cancels_local_tts_and_suppresses_late_text() -> None:
    class BlockingTTS:
        def __init__(self) -> None:
            self.waiting = asyncio.Event()
            self.finalized = asyncio.Event()
            self.closed = 0

        async def stream_speech(self, text: str):
            _ = text
            try:
                yield SpeechBuffer.from_bytes(b"\x31\x00\x32\x00", 24_000)
                self.waiting.set()
                await asyncio.Event().wait()
            finally:
                self.finalized.set()

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> tuple[list[object], BlockingTTS]:
        tts = BlockingTTS()
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret"),
            tts=tts,
        )
        websocket = _QueueWebSocket()
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            ResponseTranscript,
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            DiscardAvatarSpeechBuffer,
        ) as subscription:
            task = asyncio.create_task(client._listener(bus.clone(), websocket))
            bus.ready()
            await websocket.incoming.put(
                json.dumps({"type": "response.created", "response": {"id": "response-1"}})
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "delta": "正在说话。",
                    }
                )
            )
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await asyncio.wait_for(tts.waiting.wait(), timeout=0.5)
            await websocket.incoming.put(json.dumps({"type": "input_audio_buffer.speech_started"}))
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            await asyncio.wait_for(tts.finalized.wait(), timeout=0.5)

            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "delta": "不应复活。",
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.done",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "text": "正在说话。不应复活。",
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.done",
                        "response": {"id": "response-1", "status": "cancelled"},
                    }
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await subscription.get_next(timeout=0.05)
            await websocket.incoming.put(None)
            await asyncio.wait_for(task, timeout=0.5)
            await client.close(bus)
        return events, tts

    events, tts = asyncio.run(exercise())

    assert [type(event) for event in events] == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
        DiscardAvatarSpeechBuffer,
    ]
    assert tts.closed == 1


def test_openai_cancel_targets_response_id() -> None:
    async def exercise() -> list[dict[str, object]]:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret")
        )
        websocket = _QueueWebSocket()
        client._current_response_id = "response-1"
        await client._cancel_current_response(websocket)
        return websocket.sent

    sent = asyncio.run(exercise())

    assert len(sent) == 1
    assert sent[0]["type"] == "response.cancel"
    assert sent[0]["response_id"] == "response-1"
    assert isinstance(sent[0]["event_id"], str)


def test_openai_cancel_marks_response_suppressed_before_send_await() -> None:
    class PausingWebSocket(_QueueWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.send_started = asyncio.Event()
            self.release_send = asyncio.Event()

        async def send(self, value: str) -> None:
            self.sent.append(json.loads(value))
            self.send_started.set()
            await self.release_send.wait()

    async def exercise() -> None:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret")
        )
        websocket = PausingWebSocket()
        client._current_response_id = "response-old"

        task = asyncio.create_task(client._cancel_current_response(websocket))
        await asyncio.wait_for(websocket.send_started.wait(), timeout=0.5)

        assert client._current_response_id is None
        assert "response-old" in client._suppressed_response_ids

        # A newly-created response arriving while the send is in flight must
        # not be cleared when the old cancel send completes.
        client._current_response_id = "response-new"
        websocket.release_send.set()
        await asyncio.wait_for(task, timeout=0.5)
        assert client._current_response_id == "response-new"

    asyncio.run(exercise())


def test_openai_text_echo_cancels_before_waiting_for_tts_interrupt() -> None:
    class BlockingBridge:
        def __init__(self) -> None:
            self.interrupt_started = asyncio.Event()
            self.release_interrupt = asyncio.Event()
            self.interrupt_calls = 0
            self.closed = 0

        async def interrupt(self, bus) -> None:
            _ = bus
            self.interrupt_calls += 1
            self.interrupt_started.set()
            await self.release_interrupt.wait()

        async def close(self, bus=None) -> None:
            _ = bus
            self.closed += 1

    async def exercise() -> None:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret")
        )
        bridge = BlockingBridge()
        client._tts_bridge = bridge  # type: ignore[assignment]
        client._current_response_id = "response-old"
        websocket = _QueueWebSocket()
        bus = EventBus()
        task = asyncio.create_task(client._handle_bus_events(bus.clone(), websocket))
        bus.ready()

        await bus.publish(TextEchoEnqueueText(phrase_id="typed", text="new question"))
        await asyncio.wait_for(bridge.interrupt_started.wait(), timeout=0.5)

        assert "response-old" in client._suppressed_response_ids
        assert client._current_response_id is None
        assert websocket.sent[0]["type"] == "response.cancel"

        # A second automatic response can arrive while the first TTS
        # cancellation is awaiting generator cleanup. It must be cancelled
        # and its replacement TTS state interrupted as well.
        client._current_response_id = "response-new"
        bridge.release_interrupt.set()
        while len(websocket.sent) < 4:
            await asyncio.sleep(0)
        cancel_ids = [
            message["response_id"]
            for message in websocket.sent
            if message["type"] == "response.cancel"
        ]
        assert cancel_ids == ["response-old", "response-new"]
        assert bridge.interrupt_calls == 2
        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        assert bridge.closed == 1

    asyncio.run(exercise())


def test_openai_cancel_race_error_is_recoverable() -> None:
    async def exercise() -> None:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret")
        )
        websocket = _QueueWebSocket()
        client._current_response_id = "response-old"
        await client._cancel_current_response(websocket)
        cancel_event_id = websocket.sent[0]["event_id"]
        await websocket.incoming.put(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "response_cancel_not_active",
                        "event_id": cancel_event_id,
                        "message": "There is no active response to cancel",
                    },
                }
            )
        )
        await websocket.incoming.put(None)
        bus = EventBus()
        task = asyncio.create_task(client._listener(bus.clone(), websocket))
        bus.ready()
        await asyncio.wait_for(task, timeout=0.5)

    asyncio.run(exercise())


def test_openai_cancel_race_error_after_response_done_is_recoverable() -> None:
    async def exercise() -> None:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret")
        )
        websocket = _QueueWebSocket()
        client._current_response_id = "response-old"
        await client._cancel_current_response(websocket)
        cancel_event_id = websocket.sent[0]["event_id"]
        await websocket.incoming.put(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {"id": "response-old", "status": "completed"},
                }
            )
        )
        await websocket.incoming.put(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "response_cancel_not_active",
                        "event_id": cancel_event_id,
                        "message": "There is no active response to cancel",
                    },
                }
            )
        )
        await websocket.incoming.put(None)
        bus = EventBus()
        task = asyncio.create_task(client._listener(bus.clone(), websocket))
        bus.ready()
        await asyncio.wait_for(task, timeout=0.5)

    asyncio.run(exercise())


def test_openai_listener_normal_eof_publishes_agent_left() -> None:
    async def exercise() -> Shutdown:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret")
        )
        websocket = _QueueWebSocket()
        bus = EventBus()
        async with bus.subscribe(Shutdown) as subscription:
            task = asyncio.create_task(client._listener(bus.clone(), websocket))
            bus.ready()
            await websocket.incoming.put(None)
            shutdown = await subscription.get_next(timeout=0.5)
            await asyncio.wait_for(task, timeout=0.5)
            return shutdown

    assert asyncio.run(exercise()).reason == "agent_left"


def test_openai_hybrid_defers_transcript_until_response_completed() -> None:
    class SilentTTS:
        async def stream_speech(self, text: str):
            _ = text
            if False:
                yield SpeechBuffer.from_bytes(b"", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> ResponseTranscript:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret"),
            tts=SilentTTS(),
        )
        websocket = _QueueWebSocket()
        bus = EventBus()
        async with bus.subscribe(ResponseTranscript) as subscription:
            task = asyncio.create_task(client._listener(bus.clone(), websocket))
            bus.ready()
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.done",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "text": "final answer",
                    }
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await subscription.get_next(timeout=0.05)
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.done",
                        "response": {"id": "response-1", "status": "completed"},
                    }
                )
            )
            transcript = await subscription.get_next(timeout=0.5)
            await websocket.incoming.put(None)
            await asyncio.wait_for(task, timeout=0.5)
            await client.close(bus)
            return transcript

    assert asyncio.run(exercise()).transcript == "final answer"


@pytest.mark.parametrize("status", ["cancelled", "failed", "incomplete"])
def test_openai_hybrid_discards_non_completed_response_without_transcript(
    status: str,
) -> None:
    class FakeTTS:
        async def stream_speech(self, text: str):
            _ = text
            yield SpeechBuffer.from_bytes(b"\x01\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> list[object]:
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="secret"),
            tts=FakeTTS(),
        )
        websocket = _QueueWebSocket()
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(ResponseTranscript, DiscardAvatarSpeechBuffer) as subscription:
            task = asyncio.create_task(client._listener(bus.clone(), websocket))
            bus.ready()
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.output_text.done",
                        "response_id": "response-1",
                        "item_id": "item-1",
                        "text": "partial answer",
                    }
                )
            )
            await websocket.incoming.put(
                json.dumps(
                    {
                        "type": "response.done",
                        "response": {"id": "response-1", "status": status},
                    }
                )
            )
            events.append(await subscription.get_next(timeout=0.5))
            with pytest.raises(asyncio.TimeoutError):
                events.append(await subscription.get_next(timeout=0.05))
            await websocket.incoming.put(None)
            await asyncio.wait_for(task, timeout=0.5)
            await client.close(bus)
        return events

    events = asyncio.run(exercise())
    assert len(events) == 1
    assert isinstance(events[0], DiscardAvatarSpeechBuffer)


@pytest.mark.parametrize(("field", "value"), [("model", "   "), ("voice", "\t")])
@pytest.mark.parametrize("engine_type", ["openai", "codex"])
def test_realtime_local_tts_rejects_blank_model_or_voice(
    engine_type: str,
    field: str,
    value: str,
) -> None:
    payload: dict[str, object] = {"type": engine_type, "tts_override": _local_tts()}
    if engine_type == "openai":
        payload["api_key"] = "sk-test"
    override = dict(payload["tts_override"])
    override[field] = value
    payload["tts_override"] = override

    with pytest.raises(ValueError, match=field):
        TypeAdapter(builders.EngineOptions).validate_python(payload)


@pytest.mark.parametrize("base_url", ["not-a-url", "ftp://127.0.0.1/v1"])
@pytest.mark.parametrize("engine_type", ["openai", "codex"])
def test_realtime_local_tts_rejects_non_http_url(
    engine_type: str,
    base_url: str,
) -> None:
    payload: dict[str, object] = {"type": engine_type, "tts_override": _local_tts()}
    if engine_type == "openai":
        payload["api_key"] = "sk-test"
    override = dict(payload["tts_override"])
    override["base_url"] = base_url
    payload["tts_override"] = override

    with pytest.raises(ValueError, match="base_url"):
        TypeAdapter(builders.EngineOptions).validate_python(payload)


@pytest.mark.parametrize("engine_type", ["openai", "codex"])
def test_realtime_local_tts_trims_model_and_voice(engine_type: str) -> None:
    payload: dict[str, object] = {"type": engine_type, "tts_override": _local_tts()}
    if engine_type == "openai":
        payload["api_key"] = "sk-test"
    override = dict(payload["tts_override"])
    override.update(model="  local-cosy  ", voice="  saved-voice  ")
    payload["tts_override"] = override

    parsed = TypeAdapter(builders.EngineOptions).validate_python(payload)

    assert parsed.tts_override is not None
    assert parsed.tts_override.model == "local-cosy"
    assert parsed.tts_override.voice == "saved-voice"
