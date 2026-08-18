from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import suppress

import httpx
import pytest

from avaturn_live_streamer.conversation_engines import aliyun_bailian_client
from avaturn_live_streamer.conversation_engines.aliyun_bailian_client import (
    BailianCosyVoiceRealtimeTTS,
    BailianCosyVoiceRealtimeTTSError,
    BailianCosyVoiceTTS,
    BailianQwenASR,
    BailianQwenRealtimeASR,
    BailianRealtimeASRError,
)
from avaturn_live_streamer.conversation_engines.builders import (
    AliyunBailianProvider,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeRealtimeWebSocket:
    def __init__(
        self,
        *,
        transcript: str = "实时转写结果",
        finish_error: str | None = None,
        partials: tuple[tuple[str, str], ...] = (),
        defer_session_finished: bool = False,
    ) -> None:
        self.transcript = transcript
        self.finish_error = finish_error
        self.partials = partials
        self.defer_session_finished = defer_session_finished
        self.incoming: asyncio.Queue[str | bytes | BaseException] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.closed = False
        self.incoming.put_nowait(
            json.dumps(
                {
                    "type": "session.created",
                    "session": {"model": "qwen3-asr-flash-realtime"},
                }
            )
        )

    async def send(self, message: str) -> None:
        event = json.loads(message)
        self.sent.append(event)
        event_type = event.get("type")
        if event_type == "session.update":
            await self.incoming.put(json.dumps({"type": "session.updated"}))
        elif event_type == "session.finish":
            if self.finish_error is not None:
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "code": "invalid_request",
                                "message": self.finish_error,
                            },
                        }
                    )
                )
            else:
                for text, stash in self.partials:
                    await self.incoming.put(
                        json.dumps(
                            {
                                "type": ("conversation.item.input_audio_transcription.text"),
                                "text": text,
                                "stash": stash,
                            }
                        )
                    )
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": ("conversation.item.input_audio_transcription.completed"),
                            "transcript": self.transcript,
                        }
                    )
                )
                if not self.defer_session_finished:
                    await self.finish_session()

    async def recv(self) -> str | bytes:
        value = await self.incoming.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True

    async def finish_session(self) -> None:
        await self.incoming.put(json.dumps({"type": "session.finished"}))


class _FakeCosyVoiceRealtimeWebSocket:
    def __init__(
        self,
        *,
        audio_frames: tuple[bytes, ...] = (b"\x01", b"\x00\x02\x00"),
        finish_error: str | None = None,
        fail_run_on: int | None = None,
    ) -> None:
        self.audio_frames = list(audio_frames)
        self.finish_error = finish_error
        self.fail_run_on = fail_run_on
        self.run_count = 0
        self.incoming: asyncio.Queue[str | bytes | BaseException] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send(self, message: str) -> None:
        event = json.loads(message)
        self.sent.append(event)
        header = event["header"]
        action = header["action"]
        task_id = header["task_id"]
        if action == "run-task":
            self.run_count += 1
            if self.run_count == self.fail_run_on:
                raise ConnectionError("stale pooled websocket")
            await self.incoming.put(
                json.dumps(
                    {
                        "header": {"event": "task-started", "task_id": task_id},
                        "payload": {},
                    }
                )
            )
        elif action == "continue-task":
            await self.incoming.put(
                json.dumps(
                    {
                        "header": {"event": "result-generated", "task_id": task_id},
                        "payload": {"output": {"type": "sentence-synthesis"}},
                    }
                )
            )
            if self.audio_frames:
                await self.incoming.put(self.audio_frames.pop(0))
        elif action == "finish-task":
            input_payload = event["payload"]["input"]
            if input_payload.get("directive") == "cancel":
                return
            if self.finish_error is not None:
                await self.incoming.put(
                    json.dumps(
                        {
                            "header": {
                                "event": "task-failed",
                                "task_id": task_id,
                                "error_code": "InvalidParameter",
                                "error_message": self.finish_error,
                            },
                            "payload": {},
                        }
                    )
                )
            else:
                await self.incoming.put(
                    json.dumps(
                        {
                            "header": {"event": "task-finished", "task_id": task_id},
                            "payload": {"usage": {"characters": 4}},
                        }
                    )
                )

    async def recv(self) -> str | bytes:
        value = await self.incoming.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True


def test_qwen_asr_uses_chat_audio_data_uri_and_parses_message_content() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "transcribed text"}}]},
        )

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-custom",
            tts_voice="voice-id",
            asr_language="zh",
        )
        client = BailianQwenASR(provider, transport=httpx.MockTransport(handler))
        try:
            return await client.transcribe(SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 16_000))
        finally:
            await client.close()

    assert asyncio.run(exercise()) == "transcribed text"
    request = requests[0]
    payload = json.loads(request.content)
    assert request.url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer dashscope-key"
    assert payload["model"] == "qwen3-asr-flash-custom"
    assert payload["stream"] is False
    assert payload["asr_options"]["language"] == "zh"
    audio = payload["messages"][0]["content"][0]
    assert audio["type"] == "input_audio"
    assert audio["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_qwen_realtime_asr_streams_pcm_through_one_manual_websocket_turn() -> None:
    websocket = _FakeRealtimeWebSocket(transcript="你好, 实时世界。")
    connection: dict[str, object] = {}

    async def connector(uri: str, **kwargs: object) -> _FakeRealtimeWebSocket:
        connection["uri"] = uri
        connection["kwargs"] = kwargs
        return websocket

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            workspace_id="workspace-123",
            asr_model="qwen3-asr-flash-realtime",
            asr_language="zh",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        source = SpeechBuffer.from_bytes(b"\x01\x00\x02\x00\x03\x00", 24_000)
        await client.start()
        await client.feed(source)
        return await client.finish()

    assert asyncio.run(exercise()) == "你好, 实时世界。"
    assert connection["uri"] == (
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"
    )
    headers = connection["kwargs"]["additional_headers"]
    assert headers == {
        "Authorization": "Bearer dashscope-key",
        "OpenAI-Beta": "realtime=v1",
        "X-DashScope-WorkSpace": "workspace-123",
    }
    assert connection["kwargs"]["open_timeout"] == 30.0
    assert [event["type"] for event in websocket.sent] == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "session.finish",
    ]
    session = websocket.sent[0]["session"]
    assert session == {
        "input_audio_format": "pcm",
        "sample_rate": 16_000,
        "input_audio_transcription": {"language": "zh"},
        "turn_detection": None,
    }
    sent_pcm = base64.b64decode(websocket.sent[1]["audio"], validate=True)
    expected_pcm = (
        SpeechBuffer.from_bytes(b"\x01\x00\x02\x00\x03\x00", 24_000).resample(16_000).to_bytes()
    )
    assert sent_pcm == expected_pcm
    assert websocket.closed is True


def test_qwen_realtime_asr_transcribe_wraps_a_complete_turn() -> None:
    websocket = _FakeRealtimeWebSocket(transcript="一次调用")

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        try:
            return await client.transcribe(SpeechBuffer.silence(0.1, 16_000))
        finally:
            await client.close()

    assert asyncio.run(exercise()) == "一次调用"
    assert websocket.closed is True


def test_qwen_realtime_asr_cancel_closes_the_active_turn() -> None:
    websocket = _FakeRealtimeWebSocket()

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> None:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        await client.start()
        await client.feed(SpeechBuffer.silence(0.02, 16_000))
        await client.cancel()
        await client.close()

    asyncio.run(exercise())
    assert websocket.closed is True
    assert [event["type"] for event in websocket.sent] == [
        "session.update",
        "input_audio_buffer.append",
    ]


def test_qwen_realtime_asr_redacts_api_key_from_server_errors() -> None:
    secret = "dashscope-super-secret"
    websocket = _FakeRealtimeWebSocket(
        finish_error=f"bad Authorization: Bearer {secret}",
    )

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key=secret,
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        await client.start()
        await client.feed(SpeechBuffer.silence(0.02, 16_000))
        try:
            return await client.finish()
        finally:
            await client.close()

    with pytest.raises(BailianRealtimeASRError) as caught:
        asyncio.run(exercise())
    assert secret not in str(caught.value)
    assert "***" in str(caught.value)
    assert websocket.closed is True


def test_qwen_realtime_asr_exposes_only_stable_confirmed_partial_text() -> None:
    websocket = _FakeRealtimeWebSocket(
        transcript="今天天气很好",
        partials=(("今", "今天"), ("今", "今天天"), ("今天天气", "很好")),
    )
    partials: list[str] = []

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(
            provider,
            connector=connector,
            on_partial_transcript=partials.append,
        )
        try:
            await client.start()
            await client.feed(SpeechBuffer.silence(0.02, 16_000))
            return await client.finish()
        finally:
            await client.close()

    assert asyncio.run(exercise()) == "今天天气很好"
    assert partials == ["今", "今天天气"]


def test_qwen_realtime_asr_partial_callback_can_be_bound_after_build() -> None:
    websocket = _FakeRealtimeWebSocket(
        transcript="hello world",
        partials=(("hello", " draft"),),
    )
    partials: list[str] = []

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        client.set_partial_transcript_callback(partials.append)
        try:
            await client.start()
            await client.feed(SpeechBuffer.silence(0.02, 16_000))
            return await client.finish()
        finally:
            await client.close()

    assert asyncio.run(exercise()) == "hello world"
    assert partials == ["hello"]


def test_qwen_realtime_asr_final_transcript_does_not_wait_for_session_finished() -> None:
    websocket = _FakeRealtimeWebSocket(
        transcript="最终结果",
        defer_session_finished=True,
    )

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> tuple[str, bool, bool]:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        await client.start()
        await client.feed(SpeechBuffer.silence(0.02, 16_000))
        transcript = await asyncio.wait_for(client.finish(), timeout=0.1)
        closed_before_session_finished = websocket.closed
        await websocket.finish_session()
        for _ in range(10):
            if websocket.closed:
                break
            await asyncio.sleep(0)
        closed_after_session_finished = websocket.closed
        await client.close()
        return transcript, closed_before_session_finished, closed_after_session_finished

    assert asyncio.run(exercise()) == ("最终结果", False, True)


def test_qwen_realtime_asr_background_cleanup_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _FakeRealtimeWebSocket(
        transcript="final",
        defer_session_finished=True,
    )
    monkeypatch.setattr(
        aliyun_bailian_client,
        "_ASR_SESSION_CLEANUP_TIMEOUT_SECONDS",
        0.01,
    )

    async def connector(_uri: str, **_kwargs: object) -> _FakeRealtimeWebSocket:
        return websocket

    async def exercise() -> str:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            asr_model="qwen3-asr-flash-realtime",
        )
        client = BailianQwenRealtimeASR(provider, connector=connector)
        await client.start()
        await client.feed(SpeechBuffer.silence(0.02, 16_000))
        transcript = await client.finish()
        for _ in range(50):
            if websocket.closed:
                break
            await asyncio.sleep(0.002)
        await client.close()
        return transcript

    assert asyncio.run(exercise()) == "final"
    assert websocket.closed is True


def test_cosyvoice_stream_decodes_fragmented_base64_pcm_sse() -> None:
    requests: list[httpx.Request] = []
    first = base64.b64encode(b"\x01\x00\x02\x00").decode()
    second = base64.b64encode(b"\x03\x00").decode()
    body = (
        f'data: {{"request_id":"req-ok","output":{{"finish_reason":"null",'
        f'"audio":{{"data":"{first}"}}}}}}\n\n'
        f'data: {{"request_id":"req-ok","output":{{"finish_reason":"null",'
        f'"audio":{{"data":"{second}"}}}}}}\n\n'
        'data: {"request_id":"req-ok","output":{"finish_reason":"stop",'
        '"audio":{"data":""}}}\n\n'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream(body[:19], body[19:47], body[47:]),
        )

    async def exercise() -> bytes:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_model="cosyvoice-v3-flash-custom",
            tts_voice="cosy-voice-id",
        )
        client = BailianCosyVoiceTTS(provider, transport=httpx.MockTransport(handler))
        try:
            chunks = [chunk async for chunk in client.stream_speech("hello")]
            return SpeechBuffer.concat(chunks).to_bytes()
        finally:
            await client.close()

    assert asyncio.run(exercise()) == b"\x01\x00\x02\x00\x03\x00"
    request = requests[0]
    payload = json.loads(request.content)
    assert request.url.path == "/api/v1/services/audio/tts/SpeechSynthesizer"
    assert request.headers["X-DashScope-SSE"] == "enable"
    assert payload["model"] == "cosyvoice-v3-flash-custom"
    assert payload["input"]["voice"] == "cosy-voice-id"
    assert payload["input"]["format"] == "pcm"
    assert payload["input"]["sample_rate"] == 24_000


def test_cosyvoice_stream_rejects_eof_without_final_stop() -> None:
    encoded = base64.b64encode(b"\x01\x00\x02\x00").decode()
    body = (
        f'data: {{"request_id":"req-cut","output":{{"finish_reason":"null",'
        f'"audio":{{"data":"{encoded}"}}}}}}\n\n'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream(body),
        )

    async def exercise() -> None:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_model="cosyvoice-v3-flash",
            tts_voice="longanhuan_v3",
        )
        client = BailianCosyVoiceTTS(provider, transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(RuntimeError, match=r"req-cut.*finish_reason=stop"):
                _ = [chunk async for chunk in client.stream_speech("hello")]
        finally:
            await client.close()

    asyncio.run(exercise())


def test_cosyvoice_stream_raises_http_200_sse_error_event() -> None:
    body = (
        b'data: {"request_id":"req-failed","code":"Throttling.RateQuota",'
        b'"message":"Requests rate limit exceeded"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream(body),
        )

    async def exercise() -> None:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_model="cosyvoice-v3-flash",
            tts_voice="longanhuan_v3",
        )
        client = BailianCosyVoiceTTS(provider, transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(
                RuntimeError,
                match=r"req-failed.*Throttling\.RateQuota.*rate limit exceeded",
            ):
                _ = [chunk async for chunk in client.stream_speech("hello")]
        finally:
            await client.close()

    asyncio.run(exercise())


def test_cosyvoice_realtime_turn_uses_duplex_protocol_and_streams_binary_pcm() -> None:
    websocket = _FakeCosyVoiceRealtimeWebSocket()
    connection: dict[str, object] = {}

    async def connector(
        uri: str,
        **kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        connection["uri"] = uri
        connection["kwargs"] = kwargs
        return websocket

    async def exercise() -> bytes:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            workspace_id="workspace-123",
            tts_model="cosyvoice-v3-flash",
            tts_voice="cosyvoice-v3-flash-user-Voice_A-9",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        turn = await client.open_text_stream()
        await turn.send_text("first,")
        await turn.send_text(" second")
        await turn.finish_text()
        chunks = [chunk async for chunk in turn.stream_audio()]
        await client.close()
        return SpeechBuffer.concat(chunks).to_bytes()

    assert asyncio.run(exercise()) == b"\x01\x00\x02\x00"
    assert connection["uri"] == (
        "wss://workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
    )
    assert connection["kwargs"] == {
        "additional_headers": {
            "Authorization": "Bearer dashscope-key",
            "X-DashScope-WorkSpace": "workspace-123",
        },
        "open_timeout": 30.0,
    }
    assert websocket.closed is True
    assert [event["header"]["action"] for event in websocket.sent] == [
        "run-task",
        "continue-task",
        "continue-task",
        "finish-task",
    ]
    task_ids = {event["header"]["task_id"] for event in websocket.sent}
    assert len(task_ids) == 1
    run_task = websocket.sent[0]
    assert run_task["header"]["streaming"] == "duplex"
    assert run_task["payload"] == {
        "task_group": "audio",
        "task": "tts",
        "function": "SpeechSynthesizer",
        "model": "cosyvoice-v3-flash",
        "parameters": {
            "text_type": "PlainText",
            "voice": "cosyvoice-v3-flash-user-Voice_A-9",
            "format": "pcm",
            "sample_rate": 24_000,
        },
        "input": {},
    }
    assert [event["payload"]["input"].get("text") for event in websocket.sent[1:3]] == [
        "first,",
        " second",
    ]
    assert websocket.sent[-1]["payload"] == {"input": {}}


def test_cosyvoice_realtime_turn_failure_redacts_api_key() -> None:
    secret = "dashscope-super-secret"
    websocket = _FakeCosyVoiceRealtimeWebSocket(
        finish_error=f"bad Authorization: Bearer {secret}",
    )

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        return websocket

    async def exercise() -> None:
        provider = AliyunBailianProvider(
            api_key=secret,
            tts_model="cosyvoice-v3-flash",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        turn = await client.open_text_stream()
        await turn.send_text("hello")
        await turn.finish_text()
        try:
            _ = [chunk async for chunk in turn.stream_audio()]
        finally:
            await client.close()

    with pytest.raises(BailianCosyVoiceRealtimeTTSError) as caught:
        asyncio.run(exercise())
    assert secret not in str(caught.value)
    assert "***" in str(caught.value)
    assert websocket.closed is True


def test_cosyvoice_realtime_turn_cancel_sends_provider_cancel_directive() -> None:
    websocket = _FakeCosyVoiceRealtimeWebSocket()

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        return websocket

    async def exercise() -> None:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        turn = await client.open_text_stream()
        await turn.send_text("hello")
        await turn.cancel()
        await client.close()

    asyncio.run(exercise())
    assert websocket.closed is True
    assert websocket.sent[-1]["header"]["action"] == "finish-task"
    assert websocket.sent[-1]["payload"] == {
        "input": {"directive": "cancel"},
    }


def test_cosyvoice_realtime_reuses_one_healthy_idle_socket_across_turns() -> None:
    websocket = _FakeCosyVoiceRealtimeWebSocket(
        audio_frames=(b"\x01\x00", b"\x02\x00"),
    )
    connector_calls = 0

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        nonlocal connector_calls
        connector_calls += 1
        return websocket

    async def exercise() -> tuple[list[bytes], bool, bool]:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        rendered: list[bytes] = []
        for text in ("first", "second"):
            turn = await client.open_text_stream()
            await turn.send_text(text)
            await turn.finish_text()
            audio = [chunk async for chunk in turn.stream_audio()]
            rendered.append(SpeechBuffer.concat(audio).to_bytes())
        closed_while_idle = websocket.closed
        await client.close()
        return rendered, closed_while_idle, websocket.closed

    assert asyncio.run(exercise()) == (
        [b"\x01\x00", b"\x02\x00"],
        False,
        True,
    )
    assert connector_calls == 1
    run_tasks = [event for event in websocket.sent if event["header"]["action"] == "run-task"]
    assert len(run_tasks) == 2
    assert run_tasks[0]["header"]["task_id"] != run_tasks[1]["header"]["task_id"]


def test_cosyvoice_realtime_full_audio_queue_does_not_leak_listener_after_aclose() -> None:
    websocket = _FakeCosyVoiceRealtimeWebSocket(
        audio_frames=(b"\x01\x00", b"\x02\x00"),
    )
    connector_calls = 0

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        nonlocal connector_calls
        connector_calls += 1
        return websocket

    async def exercise() -> tuple[bytes, bool]:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(
            provider,
            connector=connector,
            audio_queue_size=1,
        )
        turn = await client.open_text_stream()
        await turn.send_text("first")
        await turn.send_text("second")
        await turn.finish_text()

        audio_stream = turn.stream_audio()
        first = await anext(audio_stream)
        listener = turn._listener_task
        async with asyncio.timeout(0.5):
            while not turn._transport_closed:
                await asyncio.sleep(0)
        await audio_stream.aclose()

        try:
            await asyncio.wait_for(asyncio.shield(listener), timeout=0.5)
            listener_completed = True
        except TimeoutError:
            listener_completed = False
        finally:
            if not listener.done():
                listener.cancel()
                with suppress(asyncio.CancelledError):
                    await listener
            await client.close()
        return first.to_bytes(), listener_completed

    assert asyncio.run(exercise()) == (b"\x01\x00", True)
    assert connector_calls == 1
    assert websocket.closed is True


def test_cosyvoice_realtime_stale_idle_socket_reconnects_exactly_once() -> None:
    stale = _FakeCosyVoiceRealtimeWebSocket(
        audio_frames=(b"\x01\x00",),
        fail_run_on=2,
    )
    fresh = _FakeCosyVoiceRealtimeWebSocket(audio_frames=(b"\x02\x00",))
    sockets = iter((stale, fresh))
    connector_calls = 0

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        nonlocal connector_calls
        connector_calls += 1
        return next(sockets)

    async def render_turn(client: BailianCosyVoiceRealtimeTTS, text: str) -> bytes:
        turn = await client.open_text_stream()
        await turn.send_text(text)
        await turn.finish_text()
        audio = [chunk async for chunk in turn.stream_audio()]
        return SpeechBuffer.concat(audio).to_bytes()

    async def exercise() -> tuple[bytes, bytes]:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        try:
            first = await render_turn(client, "first")
            second = await render_turn(client, "second")
            return first, second
        finally:
            await client.close()

    assert asyncio.run(exercise()) == (b"\x01\x00", b"\x02\x00")
    assert connector_calls == 2
    assert stale.run_count == 2
    assert stale.closed is True
    assert fresh.run_count == 1
    assert fresh.closed is True


def test_cosyvoice_realtime_stale_retry_does_not_loop_after_fresh_failure() -> None:
    stale = _FakeCosyVoiceRealtimeWebSocket(
        audio_frames=(b"\x01\x00",),
        fail_run_on=2,
    )
    failed_fresh = _FakeCosyVoiceRealtimeWebSocket(fail_run_on=1)
    sockets = iter((stale, failed_fresh))
    connector_calls = 0

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        nonlocal connector_calls
        connector_calls += 1
        return next(sockets)

    async def exercise() -> None:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        first = await client.open_text_stream()
        await first.send_text("first")
        await first.finish_text()
        _ = [chunk async for chunk in first.stream_audio()]
        try:
            with pytest.raises(
                BailianCosyVoiceRealtimeTTSError,
                match="connection failed",
            ):
                await client.open_text_stream()
        finally:
            await client.close()

    asyncio.run(exercise())
    assert connector_calls == 2
    assert stale.run_count == 2
    assert failed_fresh.run_count == 1
    assert stale.closed is True
    assert failed_fresh.closed is True


def test_cosyvoice_realtime_failed_turn_is_not_returned_to_idle_pool() -> None:
    failed = _FakeCosyVoiceRealtimeWebSocket(
        audio_frames=(),
        finish_error="provider synthesis failed",
    )
    healthy = _FakeCosyVoiceRealtimeWebSocket(audio_frames=(b"\x03\x00",))
    sockets = iter((failed, healthy))
    connector_calls = 0

    async def connector(
        _uri: str,
        **_kwargs: object,
    ) -> _FakeCosyVoiceRealtimeWebSocket:
        nonlocal connector_calls
        connector_calls += 1
        return next(sockets)

    async def exercise() -> bytes:
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            tts_voice="cloned-voice-id",
        )
        client = BailianCosyVoiceRealtimeTTS(provider, connector=connector)
        failed_turn = await client.open_text_stream()
        await failed_turn.send_text("fail")
        await failed_turn.finish_text()
        with pytest.raises(BailianCosyVoiceRealtimeTTSError):
            _ = [chunk async for chunk in failed_turn.stream_audio()]

        healthy_turn = await client.open_text_stream()
        await healthy_turn.send_text("recover")
        await healthy_turn.finish_text()
        audio = [chunk async for chunk in healthy_turn.stream_audio()]
        await client.close()
        return SpeechBuffer.concat(audio).to_bytes()

    assert asyncio.run(exercise()) == b"\x03\x00"
    assert connector_calls == 2
    assert failed.closed is True
    assert healthy.closed is True


def test_cosyvoice_voice_inventory_paginates_and_maps_ui_fields() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        page_index = payload["input"]["page_index"]
        if page_index == 0:
            voices = [
                {
                    "voice_id": "cosyvoice-v3-flash-avatar-a1",
                    "gmt_create": "2026-08-10 12:00:00",
                    "gmt_modified": "2026-08-10 12:01:00",
                    "status": "OK",
                },
                {
                    "voice_id": "cosyvoice-v2-archive-b2",
                    "gmt_create": "2026-08-09 12:00:00",
                    "gmt_modified": "2026-08-09 12:01:00",
                    "status": "DEPLOYING",
                },
            ]
        else:
            voices = [
                {
                    "voice_id": "legacy-c3",
                    "target_model": "cosyvoice-v3-flash",
                    "status": "UNDEPLOYED",
                }
            ]
        return httpx.Response(200, json={"output": {"voice_list": voices}})

    async def exercise():
        list_voices = getattr(
            aliyun_bailian_client,
            "list_cosyvoice_voices",
            None,
        )
        assert callable(list_voices), "list_cosyvoice_voices must be implemented"
        provider = AliyunBailianProvider(
            api_key="dashscope-key",
            workspace_id="workspace-a",
            tts_model="cosyvoice-v3-flash",
        )
        return await list_voices(
            provider,
            page_size=2,
            transport=httpx.MockTransport(handler),
        )

    voices = asyncio.run(exercise())

    assert [
        (
            item.id,
            item.status,
            item.compatible,
            item.created_at,
            item.modified_at,
        )
        for item in voices
    ] == [
        (
            "cosyvoice-v3-flash-avatar-a1",
            "OK",
            True,
            "2026-08-10 12:00:00",
            "2026-08-10 12:01:00",
        ),
        (
            "cosyvoice-v2-archive-b2",
            "DEPLOYING",
            False,
            "2026-08-09 12:00:00",
            "2026-08-09 12:01:00",
        ),
        ("legacy-c3", "UNDEPLOYED", True, None, None),
    ]
    assert len(requests) == 2
    for page_index, request in enumerate(requests):
        payload = json.loads(request.content)
        assert request.url == (
            "https://workspace-a.cn-beijing.maas.aliyuncs.com"
            "/api/v1/services/audio/tts/customization"
        )
        assert request.headers["Authorization"] == "Bearer dashscope-key"
        assert payload == {
            "model": "voice-enrollment",
            "input": {
                "action": "list_voice",
                "page_index": page_index,
                "page_size": 2,
            },
        }


def test_cosyvoice_voice_inventory_rejects_malformed_provider_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": {"voice_list": "invalid"}})

    async def exercise() -> None:
        provider = AliyunBailianProvider(api_key="dashscope-key")
        await aliyun_bailian_client.list_cosyvoice_voices(
            provider,
            transport=httpx.MockTransport(handler),
        )

    with pytest.raises(RuntimeError, match="voice_list"):
        asyncio.run(exercise())
