from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from tests.conversation_engines.test_custom_api_contract import (
    _connection_config,
    _custom_api_subject,
)

from avaturn_live_streamer.conversation_engines import builders
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


def test_preflight_exercises_llm_asr_and_streaming_pcm_tts_protocols() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "llm.test":
            payload = json.loads(request.content)
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer llm-key"
            assert payload["model"] == "llm-model"
            assert payload["stream"] is False
            assert payload["max_tokens"] == 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )
        if request.url.host == "asr.test":
            assert request.url.path == "/v1/audio/transcriptions"
            assert request.headers["Authorization"] == "Bearer asr-key"
            assert "multipart/form-data" in request.headers["Content-Type"]
            assert b'name="model"' in request.content
            assert b"asr-model" in request.content
            assert b'name="file"' in request.content
            assert b"speech.wav" in request.content
            return httpx.Response(200, json={"text": ""})
        if request.url.host == "tts.test":
            payload = json.loads(request.content)
            assert request.url.path == "/v1/audio/speech"
            assert request.headers["Authorization"] == "Bearer tts-key"
            assert payload["model"] == "tts-model"
            assert payload["voice"] == "clone-voice"
            assert payload["response_format"] == "pcm"
            return httpx.Response(200, content=b"\x01\x00\x02\x00")
        return httpx.Response(404)

    async def exercise():
        return await subject.preflight_custom_api(
            _connection_config(),
            transport=httpx.MockTransport(handler),
        )

    report = asyncio.run(exercise())

    assert report.status == "ready"
    assert {name: item.status for name, item in report.components.items()} == {
        "llm": "ready",
        "asr": "ready",
        "tts": "ready",
    }
    assert {request.url.host for request in requests} == {
        "llm.test",
        "asr.test",
        "tts.test",
    }


def test_probe_tts_exercises_the_selected_incremental_transport() -> None:
    subject = _custom_api_subject()

    class Turn:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.finished = False
            self.cancelled = 0

        async def send_text(self, text: str) -> None:
            self.texts.append(text)

        async def finish_text(self) -> None:
            self.finished = True

        async def stream_audio(self):
            while not self.finished:
                await asyncio.sleep(0)
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def cancel(self) -> None:
            self.cancelled += 1

    class IncrementalTTS:
        def __init__(self) -> None:
            self.turn = Turn()
            self.open_calls = 0
            self.http_calls = 0

        async def open_text_stream(self) -> Turn:
            self.open_calls += 1
            return self.turn

        async def stream_speech(self, text: str):
            _ = text
            self.http_calls += 1
            raise AssertionError("preflight must exercise the selected websocket")
            if False:
                yield SpeechBuffer.empty()

    tts = IncrementalTTS()
    asyncio.run(subject.probe_tts(tts))

    assert tts.open_calls == 1
    assert tts.turn.texts == ["连接测试"]
    assert tts.turn.finished
    assert tts.turn.cancelled == 0
    assert tts.http_calls == 0


def test_bailian_preflight_uses_one_key_for_qwen_asr_llm_and_cosyvoice(
    monkeypatch,
) -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []
    connection: dict[str, object] = {}

    class FakeSocket:
        def __init__(self) -> None:
            self.incoming: asyncio.Queue[str | bytes] = asyncio.Queue()

        async def send(self, value: str) -> None:
            event = json.loads(value)
            header = event["header"]
            action = header["action"]
            task_id = header["task_id"]
            if action == "run-task":
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
                            "header": {
                                "event": "result-generated",
                                "task_id": task_id,
                            },
                            "payload": {},
                        }
                    )
                )
                await self.incoming.put(b"\x01\x00\x02\x00")
            elif action == "finish-task":
                await self.incoming.put(
                    json.dumps(
                        {
                            "header": {"event": "task-finished", "task_id": task_id},
                            "payload": {},
                        }
                    )
                )

        async def recv(self) -> str | bytes:
            return await self.incoming.get()

        async def close(self) -> None:
            return None

    async def connect(uri: str, **kwargs):
        connection["uri"] = uri
        connection.update(kwargs)
        return FakeSocket()

    bailian = __import__(
        "avaturn_live_streamer.conversation_engines.aliyun_bailian_client",
        fromlist=["_realtime_websocket_connect"],
    )
    monkeypatch.setattr(bailian, "_realtime_websocket_connect", connect)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if request.url.path == "/compatible-mode/v1/chat/completions":
            content = payload["messages"][0]["content"]
            if isinstance(content, list):
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": ""}}]},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
            )
        return httpx.Response(404)

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen-plus-custom",
                "asr_model": "qwen3-asr-flash-custom",
                "tts_model": "cosyvoice-v3-flash-custom",
                "tts_voice": "voice-id",
            }
        }
    )

    report = asyncio.run(
        subject.preflight_custom_api(
            config,
            transport=httpx.MockTransport(handler),
        )
    )

    assert report.status == "ready"
    assert len(requests) == 2
    assert {request.headers["Authorization"] for request in requests} == {
        "Bearer one-dashscope-key"
    }
    assert {request.url.host for request in requests} == {
        "dashscope.aliyuncs.com"
    }
    assert connection["additional_headers"] == {
        "Authorization": "Bearer one-dashscope-key"
    }


def test_recent_qwen_web_search_uses_responses_api_and_streams_text_deltas() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"type":"response.output_text.delta","delta":"live "}\n\n'
                b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
                b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "enable_web_search": True,
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> str:
        try:
            chunks = [
                chunk
                async for chunk in components.llm.stream_text(
                    [{"role": "user", "content": "today's news"}]
                )
            ]
            return "".join(chunks)
        finally:
            await components.close()

    assert asyncio.run(exercise()) == "live answer"
    assert requests[0].url.path == "/compatible-mode/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["input"] == [{"role": "user", "content": "today's news"}]
    assert "messages" not in payload
    assert "enable_search" not in payload
    assert "extra_body" not in payload
    assert payload["stream"] is True


@pytest.mark.parametrize(
    ("mode", "prompt", "expected_path", "expected_search"),
    [
        ("off", "请联网搜索今天的新闻", "/compatible-mode/v1/chat/completions", False),
        ("auto", "讲一个简短的笑话", "/compatible-mode/v1/chat/completions", False),
        ("auto", "今天北京空气质量怎么样？", "/compatible-mode/v1/responses", True),
        ("auto", "最近有哪些人工智能新闻？", "/compatible-mode/v1/responses", True),
        ("auto", "请联网搜索今天的新闻", "/compatible-mode/v1/responses", True),
        ("always", "讲一个简短的笑话", "/compatible-mode/v1/responses", True),
    ],
)
def test_qwen_web_search_mode_routes_each_turn_without_an_extra_model_call(
    mode: str,
    prompt: str,
    expected_path: str,
    expected_search: bool,
) -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.7-flash",
                "web_search_mode": mode,
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> str:
        try:
            return "".join(
                [
                    chunk
                    async for chunk in components.llm.stream_text(
                        [{"role": "user", "content": prompt}]
                    )
                ]
            )
        finally:
            await components.close()

    assert asyncio.run(exercise()) == "ok"
    assert len(requests) == 1
    assert requests[0].url.path == expected_path
    payload = json.loads(requests[0].content)
    assert (payload.get("tools") == [{"type": "web_search"}]) is expected_search
    assert "enable_search" not in payload


@pytest.mark.parametrize(
    ("thinking_mode", "web_search_mode", "expected_thinking", "expected_search"),
    [
        ("fast", "off", False, False),
        ("fast", "always", False, True),
        ("deep", "off", True, False),
        ("deep", "always", True, True),
    ],
)
def test_qwen_chat_thinking_and_web_search_controls_are_independent(
    thinking_mode: str,
    web_search_mode: str,
    expected_thinking: bool,
    expected_search: bool,
) -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                # A Qwen 3 chat-completions model which does not use the
                # Responses web-search route.
                "llm_model": "qwen3-max",
                "thinking_mode": thinking_mode,
                "web_search_mode": web_search_mode,
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            _ = [
                chunk
                async for chunk in components.llm.stream_text(
                    [{"role": "user", "content": "hello"}]
                )
            ]
        finally:
            await components.close()

    asyncio.run(exercise())

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["enable_thinking"] is expected_thinking
    assert (payload.get("enable_search") is True) is expected_search


def test_qwen_fast_preflight_explicitly_disables_thinking() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}]},
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.7-flash",
                "thinking_mode": "fast",
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            await components.llm.probe()
        finally:
            await components.close()

    asyncio.run(exercise())

    payload = json.loads(requests[0].content)
    assert payload["enable_thinking"] is False


def test_auto_web_search_preflight_checks_plain_llm_without_search_overhead() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}]},
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.7-flash",
                "web_search_mode": "auto",
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            await components.llm.probe()
        finally:
            await components.close()

    asyncio.run(exercise())
    assert len(requests) == 1
    assert requests[0].url.path == "/compatible-mode/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert "tools" not in payload
    assert "enable_search" not in payload


@pytest.mark.parametrize(
    "model",
    ["qwen3.7-max", "qwen3.7-max-2026-06-08"],
)
def test_qwen37_max_web_search_never_falls_back_to_chat_completions(
    model: str,
) -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": model,
                "enable_web_search": True,
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            _ = [
                chunk
                async for chunk in components.llm.stream_text(
                    [{"role": "user", "content": "latest news"}]
                )
            ]
        finally:
            await components.close()

    asyncio.run(exercise())

    assert requests[0].url.path == "/compatible-mode/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["tools"] == [{"type": "web_search"}]
    assert "enable_search" not in payload
    assert "messages" not in payload


def test_recent_qwen_web_search_probe_parses_non_streaming_responses_output() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK"}],
                    }
                ],
            },
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.7-plus",
                "web_search_mode": "always",
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            await components.llm.probe()
        finally:
            await components.close()

    asyncio.run(exercise())

    assert requests[0].url.path == "/compatible-mode/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["stream"] is False
    assert payload["input"] == [{"role": "user", "content": "Reply OK."}]
    assert payload["max_output_tokens"] == 16


def test_recent_qwen_web_search_probe_surfaces_provider_error_code_and_message() -> None:
    subject = _custom_api_subject()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "InvalidParameter",
                    "message": "tools cannot be combined with this parameter",
                }
            },
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.7-max",
                "web_search_mode": "always",
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            with pytest.raises(
                RuntimeError,
                match=r"InvalidParameter.*tools cannot be combined",
            ):
                await components.llm.probe()
        finally:
            await components.close()

    asyncio.run(exercise())


def test_recent_qwen_web_search_probe_surfaces_top_level_provider_error() -> None:
    subject = _custom_api_subject()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "The web search request contains an invalid parameter.",
                "request_id": "safe-request-id",
            },
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.7-max",
                "web_search_mode": "always",
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            with pytest.raises(
                RuntimeError,
                match=r"InvalidParameter.*web search request contains an invalid",
            ):
                await components.llm.probe()
        finally:
            await components.close()

    asyncio.run(exercise())


def test_recent_qwen_web_search_probe_rejects_failed_responses_result() -> None:
    subject = _custom_api_subject()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "status": "failed",
                "output": [],
                "error": {
                    "code": "web_search_unavailable",
                    "message": "Web search is unavailable",
                },
            },
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.8-max",
                "web_search_mode": "always",
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            with pytest.raises(
                RuntimeError,
                match=r"web_search_unavailable.*Web search is unavailable",
            ):
                await components.llm.probe()
        finally:
            await components.close()

    asyncio.run(exercise())


def test_recent_qwen_web_search_stream_surfaces_failed_response_event() -> None:
    subject = _custom_api_subject()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"type":"response.failed","response":{"status":"failed",'
                b'"error":{"code":"search_failed","message":"Search failed"}}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen3.5-flash",
                "enable_web_search": True,
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        try:
            with pytest.raises(RuntimeError, match=r"search_failed.*Search failed"):
                _ = [
                    chunk
                    async for chunk in components.llm.stream_text(
                        [{"role": "user", "content": "latest news"}]
                    )
                ]
        finally:
            await components.close()

    asyncio.run(exercise())


def test_legacy_qwen_web_search_uses_top_level_chat_completions_field() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"legacy"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                "llm_model": "qwen-plus",
                "enable_web_search": True,
            }
        }
    )
    components = subject.build_custom_api_components(
        config,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> str:
        try:
            chunks = [
                chunk
                async for chunk in components.llm.stream_text(
                    [{"role": "user", "content": "today's news"}]
                )
            ]
            return "".join(chunks)
        finally:
            await components.close()

    assert asyncio.run(exercise()) == "legacy"
    assert requests[0].url.path == "/compatible-mode/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["enable_search"] is True
    assert "extra_body" not in payload


def test_disabled_bailian_and_generic_llm_requests_omit_web_search_fields() -> None:
    subject = _custom_api_subject()
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    configs = [
        builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "one-dashscope-key",
                }
            }
        ),
        _connection_config(),
    ]

    async def exercise() -> None:
        for config in configs:
            components = subject.build_custom_api_components(
                config,
                transport=httpx.MockTransport(handler),
            )
            try:
                _ = [
                    chunk
                    async for chunk in components.llm.stream_text(
                        [{"role": "user", "content": "hello"}]
                    )
                ]
            finally:
                await components.close()

    asyncio.run(exercise())

    assert len(payloads) == 2
    assert all("enable_search" not in payload for payload in payloads)
    assert all("extra_body" not in payload for payload in payloads)


def test_bailian_preflight_keeps_cloud_asr_and_llm_but_uses_local_tts() -> None:
    subject = _custom_api_subject()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if request.url.host == "127.0.0.1":
            assert request.url.path == "/v1/audio/speech"
            assert "Authorization" not in request.headers
            assert payload == {
                "model": "Fun-CosyVoice3-0.5B-2512",
                "input": "连接测试",
                "voice": "voice_local_clone",
                "response_format": "pcm",
                "stream": True,
            }
            return httpx.Response(200, content=b"\x01\x00\x02\x00")
        if request.url.path == "/compatible-mode/v1/chat/completions":
            content = payload["messages"][0]["content"]
            if isinstance(content, list):
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": ""}}]},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
            )
        return httpx.Response(404)

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "one-dashscope-key",
                # This hybrid-TTS test uses MockTransport and intentionally
                # exercises the legacy HTTP ASR probe. Realtime ASR has its
                # own WebSocket contract tests.
                "asr_model": "qwen3-asr-flash",
            },
            "tts_override": {
                "base_url": "http://127.0.0.1:8768/v1",
                "auth": {"mode": "none"},
                "timeout_seconds": 120,
                "model": "Fun-CosyVoice3-0.5B-2512",
                "voice": "voice_local_clone",
                "response_format": "pcm",
                "sample_rate": 24_000,
            },
        }
    )

    report = asyncio.run(
        subject.preflight_custom_api(
            config,
            transport=httpx.MockTransport(handler),
        )
    )

    assert report.status == "ready"
    assert {name: component.status for name, component in report.components.items()} == {
        "llm": "ready",
        "asr": "ready",
        "tts": "ready",
    }
    assert [request.url.host for request in requests].count("127.0.0.1") == 1
    assert not any(
        request.url.path == "/api/v1/services/audio/tts/SpeechSynthesizer"
        for request in requests
    )


def test_custom_api_error_sanitiser_redacts_cloud_and_override_keys() -> None:
    subject = _custom_api_subject()
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "cloud-secret",
            },
            "tts_override": {
                "base_url": "http://127.0.0.1:8768/v1",
                "auth": {"api_key": "local-secret"},
                "model": "Fun-CosyVoice3-0.5B-2512",
                "voice": "voice_local_clone",
            },
        }
    )

    message = subject._sanitise_error(
        RuntimeError("cloud-secret failed while using local-secret"),
        config,
    )

    assert "cloud-secret" not in message
    assert "local-secret" not in message
    assert message.count("***") == 2


def test_prepare_custom_api_reuses_the_clients_that_passed_preflight(monkeypatch) -> None:
    subject = _custom_api_subject()
    calls: list[str] = []

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            assert isinstance(audio, SpeechBuffer)
            calls.append("asr-probe")
            return ""

        async def close(self) -> None:
            calls.append("asr-close")

    class FakeLLM:
        async def probe(self) -> None:
            calls.append("llm-probe")

        async def stream_text(self, messages) -> AsyncIterator[str]:
            _ = messages
            if False:
                yield ""

        async def close(self) -> None:
            calls.append("llm-close")

    class FakeTTS:
        async def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]:
            assert text
            calls.append("tts-probe")
            yield SpeechBuffer.from_bytes(b"\x01\x00", 24_000)

        async def close(self) -> None:
            calls.append("tts-close")

    asr = FakeASR()
    llm = FakeLLM()
    tts = FakeTTS()
    components = subject.CustomAPIComponents(asr=asr, llm=llm, tts=tts)
    builds = 0

    def fake_build(config, *, transport=None):
        nonlocal builds
        _ = config, transport
        builds += 1
        return components

    monkeypatch.setattr(subject, "build_custom_api_components", fake_build)

    async def exercise():
        prepared = await subject.prepare_custom_api(_connection_config())
        assert prepared.report.status == "ready"
        assert prepared.engine._asr is asr
        assert prepared.engine._llm is llm
        assert prepared.engine._tts is tts
        await prepared.engine.close()

    asyncio.run(exercise())

    assert builds == 1
    assert calls == [
        "llm-probe",
        "asr-probe",
        "tts-probe",
        "asr-close",
        "llm-close",
        "tts-close",
    ]


def test_minimax_preflight_reports_realtime_and_redacts_auth_errors(monkeypatch) -> None:
    subject = _custom_api_subject()
    minimax = __import__(
        "avaturn_live_streamer.conversation_engines.minimax_realtime_client",
        fromlist=["_websocket_connect"],
    )

    async def forbidden_connector(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("invalid key minimax-secret")

    monkeypatch.setattr(minimax, "_websocket_connect", forbidden_connector)
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "minimax",
                "api_key": "minimax-secret",
                "realtime_model": "abab6.5s-chat",
                "voice": "male-qn-qingse",
            }
        }
    )

    report = asyncio.run(subject.preflight_custom_api(config))

    assert report.status == "failed"
    assert set(report.components) == {"realtime"}
    assert report.components["realtime"].status == "failed"
    assert "minimax-secret" not in (report.components["realtime"].error or "")
    assert "***" in (report.components["realtime"].error or "")


def test_minimax_local_tts_preflight_retains_text_realtime_and_tts(monkeypatch) -> None:
    subject = _custom_api_subject()
    minimax = __import__(
        "avaturn_live_streamer.conversation_engines.minimax_realtime_client",
        fromlist=["_websocket_connect"],
    )

    class FakeWebSocket:
        def __init__(self) -> None:
            self.incoming = iter(
                [
                    json.dumps({"type": "session.created"}),
                    json.dumps({"type": "session.updated"}),
                ]
            )
            self.sent: list[dict[str, object]] = []
            self.closed = False

        async def recv(self) -> str:
            return next(self.incoming)

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def close(self) -> None:
            self.closed = True

    websocket = FakeWebSocket()

    async def connector(*args, **kwargs):
        _ = args, kwargs
        return websocket

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == "http://127.0.0.1:8768/v1/audio/speech"
        return httpx.Response(200, content=b"\x01\x00\x02\x00")

    monkeypatch.setattr(minimax, "_websocket_connect", connector)
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "minimax",
                "api_key": "minimax-secret",
                "realtime_model": "abab6.5s-chat",
                "voice": "male-qn-qingse",
            },
            "tts_override": {
                "base_url": "http://127.0.0.1:8768/v1",
                "auth": {"mode": "none"},
                "model": "Fun-CosyVoice3-0.5B-2512",
                "voice": "voice_local_clone",
                "response_format": "pcm",
                "sample_rate": 24_000,
            },
        }
    )

    async def exercise():
        prepared = await subject.prepare_custom_api(
            config,
            transport=httpx.MockTransport(handler),
        )
        assert prepared.report.status == "ready"
        assert prepared.engine is not None
        await prepared.engine.close()
        return prepared.report

    report = asyncio.run(exercise())

    assert {name: component.status for name, component in report.components.items()} == {
        "realtime": "ready",
        "tts": "ready",
    }
    assert len(requests) == 1
    assert websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "",
                "input_audio_format": "pcm16",
                "max_response_output_tokens": 1024,
            },
        }
    ]
    assert websocket.closed


def test_minimax_local_tts_preflight_failure_redacts_and_closes_once(monkeypatch) -> None:
    subject = _custom_api_subject()
    minimax = __import__(
        "avaturn_live_streamer.conversation_engines.minimax_realtime_client",
        fromlist=["_websocket_connect"],
    )

    class FakeWebSocket:
        def __init__(self) -> None:
            self.incoming = iter(
                [
                    json.dumps({"type": "session.created"}),
                    json.dumps({"type": "session.updated"}),
                ]
            )
            self.close_calls = 0

        async def recv(self) -> str:
            return next(self.incoming)

        async def send(self, payload: str) -> None:
            _ = payload

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def close(self) -> None:
            self.close_calls += 1

    class FailingLocalTTS:
        def __init__(self, options, *, transport=None) -> None:
            _ = options, transport
            self.close_calls = 0

        async def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]:
            _ = text
            raise RuntimeError("minimax-secret failed with local-secret")
            if False:
                yield SpeechBuffer.silence(0.01, 24_000)

        async def close(self) -> None:
            self.close_calls += 1

    websocket = FakeWebSocket()
    tts_instances: list[FailingLocalTTS] = []

    async def connector(*args, **kwargs):
        _ = args, kwargs
        return websocket

    def make_tts(options, *, transport=None):
        instance = FailingLocalTTS(options, transport=transport)
        tts_instances.append(instance)
        return instance

    monkeypatch.setattr(minimax, "_websocket_connect", connector)
    monkeypatch.setattr(subject, "OpenAICompatibleTTS", make_tts)
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "minimax",
                "api_key": "minimax-secret",
            },
            "tts_override": {
                "base_url": "http://127.0.0.1:8768/v1",
                "auth": {"api_key": "local-secret"},
                "model": "Fun-CosyVoice3-0.5B-2512",
                "voice": "voice_local_clone",
            },
        }
    )

    prepared = asyncio.run(subject.prepare_custom_api(config))

    assert prepared.engine is None
    assert prepared.report.status == "failed"
    assert {name: item.status for name, item in prepared.report.components.items()} == {
        "realtime": "ready",
        "tts": "failed",
    }
    error = prepared.report.components["tts"].error or ""
    assert "minimax-secret" not in error
    assert "local-secret" not in error
    assert error.count("***") == 2
    assert websocket.close_calls == 1
    assert len(tts_instances) == 1
    assert tts_instances[0].close_calls == 1
