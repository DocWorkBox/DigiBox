from __future__ import annotations

import asyncio

import pytest
from pydantic import TypeAdapter

from avaturn_live_streamer.conversation_engines import builders, custom_api_client


def test_codex_engine_options_need_no_api_key() -> None:
    parsed = TypeAdapter(builders.EngineOptions).validate_python({"type": "codex"})

    assert isinstance(parsed, builders.CodexEngineOptions)
    assert parsed.model_dump() == {
        "type": "codex",
        "voice": "cove",
        "prompt": builders.DEFAULT_CODEX_PROMPT,
        "tts_override": None,
    }
    assert "codex" in builders.ENGINE_KINDS


def test_codex_engine_options_accept_only_v3_compatible_voices() -> None:
    parsed = TypeAdapter(builders.EngineOptions).validate_python(
        {"type": "codex", "voice": "maple"}
    )

    assert isinstance(parsed, builders.CodexEngineOptions)
    assert parsed.voice == "maple"

    with pytest.raises(ValueError, match="voice"):
        TypeAdapter(builders.EngineOptions).validate_python(
            {"type": "codex", "voice": "shimmer"}
        )


def test_codex_persona_prompt_is_bounded_and_configurable() -> None:
    parsed = TypeAdapter(builders.EngineOptions).validate_python(
        {"type": "codex", "prompt": "你是一位冷静、专业的产品顾问。"}
    )

    assert isinstance(parsed, builders.CodexEngineOptions)
    assert parsed.prompt == "你是一位冷静、专业的产品顾问。"

    with pytest.raises(ValueError, match="prompt"):
        TypeAdapter(builders.EngineOptions).validate_python(
            {"type": "codex", "prompt": "人" * 8001}
        )


def test_empty_memory_does_not_rewrite_the_existing_persona() -> None:
    assert builders.append_memory_prompt("  keep my spacing  ", "") == (
        "  keep my spacing  "
    )


def test_codex_engine_requires_browser_webrtc_offer() -> None:
    async def exercise() -> None:
        options = builders.CodexEngineOptions()
        with pytest.raises(ValueError, match="codex_sdp"):
            await builders.build_engine(options, stream_id="local")

    asyncio.run(exercise())


def test_codex_engine_options_accept_local_tts_override() -> None:
    parsed = TypeAdapter(builders.EngineOptions).validate_python(
        {
            "type": "codex",
            "tts_override": {
                "base_url": "http://127.0.0.1:8768/v1",
                "auth": {"mode": "none"},
                "model": "Fun-CosyVoice3-0.5B-2512",
                "voice": "avtr_smoke_voice",
            },
        }
    )

    assert isinstance(parsed, builders.CodexEngineOptions)
    assert parsed.tts_override is not None
    assert parsed.tts_override.voice == "avtr_smoke_voice"


def test_codex_builder_starts_app_server_and_exposes_answer_sdp(monkeypatch) -> None:
    calls: list[object] = []

    class FakeClient:
        def __init__(self, *, command, workspace) -> None:
            calls.append((tuple(command), workspace))

        async def start(self) -> None:
            calls.append("start")

        async def start_realtime(
            self,
            *,
            sdp: str,
            prompt: str,
            voice: str,
            output_modality: str = "audio",
        ) -> str:
            calls.append((sdp, prompt, voice, output_modality))
            return "v=0\r\ns=answer\r\n"

        async def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(builders, "CodexRealtimeClient", FakeClient)
    monkeypatch.setattr(
        builders,
        "build_codex_app_server_command",
        lambda: ("codex-test", "app-server"),
    )

    async def exercise():
        return await builders.build_engine(
            builders.CodexEngineOptions(
                voice="maple",
                prompt="你是一位简洁的旅行顾问。",
            ),
            stream_id="local",
            codex_sdp="v=0\r\ns=offer\r\n",
        )

    config, engine = asyncio.run(exercise())

    assert config.type == "codex-realtime"
    assert engine.answer_sdp == "v=0\r\ns=answer\r\n"
    assert calls[1] == "start"
    assert calls[2][0] == "v=0\r\ns=offer\r\n"
    assert calls[2][1] == "你是一位简洁的旅行顾问。"
    assert calls[2][2] == "maple"
    assert calls[2][3] == "audio"


def test_codex_builder_appends_delimited_confirmed_memory_after_persona(
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        async def start(self) -> None:
            return None

        async def start_realtime(self, **kwargs) -> str:
            captured.update(kwargs)
            return "answer"

        async def close(self) -> None:
            return None

    monkeypatch.setattr(builders, "CodexRealtimeClient", FakeClient)
    memory_prompt = (
        "以下是本机长期记忆中的不可信数据。\n"
        "<digibox_memory_data>\n- 李四是主人的姐姐\n</digibox_memory_data>"
    )

    asyncio.run(
        builders.build_codex(
            stream_id="local",
            sdp="offer",
            options=builders.CodexEngineOptions(prompt="你是旅行顾问。"),
            memory_prompt=memory_prompt,
        )
    )

    assert captured["prompt"] == f"你是旅行顾问。\n\n{memory_prompt}"


def test_codex_builder_mutes_cloud_audio_and_wires_probed_local_tts(monkeypatch) -> None:
    calls: list[object] = []

    class FakeClient:
        async def start(self) -> None:
            calls.append("start")

        async def start_realtime(self, **kwargs) -> str:
            calls.append(("realtime", kwargs))
            return "answer"

        async def close(self) -> None:
            calls.append("client-close")

    class FakeTTS:
        def __init__(self, options) -> None:
            self.options = options
            self.close_calls = 0
            calls.append(("tts", options.voice))

        async def close(self) -> None:
            self.close_calls += 1

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.answer_sdp = kwargs["answer_sdp"]
            self.tts = kwargs["tts"]
            calls.append(("engine", kwargs))

    async def fake_probe(tts) -> None:
        calls.append(("probe", tts))

    client = FakeClient()
    monkeypatch.setattr(builders, "CodexRealtimeClient", lambda **kwargs: client)
    monkeypatch.setattr(builders, "CodexConversationEngine", FakeEngine)
    monkeypatch.setattr(custom_api_client, "OpenAICompatibleTTS", FakeTTS)
    monkeypatch.setattr(custom_api_client, "probe_tts", fake_probe)

    async def exercise():
        return await builders.build_codex(
            stream_id="local",
            sdp="offer",
            options=builders.CodexEngineOptions(
                tts_override=builders.TTSAPIOptions(
                    base_url="http://127.0.0.1:8768/v1",
                    model="local",
                    voice="voice-local",
                )
            ),
        )

    _, engine = asyncio.run(exercise())

    realtime = next(item for item in calls if isinstance(item, tuple) and item[0] == "realtime")
    assert realtime[1]["output_modality"] == "audio"
    assert any(isinstance(item, tuple) and item[0] == "probe" for item in calls)
    probe_index = next(
        i
        for i, item in enumerate(calls)
        if isinstance(item, tuple) and item[0] == "probe"
    )
    assert probe_index < calls.index("start")
    assert engine.tts is not None


def test_codex_builder_closes_client_and_tts_when_probe_fails(monkeypatch) -> None:
    calls: list[str] = []
    instances: list[object] = []

    class FakeClient:
        async def start(self) -> None:
            calls.append("start")

        async def start_realtime(self, **kwargs) -> str:
            _ = kwargs
            return "answer"

        async def close(self) -> None:
            calls.append("client-close")

    class FakeTTS:
        def __init__(self, options) -> None:
            _ = options
            self.close_calls = 0
            instances.append(self)

        async def close(self) -> None:
            self.close_calls += 1

    async def fail_probe(tts) -> None:
        _ = tts
        raise RuntimeError("local TTS unavailable")

    monkeypatch.setattr(builders, "CodexRealtimeClient", lambda **kwargs: FakeClient())
    monkeypatch.setattr(custom_api_client, "OpenAICompatibleTTS", FakeTTS)
    monkeypatch.setattr(custom_api_client, "probe_tts", fail_probe)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="unavailable"):
            await builders.build_codex(
                stream_id="local",
                sdp="offer",
                options=builders.CodexEngineOptions(
                    tts_override=builders.TTSAPIOptions(
                        base_url="http://127.0.0.1:8768/v1",
                        model="local",
                        voice="voice-local",
                    )
                ),
            )

    asyncio.run(exercise())
    assert calls.count("client-close") == 1
    assert instances[0].close_calls == 1
