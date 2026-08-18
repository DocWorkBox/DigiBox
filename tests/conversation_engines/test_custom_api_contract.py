from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from importlib import import_module
from types import ModuleType
from uuid import uuid4

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from avaturn_live_streamer.conversation_engines import builders
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
    TextEchoEnqueueText,
    UserSpeechReceived,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


def _custom_api_subject() -> ModuleType:
    try:
        return import_module(
            "avaturn_live_streamer.conversation_engines.custom_api_client"
        )
    except ModuleNotFoundError:
        pytest.fail("custom_api_client is not implemented", pytrace=False)


def _connection_config():
    config_type = getattr(builders, "CustomAPIConnectionConfig", None)
    if config_type is None:
        pytest.fail("CustomAPIConnectionConfig is not implemented", pytrace=False)
    return config_type(
        llm={
            "base_url": "https://llm.test/v1",
            "auth": {"api_key": "llm-key"},
            "model": "llm-model",
        },
        asr={
            "base_url": "https://asr.test/v1",
            "auth": {"api_key": "asr-key"},
            "model": "asr-model",
            "language": "zh",
        },
        tts={
            "base_url": "https://tts.test/v1",
            "auth": {"api_key": "tts-key"},
            "model": "tts-model",
            "voice": "clone-voice",
            "response_format": "pcm",
            "sample_rate": 24_000,
        },
        vad={
            "rms_threshold": 0.02,
            "pre_roll_ms": 0,
            "min_speech_ms": 100,
            "silence_ms": 200,
            "max_turn_seconds": 10,
        },
        prompt="你是一位简洁的语音助手。",
    )


def test_custom_api_has_separate_connection_and_offer_models() -> None:
    config = _connection_config()
    option_type = getattr(builders, "CustomAPIEngineOptions", None)
    if option_type is None:
        pytest.fail("CustomAPIEngineOptions is not implemented", pytrace=False)

    connection_id = uuid4()
    parsed = TypeAdapter(builders.EngineOptions).validate_python(
        {"type": "custom_api", "connection_id": str(connection_id)}
    )

    assert isinstance(parsed, option_type)
    assert parsed.connection_id == connection_id
    assert str(config.llm.base_url).rstrip("/") == "https://llm.test/v1"
    assert str(config.asr.base_url).rstrip("/") == "https://asr.test/v1"
    assert str(config.tts.base_url).rstrip("/") == "https://tts.test/v1"


def test_legacy_generic_connection_shape_is_migrated_to_nested_provider() -> None:
    config = _connection_config()

    assert config.provider.kind == "generic"
    assert config.llm is config.provider.llm
    assert config.asr is config.provider.asr
    assert config.tts is config.provider.tts


def test_tts_streaming_text_is_opt_in_and_old_configs_stay_compatible() -> None:
    legacy = _connection_config().tts
    websocket = builders.TTSAPIOptions.model_validate(
        {
            "base_url": "http://127.0.0.1:8768/v1",
            "model": "Fun-CosyVoice3-0.5B-2512",
            "voice": "voice_local_clone",
            "streaming_text": "websocket",
        }
    )

    assert legacy.streaming_text == "off"
    assert websocket.streaming_text == "websocket"


def test_bailian_provider_uses_one_secret_and_editable_models() -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-secret",
                "workspace_id": "ws-123",
                "region": "cn-beijing",
                "llm_model": "editable-llm",
                "asr_model": "editable-asr",
                "tts_model": "editable-tts",
                "tts_voice": "editable-voice",
            }
        }
    )

    assert config.provider.kind == "aliyun_bailian"
    assert config.provider.api_key.get_secret_value() == "dashscope-secret"
    assert config.provider.llm_model == "editable-llm"
    assert config.provider.asr_model == "editable-asr"
    assert config.provider.tts_model == "editable-tts"
    assert config.provider.tts_voice == "editable-voice"
    assert "dashscope-secret" not in config.model_dump_json()


def test_bailian_provider_defaults_match_the_local_ui() -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
            }
        }
    )

    assert config.provider.llm_model == "qwen3.7-flash"
    assert config.provider.asr_model == "qwen3-asr-flash-realtime"
    assert config.provider.tts_model == "cosyvoice-v3-flash"
    assert config.provider.tts_voice == "longanhuan_v3"
    assert config.provider.web_search_mode == "off"
    assert config.provider.enable_web_search is False
    assert config.provider.thinking_mode == "fast"
    assert config.fast_history_turns == 6


@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_bailian_provider_accepts_independent_thinking_modes(mode: str) -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
                "thinking_mode": mode,
                "web_search_mode": "off",
            }
        }
    )

    assert config.provider.thinking_mode == mode
    assert config.provider.web_search_mode == "off"


def test_fast_history_is_capped_at_six_turns_but_deep_keeps_configured_history() -> None:
    subject = _custom_api_subject()

    class NoopASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class NoopLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            if False:
                yield ""

        async def close(self) -> None:
            return None

    class NoopTTS:
        async def stream_speech(self, text: str):
            _ = text
            if False:
                yield SpeechBuffer.empty()

        async def close(self) -> None:
            return None

    def make_engine(mode: str):
        config = builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "thinking_mode": mode,
                },
                "history_turns": 8,
                "fast_history_turns": 6,
            }
        )
        engine = subject.CustomAPIConversationEngine(
            config,
            components=subject.CustomAPIComponents(
                asr=NoopASR(),
                llm=NoopLLM(),
                tts=NoopTTS(),
            ),
        )
        engine._history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(16)
        ]
        return engine

    fast = make_engine("fast")._messages("new")
    deep = make_engine("deep")._messages("new")

    fast_history = [item for item in fast if item["role"] != "system"][:-1]
    assert [item["content"] for item in fast_history] == [
        str(index) for index in range(4, 16)
    ]
    assert [item["content"] for item in deep[:-1]] == [str(index) for index in range(16)]
    assert fast[-1] == deep[-1] == {"role": "user", "content": "new"}


def test_per_turn_memory_is_after_persona_before_history_without_pollution() -> None:
    subject = _custom_api_subject()

    class NoopASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class NoopLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            if False:
                yield ""

        async def close(self) -> None:
            return None

    class NoopTTS:
        async def stream_speech(self, text: str):
            _ = text
            if False:
                yield SpeechBuffer.empty()

        async def close(self) -> None:
            return None

    recalled: list[str] = []

    async def recall(user_text: str) -> str:
        recalled.append(user_text)
        return "<digibox_memory_recall>\n- 张三是同事\n</digibox_memory_recall>"

    engine = subject.CustomAPIConversationEngine(
        _connection_config(),
        components=subject.CustomAPIComponents(
            asr=NoopASR(),
            llm=NoopLLM(),
            tts=NoopTTS(),
        ),
        memory_recall=recall,
    )
    engine._history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
    before = list(engine._history)

    messages = asyncio.run(engine._messages_for_turn("new question"))

    assert messages[0] == {
        "role": "system",
        "content": "你是一位简洁的语音助手。",
    }
    assert messages[1]["role"] == "system"
    assert "<digibox_memory_recall>" in messages[1]["content"]
    assert messages[2:4] == before
    assert messages[-1] == {"role": "user", "content": "new question"}
    assert engine._history == before
    assert recalled == ["new question"]


def test_fast_qwen_adds_short_first_sentence_instruction_after_persona_only() -> None:
    subject = _custom_api_subject()

    class NoopASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class NoopLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            if False:
                yield ""

        async def close(self) -> None:
            return None

    class NoopTTS:
        async def stream_speech(self, text: str):
            _ = text
            if False:
                yield SpeechBuffer.empty()

        async def close(self) -> None:
            return None

    def messages_for(config):
        engine = subject.CustomAPIConversationEngine(
            config,
            components=subject.CustomAPIComponents(
                asr=NoopASR(),
                llm=NoopLLM(),
                tts=NoopTTS(),
            ),
        )
        return engine._messages("用户问题")

    fast = messages_for(
        builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "thinking_mode": "fast",
                },
                "prompt": "你是沉稳的历史老师。",
            }
        )
    )
    deep = messages_for(
        builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "thinking_mode": "deep",
                },
                "prompt": "你是沉稳的历史老师。",
            }
        )
    )
    generic = messages_for(_connection_config())

    assert fast[0] == {"role": "system", "content": "你是沉稳的历史老师。"}
    assert fast[1]["role"] == "system"
    assert "不超过12个汉字" in fast[1]["content"]
    assert "再继续展开" in fast[1]["content"]
    assert fast[-1] == {"role": "user", "content": "用户问题"}
    assert deep == [
        {"role": "system", "content": "你是沉稳的历史老师。"},
        {"role": "user", "content": "用户问题"},
    ]
    assert generic[0] == {"role": "system", "content": "你是一位简洁的语音助手。"}
    assert len([item for item in generic if item["role"] == "system"]) == 1


@pytest.mark.parametrize("mode", ["off", "auto", "always"])
def test_bailian_provider_accepts_web_search_modes(mode: str) -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
                "web_search_mode": mode,
            }
        }
    )

    assert config.provider.web_search_mode == mode
    assert config.provider.enable_web_search is (mode != "off")


@pytest.mark.parametrize(
    ("legacy_value", "expected_mode"),
    [(True, "auto"), (False, "off")],
)
def test_bailian_provider_migrates_legacy_web_search_boolean(
    legacy_value: bool,
    expected_mode: str,
) -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
                "enable_web_search": legacy_value,
            }
        }
    )

    assert config.provider.web_search_mode == expected_mode


def test_custom_api_vad_defaults_to_low_latency_hangover() -> None:
    assert builders.VADOptions().silence_ms == 320


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_id", "attacker.test/path"),
        ("workspace_id", "-invalid-label"),
        ("region", "attacker.example"),
    ],
)
def test_bailian_provider_rejects_host_injection(field: str, value: str) -> None:
    provider = {
        "kind": "aliyun_bailian",
        "api_key": "must-not-leak",
        "tts_voice": "voice-id",
        field: value,
    }

    with pytest.raises(ValidationError):
        builders.CustomAPIConnectionConfig.model_validate({"provider": provider})


def test_empty_bailian_workspace_is_normalised_to_shared_dashscope() -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
                "workspace_id": "   ",
                "tts_voice": "voice-id",
            }
        }
    )

    assert config.provider.workspace_id is None


def test_minimax_provider_has_an_independent_secret_and_fixed_realtime_options() -> None:
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "minimax",
                "api_key": "minimax-secret",
                "realtime_model": "abab-custom",
                "voice": "custom-voice",
                "timeout_seconds": 12,
            }
        }
    )

    assert config.provider.kind == "minimax"
    assert config.provider.api_key.get_secret_value() == "minimax-secret"
    assert config.provider.realtime_model == "abab-custom"
    assert config.provider.voice == "custom-voice"
    assert config.provider.timeout_seconds == 12
    assert not hasattr(config.provider, "base_url")
    assert "minimax-secret" not in config.model_dump_json()


@pytest.mark.parametrize("kind", ["aliyun_bailian", "minimax"])
def test_cloud_provider_connection_accepts_a_local_tts_override(kind: str) -> None:
    provider: dict[str, object]
    if kind == "aliyun_bailian":
        provider = {
            "kind": kind,
            "api_key": "cloud-secret",
        }
    else:
        provider = {
            "kind": kind,
            "api_key": "cloud-secret",
            "realtime_model": "abab6.5s-chat",
            "voice": "male-qn-qingse",
        }

    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": provider,
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

    assert config.tts_override is not None
    assert str(config.tts_override.base_url).rstrip("/") == "http://127.0.0.1:8768/v1"
    assert config.tts_override.model == "Fun-CosyVoice3-0.5B-2512"
    assert config.tts_override.voice == "voice_local_clone"
    assert config.tts_override.timeout_seconds == 120


def test_bailian_default_tts_exposes_incremental_stream_and_http_probe_fallback() -> None:
    subject = _custom_api_subject()
    config = builders.CustomAPIConnectionConfig.model_validate(
        {
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
                "asr_model": "qwen3-asr-flash",
            }
        }
    )
    components = subject.build_custom_api_components(config)

    assert callable(getattr(components.tts, "open_text_stream", None))
    assert callable(getattr(components.tts, "stream_speech", None))
    asyncio.run(components.close())


def test_cartesia_is_removed_from_the_public_engine_union() -> None:
    assert "custom_api" in builders.ENGINE_KINDS
    assert "cartesia" not in builders.ENGINE_KINDS

    with pytest.raises(ValidationError):
        TypeAdapter(builders.EngineOptions).validate_python(
            {"type": "cartesia", "api_key": "secret", "agent_id": "agent"}
        )


def test_pcm_stream_keeps_samples_split_across_odd_byte_chunks() -> None:
    subject = _custom_api_subject()

    async def chunks() -> AsyncIterator[bytes]:
        for chunk in (b"\x01", b"\x00\x02", b"\x00\x03\x00"):
            yield chunk

    async def exercise() -> list[SpeechBuffer]:
        return [
            chunk
            async for chunk in subject.iter_pcm_s16le(
                chunks(),
                sample_rate=24_000,
            )
        ]

    decoded = SpeechBuffer.concat(asyncio.run(exercise()))

    assert decoded.sample_rate == 24_000
    assert decoded.to_bytes() == b"\x01\x00\x02\x00\x03\x00"


def test_vad_finishes_an_utterance_after_configured_trailing_silence() -> None:
    subject = _custom_api_subject()
    detector = subject.EnergyTurnDetector(_connection_config().vad)
    speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
    silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)

    for _ in range(10):  # 200 ms of speech
        assert detector.feed(speech) is None
    for _ in range(9):  # 180 ms is shorter than the configured hangover
        assert detector.feed(silence) is None

    utterance = detector.feed(silence)

    assert isinstance(utterance, SpeechBuffer)
    assert utterance.sample_rate == 24_000
    assert float(utterance.duration) >= 0.2
    assert utterance.to_bytes().startswith(speech.to_bytes())


def test_vad_requires_continuous_minimum_speech_before_barge_in() -> None:
    subject = _custom_api_subject()
    detector = subject.EnergyTurnDetector(_connection_config().vad)
    speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
    silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)

    assert detector.feed(speech) is None
    assert detector.is_active
    assert not detector.is_speech_confirmed

    for _ in range(3):
        assert detector.feed(speech) is None
    assert not detector.is_speech_confirmed  # 80 ms is below the 100 ms threshold.

    assert detector.feed(silence) is None
    for _ in range(4):
        assert detector.feed(speech) is None
    assert not detector.is_speech_confirmed  # The silence reset the continuous onset.

    assert detector.feed(speech) is None
    assert detector.is_speech_confirmed


def test_vad_discontinuous_transients_never_emit_an_utterance() -> None:
    subject = _custom_api_subject()
    detector = subject.EnergyTurnDetector(_connection_config().vad)
    speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
    silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)

    for _ in range(5):  # 100 ms total, but never more than 20 ms continuously.
        assert detector.feed(speech) is None
        assert detector.feed(silence) is None
        assert not detector.is_speech_confirmed

    for _ in range(8):
        assert detector.feed(silence) is None
    assert detector.feed(silence) is None
    assert not detector.is_active


def test_custom_api_transient_does_not_interrupt_but_confirmed_speech_does() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class FakeLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            if False:
                yield ""

        async def close(self) -> None:
            return None

    class FakeTTS:
        async def stream_speech(self, text: str):
            _ = text
            if False:
                yield SpeechBuffer.silence(0.02, 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[int, int]:
        components = subject.CustomAPIComponents(
            asr=FakeASR(),
            llm=FakeLLM(),
            tts=FakeTTS(),
        )
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=components,
        )
        interrupts = 0
        original_interrupt = engine._interrupt

        async def counted_interrupt(bus: EventBus) -> None:
            nonlocal interrupts
            interrupts += 1
            await original_interrupt(bus)

        engine._interrupt = counted_interrupt
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        loud = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)

        await bus.publish(UserSpeechReceived(buffer=loud))
        await asyncio.sleep(0)
        after_transient = interrupts

        for _ in range(4):
            await bus.publish(UserSpeechReceived(buffer=loud))
        await asyncio.sleep(0)
        after_confirmed = interrupts

        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return after_transient, after_confirmed

    assert asyncio.run(exercise()) == (0, 1)


def test_custom_api_playback_echo_is_rejected_but_strong_barge_in_interrupts() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class FakeLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            if False:
                yield ""

        async def close(self) -> None:
            return None

    class FakeTTS:
        async def stream_speech(self, text: str):
            _ = text
            if False:
                yield SpeechBuffer.silence(0.02, 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[int, int]:
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=FakeLLM(),
                tts=FakeTTS(),
            ),
        )
        interrupts = 0
        original_interrupt = engine._interrupt

        async def counted_interrupt(bus: EventBus) -> None:
            nonlocal interrupts
            interrupts += 1
            await original_interrupt(bus)

        engine._interrupt = counted_interrupt
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        await bus.publish(SegmentPlaybackStarted(segment_id="assistant-output"))

        # A residual speaker echo is above the ordinary 0.02 RMS threshold,
        # but must not be treated as a user barge-in while playback is active.
        echo = SpeechBuffer(np.full(480, 1_000, dtype=np.int16), 24_000)
        for _ in range(20):
            await bus.publish(UserSpeechReceived(buffer=echo))
        await asyncio.sleep(0)
        after_echo = interrupts

        # Direct close-mic speech remains able to interrupt; playback guarding
        # must not turn the conversation into half-duplex audio.
        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        for _ in range(20):
            await bus.publish(UserSpeechReceived(buffer=speech))
        await asyncio.sleep(0)
        after_barge_in = interrupts

        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return after_echo, after_barge_in

    assert asyncio.run(exercise()) == (0, 1)


def test_vad_keeps_echo_guard_during_speaker_playout_tail() -> None:
    subject = _custom_api_subject()
    detector = subject.EnergyTurnDetector(_connection_config().vad)
    echo = SpeechBuffer(np.full(480, 1_000, dtype=np.int16), 24_000)

    detector.set_assistant_playback(True)
    for _ in range(10):
        assert detector.feed(echo) is None
    detector.set_assistant_playback(False)

    # WebRTC playout and acoustic echo arrive on independently buffered paths;
    # ending the playback event must not immediately restore the ordinary VAD.
    for _ in range(10):  # 200 ms of residual speaker tail.
        assert detector.feed(echo) is None
        assert not detector.is_speech_confirmed


def test_default_vad_playback_guard_has_headroom_above_moderate_echo() -> None:
    subject = _custom_api_subject()
    detector = subject.EnergyTurnDetector(builders.VADOptions())
    echo = SpeechBuffer(np.full(480, 1_000, dtype=np.int16), 24_000)

    detector.set_assistant_playback(True)
    for _ in range(20):
        assert detector.feed(echo) is None

    assert not detector.is_speech_confirmed


def test_custom_api_audio_turn_emits_ordered_latency_milestones_and_turn_metadata() -> None:
    subject = _custom_api_subject()
    events_module = import_module("avaturn_live_streamer.events")
    milestone_type = getattr(events_module, "TurnLatencyMilestone", None)
    if milestone_type is None:
        pytest.fail("TurnLatencyMilestone is not implemented", pytrace=False)

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            assert not audio.is_empty
            return "hello"

        async def close(self) -> None:
            return None

    class FakeLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            assert messages[-1] == {"role": "user", "content": "hello"}
            yield "answer?"

        async def close(self) -> None:
            return None

    class FakeTTS:
        async def stream_speech(self, text: str):
            assert text == "answer?"
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> list[object]:
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=FakeLLM(),
                tts=FakeTTS(),
            ),
        )
        bus = EventBus()
        seen: list[object] = []
        async with bus.subscribe(milestone_type, SegmentGenerationStarted) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
            silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)
            for _ in range(10):
                await bus.publish(UserSpeechReceived(buffer=speech))
            for _ in range(10):
                await bus.publish(UserSpeechReceived(buffer=silence))

            for _ in range(6):
                event = await subscription.get_next(timeout=0.5)
                assert event is not None
                seen.append(event)

            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
            await engine.close()
        return seen

    events = asyncio.run(exercise())
    milestones = [event for event in events if isinstance(event, milestone_type)]
    assert [event.phase for event in milestones] == [
        "vad_complete",
        "asr_complete",
        "llm_first_token",
        "first_speakable_text",
        "tts_first_audio",
    ]
    assert len({event.turn_id for event in milestones}) == 1
    assert milestones[0].details["vad_wait_ms"] == 200
    segment_started = next(
        event for event in events if isinstance(event, SegmentGenerationStarted)
    )
    assert segment_started.metadata == {"turn_id": milestones[0].turn_id}


def test_custom_api_turn_error_finishes_segment_is_logged_and_next_turn_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class FakeLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            yield "这是一次回答。"

        async def close(self) -> None:
            return None

    class FlakyTTS:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_speech(self, text: str):
            _ = text
            self.calls += 1
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)
            if self.calls == 1:
                raise RuntimeError("tts-key provider stream ended early")

        async def close(self) -> None:
            return None

    class CapturingLogger:
        def __init__(self) -> None:
            self.errors: list[tuple[str, dict[str, object]]] = []

        def exception(self, event: str, **kwargs: object) -> None:
            self.errors.append((event, kwargs))

    async def exercise() -> tuple[
        object,
        list[object],
        object,
        list[object],
        object,
        CapturingLogger,
    ]:
        tts = FlakyTTS()
        components = subject.CustomAPIComponents(
            asr=FakeASR(),
            llm=FakeLLM(),
            tts=tts,
        )
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=components,
        )
        logger = CapturingLogger()
        monkeypatch.setattr(subject, "_LOGGER", logger)
        bus = EventBus()
        first: list[object] = []
        second: list[object] = []
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            DiscardAvatarSpeechBuffer,
            ResponseTranscript,
        ) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()

            await bus.publish(TextEchoEnqueueText(phrase_id="first", text="first"))
            before_first = await subscription.get_next(timeout=0.5)
            for _ in range(4):
                first.append(await subscription.get_next(timeout=0.5))
            first_turn_task = engine._turn_task

            await bus.publish(TextEchoEnqueueText(phrase_id="second", text="second"))
            between_turns = await subscription.get_next(timeout=0.5)
            for _ in range(4):
                second.append(await subscription.get_next(timeout=0.5))

            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return before_first, first, between_turns, second, first_turn_task, logger

    before_first, first, between_turns, second, first_turn_task, logger = asyncio.run(
        exercise()
    )

    assert isinstance(before_first, DiscardAvatarSpeechBuffer)
    assert [type(event) for event in first] == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
        DiscardAvatarSpeechBuffer,
    ]
    assert first_turn_task is None
    assert isinstance(between_turns, DiscardAvatarSpeechBuffer)
    assert [type(event) for event in second] == [
        SegmentGenerationStarted,
        SegmentChunkGenerated,
        SegmentGenerationCompleted,
        ResponseTranscript,
    ]
    assert len(logger.errors) == 1
    event, fields = logger.errors[0]
    assert event == "custom API turn failed"
    assert "tts-key" not in str(fields)
    assert "***" in str(fields)


def test_bailian_realtime_model_selects_streaming_asr_and_http_model_keeps_fallback() -> None:
    subject = _custom_api_subject()
    aliyun = import_module(
        "avaturn_live_streamer.conversation_engines.aliyun_bailian_client"
    )

    async def exercise() -> tuple[object, object | None, object, object | None]:
        realtime_config = builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "asr_model": "qwen3-asr-flash-realtime",
                }
            }
        )
        http_config = builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "asr_model": "qwen3-asr-flash",
                }
            }
        )
        realtime = subject.build_custom_api_components(realtime_config)
        fallback = subject.build_custom_api_components(http_config)
        try:
            return (
                realtime.asr,
                realtime.asr_fallback,
                fallback.asr,
                fallback.asr_fallback,
            )
        finally:
            await realtime.close()
            await fallback.close()

    realtime_asr, realtime_http_fallback, fallback_asr, fallback_http_fallback = (
        asyncio.run(exercise())
    )

    assert isinstance(realtime_asr, aliyun.BailianQwenRealtimeASR)
    assert isinstance(realtime_http_fallback, aliyun.BailianQwenASR)
    assert realtime_http_fallback.provider.asr_model == "qwen3-asr-flash"
    assert isinstance(fallback_asr, aliyun.BailianQwenASR)
    assert fallback_http_fallback is None


@pytest.mark.parametrize("fail_at", ["feed", "finish"])
def test_realtime_asr_failure_uses_http_fallback_for_the_same_turn(
    fail_at: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _custom_api_subject()
    logged_errors: list[tuple[str, dict[str, object]]] = []

    class CapturingLogger:
        def exception(self, event: str, **kwargs: object) -> None:
            logged_errors.append((event, kwargs))

    monkeypatch.setattr(subject, "_LOGGER", CapturingLogger())

    class FailingRealtimeASR:
        def __init__(self) -> None:
            self.feed_calls = 0
            self.finish_calls = 0
            self.transcribe_calls = 0
            self.cancel_calls = 0

        async def start(self) -> None:
            return None

        async def feed(self, audio: SpeechBuffer) -> None:
            assert not audio.is_empty
            self.feed_calls += 1
            if fail_at == "feed" and self.feed_calls == 1:
                raise RuntimeError("realtime websocket send failed")

        async def finish(self) -> str:
            self.finish_calls += 1
            if fail_at == "finish":
                raise RuntimeError("realtime websocket finalization timed out")
            return "unexpected realtime result"

        async def cancel(self) -> None:
            self.cancel_calls += 1

        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            self.transcribe_calls += 1
            raise AssertionError("fallback must not retry the realtime websocket")

        async def close(self) -> None:
            return None

    class HTTPFallbackASR:
        def __init__(self) -> None:
            self.audio: list[SpeechBuffer] = []
            self.closed = 0

        async def transcribe(self, audio: SpeechBuffer) -> str:
            self.audio.append(audio)
            return "http fallback"

        async def close(self) -> None:
            self.closed += 1

    class FakeLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            assert messages[-1] == {"role": "user", "content": "http fallback"}
            yield "answer."

        async def close(self) -> None:
            return None

    class FakeTTS:
        async def stream_speech(self, text: str):
            assert text == "answer."
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[FailingRealtimeASR, HTTPFallbackASR, str]:
        realtime = FailingRealtimeASR()
        fallback = HTTPFallbackASR()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=realtime,
                llm=FakeLLM(),
                tts=FakeTTS(),
                asr_fallback=fallback,
            ),
        )
        bus = EventBus()
        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)
        async with bus.subscribe(InputTranscript) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            for _ in range(10):
                await bus.publish(UserSpeechReceived(buffer=speech))
            for _ in range(10):
                await bus.publish(UserSpeechReceived(buffer=silence))
            transcript = await subscription.get_next(timeout=0.5)
            assert transcript is not None
            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return realtime, fallback, transcript.transcript

    realtime, fallback, transcript = asyncio.run(exercise())

    assert transcript == "http fallback"
    assert realtime.transcribe_calls == 0
    assert len(fallback.audio) == 1
    assert fallback.closed == 1
    assert len(logged_errors) == 1
    assert logged_errors[0][1]["error"] == (
        "realtime websocket send failed"
        if fail_at == "feed"
        else "realtime websocket finalization timed out"
    )
    if fail_at == "feed":
        assert realtime.finish_calls == 0
        assert realtime.cancel_calls >= 1
    else:
        assert realtime.finish_calls == 1


def test_realtime_asr_starts_at_speech_onset_and_streams_before_vad_finish() -> None:
    subject = _custom_api_subject()

    class FakeRealtimeASR:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.finished = asyncio.Event()
            self.calls: list[str] = []
            self.audio: list[SpeechBuffer] = []

        async def start(self) -> None:
            self.calls.append("start")
            self.started.set()

        async def feed(self, audio: SpeechBuffer) -> None:
            self.calls.append("feed")
            self.audio.append(audio)

        async def finish(self) -> str:
            self.calls.append("finish")
            self.finished.set()
            return "hello"

        async def cancel(self) -> None:
            self.calls.append("cancel")

        async def transcribe(self, audio: SpeechBuffer) -> str:
            raise AssertionError("completed realtime turns must not use batch transcribe")

        async def close(self) -> None:
            return None

    class FakeLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            assert messages[-1] == {"role": "user", "content": "hello"}
            yield "answer."

        async def close(self) -> None:
            return None

    class FakeTTS:
        async def stream_speech(self, text: str):
            assert text == "answer."
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[FakeRealtimeASR, bytes, bytes, list[object]]:
        asr = FakeRealtimeASR()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=asr,
                llm=FakeLLM(),
                tts=FakeTTS(),
            ),
        )
        bus = EventBus()
        transcripts: list[object] = []
        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)
        async with bus.subscribe(InputTranscript) as subscription:
            task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
            bus.ready()
            for _ in range(5):
                await bus.publish(UserSpeechReceived(buffer=speech))
            await asyncio.wait_for(asr.started.wait(), timeout=0.5)
            assert asr.calls == ["start", *("feed" for _ in range(5))]

            for _ in range(5):
                await bus.publish(UserSpeechReceived(buffer=speech))
            for _ in range(10):
                await bus.publish(UserSpeechReceived(buffer=silence))
            await asyncio.wait_for(asr.finished.wait(), timeout=0.5)
            transcript = await subscription.get_next(timeout=0.5)
            assert transcript is not None
            transcripts.append(transcript)

            await bus.publish(Shutdown())
            await asyncio.wait_for(task, timeout=0.5)
        expected = SpeechBuffer.concat([*[speech] * 10, *[silence] * 10]).to_bytes()
        received = SpeechBuffer.concat(asr.audio).to_bytes()
        await engine.close()
        return asr, received, expected, transcripts

    asr, received, expected, transcripts = asyncio.run(exercise())

    assert asr.calls.count("start") == 1
    assert asr.calls.count("finish") == 1
    assert received == expected
    assert [event.transcript for event in transcripts] == ["hello"]


def test_llm_producer_runs_ahead_while_tts_consumer_streams_in_order() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class StreamingLLM:
        def __init__(self) -> None:
            self.second_delta_requested = asyncio.Event()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            yield "first."
            self.second_delta_requested.set()
            yield "second."

        async def close(self) -> None:
            return None

    class BlockingTTS:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.texts: list[str] = []

        async def stream_speech(self, text: str):
            self.texts.append(text)
            if len(self.texts) == 1:
                self.started.set()
                await self.release.wait()
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        llm = StreamingLLM()
        tts = BlockingTTS()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=llm,
                tts=tts,
            ),
        )
        bus = EventBus()
        bus.ready()
        task = asyncio.create_task(
            engine._process_text(bus, "question", engine._generation, "turn-queue")
        )
        await asyncio.wait_for(tts.started.wait(), timeout=0.5)
        await asyncio.wait_for(llm.second_delta_requested.wait(), timeout=0.5)
        tts.release.set()
        await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return tts.texts

    assert asyncio.run(exercise()) == ["first.", "second."]


def test_short_pending_text_flushes_after_idle_before_llm_completion() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class PausingLLM:
        def __init__(self) -> None:
            self.paused = asyncio.Event()
            self.release = asyncio.Event()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            yield "短句"
            self.paused.set()
            await self.release.wait()
            yield "继续。"

        async def close(self) -> None:
            return None

    class RecordingTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.started = asyncio.Event()

        async def stream_speech(self, text: str):
            self.calls.append(text)
            self.started.set()
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        llm = PausingLLM()
        tts = RecordingTTS()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=llm,
                tts=tts,
            ),
            tts_idle_flush_seconds=0.02,
        )
        bus = EventBus()
        bus.ready()
        task = asyncio.create_task(
            engine._process_text(bus, "question", engine._generation, "turn-idle")
        )
        await asyncio.wait_for(llm.paused.wait(), timeout=0.2)
        await asyncio.wait_for(tts.started.wait(), timeout=0.2)
        assert not task.done()
        llm.release.set()
        await asyncio.wait_for(task, timeout=0.2)
        await engine.close()
        return tts.calls

    assert asyncio.run(exercise()) == ["短句", "继续。"]


def test_new_delta_resets_idle_flush_and_finish_does_not_repeat_text() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class ControlledLLM:
        def __init__(self) -> None:
            self.first_sent = asyncio.Event()
            self.send_second = asyncio.Event()
            self.second_sent = asyncio.Event()
            self.finish = asyncio.Event()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            yield "短"
            self.first_sent.set()
            await self.send_second.wait()
            yield "句"
            self.second_sent.set()
            await self.finish.wait()

        async def close(self) -> None:
            return None

    class RecordingTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.started = asyncio.Event()

        async def stream_speech(self, text: str):
            self.calls.append(text)
            self.started.set()
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        llm = ControlledLLM()
        tts = RecordingTTS()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=llm,
                tts=tts,
            ),
            tts_idle_flush_seconds=0.1,
        )
        bus = EventBus()
        bus.ready()
        task = asyncio.create_task(
            engine._process_text(bus, "question", engine._generation, "turn-reset")
        )
        await asyncio.wait_for(llm.first_sent.wait(), timeout=0.2)
        await asyncio.sleep(0.04)
        llm.send_second.set()
        await asyncio.wait_for(llm.second_sent.wait(), timeout=0.2)
        await asyncio.sleep(0.07)
        assert tts.calls == []
        await asyncio.wait_for(tts.started.wait(), timeout=0.1)
        llm.finish.set()
        await asyncio.wait_for(task, timeout=0.2)
        await engine.close()
        return tts.calls

    assert asyncio.run(exercise()) == ["短句"]


def test_interruption_cancels_pending_idle_flush_without_late_audio() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class BlockingLLM:
        def __init__(self) -> None:
            self.paused = asyncio.Event()
            self.release = asyncio.Event()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            yield "不要播"
            self.paused.set()
            await self.release.wait()

        async def close(self) -> None:
            return None

    class RecordingTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def stream_speech(self, text: str):
            self.calls.append(text)
            if False:
                yield SpeechBuffer.empty()

        async def close(self) -> None:
            return None

    async def exercise() -> list[str]:
        llm = BlockingLLM()
        tts = RecordingTTS()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=llm,
                tts=tts,
            ),
            tts_idle_flush_seconds=0.05,
        )
        bus = EventBus()
        bus.ready()
        task = asyncio.create_task(
            engine._process_text(bus, "question", engine._generation, "turn-cancel")
        )
        await asyncio.wait_for(llm.paused.wait(), timeout=0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.07)
        await engine.close()
        return tts.calls

    assert asyncio.run(exercise()) == []


def test_incremental_tts_opens_one_turn_for_all_verified_llm_pieces() -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class StreamingLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            assert messages[-1] == {"role": "user", "content": "question"}
            yield "first."
            yield "second."

        async def close(self) -> None:
            return None

    class IncrementalTurn:
        def __init__(self) -> None:
            self.texts: list[str] = []
            self.finished = asyncio.Event()
            self.finish_calls = 0
            self.cancel_calls = 0

        async def send_text(self, text: str) -> None:
            self.texts.append(text)

        async def finish_text(self) -> None:
            self.finish_calls += 1
            self.finished.set()

        async def stream_audio(self):
            await self.finished.wait()
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def cancel(self) -> None:
            self.cancel_calls += 1

    class IncrementalTTS:
        def __init__(self) -> None:
            self.turn = IncrementalTurn()
            self.open_calls = 0
            self.legacy_calls: list[str] = []

        async def open_text_stream(self) -> IncrementalTurn:
            self.open_calls += 1
            return self.turn

        async def stream_speech(self, text: str):
            self.legacy_calls.append(text)
            raise AssertionError("incremental TTS must not fall back per sentence")
            if False:
                yield SpeechBuffer.empty()

        async def close(self) -> None:
            return None

    async def exercise():
        tts = IncrementalTTS()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=StreamingLLM(),
                tts=tts,
            ),
        )
        bus = EventBus()
        bus.ready()
        await engine._process_text(bus, "question", engine._generation, "turn-stream")
        await engine.close()
        return tts

    tts = asyncio.run(exercise())

    assert tts.open_calls == 1
    assert tts.turn.texts == ["first.", "second."]
    assert tts.turn.finish_calls == 1
    assert tts.turn.cancel_calls == 0
    assert tts.legacy_calls == []


@pytest.mark.parametrize("emit_audio_first", [False, True, None])
def test_incremental_tts_only_falls_back_before_first_audio(
    emit_audio_first: bool | None,
) -> None:
    subject = _custom_api_subject()

    class FakeASR:
        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            return ""

        async def close(self) -> None:
            return None

    class StreamingLLM:
        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            yield "first."
            yield "second."

        async def close(self) -> None:
            return None

    class FailingTurn:
        def __init__(self) -> None:
            self.finished = asyncio.Event()
            self.cancel_calls = 0

        async def send_text(self, text: str) -> None:
            assert text

        async def finish_text(self) -> None:
            self.finished.set()

        async def stream_audio(self):
            await self.finished.wait()
            if emit_audio_first is True:
                yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)
            if emit_audio_first is not None:
                raise RuntimeError("websocket failed")

        async def cancel(self) -> None:
            self.cancel_calls += 1

    class FallbackTTS:
        def __init__(self) -> None:
            self.turn = FailingTurn()
            self.legacy_calls: list[str] = []

        async def open_text_stream(self) -> FailingTurn:
            return self.turn

        async def stream_speech(self, text: str):
            self.legacy_calls.append(text)
            yield SpeechBuffer.from_bytes(b"\x03\x00\x04\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise():
        tts = FallbackTTS()
        engine = subject.CustomAPIConversationEngine(
            _connection_config(),
            components=subject.CustomAPIComponents(
                asr=FakeASR(),
                llm=StreamingLLM(),
                tts=tts,
            ),
        )
        bus = EventBus()
        bus.ready()
        failure = None
        try:
            await engine._process_text(
                bus,
                "question",
                engine._generation,
                "turn-fallback",
            )
        except BaseException as exc:
            failure = exc
        await engine.close()
        return tts, failure

    tts, failure = asyncio.run(exercise())

    if emit_audio_first is True:
        assert failure is not None
        assert tts.legacy_calls == []
    else:
        assert failure is None
        assert tts.legacy_calls == ["first.second."]
    assert tts.turn.cancel_calls == 1


@pytest.mark.parametrize(
    ("partials", "final_transcript"),
    [
        (("请简单介绍北京",), "请简单介绍北京"),
        (("请简单介绍", "请简单介绍北京天气"), "请简单介绍北京天气"),
    ],
)
def test_stable_partial_exact_reuses_buffered_llm_without_speculative_audio(
    partials: tuple[str, ...],
    final_transcript: str,
) -> None:
    subject = _custom_api_subject()

    class PartialRealtimeASR:
        def __init__(self) -> None:
            self.callback = None
            self.started = asyncio.Event()

        def set_partial_transcript_callback(self, callback) -> None:
            self.callback = callback

        async def emit_partial(self, transcript: str) -> None:
            assert self.callback is not None
            await self.callback(transcript)

        async def start(self) -> None:
            self.started.set()

        async def feed(self, audio: SpeechBuffer) -> None:
            assert not audio.is_empty

        async def finish(self) -> str:
            return final_transcript

        async def cancel(self) -> None:
            return None

        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            raise AssertionError("realtime ASR must finish the active turn")

        async def close(self) -> None:
            return None

    class BufferedLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.started = asyncio.Event()
            self.completed = asyncio.Event()
            self.started_prompts: asyncio.Queue[str] = asyncio.Queue()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            prompt = messages[-1]["content"]
            self.calls.append(prompt)
            self.started_prompts.put_nowait(prompt)
            self.started.set()
            yield f"answer for {prompt}."
            self.completed.set()

        async def close(self) -> None:
            return None

    class RecordingTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.started = asyncio.Event()

        async def stream_speech(self, text: str):
            self.calls.append(text)
            self.started.set()
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[list[str], list[str], list[str]]:
        config = builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "thinking_mode": "fast",
                },
                "vad": {
                    "rms_threshold": 0.02,
                    "pre_roll_ms": 0,
                    "min_speech_ms": 100,
                    "silence_ms": 200,
                    "max_turn_seconds": 10,
                },
            }
        )
        asr = PartialRealtimeASR()
        llm = BufferedLLM()
        tts = RecordingTTS()
        memory_recalls: list[str] = []

        async def recall_memory(transcript: str) -> str:
            memory_recalls.append(transcript)
            return (
                "<digibox_memory_recall>\n"
                f"- context for {transcript}\n"
                "</digibox_memory_recall>"
            )

        engine = subject.CustomAPIConversationEngine(
            config,
            components=subject.CustomAPIComponents(asr=asr, llm=llm, tts=tts),
            memory_recall=recall_memory,
        )
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)

        for _ in range(5):
            await bus.publish(UserSpeechReceived(buffer=speech))
        await asyncio.wait_for(asr.started.wait(), timeout=0.5)

        # Too little context is not safe enough to spend a provider request.
        await asr.emit_partial("你好")
        await asyncio.sleep(0)
        assert not llm.started.is_set()

        for partial in partials:
            await asr.emit_partial(partial)
            assert await asyncio.wait_for(llm.started_prompts.get(), timeout=0.5) == partial
        assert tts.calls == []
        assert engine._active_segment is None

        for _ in range(5):
            await bus.publish(UserSpeechReceived(buffer=speech))
        for _ in range(10):
            await bus.publish(UserSpeechReceived(buffer=silence))
        await asyncio.wait_for(tts.started.wait(), timeout=0.5)

        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return llm.calls, tts.calls, memory_recalls

    llm_calls, tts_calls, memory_recalls = asyncio.run(exercise())

    assert llm_calls == list(partials)
    assert tts_calls == [f"answer for {final_transcript}."]
    assert memory_recalls == list(partials)


@pytest.mark.parametrize(
    "final_transcript",
    ["请简单介绍上海", "请简单介绍北京天气"],
)
def test_mismatched_final_asr_cancels_speculation_and_only_speaks_verified_text(
    final_transcript: str,
) -> None:
    subject = _custom_api_subject()

    class PartialRealtimeASR:
        def __init__(self) -> None:
            self.callback = None
            self.started = asyncio.Event()

        def set_partial_transcript_callback(self, callback) -> None:
            self.callback = callback

        async def emit_partial(self, transcript: str) -> None:
            assert self.callback is not None
            await self.callback(transcript)

        async def start(self) -> None:
            self.started.set()

        async def feed(self, audio: SpeechBuffer) -> None:
            assert not audio.is_empty

        async def finish(self) -> str:
            return final_transcript

        async def cancel(self) -> None:
            return None

        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            raise AssertionError("realtime ASR must finish the active turn")

        async def close(self) -> None:
            return None

    class RestartingLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.speculation_started = asyncio.Event()
            self.cancelled = 0
            self.hold_speculation = asyncio.Event()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            prompt = messages[-1]["content"]
            self.calls.append(prompt)
            if len(self.calls) == 1:
                self.speculation_started.set()
                try:
                    yield "wrong speculative answer."
                    await self.hold_speculation.wait()
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise
                return
            yield "verified answer."

        async def close(self) -> None:
            return None

    class RecordingTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.started = asyncio.Event()

        async def stream_speech(self, text: str):
            self.calls.append(text)
            self.started.set()
            yield SpeechBuffer.from_bytes(b"\x01\x00\x02\x00", 24_000)

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[list[str], int, list[str]]:
        config = builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "thinking_mode": "fast",
                },
                "vad": {
                    "rms_threshold": 0.02,
                    "pre_roll_ms": 0,
                    "min_speech_ms": 100,
                    "silence_ms": 200,
                    "max_turn_seconds": 10,
                },
            }
        )
        asr = PartialRealtimeASR()
        llm = RestartingLLM()
        tts = RecordingTTS()
        engine = subject.CustomAPIConversationEngine(
            config,
            components=subject.CustomAPIComponents(asr=asr, llm=llm, tts=tts),
        )
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        silence = SpeechBuffer(np.zeros(480, dtype=np.int16), 24_000)

        for _ in range(5):
            await bus.publish(UserSpeechReceived(buffer=speech))
        await asyncio.wait_for(asr.started.wait(), timeout=0.5)
        await asr.emit_partial("请简单介绍北京")
        await asyncio.wait_for(llm.speculation_started.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert tts.calls == []

        for _ in range(5):
            await bus.publish(UserSpeechReceived(buffer=speech))
        for _ in range(10):
            await bus.publish(UserSpeechReceived(buffer=silence))
        await asyncio.wait_for(tts.started.wait(), timeout=0.5)

        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return llm.calls, llm.cancelled, tts.calls

    llm_calls, cancelled, tts_calls = asyncio.run(exercise())

    assert llm_calls == ["请简单介绍北京", final_transcript]
    assert cancelled == 1
    assert tts_calls == ["verified answer."]


def test_interruption_cancels_partial_speculation_without_synthesising_audio() -> None:
    subject = _custom_api_subject()

    class PartialRealtimeASR:
        def __init__(self) -> None:
            self.callback = None
            self.started = asyncio.Event()
            self.cancelled = 0

        def set_partial_transcript_callback(self, callback) -> None:
            self.callback = callback

        async def emit_partial(self, transcript: str) -> None:
            assert self.callback is not None
            await self.callback(transcript)

        async def start(self) -> None:
            self.started.set()

        async def feed(self, audio: SpeechBuffer) -> None:
            assert not audio.is_empty

        async def finish(self) -> str:
            raise AssertionError("interrupted capture must not finish")

        async def cancel(self) -> None:
            self.cancelled += 1

        async def transcribe(self, audio: SpeechBuffer) -> str:
            _ = audio
            raise AssertionError("interrupted capture must not transcribe")

        async def close(self) -> None:
            return None

    class BlockingLLM:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = 0
            self.release = asyncio.Event()

        async def probe(self) -> None:
            return None

        async def stream_text(self, messages):
            _ = messages
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            if False:
                yield ""

        async def close(self) -> None:
            return None

    class RecordingTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def stream_speech(self, text: str):
            self.calls.append(text)
            if False:
                yield SpeechBuffer.empty()

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[int, int, list[str]]:
        config = builders.CustomAPIConnectionConfig.model_validate(
            {
                "provider": {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "thinking_mode": "fast",
                },
                "vad": {
                    "rms_threshold": 0.02,
                    "pre_roll_ms": 0,
                    "min_speech_ms": 100,
                    "silence_ms": 200,
                    "max_turn_seconds": 10,
                },
            }
        )
        asr = PartialRealtimeASR()
        llm = BlockingLLM()
        tts = RecordingTTS()
        engine = subject.CustomAPIConversationEngine(
            config,
            components=subject.CustomAPIComponents(asr=asr, llm=llm, tts=tts),
        )
        bus = EventBus()
        task = asyncio.create_task(engine.run(bus.clone(), object()))  # type: ignore[arg-type]
        bus.ready()
        speech = SpeechBuffer(np.full(480, 8_000, dtype=np.int16), 24_000)
        for _ in range(5):
            await bus.publish(UserSpeechReceived(buffer=speech))
        await asyncio.wait_for(asr.started.wait(), timeout=0.5)
        await asr.emit_partial("请简单介绍北京")
        await asyncio.wait_for(llm.started.wait(), timeout=0.5)

        await bus.publish(Shutdown())
        await asyncio.wait_for(task, timeout=0.5)
        await engine.close()
        return asr.cancelled, llm.cancelled, tts.calls

    asr_cancelled, llm_cancelled, tts_calls = asyncio.run(exercise())

    assert asr_cancelled >= 1
    assert llm_cancelled == 1
    assert tts_calls == []
