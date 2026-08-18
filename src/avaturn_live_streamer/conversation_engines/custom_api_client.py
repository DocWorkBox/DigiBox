# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""OpenAI-compatible ASR -> LLM -> streaming TTS conversation engine."""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
import wave
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import httpx
import numpy as np

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.builders import (
    AliyunBailianProvider,
    APIAuth,
    ASRAPIOptions,
    CustomAPIConnectionConfig,
    GenericAPIProvider,
    LLMAPIOptions,
    MiniMaxProvider,
    TTSAPIOptions,
    VADOptions,
    append_memory_prompt,
)
from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
    IncrementalTextTTSTurn,
    SentenceChunker,
)
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SegmentPlaybackCancelled,
    SegmentPlaybackCompleted,
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
    TextEchoEnqueueText,
    TurnLatencyMilestone,
    TurnLatencyPhase,
    UserSpeechReceived,
)
from avaturn_live_streamer.management.types import SegmentId, make_segment_id
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.datetime import tzutcnow

_LOGGER = get_logger()
_TTS_QUEUE_END = object()
_TTS_QUEUE_CAPACITY = 4
_TTS_IDLE_FLUSH_SECONDS = 0.12
_SPECULATION_END = object()
_STABLE_PARTIAL_MIN_CHARS = 5
_STABLE_PARTIAL_RESTART_CHARS = 4
_PLAYBACK_BARGE_IN_RMS_MULTIPLIER = 2.0
_PLAYBACK_BARGE_IN_RMS_FLOOR = 0.04
_PLAYBACK_BARGE_IN_MIN_SPEECH_MS = 320.0
_PLAYBACK_ECHO_TAIL_MS = 300.0
_FAST_FIRST_SENTENCE_INSTRUCTION = (
    "语音低延迟模式:先用不超过12个汉字的直接短句回答核心结论,"
    "再继续展开;不要改变既定人格。"
)


@dataclass(frozen=True, slots=True)
class ComponentPreflight:
    status: str
    latency_ms: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CustomAPIPreflightReport:
    status: str
    components: dict[str, ComponentPreflight]


@dataclass(slots=True)
class _SpeculativeResponse:
    transcript: str
    normalised_transcript: str
    turn_id: str
    generation: int
    queue: asyncio.Queue[object]
    task: asyncio.Task[None]


class ASRBackend(Protocol):
    async def transcribe(self, audio: SpeechBuffer) -> str: ...

    async def close(self) -> None: ...


@runtime_checkable
class RealtimeASRBackend(ASRBackend, Protocol):
    async def start(self) -> None: ...

    async def feed(self, audio: SpeechBuffer) -> None: ...

    async def finish(self) -> str: ...

    async def cancel(self) -> None: ...


class LLMBackend(Protocol):
    async def probe(self) -> None: ...

    def stream_text(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


class TTSBackend(Protocol):
    def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CustomAPIComponents:
    asr: ASRBackend
    llm: LLMBackend
    tts: TTSBackend
    asr_fallback: ASRBackend | None = None

    async def close(self) -> None:
        closers = [self.asr.close(), self.llm.close(), self.tts.close()]
        if self.asr_fallback is not None and self.asr_fallback is not self.asr:
            closers.append(self.asr_fallback.close())
        await asyncio.gather(*closers)


def _endpoint_url(base_url: object, path: str) -> str:
    return f"{str(base_url).rstrip('/')}/{path.lstrip('/')}"


def _auth_headers(auth: APIAuth) -> dict[str, str]:
    secret = auth.api_key.get_secret_value()
    if auth.mode == "none" or not secret:
        return {}
    value = f"Bearer {secret}" if auth.mode == "bearer" else secret
    return {auth.header_name: value}


def _wav_bytes(audio: SpeechBuffer, sample_rate: int) -> bytes:
    pcm = audio.resample(sample_rate).to_bytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _sanitise_error(error: BaseException, config: CustomAPIConnectionConfig) -> str:
    messages = [str(error)]
    if isinstance(error, BaseExceptionGroup):
        pending = list(error.exceptions)
        while pending:
            nested = pending.pop(0)
            messages.append(str(nested))
            if isinstance(nested, BaseExceptionGroup):
                pending[0:0] = nested.exceptions
    message = " | ".join(messages)
    provider = config.provider
    if isinstance(provider, GenericAPIProvider):
        secrets = [
            provider.llm.auth.api_key.get_secret_value(),
            provider.asr.auth.api_key.get_secret_value(),
            provider.tts.auth.api_key.get_secret_value(),
        ]
    else:
        secrets = [provider.api_key.get_secret_value()]
    if config.tts_override is not None:
        secrets.append(config.tts_override.auth.api_key.get_secret_value())
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    return message[:1000]


async def _time_probe(name: str, probe) -> tuple[str, ComponentPreflight]:
    started = time.perf_counter()
    try:
        await probe()
    except Exception as exc:
        return name, ComponentPreflight(
            status="failed",
            latency_ms=round((time.perf_counter() - started) * 1000),
            error=str(exc),
        )
    return name, ComponentPreflight(
        status="ready",
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


async def probe_tts(tts: TTSBackend) -> None:
    received_audio = False
    open_text_stream = getattr(tts, "open_text_stream", None)
    if callable(open_text_stream):
        turn: IncrementalTextTTSTurn | None = None
        completed = False
        try:
            turn = await open_text_stream()
            await turn.send_text("连接测试")
            await turn.finish_text()
            async for _ in turn.stream_audio():
                received_audio = True
            completed = received_audio
        finally:
            if turn is not None and not completed:
                with suppress(Exception):
                    await turn.cancel()
    else:
        async for _ in tts.stream_speech("连接测试"):
            received_audio = True
    if not received_audio:
        raise RuntimeError("TTS response did not contain audio")


_probe_tts = probe_tts


async def preflight_custom_api(
    config: CustomAPIConnectionConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    components: CustomAPIComponents | None = None,
) -> CustomAPIPreflightReport:
    """Exercise the selected model on all three endpoints in parallel."""

    if isinstance(config.provider, MiniMaxProvider):
        prepared = await _prepare_minimax_realtime(config, transport=transport)
        if prepared.engine is not None:
            await prepared.engine.close()
        return prepared.report

    owned_components = components is None
    selected = components or build_custom_api_components(config, transport=transport)

    async def probe_llm() -> None:
        await selected.llm.probe()

    async def probe_asr() -> None:
        await selected.asr.transcribe(SpeechBuffer.silence(0.5, 16_000))

    try:
        raw = await asyncio.gather(
            _time_probe("llm", probe_llm),
            _time_probe("asr", probe_asr),
            _time_probe("tts", lambda: _probe_tts(selected.tts)),
        )
    finally:
        if owned_components:
            await selected.close()
    components = dict(raw)
    # Redact credentials from errors even when an HTTP library includes URLs
    # or headers in its exception text.
    components = {
        name: ComponentPreflight(
            status=item.status,
            latency_ms=item.latency_ms,
            error=(
                _sanitise_error(RuntimeError(item.error), config)
                if item.error is not None
                else None
            ),
        )
        for name, item in components.items()
    }
    status = "ready" if all(item.status == "ready" for item in components.values()) else "failed"
    return CustomAPIPreflightReport(status=status, components=components)


async def iter_pcm_s16le(
    chunks: AsyncIterable[bytes],
    *,
    sample_rate: int,
) -> AsyncIterator[SpeechBuffer]:
    """Turn arbitrary HTTP chunks into aligned little-endian PCM16 chunks."""

    carry = b""
    async for chunk in chunks:
        data = carry + bytes(chunk)
        even_length = len(data) & ~1
        if even_length:
            yield SpeechBuffer.from_bytes(data[:even_length], sample_rate)
        carry = data[even_length:]
    if carry:
        raise ValueError("PCM16 stream ended with half of a sample")


class EnergyTurnDetector:
    """Dependency-free RMS VAD with pre-roll and trailing-silence hangover."""

    def __init__(self, config: VADOptions) -> None:
        self.config = config
        self._pre_roll: list[SpeechBuffer] = []
        self._pre_roll_ms = 0.0
        self._chunks: list[SpeechBuffer] = []
        self._stream_cursor = 0
        self._speech_ms = 0.0
        self._onset_speech_ms = 0.0
        self._silence_ms = 0.0
        self._turn_ms = 0.0
        self.last_vad_wait_ms = 0
        self.is_active = False
        self.is_speech_confirmed = False
        self._assistant_playback_active = False
        self._echo_tail_remaining_ms = 0.0

    @staticmethod
    def _duration_ms(chunk: SpeechBuffer) -> float:
        return float(chunk.duration) * 1000.0

    def _is_speech(self, chunk: SpeechBuffer, *, guarded: bool) -> bool:
        samples = np.frombuffer(chunk.to_bytes(), dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples / 32768.0))))
        threshold = self.config.rms_threshold
        if guarded:
            threshold = max(
                threshold * _PLAYBACK_BARGE_IN_RMS_MULTIPLIER,
                _PLAYBACK_BARGE_IN_RMS_FLOOR,
            )
        return rms >= threshold

    def set_assistant_playback(self, active: bool) -> None:
        """Use stricter double-talk detection while avatar audio is audible."""

        if active == self._assistant_playback_active:
            return
        self._assistant_playback_active = active
        self._echo_tail_remaining_ms = 0.0 if active else _PLAYBACK_ECHO_TAIL_MS
        if self.is_active and not self.is_speech_confirmed:
            self._reset()

    def _remember_pre_roll(self, chunk: SpeechBuffer) -> None:
        if self.config.pre_roll_ms <= 0:
            return
        self._pre_roll.append(chunk)
        self._pre_roll_ms += self._duration_ms(chunk)
        while self._pre_roll and self._pre_roll_ms > self.config.pre_roll_ms:
            removed = self._pre_roll.pop(0)
            self._pre_roll_ms -= self._duration_ms(removed)

    def _reset(self) -> None:
        self._chunks = []
        self._stream_cursor = 0
        self._speech_ms = 0.0
        self._onset_speech_ms = 0.0
        self._silence_ms = 0.0
        self._turn_ms = 0.0
        self.is_active = False
        self.is_speech_confirmed = False

    def take_unstreamed_chunks(self) -> list[SpeechBuffer]:
        """Return active-turn chunks not yet handed to a streaming ASR."""

        chunks = self._chunks[self._stream_cursor :]
        self._stream_cursor = len(self._chunks)
        return chunks

    def feed(self, chunk: SpeechBuffer) -> SpeechBuffer | None:
        duration_ms = self._duration_ms(chunk)
        guarded = (
            self._assistant_playback_active or self._echo_tail_remaining_ms > 0
        )
        if not self._assistant_playback_active and self._echo_tail_remaining_ms > 0:
            previous_tail_ms = self._echo_tail_remaining_ms
            self._echo_tail_remaining_ms = max(
                0.0,
                previous_tail_ms - duration_ms,
            )
            if (
                self._echo_tail_remaining_ms == 0
                and self.is_active
                and not self.is_speech_confirmed
            ):
                self._reset()
        speech = self._is_speech(chunk, guarded=guarded)
        minimum_speech_ms = self.config.min_speech_ms
        if guarded:
            minimum_speech_ms = max(
                minimum_speech_ms,
                _PLAYBACK_BARGE_IN_MIN_SPEECH_MS,
            )
        if not self.is_active:
            if not speech:
                self._remember_pre_roll(chunk)
                return None
            self.is_active = True
            self.last_vad_wait_ms = 0
            self._chunks = [*self._pre_roll, chunk]
            self._pre_roll = []
            self._pre_roll_ms = 0.0
            self._speech_ms = duration_ms
            self._onset_speech_ms = duration_ms
            self._turn_ms = sum(self._duration_ms(item) for item in self._chunks)
            self.is_speech_confirmed = (
                self._onset_speech_ms >= minimum_speech_ms
            )
            return None

        self._chunks.append(chunk)
        self._turn_ms += duration_ms
        if speech:
            self._speech_ms += duration_ms
            self._silence_ms = 0.0
            if not self.is_speech_confirmed:
                self._onset_speech_ms += duration_ms
                self.is_speech_confirmed = (
                    self._onset_speech_ms >= minimum_speech_ms
                )
        else:
            self._silence_ms += duration_ms
            if not self.is_speech_confirmed:
                self._onset_speech_ms = 0.0

        complete = (
            self._silence_ms >= self.config.silence_ms
            or self._turn_ms >= self.config.max_turn_seconds * 1000
        )
        if not complete:
            return None
        chunks = self._chunks
        valid = self.is_speech_confirmed
        self.last_vad_wait_ms = round(self._silence_ms)
        self._reset()
        return SpeechBuffer.concat(chunks) if valid else None


class OpenAICompatibleASR:
    def __init__(
        self,
        options: ASRAPIOptions,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.options = options
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=options.timeout_seconds,
        )

    async def transcribe(self, audio: SpeechBuffer) -> str:
        data = {"model": self.options.model}
        if self.options.language:
            data["language"] = self.options.language
        response = await self._client.post(
            _endpoint_url(self.options.base_url, self.options.path),
            headers=_auth_headers(self.options.auth),
            data=data,
            files={
                "file": (
                    "speech.wav",
                    _wav_bytes(audio, self.options.sample_rate),
                    "audio/wav",
                )
            },
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError("ASR response is missing text")
        return text.strip()

    async def close(self) -> None:
        await self._client.aclose()


class OpenAICompatibleLLM:
    def __init__(
        self,
        options: LLMAPIOptions,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        enable_web_search: bool = False,
        enable_thinking: bool | None = None,
    ) -> None:
        self.options = options
        self._enable_web_search = enable_web_search
        self._enable_thinking = enable_thinking
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=options.timeout_seconds,
        )

    async def probe(self) -> None:
        payload: dict[str, object] = {
            "model": self.options.model,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
        }
        if self._enable_web_search:
            payload["enable_search"] = True
        if self._enable_thinking is not None:
            payload["enable_thinking"] = self._enable_thinking
        response = await self._client.post(
            _endpoint_url(self.options.base_url, self.options.path),
            headers=_auth_headers(self.options.auth),
            json=payload,
        )
        response.raise_for_status()
        choices = response.json().get("choices")
        if not isinstance(choices, list):
            raise RuntimeError("LLM response is missing choices")

    async def stream_text(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {
            "model": self.options.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.options.max_tokens,
            "temperature": self.options.temperature,
        }
        if self._enable_web_search:
            payload["enable_search"] = True
        if self._enable_thinking is not None:
            payload["enable_thinking"] = self._enable_thinking
        async with self._client.stream(
            "POST",
            _endpoint_url(self.options.base_url, self.options.path),
            headers=_auth_headers(self.options.auth),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                value = line.strip()
                if not value:
                    continue
                if value.startswith("data:"):
                    value = value[5:].strip()
                if value == "[DONE]":
                    return
                try:
                    event = json.loads(value)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                content = delta.get("content") if isinstance(delta, dict) else None
                if content is None and isinstance(choice, dict):
                    message = choice.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str) and content:
                    yield content

    async def close(self) -> None:
        await self._client.aclose()


_RECENT_QWEN_SEARCH_MODEL = re.compile(
    r"^qwen3\.(?P<minor>\d+)-(?:max|plus|flash)(?:$|-)",
    re.IGNORECASE,
)
_QWEN3_MODEL = re.compile(r"^qwen3(?:[.-]|$)", re.IGNORECASE)


def _uses_responses_web_search(model: str) -> bool:
    """Return whether web search for this Qwen generation needs Responses API."""

    match = _RECENT_QWEN_SEARCH_MODEL.match(model.strip())
    return match is not None and int(match.group("minor")) >= 5


def _qwen_thinking_control(model: str, mode: str) -> bool | None:
    """Return an explicit Qwen 3 thinking flag without polluting other APIs."""

    if _QWEN3_MODEL.match(model.strip()) is None:
        return None
    return mode == "deep"


_AUTO_WEB_SEARCH_PATTERN = re.compile(
    r"(?:"
    r"联网|搜索|查一下|查找|查询|最新|今天|最近|近期|当前|目前|新闻|天气|实时|价格|汇率|股价|比分|赛程"
    r"|\b(?:search|look\s+up|latest|recent|today|news|weather|price|stock|score|schedule)\b"
    r")",
    re.IGNORECASE,
)


def should_use_web_search(messages: list[dict[str, str]]) -> bool:
    """Route only freshness/search-shaped user turns through web search."""

    user_text = next(
        (
            message.get("content", "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    return bool(_AUTO_WEB_SEARCH_PATTERN.search(user_text))


class QwenResponsesWebSearchLLM:
    """Qwen web-search client for models served by the Responses API."""

    def __init__(
        self,
        options: LLMAPIOptions,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.options = options
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=options.timeout_seconds,
        )

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        return {
            "model": self.options.model,
            "input": messages,
            "tools": [{"type": "web_search"}],
            "stream": stream,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }

    @staticmethod
    def _raise_for_failure(payload: dict[str, object]) -> None:
        response = payload.get("response")
        result = response if isinstance(response, dict) else payload
        if (
            payload.get("type") != "response.failed"
            and result.get("status") != "failed"
        ):
            return
        error = result.get("error")
        if isinstance(error, dict):
            code = error.get("code") or "unknown-error"
            message = error.get("message") or "no error message"
        else:
            code = "unknown-error"
            message = "no error message"
        raise RuntimeError(f"Qwen Responses failed: {code}: {message}")

    @staticmethod
    def _raise_for_http_error(response: httpx.Response) -> None:
        if response.is_success:
            return
        code = "http_error"
        message = response.reason_phrase or "request failed"
        try:
            payload = response.json()
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            detail = error if isinstance(error, dict) else payload
            if isinstance(detail.get("code"), str):
                code = detail["code"][:120]
            if isinstance(detail.get("message"), str):
                message = detail["message"][:1000]
        raise RuntimeError(
            f"Qwen Responses HTTP {response.status_code}: {code}: {message}"
        )

    async def probe(self) -> None:
        response = await self._client.post(
            _endpoint_url(self.options.base_url, "/responses"),
            headers=_auth_headers(self.options.auth),
            json=self._payload(
                [{"role": "user", "content": "Reply OK."}],
                stream=False,
                max_output_tokens=16,
                temperature=0,
            ),
        )
        self._raise_for_http_error(response)
        payload = response.json()
        self._raise_for_failure(payload)
        output = payload.get("output")
        if not isinstance(output, list):
            raise RuntimeError("Qwen Responses result is missing output")

    async def stream_text(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            _endpoint_url(self.options.base_url, "/responses"),
            headers=_auth_headers(self.options.auth),
            json=self._payload(
                messages,
                stream=True,
                max_output_tokens=self.options.max_tokens,
                temperature=self.options.temperature,
            ),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                value = line.strip()
                if not value:
                    continue
                if value.startswith("data:"):
                    value = value[5:].strip()
                if value == "[DONE]":
                    return
                try:
                    event = json.loads(value)
                except json.JSONDecodeError:
                    continue
                self._raise_for_failure(event)
                if event.get("type") != "response.output_text.delta":
                    continue
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    yield delta

    async def close(self) -> None:
        await self._client.aclose()


class RoutedQwenLLM:
    """Select plain or web-search Qwen per turn without another model call."""

    def __init__(
        self,
        plain: LLMBackend,
        search: LLMBackend,
        mode: str,
    ) -> None:
        self._plain = plain
        self._search = search
        self._mode = mode

    async def probe(self) -> None:
        # Auto mode keeps connection tests fast and resilient: search is only
        # exercised by a turn whose text actually needs it. Always mode remains
        # an explicit end-to-end search connectivity test.
        await (self._search if self._mode == "always" else self._plain).probe()

    async def stream_text(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        use_search = self._mode == "always" or (
            self._mode == "auto" and should_use_web_search(messages)
        )
        backend = self._search if use_search else self._plain
        async for chunk in backend.stream_text(messages):
            yield chunk

    async def close(self) -> None:
        await asyncio.gather(self._plain.close(), self._search.close())


class _IncrementalTTSWithFallback:
    """Expose one duplex turn while retaining HTTP for probes and safe fallback."""

    def __init__(self, incremental: Any, fallback: TTSBackend) -> None:
        self._incremental = incremental
        self._fallback = fallback

    async def open_text_stream(self) -> IncrementalTextTTSTurn:
        return await self._incremental.open_text_stream()

    def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]:
        return self._fallback.stream_speech(text)

    async def close(self) -> None:
        await asyncio.gather(self._incremental.close(), self._fallback.close())


class OpenAICompatibleTTS:
    def __init__(
        self,
        options: TTSAPIOptions,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.options = options
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=options.timeout_seconds,
        )
        self._streaming_client = None
        if options.streaming_text == "websocket":
            from avaturn_live_streamer.conversation_engines.local_cosyvoice_client import (
                LocalCosyVoiceStreamingClient,
            )

            secret = (
                ""
                if options.auth.mode == "none"
                else options.auth.api_key.get_secret_value()
            )
            self._streaming_client = LocalCosyVoiceStreamingClient(
                base_url=str(options.base_url),
                model=options.model,
                voice=options.voice,
                api_key=secret,
                timeout_seconds=options.timeout_seconds,
            )
            # Keep legacy/generic providers on HTTP by leaving this attribute
            # absent unless the configuration explicitly opts into duplex text.
            self.open_text_stream = self._streaming_client.open_text_stream

    async def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]:
        payload: dict[str, object] = {
            "model": self.options.model,
            "input": text,
            "voice": self.options.voice,
            "response_format": self.options.response_format,
            "stream": True,
        }
        if self.options.instructions:
            payload["instructions"] = self.options.instructions
        async with self._client.stream(
            "POST",
            _endpoint_url(self.options.base_url, self.options.path),
            headers=_auth_headers(self.options.auth),
            json=payload,
        ) as response:
            response.raise_for_status()
            if self.options.response_format == "pcm":
                async for chunk in iter_pcm_s16le(
                    response.aiter_bytes(),
                    sample_rate=self.options.sample_rate,
                ):
                    yield chunk
                return
            raw = await response.aread()
            with wave.open(io.BytesIO(raw), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                    raise RuntimeError("WAV TTS output must be mono PCM16")
                yield SpeechBuffer.from_bytes(wav.readframes(wav.getnframes()), wav.getframerate())

    async def close(self) -> None:
        closers = [self._client.aclose()]
        if self._streaming_client is not None:
            closers.append(self._streaming_client.close())
        await asyncio.gather(*closers)


def build_custom_api_components(
    config: CustomAPIConnectionConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CustomAPIComponents:
    provider = config.provider
    if isinstance(provider, GenericAPIProvider):
        return CustomAPIComponents(
            asr=OpenAICompatibleASR(provider.asr, transport=transport),
            llm=OpenAICompatibleLLM(provider.llm, transport=transport),
            tts=OpenAICompatibleTTS(provider.tts, transport=transport),
        )
    if isinstance(provider, AliyunBailianProvider):
        from avaturn_live_streamer.conversation_engines.aliyun_bailian_client import (
            BailianCosyVoiceRealtimeTTS,
            BailianCosyVoiceTTS,
            BailianQwenASR,
            BailianQwenRealtimeASR,
            bailian_base_url,
        )

        llm = LLMAPIOptions(
            base_url=f"{bailian_base_url(provider)}/compatible-mode/v1",
            auth=APIAuth(api_key=provider.api_key),
            timeout_seconds=provider.timeout_seconds,
            model=provider.llm_model,
        )
        enable_thinking = _qwen_thinking_control(
            provider.llm_model,
            provider.thinking_mode,
        )
        plain_llm: LLMBackend = OpenAICompatibleLLM(
            llm,
            transport=transport,
            enable_thinking=enable_thinking,
        )
        if provider.web_search_mode == "off":
            llm_client = plain_llm
        else:
            search_llm: LLMBackend
            if _uses_responses_web_search(provider.llm_model):
                search_llm = QwenResponsesWebSearchLLM(
                    llm,
                    transport=transport,
                )
            else:
                search_llm = OpenAICompatibleLLM(
                    llm,
                    transport=transport,
                    enable_web_search=True,
                    enable_thinking=enable_thinking,
                )
            llm_client = RoutedQwenLLM(
                plain_llm,
                search_llm,
                provider.web_search_mode,
            )
        realtime_asr = provider.asr_model.strip().lower().endswith("-realtime")
        fallback_provider = (
            provider.model_copy(
                update={
                    "asr_model": provider.asr_model.strip()[: -len("-realtime")]
                }
            )
            if realtime_asr
            else provider
        )
        return CustomAPIComponents(
            asr=(
                BailianQwenRealtimeASR(provider)
                if realtime_asr
                else BailianQwenASR(provider, transport=transport)
            ),
            llm=llm_client,
            tts=(
                OpenAICompatibleTTS(config.tts_override, transport=transport)
                if config.tts_override is not None
                else _IncrementalTTSWithFallback(
                    BailianCosyVoiceRealtimeTTS(provider),
                    BailianCosyVoiceTTS(provider, transport=transport),
                )
            ),
            asr_fallback=(
                BailianQwenASR(fallback_provider, transport=transport)
                if realtime_asr
                else None
            ),
        )
    raise TypeError(f"unsupported custom API provider: {type(provider).__name__}")


class CustomAPIConversationEngine:
    def __init__(
        self,
        config: CustomAPIConnectionConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        components: CustomAPIComponents | None = None,
        tts_idle_flush_seconds: float = _TTS_IDLE_FLUSH_SECONDS,
        memory_recall: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        if tts_idle_flush_seconds <= 0:
            raise ValueError("tts_idle_flush_seconds must be positive")
        self.config = config
        self._components = components or build_custom_api_components(
            config,
            transport=transport,
        )
        self._asr = self._components.asr
        self._asr_fallback = self._components.asr_fallback
        self._llm = self._components.llm
        self._tts = self._components.tts
        self._tts_idle_flush_seconds = tts_idle_flush_seconds
        self._memory_recall = memory_recall
        self._vad = EnergyTurnDetector(config.vad)
        self._turn_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._active_segment: SegmentId | None = None
        self._active_turn_id: str | None = None
        self._history: list[dict[str, str]] = []
        self._realtime_asr = (
            self._asr if isinstance(self._asr, RealtimeASRBackend) else None
        )
        self._capture_turn_id: str | None = None
        self._capture_generation: int | None = None
        self._capture_started = False
        self._capture_stream_failed = False
        self._speculative_response: _SpeculativeResponse | None = None
        self._playback_segments: set[SegmentId] = set()

    @property
    def _thinking_mode(self) -> str:
        provider = self.config.provider
        if isinstance(provider, AliyunBailianProvider):
            return provider.thinking_mode
        return "fast"

    @property
    def _can_speculate_partial(self) -> bool:
        provider = self.config.provider
        return (
            isinstance(provider, AliyunBailianProvider)
            and provider.thinking_mode == "fast"
        )

    def _set_partial_transcript_callback(
        self,
        callback: Callable[[str], Awaitable[None]] | None,
    ) -> None:
        setter = getattr(self._realtime_asr, "set_partial_transcript_callback", None)
        if callable(setter):
            setter(callback)

    async def _cancel_speculation(self) -> None:
        response = self._speculative_response
        self._speculative_response = None
        if response is None:
            return
        if not response.task.done():
            response.task.cancel()
        await asyncio.gather(response.task, return_exceptions=True)

    async def _on_stable_partial(
        self,
        transcript: str,
        *,
        turn_id: str,
        generation: int,
    ) -> None:
        normalised = "".join(
            char.casefold() for char in transcript if char.isalnum()
        )
        if (
            not self._can_speculate_partial
            or generation != self._generation
            or self._capture_turn_id != turn_id
            or self._capture_generation != generation
            or len(normalised) < _STABLE_PARTIAL_MIN_CHARS
        ):
            return

        current = self._speculative_response
        if current is not None:
            if current.turn_id == turn_id and current.generation == generation:
                if normalised == current.normalised_transcript:
                    return
                if current.normalised_transcript.startswith(normalised):
                    return
                if normalised.startswith(current.normalised_transcript) and (
                    len(normalised) - len(current.normalised_transcript)
                    < _STABLE_PARTIAL_RESTART_CHARS
                ):
                    return
            await self._cancel_speculation()

        queue: asyncio.Queue[object] = asyncio.Queue()
        messages = await self._messages_for_turn(transcript)

        async def produce() -> None:
            try:
                async for delta in self._llm.stream_text(messages):
                    queue.put_nowait(delta)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(_SPECULATION_END)

        task = asyncio.create_task(
            produce(),
            name=f"CustomAPIConversationEngine.speculation.{turn_id}",
        )
        self._speculative_response = _SpeculativeResponse(
            transcript=transcript,
            normalised_transcript=normalised,
            turn_id=turn_id,
            generation=generation,
            queue=queue,
            task=task,
        )

    async def _take_validated_speculation(
        self,
        transcript: str,
        *,
        turn_id: str,
        generation: int,
    ) -> AsyncIterator[str] | None:
        response = self._speculative_response
        if response is None:
            return None
        normalised = "".join(
            char.casefold() for char in transcript if char.isalnum()
        )
        valid = (
            response.turn_id == turn_id
            and response.generation == generation
            and normalised == response.normalised_transcript
        )
        if not valid:
            await self._cancel_speculation()
            return None
        self._speculative_response = None

        async def stream() -> AsyncIterator[str]:
            try:
                while True:
                    item = await response.queue.get()
                    if item is _SPECULATION_END:
                        return
                    if isinstance(item, BaseException):
                        raise item
                    assert isinstance(item, str)
                    yield item
            finally:
                if not response.task.done():
                    response.task.cancel()
                await asyncio.gather(response.task, return_exceptions=True)

        return stream()

    @staticmethod
    async def _publish_milestone(
        bus: EventBus,
        turn_id: str | None,
        phase: TurnLatencyPhase,
        **details: int | float | str,
    ) -> None:
        if turn_id is None:
            return
        await bus.publish(
            TurnLatencyMilestone(
                turn_id=turn_id,
                phase=phase,
                at_monotonic=time.perf_counter(),
                details=details,
            )
        )

    async def _finish_segment(self, bus: EventBus) -> None:
        if self._active_segment is not None:
            segment = self._active_segment
            self._active_segment = None
            await bus.publish(SegmentGenerationCompleted(segment_id=segment))

    async def _interrupt(self, bus: EventBus) -> None:
        self._generation += 1
        task = self._turn_task
        turn_id = self._active_turn_id
        self._turn_task = None
        self._active_turn_id = None
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await self._publish_milestone(bus, turn_id, "interrupted")
        await self._cancel_realtime_capture()
        await self._finish_segment(bus)
        await bus.publish(DiscardAvatarSpeechBuffer())

    async def _cancel_realtime_capture(self) -> None:
        asr = self._realtime_asr
        started = self._capture_started
        self._capture_started = False
        self._capture_turn_id = None
        self._capture_generation = None
        self._capture_stream_failed = False
        self._set_partial_transcript_callback(None)
        await self._cancel_speculation()
        if asr is not None and started:
            await asr.cancel()

    async def _streaming_asr_failed(self, exc: BaseException) -> None:
        _LOGGER.exception(
            "custom API realtime ASR stream failed; falling back to full-turn ASR",
            error=_sanitise_error(exc, self.config),
        )
        asr = self._realtime_asr
        self._capture_started = False
        self._capture_stream_failed = True
        self._set_partial_transcript_callback(None)
        await self._cancel_speculation()
        if asr is not None:
            with suppress(Exception):
                await asr.cancel()

    async def _begin_realtime_capture(
        self,
        bus: EventBus,
        chunks: list[SpeechBuffer],
    ) -> None:
        asr = self._realtime_asr
        if asr is None:
            return
        self._capture_turn_id = uuid4().hex
        self._capture_generation = self._generation
        self._capture_stream_failed = False
        turn_id = self._capture_turn_id
        generation = self._capture_generation

        async def on_partial(transcript: str) -> None:
            assert turn_id is not None and generation is not None
            await self._on_stable_partial(
                transcript,
                turn_id=turn_id,
                generation=generation,
            )

        self._set_partial_transcript_callback(on_partial)
        try:
            await asr.start()
            self._capture_started = True
            for chunk in chunks:
                await asr.feed(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._streaming_asr_failed(exc)

    async def _feed_realtime_capture(self, chunks: list[SpeechBuffer]) -> None:
        asr = self._realtime_asr
        if asr is None or not self._capture_started:
            return
        try:
            for chunk in chunks:
                await asr.feed(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._streaming_asr_failed(exc)

    def _messages(
        self,
        user_text: str,
        *,
        memory_prompt: str = "",
    ) -> list[dict[str, str]]:
        history_turns = self.config.history_turns
        if self._thinking_mode == "fast" and isinstance(
            self.config.provider,
            AliyunBailianProvider,
        ):
            history_turns = min(history_turns, self.config.fast_history_turns)
        history = self._history[-history_turns * 2 :] if history_turns else []
        messages: list[dict[str, str]] = []
        if self.config.prompt.strip():
            messages.append({"role": "system", "content": self.config.prompt.strip()})
        if memory_prompt.strip():
            messages.append({"role": "system", "content": memory_prompt.strip()})
        provider = self.config.provider
        if (
            isinstance(provider, AliyunBailianProvider)
            and provider.thinking_mode == "fast"
            and provider.llm_model.strip().lower().startswith("qwen")
        ):
            messages.append(
                {"role": "system", "content": _FAST_FIRST_SENTENCE_INSTRUCTION}
            )
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _messages_for_turn(self, user_text: str) -> list[dict[str, str]]:
        memory_prompt = ""
        if self._memory_recall is not None:
            try:
                memory_prompt = await self._memory_recall(user_text)
            except Exception as exc:
                _LOGGER.warning(
                    "local memory recall failed; continuing without it",
                    error=str(exc),
                )
        return self._messages(user_text, memory_prompt=memory_prompt)

    async def _publish_tts_audio(
        self,
        bus: EventBus,
        audio: SpeechBuffer,
        generation: int,
        turn_id: str | None,
        timing_state: set[str],
    ) -> bool:
        if generation != self._generation:
            return False
        if "tts_first_audio" not in timing_state:
            timing_state.add("tts_first_audio")
            await self._publish_milestone(bus, turn_id, "tts_first_audio")
        if self._active_segment is None:
            self._active_segment = make_segment_id()
            metadata = {"turn_id": turn_id} if turn_id is not None else {}
            await bus.publish(
                SegmentGenerationStarted(
                    segment_id=self._active_segment,
                    metadata=metadata,
                )
            )
        await bus.publish(
            SegmentChunkGenerated(segment_id=self._active_segment, buffer=audio)
        )
        return True

    async def _speak_piece(
        self,
        bus: EventBus,
        text: str,
        generation: int,
        turn_id: str | None,
        timing_state: set[str],
    ) -> None:
        async for audio in self._tts.stream_speech(text):
            if not await self._publish_tts_audio(
                bus,
                audio,
                generation,
                turn_id,
                timing_state,
            ):
                return

    async def _speak_incremental_queue(
        self,
        bus: EventBus,
        queue: asyncio.Queue[str | object],
        generation: int,
        turn_id: str | None,
        timing_state: set[str],
        open_text_stream: Callable[[], Awaitable[IncrementalTextTTSTurn]],
    ) -> None:
        turn: IncrementalTextTTSTurn | None = None
        pieces: list[str] = []
        input_consumed = False

        async def send_text() -> None:
            nonlocal input_consumed
            assert turn is not None
            while True:
                piece = await queue.get()
                if piece is _TTS_QUEUE_END:
                    input_consumed = True
                    await turn.finish_text()
                    return
                assert isinstance(piece, str)
                pieces.append(piece)
                await turn.send_text(piece)

        async def receive_audio() -> None:
            assert turn is not None
            async for audio in turn.stream_audio():
                published = await self._publish_tts_audio(
                    bus,
                    audio,
                    generation,
                    turn_id,
                    timing_state,
                )
                if not published:
                    return

        try:
            turn = await open_text_stream()
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(send_text())
                tasks.create_task(receive_audio())
            if "tts_first_audio" not in timing_state:
                raise RuntimeError("incremental TTS completed without audio")
            return
        except asyncio.CancelledError:
            if turn is not None:
                with suppress(Exception):
                    await turn.cancel()
            raise
        except BaseException:
            if turn is not None:
                with suppress(Exception):
                    await turn.cancel()
            if "tts_first_audio" in timing_state:
                # Retrying after audible PCM would repeat the beginning of the
                # answer.  Fail the turn instead of producing duplicate speech.
                raise
            if generation != self._generation:
                return

            if not input_consumed:
                while True:
                    piece = await queue.get()
                    if piece is _TTS_QUEUE_END:
                        break
                    assert isinstance(piece, str)
                    pieces.append(piece)
            fallback_text = "".join(pieces).strip()
            if not fallback_text:
                raise
            _LOGGER.warning(
                "incremental TTS failed before first audio; using HTTP once"
            )
            await self._speak_piece(
                bus,
                fallback_text,
                generation,
                turn_id,
                timing_state,
            )

    async def _process_text(
        self,
        bus: EventBus,
        user_text: str,
        generation: int,
        turn_id: str | None = None,
        llm_stream: AsyncIterator[str] | None = None,
    ) -> None:
        if not user_text.strip() or generation != self._generation:
            return
        messages = (
            await self._messages_for_turn(user_text)
            if llm_stream is None
            else self._messages(user_text)
        )
        self._history.append({"role": "user", "content": user_text})
        chunker = SentenceChunker(
            first_max_chars=8,
            max_chars=24,
            soft_min_chars=4,
        )
        assistant_parts: list[str] = []
        timing_state: set[str] = set()
        queue: asyncio.Queue[str | object] = asyncio.Queue(
            maxsize=_TTS_QUEUE_CAPACITY
        )

        async def consume_speech() -> None:
            open_text_stream = getattr(self._tts, "open_text_stream", None)
            if callable(open_text_stream):
                await self._speak_incremental_queue(
                    bus,
                    queue,
                    generation,
                    turn_id,
                    timing_state,
                    open_text_stream,
                )
                return
            while True:
                piece = await queue.get()
                if piece is _TTS_QUEUE_END:
                    return
                assert isinstance(piece, str)
                await self._speak_piece(
                    bus,
                    piece,
                    generation,
                    turn_id,
                    timing_state,
                )

        async def enqueue_pieces(pieces: list[str]) -> None:
            if pieces and "first_speakable_text" not in timing_state:
                timing_state.add("first_speakable_text")
                await self._publish_milestone(
                    bus,
                    turn_id,
                    "first_speakable_text",
                )
            for piece in pieces:
                await queue.put(piece)

        async with asyncio.TaskGroup() as group:
            group.create_task(
                consume_speech(),
                name="CustomAPIConversationEngine.tts_consumer",
            )
            deltas = llm_stream if llm_stream is not None else self._llm.stream_text(messages)
            iterator = deltas.__aiter__()
            pending_delta: asyncio.Future[str] | None = None
            try:
                while True:
                    pending_delta = asyncio.ensure_future(anext(iterator))
                    timeout = (
                        self._tts_idle_flush_seconds
                        if chunker.has_pending
                        else None
                    )
                    done, _ = await asyncio.wait(
                        (pending_delta,),
                        timeout=timeout,
                    )
                    if not done:
                        await enqueue_pieces(chunker.flush_pending())
                        await asyncio.wait((pending_delta,))
                    try:
                        delta = pending_delta.result()
                    except StopAsyncIteration:
                        break
                    pending_delta = None
                    if generation != self._generation:
                        await queue.put(_TTS_QUEUE_END)
                        return
                    assistant_parts.append(delta)
                    if delta and "llm_first_token" not in timing_state:
                        timing_state.add("llm_first_token")
                        await self._publish_milestone(
                            bus,
                            turn_id,
                            "llm_first_token",
                            thinking_mode=self._thinking_mode,
                        )
                    await enqueue_pieces(chunker.feed(delta))
            finally:
                if pending_delta is not None:
                    if not pending_delta.done():
                        pending_delta.cancel()
                    await asyncio.gather(pending_delta, return_exceptions=True)
            await enqueue_pieces(chunker.finish())
            await queue.put(_TTS_QUEUE_END)
        assistant_text = "".join(assistant_parts).strip()
        await self._finish_segment(bus)
        if assistant_text:
            self._history.append({"role": "assistant", "content": assistant_text})
            await bus.publish(
                ResponseTranscript(
                    transcript=assistant_text,
                    timestamp=tzutcnow().timestamp(),
                )
            )

    async def _process_audio(
        self,
        bus: EventBus,
        utterance: SpeechBuffer,
        generation: int,
        turn_id: str,
        asr: ASRBackend | None = None,
    ) -> None:
        text = await (asr or self._asr).transcribe(utterance)
        await self._publish_milestone(bus, turn_id, "asr_complete")
        if generation != self._generation or not text:
            return
        await bus.publish(InputTranscript(transcript=text, timestamp=tzutcnow().timestamp()))
        await self._process_text(bus, text, generation, turn_id)

    async def _process_realtime_audio(
        self,
        bus: EventBus,
        utterance: SpeechBuffer,
        generation: int,
        turn_id: str,
    ) -> None:
        asr = self._realtime_asr
        assert asr is not None
        try:
            text = await asr.finish()
        except asyncio.CancelledError:
            await self._cancel_speculation()
            raise
        except Exception as exc:
            fallback = self._asr_fallback
            if fallback is None:
                await self._cancel_speculation()
                raise
            _LOGGER.exception(
                "custom API realtime ASR finalization failed; using HTTP fallback",
                error=_sanitise_error(exc, self.config),
            )
            with suppress(Exception):
                await asr.cancel()
            try:
                text = await fallback.transcribe(utterance)
            except BaseException:
                await self._cancel_speculation()
                raise
        await self._publish_milestone(bus, turn_id, "asr_complete")
        if generation != self._generation or not text:
            await self._cancel_speculation()
            return
        speculative_stream = await self._take_validated_speculation(
            text,
            turn_id=turn_id,
            generation=generation,
        )
        await bus.publish(InputTranscript(transcript=text, timestamp=tzutcnow().timestamp()))
        await self._process_text(
            bus,
            text,
            generation,
            turn_id,
            llm_stream=speculative_stream,
        )

    async def _run_turn(self, bus: EventBus, coroutine, turn_id: str | None) -> None:
        current = asyncio.current_task()
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOGGER.exception(
                "custom API turn failed",
                error=_sanitise_error(exc, self.config),
            )
            await self._finish_segment(bus)
            await bus.publish(DiscardAvatarSpeechBuffer())
            await self._publish_milestone(bus, turn_id, "failed")
        finally:
            if self._turn_task is current:
                self._turn_task = None
                self._active_turn_id = None

    def _start_turn(self, bus: EventBus, coroutine, turn_id: str | None = None) -> None:
        self._active_turn_id = turn_id
        self._turn_task = asyncio.create_task(self._run_turn(bus, coroutine, turn_id))

    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        _ = clocks
        async with bus.subscribe(
            UserSpeechReceived,
            TextEchoEnqueueText,
            SegmentPlaybackStarted,
            SegmentPlaybackCompleted,
            SegmentPlaybackInterrupted,
            SegmentPlaybackCancelled,
            Shutdown,
        ) as sub:
            bus.ready()
            async for event in sub:
                match event:
                    case UserSpeechReceived(buffer=buffer):
                        was_confirmed = self._vad.is_speech_confirmed
                        utterance = self._vad.feed(buffer)
                        if not was_confirmed and self._vad.is_speech_confirmed:
                            await self._interrupt(bus)
                            if self._realtime_asr is not None:
                                await self._begin_realtime_capture(
                                    bus,
                                    self._vad.take_unstreamed_chunks()
                                )
                        elif was_confirmed and self._realtime_asr is not None:
                            chunks = (
                                [buffer]
                                if utterance is not None
                                else self._vad.take_unstreamed_chunks()
                            )
                            await self._feed_realtime_capture(chunks)
                        if utterance is not None:
                            generation = (
                                self._capture_generation
                                if self._capture_generation is not None
                                else self._generation
                            )
                            turn_id = self._capture_turn_id or uuid4().hex
                            realtime_started = self._capture_started
                            stream_failed = self._capture_stream_failed
                            self._set_partial_transcript_callback(None)
                            self._capture_turn_id = None
                            self._capture_generation = None
                            self._capture_started = False
                            self._capture_stream_failed = False
                            await self._publish_milestone(
                                bus,
                                turn_id,
                                "vad_complete",
                                vad_wait_ms=self._vad.last_vad_wait_ms,
                            )
                            if realtime_started:
                                self._start_turn(
                                    bus,
                                    self._process_realtime_audio(
                                        bus,
                                        utterance,
                                        generation,
                                        turn_id,
                                    ),
                                    turn_id,
                                )
                            else:
                                self._start_turn(
                                    bus,
                                    self._process_audio(
                                        bus,
                                        utterance,
                                        generation,
                                        turn_id,
                                        (
                                            self._asr_fallback
                                            if stream_failed
                                            else None
                                        ),
                                    ),
                                    turn_id,
                                )
                    case TextEchoEnqueueText(text=text):
                        await self._interrupt(bus)
                        generation = self._generation
                        turn_id = uuid4().hex
                        self._start_turn(
                            bus,
                            self._process_text(bus, text, generation, turn_id),
                            turn_id,
                        )
                    case SegmentPlaybackStarted(segment_id=segment_id):
                        self._playback_segments.add(segment_id)
                        self._vad.set_assistant_playback(True)
                    case (
                        SegmentPlaybackCompleted(segment_id=segment_id)
                        | SegmentPlaybackInterrupted(segment_id=segment_id)
                        | SegmentPlaybackCancelled(segment_id=segment_id)
                    ):
                        self._playback_segments.discard(segment_id)
                        self._vad.set_assistant_playback(bool(self._playback_segments))
                    case Shutdown():
                        await self._interrupt(bus)
                        return

    async def close(self) -> None:
        task = self._turn_task
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._cancel_realtime_capture()
        await self._components.close()

    async def __call__(self, bus: EventBus, clocks: StreamClocks) -> None:
        try:
            await self.run(bus, clocks)
        finally:
            await self.close()


@dataclass(frozen=True, slots=True)
class PreparedCustomAPIConnection:
    report: CustomAPIPreflightReport
    engine: Any | None


async def _prepare_minimax_realtime(
    config: CustomAPIConnectionConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    memory_prompt: str = "",
) -> PreparedCustomAPIConnection:
    from avaturn_live_streamer.conversation_engines.minimax_realtime_client import (
        MiniMaxRealtimeClient,
        MiniMaxRealtimeConversationEngine,
    )

    config = config.model_copy(
        update={"prompt": append_memory_prompt(config.prompt, memory_prompt)}
    )
    provider = config.provider
    if not isinstance(provider, MiniMaxProvider):
        raise TypeError("MiniMax preparation requires a minimax provider")

    tts = (
        OpenAICompatibleTTS(config.tts_override, transport=transport)
        if config.tts_override is not None
        else None
    )
    client = MiniMaxRealtimeClient(
        provider,
        prompt=config.prompt,
        text_only=tts is not None,
    )
    probes = [_time_probe("realtime", client.start)]
    if tts is not None:
        probes.append(_time_probe("tts", lambda: _probe_tts(tts)))
    try:
        raw = await asyncio.gather(*probes)
    except BaseException:
        await client.close()
        if tts is not None:
            await tts.close()
        raise

    components = {
        name: ComponentPreflight(
            status=item.status,
            latency_ms=item.latency_ms,
            error=(
                _sanitise_error(RuntimeError(item.error), config)
                if item.error is not None
                else None
            ),
        )
        for name, item in raw
    }
    status = (
        "ready"
        if all(component.status == "ready" for component in components.values())
        else "failed"
    )
    report = CustomAPIPreflightReport(status=status, components=components)
    if status != "ready":
        await client.close()
        if tts is not None:
            await tts.close()
        return PreparedCustomAPIConnection(
            report=report,
            engine=None,
        )
    return PreparedCustomAPIConnection(
        report=report,
        engine=MiniMaxRealtimeConversationEngine(
            client=client,
            config=config,
            tts=tts,
        ),
    )


async def prepare_custom_api(
    config: CustomAPIConnectionConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    memory_prompt: str = "",
    memory_recall: Callable[[str], Awaitable[str]] | None = None,
) -> PreparedCustomAPIConnection:
    """Probe one component set and retain that same set for the session."""

    if isinstance(config.provider, MiniMaxProvider):
        return await _prepare_minimax_realtime(
            config,
            transport=transport,
            memory_prompt=memory_prompt,
        )

    config = config.model_copy(
        update={"prompt": append_memory_prompt(config.prompt, memory_prompt)}
    )
    components = build_custom_api_components(config, transport=transport)
    try:
        report = await preflight_custom_api(
            config,
            components=components,
        )
    except BaseException:
        await components.close()
        raise
    if report.status != "ready":
        await components.close()
        return PreparedCustomAPIConnection(report=report, engine=None)
    return PreparedCustomAPIConnection(
        report=report,
        engine=CustomAPIConversationEngine(
            config,
            components=components,
            memory_recall=memory_recall,
        ),
    )
