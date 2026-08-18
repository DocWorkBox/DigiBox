# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""DashScope/Bailian speech clients used by the custom API engine."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from websockets.asyncio.client import connect as websockets_connect

from avaturn_live_streamer.conversation_engines.builders import (
    AliyunBailianProvider,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer

type BailianProvider = AliyunBailianProvider

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com"
_CHAT_COMPLETIONS_PATH = "/compatible-mode/v1/chat/completions"
_COSYVOICE_TTS_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
_COSYVOICE_REALTIME_PATH = "/api-ws/v1/inference"
_VOICE_CUSTOMIZATION_PATH = "/api/v1/services/audio/tts/customization"
_UPLOAD_POLICY_PATH = "/api/v1/uploads"
_QWEN_REALTIME_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
_MIN_REALTIME_RESAMPLE_MS = 20
_ASR_SESSION_CLEANUP_TIMEOUT_SECONDS = 2.0
_COSYVOICE_AUDIO_QUEUE_SIZE = 32
_COSYVOICE_MAX_TEXT_CHARS = 20_000
_COSYVOICE_MAX_TURN_CHARS = 200_000
_MAX_COSYVOICE_INVENTORY = 1_000
_COSYVOICE_VOICE_STATES = frozenset({"DEPLOYING", "OK", "UNDEPLOYED"})


@dataclass(frozen=True, slots=True)
class VoiceCloneResult:
    voice_id: str
    preview_url: str | None = None


@dataclass(frozen=True, slots=True)
class CosyVoiceInventoryVoice:
    """Stable, UI-safe view of one provider-managed custom voice."""

    id: str
    status: str
    compatible: bool
    created_at: str | None = None
    modified_at: str | None = None


def bailian_base_url(provider: BailianProvider) -> str:
    workspace_id = (provider.workspace_id or "").strip()
    if not workspace_id:
        return _DASHSCOPE_BASE_URL
    return f"https://{workspace_id}.{provider.region}.maas.aliyuncs.com"


def _bearer_headers(provider: BailianProvider, *, sse: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {provider.api_key.get_secret_value()}",
    }
    if sse:
        headers["X-DashScope-SSE"] = "enable"
    return headers


def _wav_bytes(audio: SpeechBuffer, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.resample(sample_rate).to_bytes())
    return output.getvalue()


async def _iter_sse_json(response: httpx.Response) -> AsyncIterator[dict[str, object]]:
    async for line in response.aiter_lines():
        value = line.strip()
        if not value or value.startswith(":"):
            continue
        if value.startswith("data:"):
            value = value[5:].strip()
        if value == "[DONE]":
            return
        try:
            event = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


async def _iter_pcm_s16le(
    chunks: AsyncIterable[bytes],
    *,
    sample_rate: int,
) -> AsyncIterator[SpeechBuffer]:
    carry = b""
    async for chunk in chunks:
        data = carry + bytes(chunk)
        even_length = len(data) & ~1
        if even_length:
            yield SpeechBuffer.from_bytes(data[:even_length], sample_rate)
        carry = data[even_length:]
    if carry:
        raise ValueError("PCM16 stream ended with half of a sample")


class BailianRealtimeASRError(RuntimeError):
    """A safe, credential-redacted Qwen Realtime ASR failure."""


class BailianCosyVoiceRealtimeTTSError(RuntimeError):
    """A safe, credential-redacted CosyVoice WebSocket failure."""


class _RealtimeWebSocket(Protocol):
    async def recv(self) -> str | bytes: ...

    async def send(self, message: str) -> None: ...

    async def close(self) -> None: ...


type _RealtimeConnector = Callable[..., Awaitable[_RealtimeWebSocket]]
type _PartialTranscriptCallback = Callable[[str], Awaitable[None] | None]


async def _realtime_websocket_connect(
    uri: str,
    **kwargs: object,
) -> _RealtimeWebSocket:
    return await websockets_connect(  # type: ignore[arg-type, return-value]
        uri,
        **kwargs,
    )


def _parse_realtime_event(raw: str | bytes) -> dict[str, object]:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BailianRealtimeASRError("Qwen Realtime ASR returned invalid JSON") from exc
    if not isinstance(event, dict):
        raise BailianRealtimeASRError("Qwen Realtime ASR event must be an object")
    return event


def _parse_cosyvoice_realtime_event(raw: str | bytes) -> dict[str, object]:
    if isinstance(raw, bytes):
        raise BailianCosyVoiceRealtimeTTSError("CosyVoice returned audio before task-started")
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BailianCosyVoiceRealtimeTTSError("CosyVoice returned invalid JSON") from exc
    if not isinstance(event, dict):
        raise BailianCosyVoiceRealtimeTTSError("CosyVoice event must be an object")
    return event


def _realtime_pcm16(audio: SpeechBuffer, sample_rate: int) -> bytes:
    """Resample one realtime frame without exposing tiny-frame soxr startup noise."""

    raw = audio.to_bytes()
    if not raw or audio.sample_rate == sample_rate:
        return raw
    source_samples = len(raw) // 2
    minimum_samples = max(
        1,
        audio.sample_rate * _MIN_REALTIME_RESAMPLE_MS // 1000,
    )
    if source_samples >= minimum_samples:
        return audio.resample(sample_rate).to_bytes()

    # The Windows soxr QQ backend can return an uninitialised first sample when
    # its first call follows Torch kernels and the input is only a few samples.
    # A normal capture frame is 20 ms; zero-pad shorter fragments to that size,
    # then trim back to the fragment's exact destination duration.
    padded = SpeechBuffer.from_bytes(
        raw + b"\x00\x00" * (minimum_samples - source_samples),
        audio.sample_rate,
    )
    target_samples = (source_samples * sample_rate + audio.sample_rate // 2) // audio.sample_rate
    return padded.resample(sample_rate).to_bytes()[: target_samples * 2]


class BailianQwenRealtimeASR:
    """One manual Qwen Realtime ASR WebSocket session per utterance."""

    sample_rate = 16_000

    def __init__(
        self,
        provider: BailianProvider,
        *,
        connector: _RealtimeConnector | None = None,
        on_partial_transcript: _PartialTranscriptCallback | None = None,
    ) -> None:
        self.provider = provider
        self._connector = connector
        self._partial_transcript_callback = on_partial_transcript
        self._turn_partial_transcript_callback: _PartialTranscriptCallback | None = None
        self._websocket: _RealtimeWebSocket | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._result: asyncio.Future[str] | None = None
        self._session_finished: asyncio.Event | None = None
        self._stable_partial_transcript = ""
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._send_lock = asyncio.Lock()

    @property
    def stable_partial_transcript(self) -> str:
        """Latest provider-confirmed prefix, excluding the revisable ``stash``."""

        return self._stable_partial_transcript

    def set_partial_transcript_callback(
        self,
        callback: _PartialTranscriptCallback | None,
    ) -> None:
        """Bind the observer captured by the next ASR turn.

        Capturing at ``start`` prevents a late engine rebind from redirecting
        partials that belong to an already active utterance.
        """

        self._partial_transcript_callback = callback

    def _sanitise(self, value: object) -> str:
        message = str(value)
        secret = self.provider.api_key.get_secret_value()
        if secret:
            message = message.replace(secret, "***")
        return message[:1000]

    def _error_from_event(self, event: dict[str, object]) -> BailianRealtimeASRError:
        error = event.get("error")
        detail = error if isinstance(error, dict) else event
        code = detail.get("code") or detail.get("type") or "unknown-error"
        message = detail.get("message") or "no error message"
        safe = self._sanitise(f"{code}: {message}")
        return BailianRealtimeASRError(f"Qwen Realtime ASR failed: {safe}")

    @staticmethod
    def _event(event_type: str, **payload: object) -> dict[str, object]:
        return {
            "event_id": f"event_{uuid4().hex}",
            "type": event_type,
            **payload,
        }

    def _require_websocket(self) -> _RealtimeWebSocket:
        if self._websocket is None:
            raise BailianRealtimeASRError("Qwen Realtime ASR turn is not active")
        return self._websocket

    async def _send(self, event: dict[str, object]) -> None:
        websocket = self._require_websocket()
        async with self._send_lock:
            await websocket.send(json.dumps(event, ensure_ascii=False))

    async def _wait_for_setup_event(self, expected_type: str) -> None:
        websocket = self._require_websocket()
        async with asyncio.timeout(self.provider.timeout_seconds):
            while True:
                event = _parse_realtime_event(await websocket.recv())
                if event.get("type") == "error":
                    raise self._error_from_event(event)
                if event.get("type") == expected_type:
                    return

    async def _listen(self) -> None:
        websocket = self._require_websocket()
        result = self._result
        session_finished = self._session_finished
        assert result is not None
        assert session_finished is not None
        stable_partial = ""
        try:
            while True:
                event = _parse_realtime_event(await websocket.recv())
                event_type = event.get("type")
                if event_type == "error":
                    raise self._error_from_event(event)
                if event_type == "conversation.item.input_audio_transcription.text":
                    value = event.get("text")
                    candidate = value.strip() if isinstance(value, str) else ""
                    if (
                        candidate
                        and candidate != stable_partial
                        and candidate.startswith(stable_partial)
                    ):
                        stable_partial = candidate
                        if self._result is result:
                            self._stable_partial_transcript = candidate
                        callback = self._turn_partial_transcript_callback
                        if callback is not None:
                            try:
                                callback_result = callback(candidate)
                                if callback_result is not None:
                                    await callback_result
                            except Exception:
                                # Observers must never delay or fail final recognition.
                                pass
                elif event_type == ("conversation.item.input_audio_transcription.completed"):
                    value = event.get("transcript")
                    transcript = value.strip() if isinstance(value, str) else ""
                    if not result.done():
                        result.set_result(transcript)
                elif event_type == ("conversation.item.input_audio_transcription.failed"):
                    raise self._error_from_event(event)
                elif event_type == "session.finished":
                    session_finished.set()
                    if not result.done():
                        result.set_result("")
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            session_finished.set()
            if not result.done():
                if isinstance(exc, BailianRealtimeASRError):
                    result.set_exception(exc)
                else:
                    result.set_exception(
                        BailianRealtimeASRError(
                            f"Qwen Realtime ASR receive failed: {self._sanitise(exc)}"
                        )
                    )

    def _detach_turn(
        self,
    ) -> tuple[
        _RealtimeWebSocket,
        asyncio.Task[None],
        asyncio.Future[str],
        asyncio.Event,
    ]:
        websocket = self._websocket
        listener = self._listener_task
        result = self._result
        session_finished = self._session_finished
        assert websocket is not None
        assert listener is not None
        assert result is not None
        assert session_finished is not None
        self._websocket = None
        self._listener_task = None
        self._result = None
        self._session_finished = None
        self._turn_partial_transcript_callback = None
        return websocket, listener, result, session_finished

    async def _cleanup_finished_turn(
        self,
        websocket: _RealtimeWebSocket,
        listener: asyncio.Task[None],
        result: asyncio.Future[str],
        session_finished: asyncio.Event,
    ) -> None:
        cleanup_timeout = min(
            self.provider.timeout_seconds,
            _ASR_SESSION_CLEANUP_TIMEOUT_SECONDS,
        )
        try:
            async with asyncio.timeout(cleanup_timeout):
                await session_finished.wait()
                await listener
        except (TimeoutError, asyncio.CancelledError):
            if not listener.done():
                listener.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await listener
        except Exception:
            # The final transcript is already resolved; a late teardown error
            # must not surface as an unhandled background-task exception.
            pass
        finally:
            if result.done():
                with suppress(asyncio.CancelledError, Exception):
                    result.exception()
            with suppress(Exception):
                await websocket.close()

    def _schedule_finished_turn_cleanup(
        self,
        turn: tuple[
            _RealtimeWebSocket,
            asyncio.Task[None],
            asyncio.Future[str],
            asyncio.Event,
        ],
    ) -> None:
        task = asyncio.create_task(
            self._cleanup_finished_turn(*turn),
            name="BailianQwenRealtimeASR.cleanup",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _dispose_turn(self) -> None:
        websocket = self._websocket
        listener = self._listener_task
        result = self._result
        session_finished = self._session_finished
        self._websocket = None
        self._listener_task = None
        self._result = None
        self._session_finished = None
        self._turn_partial_transcript_callback = None
        if listener is not None and listener is not asyncio.current_task():
            if not listener.done():
                listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener
        if result is not None:
            if not result.done():
                result.cancel()
            else:
                with suppress(asyncio.CancelledError, Exception):
                    result.exception()
        if session_finished is not None:
            session_finished.set()
        if websocket is not None:
            with suppress(Exception):
                await websocket.close()

    async def start(self) -> None:
        if self._websocket is not None:
            raise BailianRealtimeASRError("Qwen Realtime ASR turn is already active")
        self._turn_partial_transcript_callback = self._partial_transcript_callback
        connector = self._connector or _realtime_websocket_connect
        url = f"{_QWEN_REALTIME_URL}?{urlencode({'model': self.provider.asr_model})}"
        headers = {
            "Authorization": f"Bearer {self.provider.api_key.get_secret_value()}",
            "OpenAI-Beta": "realtime=v1",
        }
        workspace_id = (self.provider.workspace_id or "").strip()
        if workspace_id:
            headers["X-DashScope-WorkSpace"] = workspace_id
        try:
            self._websocket = await connector(
                url,
                additional_headers=headers,
                open_timeout=self.provider.timeout_seconds,
            )
            await self._wait_for_setup_event("session.created")
            transcription: dict[str, object] = {}
            if self.provider.asr_language:
                transcription["language"] = self.provider.asr_language
            await self._send(
                self._event(
                    "session.update",
                    session={
                        "input_audio_format": "pcm",
                        "sample_rate": self.sample_rate,
                        "input_audio_transcription": transcription,
                        "turn_detection": None,
                    },
                )
            )
            await self._wait_for_setup_event("session.updated")
            self._result = asyncio.get_running_loop().create_future()
            self._session_finished = asyncio.Event()
            self._stable_partial_transcript = ""
            self._listener_task = asyncio.create_task(
                self._listen(),
                name="BailianQwenRealtimeASR.listener",
            )
        except asyncio.CancelledError:
            await self._dispose_turn()
            raise
        except BaseException as exc:
            await self._dispose_turn()
            if isinstance(exc, BailianRealtimeASRError):
                raise
            raise BailianRealtimeASRError(
                f"Qwen Realtime ASR connection failed: {self._sanitise(exc)}"
            ) from exc

    async def feed(self, audio: SpeechBuffer) -> None:
        pcm = _realtime_pcm16(audio, self.sample_rate)
        if not pcm:
            return
        try:
            await self._send(
                self._event(
                    "input_audio_buffer.append",
                    audio=base64.b64encode(pcm).decode("ascii"),
                )
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._dispose_turn()
            if isinstance(exc, BailianRealtimeASRError):
                raise
            raise BailianRealtimeASRError(
                f"Qwen Realtime ASR send failed: {self._sanitise(exc)}"
            ) from exc

    async def finish(self) -> str:
        result = self._result
        if result is None:
            raise BailianRealtimeASRError("Qwen Realtime ASR turn is not active")
        try:
            await self._send(self._event("input_audio_buffer.commit"))
            await self._send(self._event("session.finish"))
            async with asyncio.timeout(self.provider.timeout_seconds):
                transcript = (await asyncio.shield(result)).strip()
            turn = self._detach_turn()
            self._schedule_finished_turn_cleanup(turn)
            # Give an already queued session.finished event one scheduling turn
            # without putting it back on the transcript critical path.
            await asyncio.sleep(0)
            return transcript
        except asyncio.CancelledError:
            await self._dispose_turn()
            raise
        except BaseException as exc:
            await self._dispose_turn()
            if isinstance(exc, BailianRealtimeASRError):
                raise
            raise BailianRealtimeASRError(
                f"Qwen Realtime ASR finalization failed: {self._sanitise(exc)}"
            ) from exc

    async def cancel(self) -> None:
        await self._dispose_turn()

    async def transcribe(self, audio: SpeechBuffer) -> str:
        await self.start()
        try:
            await self.feed(audio)
            return await self.finish()
        finally:
            if self._websocket is not None:
                await self.cancel()

    async def close(self) -> None:
        await self.cancel()
        cleanup_tasks = tuple(self._cleanup_tasks)
        for task in cleanup_tasks:
            task.cancel()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._partial_transcript_callback = None


class BailianQwenASR:
    def __init__(
        self,
        provider: BailianProvider,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=provider.timeout_seconds,
        )

    async def transcribe(self, audio: SpeechBuffer) -> str:
        encoded = base64.b64encode(_wav_bytes(audio, 16_000)).decode("ascii")
        payload: dict[str, object] = {
            "model": self.provider.asr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/wav;base64,{encoded}",
                            },
                        }
                    ],
                }
            ],
            "stream": False,
        }
        if self.provider.asr_language:
            payload["asr_options"] = {
                "language": self.provider.asr_language,
                "enable_itn": False,
            }
        response = await self._client.post(
            f"{bailian_base_url(self.provider)}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(self.provider),
            json=payload,
        )
        response.raise_for_status()
        choices = response.json().get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Qwen ASR response is missing choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("Qwen ASR response is missing message content")
        return content.strip()

    async def close(self) -> None:
        await self._client.aclose()


def _cosyvoice_realtime_url(provider: BailianProvider) -> str:
    base_url = bailian_base_url(provider)
    if base_url.startswith("https://"):
        base_url = f"wss://{base_url.removeprefix('https://')}"
    return f"{base_url}{_COSYVOICE_REALTIME_PATH}"


class BailianCosyVoiceRealtimeTTSTurn:
    """One duplex CosyVoice task accepting incremental text and yielding PCM."""

    sample_rate = 24_000

    def __init__(
        self,
        provider: BailianProvider,
        websocket: _RealtimeWebSocket,
        task_id: str,
        *,
        audio_queue_size: int,
        on_release: Callable[
            [BailianCosyVoiceRealtimeTTSTurn, _RealtimeWebSocket, bool],
            Awaitable[None],
        ],
    ) -> None:
        self.provider = provider
        self.task_id = task_id
        self._websocket = websocket
        # Keep one queue slot reserved for the terminal marker. Without it a
        # healthy task-finished socket can return to the pool while its old
        # listener remains blocked forever trying to append ``None`` behind a
        # full set of audio frames that an abandoned consumer will never drain.
        self._audio_queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue(
            maxsize=audio_queue_size + 1
        )
        self._audio_slots = asyncio.Semaphore(audio_queue_size)
        self._on_release = on_release
        self._send_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._listener_task = asyncio.create_task(
            self._listen(),
            name=f"BailianCosyVoiceRealtimeTTS.listener.{task_id}",
        )
        self._input_finished = False
        self._transport_closed = False
        self._audio_consumer_started = False
        self._sent_characters = 0

    def _sanitise(self, value: object) -> str:
        message = str(value)
        secret = self.provider.api_key.get_secret_value()
        if secret:
            message = message.replace(secret, "***")
        return message[:1000]

    def _error_from_event(
        self,
        event: dict[str, object],
    ) -> BailianCosyVoiceRealtimeTTSError:
        header = event.get("header")
        payload = event.get("payload")
        header = header if isinstance(header, dict) else {}
        payload = payload if isinstance(payload, dict) else {}
        code = header.get("error_code") or payload.get("code") or "unknown-error"
        message = header.get("error_message") or payload.get("message") or "no error message"
        return BailianCosyVoiceRealtimeTTSError(
            f"CosyVoice realtime task failed: {self._sanitise(f'{code}: {message}')}"
        )

    def _command(self, action: str, input_payload: dict[str, object]) -> str:
        return json.dumps(
            {
                "header": {
                    "action": action,
                    "task_id": self.task_id,
                    "streaming": "duplex",
                },
                "payload": {"input": input_payload},
            },
            ensure_ascii=False,
        )

    async def _send_command(
        self,
        action: str,
        input_payload: dict[str, object],
    ) -> None:
        if self._transport_closed:
            raise BailianCosyVoiceRealtimeTTSError("CosyVoice realtime turn is closed")
        try:
            async with self._send_lock:
                await self._websocket.send(self._command(action, input_payload))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            error = BailianCosyVoiceRealtimeTTSError(
                f"CosyVoice realtime send failed: {self._sanitise(exc)}"
            )
            await self._abort(error)
            raise error from exc

    def _validate_event_header(self, event: dict[str, object]) -> str:
        header = event.get("header")
        if not isinstance(header, dict):
            raise BailianCosyVoiceRealtimeTTSError("CosyVoice event is missing header")
        event_task_id = header.get("task_id")
        if event_task_id != self.task_id:
            raise BailianCosyVoiceRealtimeTTSError(
                "CosyVoice event task_id does not match the active task"
            )
        event_type = header.get("event")
        if not isinstance(event_type, str):
            raise BailianCosyVoiceRealtimeTTSError("CosyVoice event is missing header.event")
        return event_type

    def _force_terminal(self, item: BaseException | None) -> None:
        while True:
            try:
                self._audio_queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    discarded = self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                if isinstance(discarded, bytes):
                    self._audio_slots.release()

    async def _release_transport(self, *, reusable: bool) -> None:
        async with self._close_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
            await self._on_release(self, self._websocket, reusable)

    async def _abort(self, error: BaseException) -> None:
        listener = self._listener_task
        if listener is not asyncio.current_task() and not listener.done():
            listener.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await listener
        self._force_terminal(error)
        await self._release_transport(reusable=False)

    async def _listen(self) -> None:
        terminal_queued = False
        try:
            while True:
                raw = await self._websocket.recv()
                if isinstance(raw, bytes):
                    if raw:
                        await self._audio_slots.acquire()
                        self._audio_queue.put_nowait(raw)
                    continue
                event = _parse_cosyvoice_realtime_event(raw)
                event_type = self._validate_event_header(event)
                if event_type == "task-finished":
                    await self._release_transport(reusable=True)
                    self._audio_queue.put_nowait(None)
                    terminal_queued = True
                    return
                if event_type in {"task-failed", "error"}:
                    raise self._error_from_event(event)
                if event_type != "result-generated":
                    raise BailianCosyVoiceRealtimeTTSError(
                        f"Unexpected CosyVoice event: {self._sanitise(event_type)}"
                    )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            error = (
                exc
                if isinstance(exc, BailianCosyVoiceRealtimeTTSError)
                else BailianCosyVoiceRealtimeTTSError(
                    f"CosyVoice realtime receive failed: {self._sanitise(exc)}"
                )
            )
            await self._release_transport(reusable=False)
            self._force_terminal(error)
            terminal_queued = True
        finally:
            if not terminal_queued and not self._transport_closed:
                self._force_terminal(
                    BailianCosyVoiceRealtimeTTSError(
                        "CosyVoice realtime stream closed before task-finished"
                    )
                )
            await self._release_transport(reusable=False)

    async def send_text(self, text: str) -> None:
        if self._input_finished:
            raise BailianCosyVoiceRealtimeTTSError("Cannot send text after finish_text")
        if not text:
            return
        if len(text) > _COSYVOICE_MAX_TEXT_CHARS:
            raise ValueError(
                f"CosyVoice text increment exceeds {_COSYVOICE_MAX_TEXT_CHARS} characters"
            )
        if self._sent_characters + len(text) > _COSYVOICE_MAX_TURN_CHARS:
            raise ValueError(f"CosyVoice turn exceeds {_COSYVOICE_MAX_TURN_CHARS} characters")
        await self._send_command("continue-task", {"text": text})
        self._sent_characters += len(text)

    async def finish_text(self) -> None:
        if self._input_finished:
            return
        self._input_finished = True
        await self._send_command("finish-task", {})

    async def stream_audio(self) -> AsyncIterator[SpeechBuffer]:
        if self._audio_consumer_started:
            raise BailianCosyVoiceRealtimeTTSError(
                "CosyVoice realtime audio can only be consumed once"
            )
        self._audio_consumer_started = True
        carry = b""
        try:
            while True:
                item = await self._audio_queue.get()
                if item is None:
                    if carry:
                        raise BailianCosyVoiceRealtimeTTSError(
                            "CosyVoice PCM16 stream ended with half of a sample"
                        )
                    return
                if isinstance(item, BaseException):
                    raise item
                self._audio_slots.release()
                data = carry + item
                even_length = len(data) & ~1
                if even_length:
                    yield SpeechBuffer.from_bytes(
                        data[:even_length],
                        self.sample_rate,
                    )
                carry = data[even_length:]
        finally:
            if not self._transport_closed:
                await self.cancel()

    async def cancel(self) -> None:
        if self._transport_closed:
            return
        self._input_finished = True
        with suppress(Exception):
            await self._send_command("finish-task", {"directive": "cancel"})
        listener = self._listener_task
        if listener is not asyncio.current_task() and not listener.done():
            listener.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await listener
        while True:
            try:
                discarded = self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(discarded, bytes):
                self._audio_slots.release()
        self._force_terminal(None)
        await self._release_transport(reusable=False)


class BailianCosyVoiceRealtimeTTS:
    """Factory for turn-scoped official CosyVoice duplex WebSocket tasks."""

    sample_rate = 24_000

    def __init__(
        self,
        provider: BailianProvider,
        *,
        connector: _RealtimeConnector | None = None,
        audio_queue_size: int = _COSYVOICE_AUDIO_QUEUE_SIZE,
    ) -> None:
        if audio_queue_size < 1:
            raise ValueError("audio_queue_size must be at least 1")
        self.provider = provider
        self._connector = connector
        self._audio_queue_size = audio_queue_size
        self._turns: set[BailianCosyVoiceRealtimeTTSTurn] = set()
        self._idle_websocket: _RealtimeWebSocket | None = None
        self._open_lock = asyncio.Lock()
        self._pool_lock = asyncio.Lock()
        self._closed = False

    def _sanitise(self, value: object) -> str:
        message = str(value)
        secret = self.provider.api_key.get_secret_value()
        if secret:
            message = message.replace(secret, "***")
        return message[:1000]

    async def _release_turn(
        self,
        turn: BailianCosyVoiceRealtimeTTSTurn,
        websocket: _RealtimeWebSocket,
        reusable: bool,
    ) -> None:
        close_websocket = not reusable
        async with self._pool_lock:
            self._turns.discard(turn)
            if reusable and not self._closed and self._idle_websocket is None:
                self._idle_websocket = websocket
                close_websocket = False
            elif reusable:
                close_websocket = True
        if close_websocket:
            with suppress(Exception):
                await websocket.close()

    async def _take_idle_websocket(self) -> _RealtimeWebSocket | None:
        async with self._pool_lock:
            websocket = self._idle_websocket
            self._idle_websocket = None
            return websocket

    def _setup_error(
        self,
        event: dict[str, object],
    ) -> BailianCosyVoiceRealtimeTTSError:
        header = event.get("header")
        header = header if isinstance(header, dict) else {}
        code = header.get("error_code") or "unknown-error"
        message = header.get("error_message") or "no error message"
        return BailianCosyVoiceRealtimeTTSError(
            f"CosyVoice realtime task setup failed: {self._sanitise(f'{code}: {message}')}"
        )

    async def _wait_for_task_started(
        self,
        websocket: _RealtimeWebSocket,
        task_id: str,
    ) -> None:
        async with asyncio.timeout(self.provider.timeout_seconds):
            while True:
                event = _parse_cosyvoice_realtime_event(await websocket.recv())
                header = event.get("header")
                if not isinstance(header, dict):
                    raise BailianCosyVoiceRealtimeTTSError(
                        "CosyVoice setup event is missing header"
                    )
                if header.get("task_id") != task_id:
                    raise BailianCosyVoiceRealtimeTTSError(
                        "CosyVoice setup task_id does not match the active task"
                    )
                event_type = header.get("event")
                if event_type == "task-started":
                    return
                if event_type in {"task-failed", "error"}:
                    raise self._setup_error(event)
                raise BailianCosyVoiceRealtimeTTSError(
                    f"Unexpected CosyVoice setup event: {self._sanitise(event_type)}"
                )

    async def open_text_stream(self) -> BailianCosyVoiceRealtimeTTSTurn:
        async with self._open_lock:
            if self._closed:
                raise BailianCosyVoiceRealtimeTTSError("CosyVoice realtime client is closed")
            connector = self._connector or _realtime_websocket_connect
            workspace_id = (self.provider.workspace_id or "").strip()
            headers = {
                "Authorization": f"Bearer {self.provider.api_key.get_secret_value()}",
            }
            if workspace_id:
                headers["X-DashScope-WorkSpace"] = workspace_id

            websocket = await self._take_idle_websocket()
            retry_fresh_connection = websocket is not None
            while True:
                task_id = str(uuid4())
                run_task = {
                    "header": {
                        "action": "run-task",
                        "task_id": task_id,
                        "streaming": "duplex",
                    },
                    "payload": {
                        "task_group": "audio",
                        "task": "tts",
                        "function": "SpeechSynthesizer",
                        "model": self.provider.tts_model,
                        "parameters": {
                            "text_type": "PlainText",
                            # Clone IDs are opaque provider identifiers. Never
                            # normalise or replace a selected clone with stock.
                            "voice": self.provider.tts_voice,
                            "format": "pcm",
                            "sample_rate": self.sample_rate,
                        },
                        "input": {},
                    },
                }
                try:
                    if websocket is None:
                        websocket = await connector(
                            _cosyvoice_realtime_url(self.provider),
                            additional_headers=headers,
                            open_timeout=self.provider.timeout_seconds,
                        )
                    await websocket.send(json.dumps(run_task, ensure_ascii=False))
                    await self._wait_for_task_started(websocket, task_id)
                except asyncio.CancelledError:
                    if websocket is not None:
                        with suppress(Exception):
                            await websocket.close()
                    raise
                except BaseException as exc:
                    if websocket is not None:
                        with suppress(Exception):
                            await websocket.close()
                    websocket = None
                    if retry_fresh_connection:
                        retry_fresh_connection = False
                        continue
                    if isinstance(exc, BailianCosyVoiceRealtimeTTSError):
                        raise
                    raise BailianCosyVoiceRealtimeTTSError(
                        f"CosyVoice realtime connection failed: {self._sanitise(exc)}"
                    ) from exc

                turn = BailianCosyVoiceRealtimeTTSTurn(
                    self.provider,
                    websocket,
                    task_id,
                    audio_queue_size=self._audio_queue_size,
                    on_release=self._release_turn,
                )
                self._turns.add(turn)
                return turn

    async def close(self) -> None:
        async with self._open_lock:
            if self._closed:
                return
            self._closed = True
            async with self._pool_lock:
                idle_websocket = self._idle_websocket
                self._idle_websocket = None
                turns = tuple(self._turns)
            if idle_websocket is not None:
                with suppress(Exception):
                    await idle_websocket.close()
            if turns:
                await asyncio.gather(
                    *(turn.cancel() for turn in turns),
                    return_exceptions=True,
                )


class BailianCosyVoiceTTS:
    sample_rate = 24_000

    def __init__(
        self,
        provider: AliyunBailianProvider,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=provider.timeout_seconds,
        )

    async def stream_speech(self, text: str) -> AsyncIterator[SpeechBuffer]:
        payload = {
            "model": self.provider.tts_model,
            "input": {
                "text": text,
                "voice": self.provider.tts_voice,
                "format": "pcm",
                "sample_rate": self.sample_rate,
            },
        }
        async with self._client.stream(
            "POST",
            f"{bailian_base_url(self.provider)}{_COSYVOICE_TTS_PATH}",
            headers=_bearer_headers(self.provider, sse=True),
            json=payload,
        ) as response:
            response.raise_for_status()
            request_id = "unknown-request"
            completed = False

            async def audio_chunks() -> AsyncIterator[bytes]:
                nonlocal completed, request_id
                async for event in _iter_sse_json(response):
                    event_request_id = event.get("request_id")
                    if isinstance(event_request_id, str) and event_request_id:
                        request_id = event_request_id
                    code = event.get("code")
                    message = event.get("message")
                    if code is not None or message is not None:
                        raise RuntimeError(
                            "DashScope TTS stream "
                            f"{request_id} failed: {code or 'unknown-error'}: "
                            f"{message or 'no error message'}"
                        )
                    output = event.get("output")
                    finish_reason = (
                        output.get("finish_reason") if isinstance(output, dict) else None
                    )
                    if finish_reason == "stop":
                        completed = True
                    elif finish_reason not in (None, "null"):
                        raise RuntimeError(
                            "DashScope TTS stream "
                            f"{request_id} ended with finish_reason={finish_reason}"
                        )
                    audio = output.get("audio") if isinstance(output, dict) else None
                    data = audio.get("data") if isinstance(audio, dict) else None
                    if isinstance(data, str) and data:
                        yield base64.b64decode(data, validate=True)

            async for chunk in _iter_pcm_s16le(
                audio_chunks(),
                sample_rate=self.sample_rate,
            ):
                yield chunk
            if not completed:
                raise RuntimeError(
                    f"DashScope TTS stream {request_id} ended before finish_reason=stop"
                )

    async def close(self) -> None:
        await self._client.aclose()


async def list_cosyvoice_voices(
    provider: AliyunBailianProvider,
    *,
    prefix: str | None = None,
    page_size: int = 100,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[CosyVoiceInventoryVoice]:
    """List provider-managed voices without issuing one detail query per voice."""

    if not 1 <= page_size <= _MAX_COSYVOICE_INVENTORY:
        raise ValueError(f"page_size must be between 1 and {_MAX_COSYVOICE_INVENTORY}")

    voices: list[CosyVoiceInventoryVoice] = []
    seen_ids: set[str] = set()
    max_pages = (_MAX_COSYVOICE_INVENTORY + page_size - 1) // page_size
    target_model = provider.tts_model.strip()
    compatible_prefix = f"{target_model}-"

    async with httpx.AsyncClient(
        transport=transport,
        timeout=provider.timeout_seconds,
    ) as client:
        for page_index in range(max_pages):
            request_input: dict[str, object] = {
                "action": "list_voice",
                "page_index": page_index,
                "page_size": page_size,
            }
            if prefix is not None:
                request_input["prefix"] = prefix
            response = await client.post(
                f"{bailian_base_url(provider)}{_VOICE_CUSTOMIZATION_PATH}",
                headers=_bearer_headers(provider),
                json={
                    "model": "voice-enrollment",
                    "input": request_input,
                },
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("CosyVoice voice inventory response is not valid JSON") from exc
            output = payload.get("output") if isinstance(payload, dict) else None
            page = output.get("voice_list") if isinstance(output, dict) else None
            if not isinstance(page, list):
                raise RuntimeError(
                    "CosyVoice voice inventory response is missing output.voice_list"
                )

            new_ids = 0
            for raw_voice in page:
                if not isinstance(raw_voice, dict):
                    raise RuntimeError("CosyVoice voice inventory contains an invalid voice entry")
                voice_id = raw_voice.get("voice_id")
                if not isinstance(voice_id, str) or not voice_id.strip():
                    raise RuntimeError("CosyVoice voice inventory entry is missing voice_id")
                voice_id = voice_id.strip()
                if voice_id in seen_ids:
                    continue

                raw_status = raw_voice.get("status")
                status = (
                    raw_status.strip().upper()
                    if isinstance(raw_status, str) and raw_status.strip()
                    else "UNKNOWN"
                )
                if status not in _COSYVOICE_VOICE_STATES:
                    status = "UNKNOWN"

                created_at = raw_voice.get("gmt_create")
                if created_at is not None and not isinstance(created_at, str):
                    raise RuntimeError("CosyVoice voice inventory entry has an invalid gmt_create")
                modified_at = raw_voice.get("gmt_modified")
                if modified_at is not None and not isinstance(modified_at, str):
                    raise RuntimeError(
                        "CosyVoice voice inventory entry has an invalid gmt_modified"
                    )

                explicit_target = raw_voice.get("target_model")
                compatible = (
                    explicit_target.strip() == target_model
                    if isinstance(explicit_target, str) and explicit_target.strip()
                    else voice_id.startswith(compatible_prefix)
                )
                voices.append(
                    CosyVoiceInventoryVoice(
                        id=voice_id,
                        status=status,
                        compatible=compatible,
                        created_at=created_at,
                        modified_at=modified_at,
                    )
                )
                seen_ids.add(voice_id)
                new_ids += 1
                if len(voices) >= _MAX_COSYVOICE_INVENTORY:
                    return voices

            if len(page) < page_size:
                break
            if page and new_ids == 0:
                raise RuntimeError("CosyVoice voice inventory pagination repeated a page")

    return voices


async def clone_cosyvoice(
    provider: AliyunBailianProvider,
    *,
    audio_url: str | None = None,
    filename: str | None = None,
    content_type: str | None = None,
    audio: bytes | None = None,
    prefix: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> VoiceCloneResult:
    async with httpx.AsyncClient(
        transport=transport,
        timeout=provider.timeout_seconds,
    ) as client:
        local_audio_supplied = any(value is not None for value in (filename, content_type, audio))
        if audio_url is not None and local_audio_supplied:
            raise ValueError("provide either audio_url or local audio, not both")
        if audio_url is None:
            if not filename or not content_type or not audio:
                raise ValueError(
                    "filename, content_type, and non-empty audio are required for local upload"
                )
            safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
            if not safe_filename:
                raise ValueError("local audio filename is required")
            policy_response = await client.get(
                f"{_DASHSCOPE_BASE_URL}{_UPLOAD_POLICY_PATH}",
                headers=_bearer_headers(provider),
                params={"action": "getPolicy", "model": "voice-enrollment"},
            )
            policy_response.raise_for_status()
            payload = policy_response.json()
            policy = payload.get("data")
            if not isinstance(policy, dict):
                raise RuntimeError("DashScope upload policy response is missing data")
            required_fields = (
                "policy",
                "signature",
                "upload_dir",
                "upload_host",
                "oss_access_key_id",
                "x_oss_object_acl",
                "x_oss_forbid_overwrite",
            )
            values = {name: policy.get(name) for name in required_fields}
            missing = [
                name for name, value in values.items() if not isinstance(value, str) or not value
            ]
            if missing:
                raise RuntimeError("DashScope upload policy is missing " + ", ".join(missing))
            key = f"{values['upload_dir'].rstrip('/')}/{safe_filename}"
            upload_response = await client.post(
                values["upload_host"],
                data={
                    "OSSAccessKeyId": values["oss_access_key_id"],
                    "policy": values["policy"],
                    "Signature": values["signature"],
                    "key": key,
                    "x-oss-object-acl": values["x_oss_object_acl"],
                    "x-oss-forbid-overwrite": values["x_oss_forbid_overwrite"],
                    "success_action_status": "200",
                },
                files={"file": (safe_filename, audio, content_type)},
            )
            upload_response.raise_for_status()
            audio_url = f"oss://{key}"

        headers = _bearer_headers(provider)
        headers["X-DashScope-OssResourceResolve"] = "enable"
        response = await client.post(
            f"{bailian_base_url(provider)}{_VOICE_CUSTOMIZATION_PATH}",
            headers=headers,
            json={
                "model": "voice-enrollment",
                "input": {
                    "action": "create_voice",
                    "target_model": provider.tts_model,
                    "prefix": prefix,
                    "url": audio_url,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
    output = payload.get("output")
    voice_id = output.get("voice_id") if isinstance(output, dict) else None
    if not isinstance(voice_id, str) or not voice_id:
        raise RuntimeError("CosyVoice clone response is missing output.voice_id")
    return VoiceCloneResult(voice_id=voice_id)
