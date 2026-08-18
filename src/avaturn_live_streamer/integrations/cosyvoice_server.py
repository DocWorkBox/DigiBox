# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Loopback OpenAI-compatible streaming wrapper for CosyVoice3.

The module intentionally imports neither Torch nor CosyVoice at import time,
so the main AVTR environment can expose/configure the integration even when
the separate CosyVoice environment has not been installed yet.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import ipaddress
import os
import queue
import re
import shutil
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = "Fun-CosyVoice3-0.5B-2512"
SAMPLE_RATE = 24_000
COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_VOICE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_PROMPT_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_REFERENCE_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
_MIN_REFERENCE_SECONDS = 3.0
_MAX_REFERENCE_SECONDS = 30.0
_MIN_REFERENCE_SAMPLE_RATE = 16_000
_MIN_REFERENCE_RMS = 0.003
_MIN_REFERENCE_PEAK = 0.01
_CACHE_FRAME_RATE = 50.0
_MIN_PROMPT_TOKENS = 20
_MIN_PROMPT_FEATURE_FRAMES = int(_MIN_REFERENCE_SECONDS * _CACHE_FRAME_RATE)
_MIN_ACOUSTIC_VARIANCE = 1e-6
_SPEAKER_CACHE_ENV = "AVTR1_COSYVOICE_SPEAKER_CACHE"
_SPEAKER_CACHE_MIGRATION_SUFFIX = ".avtr1-migrated-v1"


class ModelBusyError(RuntimeError):
    """Raised when an explicit unload races with an active model request."""


async def _gather_cancel_on_error(*awaitables: Awaitable[None]) -> None:
    """Run sibling coroutines with TaskGroup-like cancellation on Python 3.10."""

    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _is_loopback_request(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    try:
        address = ipaddress.ip_address(client.host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped is not None and mapped.is_loopback)


def _release_runtime_memory() -> None:
    """Collect dead model objects and return unused CUDA blocks to the driver."""

    gc.collect()
    try:
        import torch
    except (ImportError, OSError):
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        available = bool(cuda.is_available())
    except Exception:
        return
    if not available:
        return
    for cleanup_name in ("empty_cache", "ipc_collect"):
        cleanup = getattr(cuda, cleanup_name, None)
        if not callable(cleanup):
            continue
        try:
            cleanup()
        except Exception:
            # The model references are already gone.  An unavailable optional
            # CUDA cache operation must not turn a successful unload into a 500.
            continue


@dataclass(frozen=True)
class ReferenceAudioInfo:
    duration_seconds: float
    sample_rate: int
    rms: float
    peak: float


def _inspect_reference_audio(path: Path) -> ReferenceAudioInfo:
    """Decode an upload before the CosyVoice frontend sees it."""

    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise ValueError("soundfile is required to decode reference audio") from exc
    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("reference audio could not be decoded by soundfile") from exc
    waveform = np.asarray(samples, dtype=np.float64)
    if waveform.size == 0 or waveform.shape[0] == 0 or not np.isfinite(waveform).all():
        raise ValueError("reference audio is empty or contains invalid samples")
    return ReferenceAudioInfo(
        duration_seconds=float(waveform.shape[0] / int(sample_rate)),
        sample_rate=int(sample_rate),
        rms=float(np.sqrt(np.mean(np.square(waveform)))),
        peak=float(np.max(np.abs(waveform))),
    )


def _validate_reference_audio(info: ReferenceAudioInfo) -> None:
    if info.duration_seconds < _MIN_REFERENCE_SECONDS:
        raise ValueError("reference audio must be at least 3 seconds long")
    if info.duration_seconds > _MAX_REFERENCE_SECONDS:
        raise ValueError("reference audio must be no longer than 30 seconds")
    if info.sample_rate < _MIN_REFERENCE_SAMPLE_RATE:
        raise ValueError("reference audio must have a sample rate of at least 16 kHz")
    if info.rms < _MIN_REFERENCE_RMS or info.peak < _MIN_REFERENCE_PEAK:
        raise ValueError("reference audio is silent or has too little energy")


def _as_numpy(value: Any) -> np.ndarray:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value)


def _first_int(value: Any) -> int | None:
    try:
        values = _as_numpy(value).reshape(-1)
        if values.size != 1:
            return None
        return int(values[0])
    except (TypeError, ValueError, OverflowError):
        return None


def _voice_quality_from_cache(spk2info: Any, voice_id: str) -> dict[str, Any]:
    """Classify cached speaker features without mutating persisted state."""

    if not isinstance(spk2info, dict) or voice_id not in spk2info:
        return {
            "quality": "unknown",
            "selectable": True,
            "reference_duration_seconds": None,
            "warning": "cached voice metadata is unavailable; legacy voice remains selectable",
        }
    info = spk2info[voice_id]
    if not isinstance(info, dict):
        return {
            "quality": "unknown",
            "selectable": True,
            "reference_duration_seconds": None,
            "warning": "cached voice metadata is unavailable; legacy voice remains selectable",
        }
    token_count = _first_int(info.get("llm_prompt_speech_token_len"))
    feature_frames = _first_int(info.get("prompt_speech_feat_len"))
    features = info.get("prompt_speech_feat")
    if token_count is None or feature_frames is None or features is None:
        return {
            "quality": "unknown",
            "selectable": True,
            "reference_duration_seconds": None,
            "warning": "cached voice metadata is incomplete; legacy voice remains selectable",
        }
    duration = feature_frames / _CACHE_FRAME_RATE
    try:
        acoustic_features = _as_numpy(features).astype(np.float64, copy=False)
        acoustic_variance = float(np.var(acoustic_features))
        valid_acoustics = np.isfinite(acoustic_features).all() and acoustic_variance >= _MIN_ACOUSTIC_VARIANCE
    except (TypeError, ValueError):
        valid_acoustics = False
    valid_lengths = (
        token_count >= _MIN_PROMPT_TOKENS
        and feature_frames >= _MIN_PROMPT_FEATURE_FRAMES
        and _MIN_REFERENCE_SECONDS <= duration <= _MAX_REFERENCE_SECONDS
    )
    if valid_lengths and valid_acoustics:
        return {
            "quality": "ready",
            "selectable": True,
            "reference_duration_seconds": duration,
            "warning": None,
        }
    return {
        "quality": "invalid",
        "selectable": False,
        "reference_duration_seconds": duration,
        "warning": "cached speaker features fail minimum duration, token, or acoustic-quality checks",
    }


def _voice_quality(model: Any, voice_id: str) -> dict[str, Any]:
    frontend = getattr(model, "frontend", None)
    return _voice_quality_from_cache(getattr(frontend, "spk2info", None), voice_id)


def iter_pcm16_chunks(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Align arbitrary producer chunks to complete little-endian PCM16 samples."""

    carry = b""
    try:
        for chunk in chunks:
            data = carry + bytes(chunk)
            even_length = len(data) & ~1
            if even_length:
                yield data[:even_length]
            carry = data[even_length:]
        if carry:
            raise ValueError("PCM16 stream ended with an odd half-sample byte")
    finally:
        close_chunks = getattr(chunks, "close", None)
        if callable(close_chunks):
            close_chunks()


class _ManagedSpeechStream(Iterator[bytes]):
    """Own a model reservation even before the first audio chunk is requested."""

    def __init__(
        self,
        source: Iterator[bytes],
        on_close: Callable[[], None],
    ) -> None:
        self._source: Iterator[bytes] | None = source
        self._on_close: Callable[[], None] | None = on_close

    def __iter__(self) -> _ManagedSpeechStream:
        return self

    def __next__(self) -> bytes:
        source = self._source
        if source is None:
            raise StopIteration
        try:
            return next(source)
        except BaseException:
            try:
                self.close()
            finally:
                raise

    def close(self) -> None:
        source = self._source
        callback = self._on_close
        if source is None and callback is None:
            return
        self._source = None
        self._on_close = None
        try:
            close_source = getattr(source, "close", None)
            if callable(close_source):
                close_source()
        finally:
            # Drop the generator closure (and its model reference) before the
            # reservation becomes available to a concurrent release request.
            del source
            if callback is not None:
                callback()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class SpeechRequest(BaseModel):
    model: str = Field(min_length=1)
    input: str = Field(min_length=1, max_length=20_000)
    voice: str = Field(min_length=1, max_length=256)
    response_format: Literal["pcm"] = "pcm"
    stream: bool = True


class StreamingSpeechRequest(BaseModel):
    model: str = Field(min_length=1)
    voice: str = Field(min_length=1, max_length=256)


class VoiceService(Protocol):
    def health(self) -> dict[str, Any]: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def list_voices(self) -> list[dict[str, Any]]: ...

    def add_voice(
        self,
        *,
        name: str,
        transcript: str,
        reference_audio: bytes,
        reference_filename: str | None = None,
    ) -> dict[str, Any]: ...

    def delete_voice(self, voice_id: str) -> dict[str, Any]: ...

    def stream_speech(self, request: SpeechRequest) -> Iterable[bytes]: ...

    def stream_speech_incremental(
        self,
        request: StreamingSpeechRequest,
        text_chunks: Iterable[str],
    ) -> Iterable[bytes]: ...

    def release(self) -> dict[str, Any]: ...


class CosyVoice3Service:
    """Lazy adapter around the official ``AutoModel`` API."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir).resolve()
        legacy_cache = self.model_dir / "spk2info.pt"
        configured_cache = os.environ.get(_SPEAKER_CACHE_ENV)
        self.speaker_cache_path = (
            Path(configured_cache).expanduser().resolve()
            if configured_cache
            else legacy_cache
        )
        self._speaker_cache_is_overlay = (
            self.speaker_cache_path.resolve(strict=False)
            != legacy_cache.resolve(strict=False)
        )
        if self._speaker_cache_is_overlay:
            self._migrate_speaker_cache_once(legacy_cache)
        self._model: Any | None = None
        self._load_error: str | None = None
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._token_hop_len_base: int | None = None
        self._active_requests = 0
        self._released = False

    @property
    def _speaker_cache_migration_marker(self) -> Path:
        return self.speaker_cache_path.with_name(
            self.speaker_cache_path.name + _SPEAKER_CACHE_MIGRATION_SUFFIX
        )

    def _migrate_speaker_cache_once(self, legacy_cache: Path) -> None:
        """Seed the writable cache once, without retaining a legacy fallback."""

        target = self.speaker_cache_path
        target.parent.mkdir(parents=True, exist_ok=True)
        marker = self._speaker_cache_migration_marker
        if marker.is_file():
            return

        if not target.exists() and legacy_cache.is_file():
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".migrating",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    with legacy_cache.open("rb") as source_file:
                        shutil.copyfileobj(source_file, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                if not target.exists():
                    os.replace(temp_path, target)
                    temp_path = None
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        marker_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{marker.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                marker_temp = Path(temp_file.name)
                temp_file.write("legacy speaker cache migrated\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(marker_temp, marker)
            marker_temp = None
        finally:
            if marker_temp is not None:
                marker_temp.unlink(missing_ok=True)

    def _load_speaker_cache(self, *, map_location: Any = "cpu") -> dict[Any, Any]:
        if not self.speaker_cache_path.is_file():
            return {}
        import torch

        speaker_cache = torch.load(
            self.speaker_cache_path,
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(speaker_cache, dict):
            raise RuntimeError("CosyVoice speaker cache is invalid")
        return speaker_cache

    def _apply_speaker_cache_overlay(self, model: Any) -> None:
        if not self._speaker_cache_is_overlay:
            return
        frontend = getattr(model, "frontend", None)
        if frontend is None:
            raise RuntimeError("CosyVoice frontend is unavailable")
        # Replace, do not merge: otherwise a voice deleted from the writable
        # cache would be resurrected from the immutable model on every load.
        frontend.spk2info = self._load_speaker_cache(
            map_location=getattr(frontend, "device", "cpu")
        )

    def _reset_token_hop_len(self, model: Any) -> None:
        runtime_model = getattr(model, "model", None)
        current = getattr(runtime_model, "token_hop_len", None)
        if (
            self._token_hop_len_base is None
            and isinstance(current, int)
            and not isinstance(current, bool)
            and current > 0
        ):
            self._token_hop_len_base = current
        if self._token_hop_len_base is not None and runtime_model is not None:
            runtime_model.token_hop_len = self._token_hop_len_base

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from cosyvoice.cli.cosyvoice import AutoModel

                model = AutoModel(
                    model_dir=str(self.model_dir),
                    fp16=True,
                    load_trt=False,
                    load_vllm=False,
                )
                self._apply_speaker_cache_overlay(model)
                self._reset_token_hop_len(model)
                self._model = model
                self._load_error = None
                self._released = False
            except Exception as exc:
                self._load_error = str(exc)
                raise
            return self._model

    def health(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._model is not None
            status = "ready" if loaded else ("released" if self._released else "not_loaded")
            return {
                "service": "cosyvoice",
                "status": status,
                "model": MODEL_ID,
                "sample_rate": SAMPLE_RATE,
                "model_dir": str(self.model_dir),
                "speaker_cache_path": str(self.speaker_cache_path),
                "loaded": loaded,
                "released": self._released,
                "active_requests": self._active_requests,
                "error": self._load_error,
            }

    def release(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("CosyVoice has an active model request")
        try:
            if self._active_requests:
                raise ModelBusyError(
                    f"CosyVoice has {self._active_requests} active speech stream(s)"
                )
            self._model = None
            self._token_hop_len_base = None
            self._load_error = None
            self._released = True
            _release_runtime_memory()
            return {
                "service": "cosyvoice",
                "status": "released",
                "released": True,
                "loaded": False,
                "active_requests": 0,
            }
        finally:
            self._lock.release()

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": MODEL_ID, "object": "model", "owned_by": "FunAudioLLM"}]

    def list_voices(self) -> list[dict[str, Any]]:
        with self._lock:
            model = self._model
            if model is None:
                speaker_cache = self._load_speaker_cache()
                if not speaker_cache:
                    return []
                voice_ids = speaker_cache
            else:
                frontend = getattr(model, "frontend", None)
                speaker_cache = getattr(frontend, "spk2info", None)
                voice_ids = model.list_available_spks()
            voices: list[dict[str, Any]] = []
            for voice_id in voice_ids:
                identifier = str(voice_id)
                voices.append(
                    {
                        "id": identifier,
                        "name": identifier,
                        "object": "audio.voice",
                        "deletable": True,
                        **_voice_quality_from_cache(speaker_cache, identifier),
                    }
                )
            return voices

    def _persist_speaker_cache(self, model: Any) -> None:
        """Atomically replace CosyVoice's shared speaker cache."""

        frontend = getattr(model, "frontend", None)
        speaker_cache = getattr(frontend, "spk2info", None)
        if not isinstance(speaker_cache, dict):
            raise RuntimeError("CosyVoice speaker cache is unavailable")
        import torch

        target = self.speaker_cache_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=".spk2info.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            torch.save(speaker_cache, str(temp_path))
            os.replace(temp_path, target)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def delete_voice(self, voice_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", voice_id):
            raise ValueError("invalid voice id")
        with self._lock:
            model = self._load()
            frontend = getattr(model, "frontend", None)
            speaker_cache = getattr(frontend, "spk2info", None)
            if not isinstance(speaker_cache, dict):
                raise RuntimeError("CosyVoice speaker cache is unavailable")
            if voice_id not in speaker_cache:
                raise KeyError(voice_id)
            previous = speaker_cache.pop(voice_id)
            try:
                self._persist_speaker_cache(model)
            except BaseException:
                speaker_cache[voice_id] = previous
                raise
        return {
            "id": voice_id,
            "object": "audio.voice.deleted",
            "deleted": True,
        }

    @staticmethod
    def _voice_id(name: str) -> str:
        source_name = name.strip()
        cleaned = _VOICE_ID_RE.sub("_", source_name).strip("._-")
        if not cleaned:
            digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:12]
            cleaned = f"voice_{digest}"
        return cleaned[:128]

    def add_voice(
        self,
        *,
        name: str,
        transcript: str,
        reference_audio: bytes,
        reference_filename: str | None = None,
    ) -> dict[str, Any]:
        prompt_transcript = transcript.strip()
        if not prompt_transcript:
            raise ValueError("reference transcript cannot be empty")
        if _PROMPT_CONTROL_TOKEN_RE.search(prompt_transcript):
            raise ValueError("reference transcript cannot contain prompt control tokens")
        voice_id = self._voice_id(name)
        suffix = Path(reference_filename or "").suffix.lower()
        if suffix not in _REFERENCE_AUDIO_SUFFIXES:
            raise ValueError("reference audio must be WAV, FLAC, OGG, or MP3")
        temp_path: Path | None = None
        try:
            # CosyVoice opens the reference more than once for tokenizer,
            # speaker embedding and acoustic features. A real closed file is
            # therefore required on Windows; a decoded tensor or one-shot
            # BytesIO cannot satisfy the official frontend contract.
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(reference_audio)
                temp_path = Path(temp_file.name)
            _validate_reference_audio(_inspect_reference_audio(temp_path))

            with self._lock:
                model = self._load()
                frontend = getattr(model, "frontend", None)
                speaker_cache = getattr(frontend, "spk2info", None)
                if not isinstance(speaker_cache, dict):
                    speaker_cache = None
                missing = object()
                previous = (
                    speaker_cache.get(voice_id, missing)
                    if speaker_cache is not None
                    else missing
                )
                try:
                    model.add_zero_shot_spk(
                        COSYVOICE3_PROMPT_PREFIX + prompt_transcript,
                        str(temp_path),
                        voice_id,
                    )
                    quality = _voice_quality(model, voice_id)
                    if quality["quality"] != "ready":
                        raise ValueError(
                            "extracted voice features are invalid; use a clear 3-10 second "
                            "reference whose transcript exactly matches the speech"
                        )
                    if self._speaker_cache_is_overlay:
                        self._persist_speaker_cache(model)
                    else:
                        model.save_spkinfo()
                except Exception:
                    if speaker_cache is not None:
                        if previous is missing:
                            speaker_cache.pop(voice_id, None)
                        else:
                            speaker_cache[voice_id] = previous
                    raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return {
            "id": voice_id,
            "name": name.strip(),
            "object": "audio.voice",
            **quality,
        }

    def _stream_speech_input(
        self,
        *,
        model_id: str,
        voice: str,
        text_input: str | Iterator[str],
    ) -> Iterable[bytes]:
        if model_id not in {MODEL_ID, str(self.model_dir), self.model_dir.name}:
            raise ValueError(f"unsupported model: {model_id}")
        with self._lock:
            model = self._load()
            if voice not in model.list_available_spks():
                raise ValueError(f"unknown voice: {voice}")
            quality = _voice_quality(model, voice)
            if quality["quality"] == "invalid":
                raise ValueError(
                    f"voice {voice!r} has invalid cached reference features; "
                    "recreate it from a clear 3-10 second reference"
                )
            # Stored zero-shot speaker features make the reference inputs unused.
            # Keep them present for the official API signature.
            import torch

            empty_prompt = torch.zeros(1, 0)
            self._active_requests += 1

        def output_chunks() -> Iterator[bytes]:
            outputs: Iterator[dict[str, Any]] | None = None
            with self._inference_lock:
                self._reset_token_hop_len(model)
                try:
                    text_frontend = bool(
                        getattr(
                            getattr(model, "frontend", None),
                            "text_frontend",
                            "",
                        )
                    )
                    outputs = iter(
                        model.inference_zero_shot(
                            text_input,
                            "",
                            empty_prompt,
                            zero_shot_spk_id=voice,
                            stream=True,
                            text_frontend=text_frontend,
                        )
                    )
                    for output in outputs:
                        speech = output["tts_speech"].detach().float().cpu().numpy()
                        pcm = np.clip(speech.reshape(-1), -1.0, 1.0)
                        yield (pcm * 32767.0).astype("<i2", copy=False).tobytes()
                finally:
                    close_outputs = getattr(outputs, "close", None)
                    if callable(close_outputs):
                        with suppress(Exception):
                            close_outputs()
                    self._reset_token_hop_len(model)

        def release_reservation() -> None:
            with self._lock:
                self._active_requests -= 1

        return _ManagedSpeechStream(output_chunks(), release_reservation)

    def stream_speech(self, request: SpeechRequest) -> Iterable[bytes]:
        return self._stream_speech_input(
            model_id=request.model,
            voice=request.voice,
            text_input=request.input,
        )

    def stream_speech_incremental(
        self,
        request: StreamingSpeechRequest,
        text_chunks: Iterable[str],
    ) -> Iterable[bytes]:
        def text_generator() -> Iterator[str]:
            for text in text_chunks:
                if not isinstance(text, str):
                    raise TypeError("incremental TTS text chunks must be strings")
                if text:
                    yield text

        return self._stream_speech_input(
            model_id=request.model,
            voice=request.voice,
            text_input=text_generator(),
        )


def create_app(*, service: VoiceService | None = None) -> FastAPI:
    if service is None:
        model_dir = os.environ.get(
            "AVTR_COSYVOICE_MODEL_DIR",
            str(Path.cwd() / "models" / MODEL_ID),
        )
        service = CosyVoice3Service(model_dir)

    app = FastAPI(title="AVTR CosyVoice3 Local TTS", docs_url="/docs")
    allowed_origins = [
        origin.strip()
        for origin in os.environ.get(
            "AVTR_COSYVOICE_ALLOWED_ORIGINS",
            "http://localhost:7860,http://127.0.0.1:7860",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return service.health()

    @app.post("/release")
    async def release_runtime(request: Request) -> dict[str, Any]:
        if not _is_loopback_request(request):
            raise HTTPException(status_code=403, detail="release is restricted to loopback")
        try:
            return await asyncio.to_thread(service.release)
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": service.list_models()}

    @app.get("/v1/audio/voices")
    async def voices() -> dict[str, Any]:
        try:
            return {"object": "list", "data": await asyncio.to_thread(service.list_voices)}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/audio/voices")
    async def add_voice(
        name: Annotated[str, Form(min_length=1, max_length=128)],
        transcript: Annotated[str, Form(min_length=1, max_length=2000)],
        consent: Annotated[bool, Form()],
        reference_audio: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        if consent is not True:
            raise HTTPException(
                status_code=422,
                detail="consent must confirm that this voice is authorised for cloning",
            )
        audio = await reference_audio.read()
        if not audio:
            raise HTTPException(status_code=422, detail="reference_audio cannot be empty")
        if len(audio) > 30 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="reference_audio is too large")
        try:
            voice = await asyncio.to_thread(
                service.add_voice,
                name=name,
                transcript=transcript,
                reference_audio=audio,
                reference_filename=reference_audio.filename,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return voice

    @app.delete("/v1/audio/voices/{voice_id}")
    async def delete_voice(voice_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.delete_voice, voice_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown voice: {voice_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> StreamingResponse:
        try:
            chunks = service.stream_speech(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(
            iter_pcm16_chunks(chunks),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": str(SAMPLE_RATE),
                "X-Audio-Sample-Format": "s16le",
            },
        )

    @app.websocket("/v1/audio/speech/stream")
    async def speech_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        text_end = object()
        audio_end = object()
        stop_event = threading.Event()
        text_queue: queue.Queue[str | object] = queue.Queue(maxsize=8)
        audio_queue: queue.Queue[bytes | BaseException | object] = queue.Queue(maxsize=4)
        chunks: Iterable[bytes] | None = None
        worker: asyncio.Task[None] | None = None

        class ClientCancelledError(Exception):
            pass

        def force_queue_item(target: queue.Queue, item: object) -> None:
            while True:
                try:
                    target.put_nowait(item)
                    return
                except queue.Full:
                    with suppress(queue.Empty):
                        target.get_nowait()

        def text_source() -> Iterator[str]:
            while not stop_event.is_set():
                try:
                    item = text_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is text_end:
                    return
                assert isinstance(item, str)
                yield item

        async def enqueue_text(item: str | object) -> None:
            while not stop_event.is_set():
                try:
                    text_queue.put_nowait(item)
                    return
                except queue.Full:
                    await asyncio.sleep(0.005)
            raise ClientCancelledError

        async def receive_text() -> None:
            ended = False
            try:
                while True:
                    message = await websocket.receive_json()
                    kind = message.get("type") if isinstance(message, dict) else None
                    if kind == "append":
                        text = message.get("text")
                        if not isinstance(text, str) or not text:
                            raise ValueError("append.text must be a non-empty string")
                        await enqueue_text(text)
                        continue
                    if kind == "finish":
                        await enqueue_text(text_end)
                        ended = True
                        return
                    if kind == "cancel":
                        await enqueue_text(text_end)
                        ended = True
                        raise ClientCancelledError
                    raise ValueError(f"unsupported incremental TTS message: {kind!r}")
            finally:
                if not ended:
                    # Cancellation must not leave an uncancellable to_thread
                    # queue.put blocked behind a full producer queue.
                    force_queue_item(text_queue, text_end)

        def produce_audio() -> None:
            assert chunks is not None
            try:
                for chunk in iter_pcm16_chunks(chunks):
                    while not stop_event.is_set():
                        try:
                            audio_queue.put(chunk, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop_event.is_set():
                        break
            except BaseException as exc:
                if not stop_event.is_set():
                    force_queue_item(audio_queue, exc)
            finally:
                close_chunks = getattr(chunks, "close", None)
                if callable(close_chunks):
                    with suppress(Exception):
                        close_chunks()
                if stop_event.is_set():
                    force_queue_item(audio_queue, audio_end)
                else:
                    audio_queue.put(audio_end)

        async def forward_audio() -> None:
            while True:
                item = await asyncio.to_thread(audio_queue.get)
                if item is audio_end:
                    return
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, bytes)
                await websocket.send_bytes(item)

        try:
            message = await websocket.receive_json()
            if not isinstance(message, dict) or message.get("type") != "start":
                raise ValueError("first incremental TTS message must be start")
            request = StreamingSpeechRequest.model_validate(message)
            chunks = await asyncio.to_thread(
                service.stream_speech_incremental,
                request,
                text_source(),
            )
            await websocket.send_json(
                {
                    "type": "started",
                    "sample_rate": SAMPLE_RATE,
                    "sample_format": "s16le",
                }
            )
            worker = asyncio.create_task(asyncio.to_thread(produce_audio))
            await _gather_cancel_on_error(receive_text(), forward_audio())
            await worker
            await websocket.send_json({"type": "completed"})
        except (WebSocketDisconnect, asyncio.CancelledError, ClientCancelledError):
            pass
        except Exception as exc:
            with suppress(Exception):
                await websocket.send_json(
                    {"type": "failed", "message": str(exc)[:500]}
                )
        finally:
            stop_event.set()
            force_queue_item(text_queue, text_end)
            if worker is not None:
                with suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(worker), timeout=5.0)
            with suppress(Exception):
                await websocket.close()

    return app


app = create_app()
