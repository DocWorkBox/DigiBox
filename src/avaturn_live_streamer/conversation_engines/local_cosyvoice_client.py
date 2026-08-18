# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Turn-scoped duplex client for the local CosyVoice clone service."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer


class _WebSocket(Protocol):
    async def send(self, value: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


Connector = Callable[..., Awaitable[_WebSocket]]


def _stream_url(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    if parts.scheme not in {"http", "https", "ws", "wss"} or not parts.netloc:
        raise ValueError("local CosyVoice base URL must be HTTP(S) or WS(S)")
    scheme = "wss" if parts.scheme in {"https", "wss"} else "ws"
    path = f"{parts.path.rstrip('/')}/audio/speech/stream"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


class LocalCosyVoiceTextTurn:
    def __init__(
        self,
        websocket: _WebSocket,
        *,
        sample_rate: int,
        secret: str,
        on_close: Callable[[LocalCosyVoiceTextTurn], None],
    ) -> None:
        self._websocket = websocket
        self._sample_rate = sample_rate
        self._secret = secret
        self._on_close = on_close
        self._closed = False
        self._finished = False

    async def _send(self, payload: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("local CosyVoice text stream is closed")
        await self._websocket.send(json.dumps(payload, ensure_ascii=False))

    async def send_text(self, text: str) -> None:
        if self._finished:
            raise RuntimeError("local CosyVoice text stream is already finished")
        if text:
            await self._send({"type": "append", "text": text})

    async def finish_text(self) -> None:
        if self._finished or self._closed:
            return
        self._finished = True
        await self._send({"type": "finish"})

    def _redact(self, value: object) -> str:
        message = str(value)
        if self._secret:
            message = message.replace(self._secret, "***")
        return message[:500]

    async def stream_audio(self) -> AsyncIterator[SpeechBuffer]:
        try:
            while not self._closed:
                message = await self._websocket.recv()
                if isinstance(message, bytes):
                    if len(message) % 2:
                        raise RuntimeError("local CosyVoice returned unaligned PCM16 audio")
                    if message:
                        yield SpeechBuffer.from_bytes(message, self._sample_rate)
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("local CosyVoice returned invalid JSON") from exc
                event_type = event.get("type") if isinstance(event, dict) else None
                if event_type == "completed":
                    return
                if event_type == "failed":
                    detail = event.get("message", "local CosyVoice synthesis failed")
                    raise RuntimeError(self._redact(detail))
                raise RuntimeError(f"unexpected local CosyVoice event: {event_type!r}")
        finally:
            await self._close()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._websocket.close()
        finally:
            self._on_close(self)

    async def cancel(self) -> None:
        if self._closed:
            return
        with suppress(Exception):
            await self._send({"type": "cancel"})
        await self._close()


class LocalCosyVoiceStreamingClient:
    """Open one duplex text stream per response while retaining clone identity."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        voice: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        connector: Connector | None = None,
    ) -> None:
        if not model.strip() or not voice.strip():
            raise ValueError("model and voice are required")
        self._url = _stream_url(base_url)
        self._model = model
        self._voice = voice
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._connector = connector
        self._turns: set[LocalCosyVoiceTextTurn] = set()
        self._closed = False

    async def _connect(self, **kwargs: Any) -> _WebSocket:
        connector = self._connector
        if connector is None:
            from websockets.asyncio.client import connect

            connector = connect
        return await connector(self._url, **kwargs)

    def _forget(self, turn: LocalCosyVoiceTextTurn) -> None:
        self._turns.discard(turn)

    def _redact(self, value: object) -> str:
        message = str(value)
        if self._api_key:
            message = message.replace(self._api_key, "***")
        return message[:500]

    async def open_text_stream(self) -> LocalCosyVoiceTextTurn:
        if self._closed:
            raise RuntimeError("local CosyVoice client is closed")
        headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self._api_key
            else None
        )
        try:
            websocket = await self._connect(
                additional_headers=headers,
                open_timeout=self._timeout_seconds,
                close_timeout=min(self._timeout_seconds, 5.0),
                max_size=None,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise RuntimeError(
                f"local CosyVoice connection failed: {self._redact(exc)}"
            ) from exc
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "model": self._model,
                        "voice": self._voice,
                    },
                    ensure_ascii=False,
                )
            )
            raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=self._timeout_seconds,
            )
            if not isinstance(raw, str):
                raise RuntimeError("local CosyVoice start response must be JSON")
            event = json.loads(raw)
            if not isinstance(event, dict) or event.get("type") != "started":
                detail = event.get("message", event) if isinstance(event, dict) else event
                message = str(detail)
                if self._api_key:
                    message = message.replace(self._api_key, "***")
                raise RuntimeError(message[:500])
            sample_rate = event.get("sample_rate")
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise RuntimeError("local CosyVoice start response has invalid sample_rate")
            turn = LocalCosyVoiceTextTurn(
                websocket,
                sample_rate=sample_rate,
                secret=self._api_key,
                on_close=self._forget,
            )
            self._turns.add(turn)
            return turn
        except BaseException:
            await websocket.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        turns = tuple(self._turns)
        if turns:
            await asyncio.gather(
                *(turn.cancel() for turn in turns),
                return_exceptions=True,
            )
