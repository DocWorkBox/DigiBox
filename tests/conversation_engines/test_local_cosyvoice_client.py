from __future__ import annotations

import asyncio
import json

import pytest


class _FakeWebSocket:
    def __init__(self, incoming: list[str | bytes]) -> None:
        self.incoming: asyncio.Queue[str | bytes] = asyncio.Queue()
        for item in incoming:
            self.incoming.put_nowait(item)
        self.sent: list[dict[str, object]] = []
        self.closed = 0

    async def send(self, value: str) -> None:
        self.sent.append(json.loads(value))

    async def recv(self) -> str | bytes:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed += 1


def test_local_cosyvoice_websocket_turn_preserves_selected_clone_and_audio() -> None:
    from avaturn_live_streamer.conversation_engines.local_cosyvoice_client import (
        LocalCosyVoiceStreamingClient,
    )

    socket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "started",
                    "sample_rate": 24_000,
                    "sample_format": "s16le",
                }
            ),
            b"\x01\x00\x02\x00",
            json.dumps({"type": "completed"}),
        ]
    )
    connection: dict[str, object] = {}

    async def connect(uri: str, **kwargs):
        connection["uri"] = uri
        connection.update(kwargs)
        return socket

    async def exercise():
        client = LocalCosyVoiceStreamingClient(
            base_url="http://127.0.0.1:8768/v1",
            model="Fun-CosyVoice3-0.5B-2512",
            voice="voice_local_clone",
            api_key="local-secret",
            connector=connect,
        )
        turn = await client.open_text_stream()
        await turn.send_text("第一段")
        await turn.send_text("第二段")
        await turn.finish_text()
        audio = [item async for item in turn.stream_audio()]
        await client.close()
        return audio

    audio = asyncio.run(exercise())

    assert connection["uri"] == "ws://127.0.0.1:8768/v1/audio/speech/stream"
    assert connection["additional_headers"] == {
        "Authorization": "Bearer local-secret"
    }
    assert socket.sent == [
        {
            "type": "start",
            "model": "Fun-CosyVoice3-0.5B-2512",
            "voice": "voice_local_clone",
        },
        {"type": "append", "text": "第一段"},
        {"type": "append", "text": "第二段"},
        {"type": "finish"},
    ]
    assert len(audio) == 1
    assert audio[0].sample_rate == 24_000
    assert audio[0].to_bytes() == b"\x01\x00\x02\x00"
    assert socket.closed == 1


def test_local_cosyvoice_websocket_failure_redacts_key_and_cancel_closes() -> None:
    from avaturn_live_streamer.conversation_engines.local_cosyvoice_client import (
        LocalCosyVoiceStreamingClient,
    )

    socket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "started",
                    "sample_rate": 24_000,
                    "sample_format": "s16le",
                }
            ),
            json.dumps(
                {
                    "type": "failed",
                    "message": "bad local-secret",
                }
            ),
        ]
    )

    async def connect(_uri: str, **_kwargs):
        return socket

    async def exercise() -> str:
        client = LocalCosyVoiceStreamingClient(
            base_url="http://127.0.0.1:8768/v1",
            model="model",
            voice="voice_local_clone",
            api_key="local-secret",
            connector=connect,
        )
        turn = await client.open_text_stream()
        with pytest.raises(RuntimeError) as caught:
            _ = [item async for item in turn.stream_audio()]
        await client.close()
        return str(caught.value)

    message = asyncio.run(exercise())
    assert "local-secret" not in message
    assert "***" in message
    assert socket.closed == 1


def test_local_cosyvoice_connector_failure_redacts_authorization_secret() -> None:
    from avaturn_live_streamer.conversation_engines.local_cosyvoice_client import (
        LocalCosyVoiceStreamingClient,
    )

    async def connect(_uri: str, **_kwargs):
        raise RuntimeError("handshake rejected for local-secret")

    async def exercise() -> str:
        client = LocalCosyVoiceStreamingClient(
            base_url="http://127.0.0.1:8768/v1",
            model="model",
            voice="voice_local_clone",
            api_key="local-secret",
            connector=connect,
        )
        with pytest.raises(RuntimeError) as caught:
            await client.open_text_stream()
        return str(caught.value)

    message = asyncio.run(exercise())
    assert "local-secret" not in message
    assert "***" in message
