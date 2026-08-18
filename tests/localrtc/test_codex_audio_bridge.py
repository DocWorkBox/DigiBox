from __future__ import annotations

import asyncio
import json
import struct

import numpy as np
import pytest

from avaturn_live_streamer.localrtc import peer as peer_module


class _FakePeerConnection:
    connectionState = "new"  # noqa: N815 - mirrors aiortc

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def addTrack(self, _track: object) -> None:
        return None

    def on(self, event: str, handler: object | None = None):
        if handler is not None:
            self.handlers[event] = handler
            return handler

        def decorator(callback: object) -> object:
            self.handlers[event] = callback
            return callback

        return decorator


class _FakeDataChannel:
    def __init__(self, label: str) -> None:
        self.label = label
        self.handlers: dict[str, object] = {}

    def on(self, event: str):
        def decorator(callback: object) -> object:
            self.handlers[event] = callback
            return callback

        return decorator

    def emit_message(self, message: bytes | str) -> None:
        callback = self.handlers["message"]
        assert callable(callback)
        callback(message)


def _pcm_packet(samples: np.ndarray, sample_rate: int = 24_000) -> bytes:
    return struct.pack("<I", sample_rate) + samples.astype("<i2").tobytes()


def test_codex_audio_packet_decoder_preserves_pcm_and_sample_rate() -> None:
    decoder = getattr(peer_module, "decode_codex_audio_packet", None)
    assert callable(decoder), "Codex bridge packet decoder is not implemented"

    expected = np.array([-32768, -100, 0, 100, 32767], dtype=np.int16)
    decoded = decoder(_pcm_packet(expected))

    assert decoded.sample_rate == 24_000
    assert decoded.to_bytes() == expected.astype("<i2").tobytes()


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        struct.pack("<I", 0) + b"\x00\x00",
        struct.pack("<I", 24_000) + b"\x00",
    ],
)
def test_codex_audio_packet_decoder_rejects_malformed_packets(packet: bytes) -> None:
    decoder = getattr(peer_module, "decode_codex_audio_packet", None)
    assert callable(decoder), "Codex bridge packet decoder is not implemented"

    with pytest.raises(ValueError):
        decoder(packet)


def test_codex_control_parser_accepts_only_supported_events() -> None:
    parser = getattr(peer_module, "parse_codex_bridge_control", None)
    assert callable(parser), "Codex bridge control parser is not implemented"

    speech = parser(json.dumps({"type": "speech_started"}))
    done = parser(json.dumps({"type": "output_audio_done", "item_id": "item-7"}))

    assert speech.type == "speech_started"
    assert speech.item_id is None
    assert done.type == "output_audio_done"
    assert done.item_id == "item-7"
    with pytest.raises(ValueError):
        parser(json.dumps({"type": "arbitrary_browser_command"}))


def test_localrtc_routes_only_the_named_codex_bridge_channel() -> None:
    assert hasattr(peer_module.LocalRTC, "recv_codex_bridge_message"), (
        "LocalRTC does not expose the Codex audio bridge queue"
    )

    async def exercise() -> None:
        pc = _FakePeerConnection()
        rtc = peer_module.LocalRTC(pc)  # type: ignore[arg-type]
        on_datachannel = pc.handlers.get("datachannel")
        assert callable(on_datachannel)

        ignored = _FakeDataChannel("unrelated")
        on_datachannel(ignored)
        assert "message" not in ignored.handlers

        channel = _FakeDataChannel("avtr-codex-audio")
        on_datachannel(channel)
        channel.emit_message(_pcm_packet(np.array([1, -1], dtype=np.int16)))

        message = await asyncio.wait_for(rtc.recv_codex_bridge_message(), timeout=0.1)
        assert message.sample_rate == 24_000
        assert message.to_bytes() == np.array([1, -1], dtype="<i2").tobytes()

    asyncio.run(exercise())
