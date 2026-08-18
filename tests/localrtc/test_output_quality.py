from __future__ import annotations

import asyncio
import inspect

import pytest
from aiortc import RTCPeerConnection
from aiortc.codecs import h264

from avaturn_live_streamer import local_stream_cli
from avaturn_live_streamer.localrtc.peer import LocalRTC


@pytest.mark.parametrize(
    ("quality", "aspect", "expected"),
    [
        ("smooth", "16:9", (360, 640)),
        ("smooth", "9:16", (360, 202)),
        ("smooth", "1:1", (360, 360)),
        ("smooth", "4:3", (360, 480)),
        ("smooth", "3:4", (360, 270)),
        ("balanced", "16:9", (540, 960)),
        ("balanced", "9:16", (540, 302)),
        ("balanced", "1:1", (540, 540)),
        ("balanced", "4:3", (540, 720)),
        ("balanced", "3:4", (540, 404)),
        ("ultra", "16:9", (720, 1280)),
        ("ultra", "9:16", (720, 404)),
        ("ultra", "1:1", (720, 720)),
        ("ultra", "4:3", (720, 960)),
        ("ultra", "3:4", (720, 540)),
    ],
)
def test_output_quality_resolves_even_aspect_dimensions(
    quality: str,
    aspect: str,
    expected: tuple[int, int],
) -> None:
    resolver = getattr(local_stream_cli, "_output_dimensions", None)
    assert callable(resolver), "output quality dimension resolver is not implemented"

    assert resolver(aspect, quality) == expected


def test_offer_body_carries_output_quality() -> None:
    assert "output_quality" in local_stream_cli._OfferBody.model_fields, (
        "the signaling offer does not carry the selected output quality"
    )

    body = local_stream_cli._OfferBody.model_validate(
        {
            "engine": {"type": "codex"},
            "sdp": "browser-offer",
            "type": "offer",
            "avatar_id": "avatar",
            "background_id": "background",
            "output_aspect": "9:16",
            "output_quality": "balanced",
        }
    )

    assert body.output_quality == "balanced"


def test_offer_body_carries_independent_rtx_super_resolution_switch() -> None:
    assert "rtx_super_resolution" in local_stream_cli._OfferBody.model_fields

    body = local_stream_cli._OfferBody.model_validate(
        {
            "engine": {"type": "codex"},
            "sdp": "browser-offer",
            "type": "offer",
            "avatar_id": "avatar",
            "background_id": "background",
            "output_aspect": "16:9",
            "output_quality": "smooth",
            "rtx_super_resolution": True,
        }
    )

    assert body.rtx_super_resolution is True


@pytest.mark.parametrize(
    ("quality", "aspect", "expected"),
    [
        ("smooth", "16:9", (720, 1280)),
        ("smooth", "9:16", (720, 404)),
        ("balanced", "16:9", (1080, 1920)),
        ("balanced", "1:1", (1080, 1080)),
        ("ultra", "16:9", (1080, 1920)),
        ("ultra", "4:3", (1080, 1440)),
    ],
)
def test_rtx_super_resolution_scales_quality_and_caps_at_1080p(
    quality: str,
    aspect: str,
    expected: tuple[int, int],
) -> None:
    assert local_stream_cli._output_dimensions(aspect, quality, True) == expected


def test_localrtc_prefers_h264_and_accepts_a_preset_bitrate(monkeypatch) -> None:
    assert "video_bitrate_bps" in inspect.signature(LocalRTC).parameters, (
        "LocalRTC does not accept the selected video bitrate"
    )

    monkeypatch.setattr(h264, "DEFAULT_BITRATE", 1_000_000)
    monkeypatch.setattr(h264, "MAX_BITRATE", 3_000_000)

    async def exercise() -> None:
        pc = RTCPeerConnection()
        try:
            LocalRTC(pc, video_bitrate_bps=2_500_000)

            video = next(
                transceiver
                for transceiver in pc.getTransceivers()
                if transceiver.kind == "video"
            )
            preferred = video._preferred_codecs
            assert preferred, "the video transceiver has no explicit codec preference"
            assert preferred[0].mimeType.lower() == "video/h264"
            assert all(codec.mimeType.lower() == "video/h264" for codec in preferred)
            assert h264.DEFAULT_BITRATE == 2_500_000
            assert h264.MAX_BITRATE >= 2_500_000
        finally:
            await pc.close()

    asyncio.run(exercise())
