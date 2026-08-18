from __future__ import annotations

import threading
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from avtr1_renderer.api import app as api_app
from avtr1_renderer.idle_loop import (
    IDLE_LOOP_FPS,
    IDLE_LOOP_LOSSLESS,
    IDLE_LOOP_METHOD,
    IDLE_LOOP_SOURCE_FRAMES,
    encode_idle_loop_webp,
    generate_avatar_idle_loop,
)
from avtr1_renderer.pipeline import _straight_rgba_uint8


def _rgba_frame(
    *,
    straight_rgb: tuple[int, int, int] = (200, 100, 50),
    alpha: int = 128,
    height: int = 8,
    width: int = 8,
) -> np.ndarray:
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[:, :, :3] = straight_rgb
    frame[:, :, 3] = alpha
    return frame


def _webp_animation_frame_durations(payload: bytes) -> list[int]:
    """Read ANMF durations without relying on Pillow's per-frame metadata."""

    durations: list[int] = []
    offset = 12  # RIFF size + WEBP signature
    while offset + 8 <= len(payload):
        chunk_type = payload[offset : offset + 4]
        chunk_size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        chunk_payload = offset + 8
        if chunk_type == b"ANMF":
            durations.append(
                int.from_bytes(
                    payload[chunk_payload + 12 : chunk_payload + 15],
                    "little",
                )
            )
        offset = chunk_payload + chunk_size + (chunk_size % 2)
    return durations


def _distinct_idle_source_frames(count: int) -> list[Image.Image]:
    """Build production-shaped RGBA values that expose frame loss or reordering."""

    y, x = np.indices((6, 8))
    frames: list[Image.Image] = []
    for index in range(count):
        rgba = np.empty((6, 8, 4), dtype=np.uint8)
        rgba[:, :, 0] = (index * 17 + x * 13 + y * 7) % 256
        rgba[:, :, 1] = (index * 29 + x * 5 + y * 19) % 256
        rgba[:, :, 2] = (index * 43 + x * 11 + y * 3) % 256
        rgba[:, :, 3] = (index * 31 + x * 23 + y * 37) % 256

        # Match _straight_rgba_uint8's real output contract: transparent
        # pixels carry zero RGB. Keep explicit semi-transparent and opaque
        # samples so alpha and edge colour must also survive the round-trip.
        rgba[0, 0] = (0, 0, 0, 0)
        rgba[0, 1] = (17 + index % 100, 88, 203, 128)
        rgba[0, 2] = (91, 55 + index % 200, 144, 255)
        rgba[rgba[:, :, 3] == 0, :3] = 0
        frames.append(Image.fromarray(rgba, mode="RGBA"))
    return frames


def test_direct_rgba_export_unpremultiplies_hair_edges_before_quantising() -> None:
    alpha = torch.tensor([[[[0.5, 0.0]]]], dtype=torch.float32)
    premultiplied = torch.tensor(
        [[[[0.4, 0.0]], [[0.2, 0.0]], [[0.1, 0.0]]]],
        dtype=torch.float32,
    )

    rgba = _straight_rgba_uint8(premultiplied, alpha).numpy()

    assert rgba.shape == (1, 1, 2, 4)
    assert np.allclose(rgba[0, 0, 0], [204, 102, 51, 128], atol=1)
    assert np.array_equal(rgba[0, 0, 1], [0, 0, 0, 0])


def test_idle_webp_ping_pongs_and_loops_forever() -> None:
    frames = [
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 255, 0, 192)),
        Image.new("RGBA", (8, 8), (0, 0, 255, 128)),
    ]

    payload = encode_idle_loop_webp(frames)
    animated = Image.open(BytesIO(payload))

    assert payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
    assert animated.is_animated
    assert animated.n_frames == 4
    assert animated.info["loop"] == 0
    first_anmf = payload.index(b"ANMF")
    first_payload = first_anmf + 8
    assert int.from_bytes(payload[first_payload + 12 : first_payload + 15], "little") == 50


def test_default_idle_recipe_is_78_lossless_rgba_frames_over_3900ms() -> None:
    assert IDLE_LOOP_LOSSLESS is True
    assert IDLE_LOOP_METHOD == 0
    assert IDLE_LOOP_FPS == 20
    assert IDLE_LOOP_SOURCE_FRAMES == 40

    source_frames = _distinct_idle_source_frames(IDLE_LOOP_SOURCE_FRAMES)
    expected_frames = [*source_frames, *source_frames[-2:0:-1]]
    payload = encode_idle_loop_webp(source_frames)
    durations = _webp_animation_frame_durations(payload)
    decoded = Image.open(BytesIO(payload))

    assert len(expected_frames) == 78
    assert decoded.n_frames == 78
    assert durations == [50] * 78
    assert sum(durations) == 3900

    for index, expected in enumerate(expected_frames):
        decoded.seek(index)
        actual_rgba = np.asarray(decoded.convert("RGBA"))
        expected_rgba = np.asarray(expected)
        assert np.array_equal(actual_rgba, expected_rgba), (
            f"frame {index} changed during lossless WebP round-trip"
        )


def test_idle_generation_uses_silent_listening_track_and_transparent_output() -> None:
    calls = []
    guard_events: list[str] = []

    @contextmanager
    def inference_guard():
        guard_events.append("enter")
        try:
            yield
        finally:
            guard_events.append("exit")

    class FakePipeline:
        _motion_generator = SimpleNamespace(
            chunk_size=2,
            future_size=1,
            frame_len=4,
            audio_shift=0,
        )

        def process_transparent_rgba_chunk(self, avatar, chunk, state, options):
            calls.append((avatar, chunk, state, options))
            index = len(calls)
            frames = [
                _rgba_frame(straight_rgb=(40 * index, 80, 120)),
                _rgba_frame(straight_rgb=(40 * index, 100, 120)),
            ]
            return f"state-{index}", iter(frames)

    avatar = SimpleNamespace(id="person")
    payload = generate_avatar_idle_loop(
        FakePipeline(),
        avatar,
        source_frames=5,
        fps=25,
        inference_guard=inference_guard,
    )

    assert payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
    assert len(calls) == 4
    assert guard_events == ["enter", "exit"] * 4
    assert calls[0][2] is None
    assert calls[1][2] == "state-1"
    for _avatar, chunk, _state, options in calls:
        assert not np.any(chunk.audio_speech)
        assert not np.any(chunk.audio_listen)
        assert options.cfg_self_audio == 0.0
        assert options.cfg_other_audio == 2.0


def test_idle_loop_route_generates_once_then_reuses_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    avatar = SimpleNamespace(id="person")
    calls = []
    payload = b"RIFF\x10\x00\x00\x00WEBPidle-loop"

    def fake_generate(pipeline, selected_avatar, **_kwargs):
        calls.append((pipeline, selected_avatar))
        return payload

    pipeline = object()
    monkeypatch.setattr(api_app, "generate_avatar_idle_loop", fake_generate)
    monkeypatch.setattr(api_app.app.state, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(api_app.app.state, "registry", {"person": avatar}, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_assets_root", tmp_path, raising=False)
    client = TestClient(api_app.app, raise_server_exceptions=False)

    first = client.get("/avatars/person/idle-loop")
    second = client.get("/avatars/person/idle-loop")

    assert first.status_code == 200
    assert first.headers["content-type"] == "image/webp"
    assert first.content == payload
    assert second.status_code == 200
    assert calls == [(pipeline, avatar)]
    assert (tmp_path / "idle_loops" / "v3" / "person.webp").read_bytes() == payload


def test_idle_loop_route_returns_retryable_busy_without_starting_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    avatar = SimpleNamespace(id="person")

    def fake_generate(*args, **kwargs):
        calls.append((args, kwargs))
        return b"RIFF\x10\x00\x00\x00WEBPidle-loop"

    monkeypatch.setattr(api_app, "generate_avatar_idle_loop", fake_generate)
    monkeypatch.setattr(api_app.app.state, "pipeline", object(), raising=False)
    monkeypatch.setattr(api_app.app.state, "registry", {"person": avatar}, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_assets_root", tmp_path, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_asset_lock", None, raising=False)
    monkeypatch.setattr(
        api_app.app.state,
        "renderer_activity",
        SimpleNamespace(has_active_live=True),
        raising=False,
    )
    client = TestClient(api_app.app, raise_server_exceptions=False)

    response = client.get("/avatars/person/idle-loop")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert calls == []


def test_live_waiter_prevents_a_second_idle_inference_chunk() -> None:
    gate = api_app._RendererActivityGate()
    live_registered = threading.Event()
    live_entered = threading.Event()
    release_live = threading.Event()

    def run_live() -> None:
        gate.begin_live()
        live_registered.set()
        try:
            with gate.live_inference():
                live_entered.set()
                assert release_live.wait(timeout=2)
        finally:
            gate.end_live()

    with gate.idle_inference():
        worker = threading.Thread(target=run_live)
        worker.start()
        assert live_registered.wait(timeout=2)
        assert not live_entered.is_set()

    assert live_entered.wait(timeout=2)
    with pytest.raises(api_app._RendererBusyError), gate.idle_inference():
        pass

    release_live.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert not gate.has_active_live
    with gate.idle_inference():
        pass


def test_live_operation_releases_activity_after_failure() -> None:
    gate = api_app._RendererActivityGate()
    gate.begin_live()

    def fail() -> None:
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        api_app._run_reserved_live(gate, fail)

    assert not gate.has_active_live
    with gate.idle_inference():
        pass


def test_avatar_upload_generates_idle_loop_and_returns_its_url(
    monkeypatch,
    tmp_path: Path,
) -> None:
    asset = api_app._NormalisedUserAsset(
        kind="avatar",
        asset_id="user_person_123456789abc",
        png_bytes=b"png",
    )

    class FakePipeline:
        _backgrounds: ClassVar[dict[str, object]] = {}

        def prepare_avatar(self, path, *, avatar_id):
            return SimpleNamespace(id=avatar_id, path=path)

    pipeline = FakePipeline()
    generated = []

    monkeypatch.setattr(api_app, "_normalise_user_image", lambda *args, **kwargs: asset)
    monkeypatch.setattr(
        api_app,
        "_ensure_avatar_idle_loop",
        lambda **kwargs: generated.append(kwargs) or (tmp_path / "idle_loops" / "v3" / "voice.webp"),
    )
    monkeypatch.setattr(api_app.app.state, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(api_app.app.state, "registry", {}, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_assets_root", tmp_path, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_asset_lock", None, raising=False)
    client = TestClient(
        api_app.app,
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/assets/avatar",
        files={"file": ("person.png", b"source", "image/png")},
        data={"preserve_background": "false"},
    )

    assert response.status_code == 200
    assert response.json()["idle_loop_url"] == "/avatars/user_person_123456789abc/idle-loop?recipe=v3"
    assert response.json()["idle_loop_ready"] is True
    assert len(generated) == 1
    assert generated[0]["avatar_id"] == "user_person_123456789abc"


def test_avatar_upload_keeps_static_fallback_while_live_rendering_is_active(
    monkeypatch,
    tmp_path: Path,
) -> None:
    asset = api_app._NormalisedUserAsset(
        kind="avatar",
        asset_id="user_person_123456789abc",
        png_bytes=b"png",
    )

    class FakePipeline:
        _backgrounds: ClassVar[dict[str, object]] = {}

        def prepare_avatar(self, path, *, avatar_id):
            return SimpleNamespace(id=avatar_id, path=path)

    generated = []

    def fake_generate(*args, **kwargs):
        generated.append((args, kwargs))
        return b"RIFF\x10\x00\x00\x00WEBPidle-loop"

    monkeypatch.setattr(api_app, "_normalise_user_image", lambda *args, **kwargs: asset)
    monkeypatch.setattr(api_app, "generate_avatar_idle_loop", fake_generate)
    monkeypatch.setattr(api_app.app.state, "pipeline", FakePipeline(), raising=False)
    monkeypatch.setattr(api_app.app.state, "registry", {}, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_assets_root", tmp_path, raising=False)
    monkeypatch.setattr(api_app.app.state, "user_asset_lock", None, raising=False)
    monkeypatch.setattr(
        api_app.app.state,
        "renderer_activity",
        SimpleNamespace(has_active_live=True),
        raising=False,
    )
    client = TestClient(
        api_app.app,
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/assets/avatar",
        files={"file": ("person.png", b"source", "image/png")},
        data={"preserve_background": "false"},
    )

    assert response.status_code == 200
    assert response.json()["idle_loop_ready"] is False
    assert response.json()["idle_loop_url"].endswith("/idle-loop?recipe=v3")
    assert generated == []
    assert not (tmp_path / "idle_loops" / "v3" / f"{asset.asset_id}.webp").exists()


def test_deleting_avatar_moves_cached_idle_loop_to_recoverable_trash(tmp_path: Path) -> None:
    avatar_id = "user_person_123456789abc"
    portrait_dir = tmp_path / "reference_frames"
    idle_dir = tmp_path / "idle_loops" / "v3"
    portrait_dir.mkdir(parents=True)
    idle_dir.mkdir(parents=True)
    (portrait_dir / f"{avatar_id}.png").write_bytes(b"portrait")
    (idle_dir / f"{avatar_id}.webp").write_bytes(b"idle")

    result = api_app._trash_user_avatar(
        avatar_id,
        registry={avatar_id: object()},
        user_root=tmp_path,
    )

    assert result["deleted"] is True
    assert not (idle_dir / f"{avatar_id}.webp").exists()
    assert len(
        list((tmp_path / ".trash" / "idle_loops" / "v3").glob(f"{avatar_id}.*.webp"))
    ) == 1
