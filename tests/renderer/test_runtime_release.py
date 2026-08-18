from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from avtr1_renderer.api import app as subject


def _runtime_state(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=object(),
        registry={"person": object()},
        current_samples=3200,
        future_samples=6400,
        health=object(),
        runtime_status="ready",
        runtime_error=None,
        runtime_lock=None,
        user_asset_lock=None,
        renderer_activity=subject._RendererActivityGate(),
        portraits_dir=tmp_path / "bundled",
        user_assets_root=tmp_path / "user_assets",
    )


def test_release_renderer_runtime_drops_all_model_references_and_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _runtime_state(tmp_path)
    fake_app = SimpleNamespace(state=state)
    cache_calls: list[str] = []
    monkeypatch.setattr(
        subject,
        "_release_accelerator_memory",
        lambda: cache_calls.append("released"),
    )

    result = subject._release_renderer_runtime(fake_app)

    assert result == {
        "status": "released",
        "released": True,
        "loaded": False,
        "active_requests": 0,
    }
    assert state.pipeline is None
    assert state.registry == {}
    assert state.current_samples is None
    assert state.future_samples is None
    assert state.health is None
    assert state.runtime_status == "released"
    assert cache_calls == ["released"]


def test_release_renderer_runtime_refuses_while_live_inference_is_active(
    tmp_path: Path,
) -> None:
    state = _runtime_state(tmp_path)
    state.renderer_activity.begin_live()
    try:
        with pytest.raises(subject._RendererBusyError, match="live"):
            subject._release_renderer_runtime(SimpleNamespace(state=state))
    finally:
        state.renderer_activity.end_live()


def test_ensure_renderer_runtime_lazy_loads_once_after_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _runtime_state(tmp_path)
    state.pipeline = None
    state.registry = {}
    state.current_samples = None
    state.future_samples = None
    state.health = None
    state.runtime_status = "released"
    fake_app = SimpleNamespace(state=state)
    pipeline = SimpleNamespace(
        _motion_generator=SimpleNamespace(
            chunk_size=5,
            frame_len=640,
            future_size=3,
            audio_shift=160,
        ),
        _backgrounds={"bg": object()},
    )
    loaded = subject._LoadedRendererRuntime(
        pipeline=pipeline,
        registry={"person": object()},
        current_samples=3200,
        future_samples=2080,
        health=object(),
    )
    load_calls: list[tuple[Path, Path]] = []

    def fake_load(*, portraits_dir: Path, user_root: Path):
        load_calls.append((portraits_dir, user_root))
        return loaded

    monkeypatch.setattr(subject, "_load_renderer_runtime", fake_load)

    first = asyncio.run(subject._ensure_renderer_runtime_loaded(fake_app))
    second = asyncio.run(subject._ensure_renderer_runtime_loaded(fake_app))

    assert first == (pipeline, loaded.registry)
    assert second == first
    assert load_calls == [(state.portraits_dir, state.user_assets_root)]
    assert state.runtime_status == "ready"


def test_release_route_is_loopback_only_and_reports_released(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _runtime_state(tmp_path)
    for name, value in vars(state).items():
        monkeypatch.setattr(subject.app.state, name, value, raising=False)
    monkeypatch.setattr(subject, "_release_accelerator_memory", lambda: None)

    local = TestClient(
        subject.app,
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    ).post("/release")
    remote = TestClient(
        subject.app,
        client=("192.0.2.15", 50000),
        raise_server_exceptions=False,
    ).post("/release")

    assert local.status_code == 200
    assert local.json() == {
        "service": "renderer",
        "status": "released",
        "released": True,
        "loaded": False,
        "active_requests": 0,
    }
    assert remote.status_code == 403


def test_released_renderer_lists_disk_assets_and_serves_cached_idle_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    portraits_dir = tmp_path / "bundled"
    user_root = tmp_path / "user_assets"
    (user_root / "reference_frames").mkdir(parents=True)
    (user_root / "backgrounds").mkdir(parents=True)
    (user_root / "idle_loops" / "v3").mkdir(parents=True)
    portraits_dir.mkdir()
    (portraits_dir / "bundled_person.png").write_bytes(b"png")
    avatar_id = "user_person_123456789abc"
    (user_root / "reference_frames" / f"{avatar_id}.png").write_bytes(b"png")
    (user_root / "backgrounds" / "user_bg_123456789abc.png").write_bytes(b"png")
    payload = b"RIFF\x10\x00\x00\x00WEBPcached-loop"
    (user_root / "idle_loops" / "v3" / f"{avatar_id}.webp").write_bytes(payload)

    monkeypatch.setattr(subject.app.state, "pipeline", None, raising=False)
    monkeypatch.setattr(subject.app.state, "registry", {}, raising=False)
    monkeypatch.setattr(subject.app.state, "runtime_status", "released", raising=False)
    monkeypatch.setattr(subject.app.state, "portraits_dir", portraits_dir, raising=False)
    monkeypatch.setattr(subject.app.state, "user_assets_root", user_root, raising=False)
    monkeypatch.setattr(
        subject,
        "_bundled_background_paths",
        lambda: {"theme": tmp_path / "theme.png"},
    )

    async def forbidden_load(_app):
        raise AssertionError("cached UI assets must not reload the renderer")

    monkeypatch.setattr(subject, "_ensure_renderer_runtime_loaded", forbidden_load)
    client = TestClient(subject.app, raise_server_exceptions=False)

    listing = client.get("/avatars")
    idle = client.get(f"/avatars/{avatar_id}/idle-loop")

    assert listing.status_code == 200
    assert listing.json() == {
        "avatars": ["bundled_person", avatar_id],
        "backgrounds": ["theme", "user_bg_123456789abc"],
        "loaded": False,
    }
    assert idle.status_code == 200
    assert idle.content == payload


def test_health_stays_ready_for_orchestrator_while_models_are_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject.app.state, "pipeline", None, raising=False)
    monkeypatch.setattr(subject.app.state, "health", None, raising=False)
    monkeypatch.setattr(subject.app.state, "runtime_status", "released", raising=False)

    response = TestClient(subject.app, raise_server_exceptions=False).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "released", "loaded": False}
