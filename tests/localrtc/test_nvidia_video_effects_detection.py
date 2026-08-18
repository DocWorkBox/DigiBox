from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from avaturn_live_streamer import local_stream_cli


def _detector() -> Callable[..., dict[str, Any]]:
    module_name = "avaturn_live_streamer.integrations.nvidia_video_effects"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} is not implemented", pytrace=False)
    detect = getattr(module, "detect_nvidia_video_effects", None)
    assert callable(detect), "NVIDIA video-effects capability detector is not implemented"
    return detect


def _same_path(left: str | Path, right: Path) -> bool:
    return str(Path(left)).replace("/", "\\").casefold() == str(right).replace("/", "\\").casefold()


def test_detects_the_official_nvvfx_python_binding_without_loading_it() -> None:
    looked_up: list[str] = []

    def find_spec(module_name: str) -> object | None:
        looked_up.append(module_name)
        return object()

    result = _detector()({}, lambda _candidate: False, find_spec)

    assert result["available"] is True
    assert result["backend"] == "nvidia_vfx_python"
    assert looked_up == ["nvvfx"]
    assert "nvvfx" in str(result["reason"]).casefold()


def test_a_dll_without_the_nvvfx_binding_does_not_masquerade_as_usable_rtx() -> None:
    sdk_root = Path(r"C:\Program Files\NVIDIA Corporation\RTX Video SDK")
    runtime = sdk_root / "bin" / "x64" / "NVVideoEffects.dll"

    result = _detector()(
        {"NVIDIA_RTX_VIDEO_SDK_PATH": str(sdk_root)},
        lambda candidate: _same_path(candidate, runtime),
        lambda _module_name: None,
    )

    assert result["available"] is False
    assert result["backend"] is None
    reason = str(result["reason"]).casefold()
    assert "nvvfx" in reason
    assert "dll" in reason


def test_reports_explicitly_unavailable_when_no_nvidia_runtime_exists() -> None:
    result = _detector()({}, lambda _candidate: False, lambda _module_name: None)

    assert result["available"] is False
    assert result["backend"] is None
    reason = str(result["reason"]).strip().casefold()
    assert reason
    assert "nvvfx" in reason


def test_nvidia_vsr_is_an_independent_switch_not_a_quality_preset() -> None:
    assert "output_quality" in local_stream_cli._OfferBody.model_fields
    assert "rtx_super_resolution" in local_stream_cli._OfferBody.model_fields

    with pytest.raises(ValidationError):
        local_stream_cli._OfferBody.model_validate(
            {
                "engine": {"type": "codex"},
                "sdp": "browser-offer",
                "type": "offer",
                "avatar_id": "avatar",
                "background_id": "background",
                "output_aspect": "16:9",
                "output_quality": "nvidia_vsr",
                "rtx_super_resolution": True,
            }
        )


def test_streamer_preserves_safe_renderer_preflight_error_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = {
        "code": "nvidia_vsr_model_load_failed",
        "message": "NVIDIA VSR model files could not be loaded.",
    }

    class FakeHTTPClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeHTTPClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def post(self, url: str, **_kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            return httpx.Response(503, json={"detail": detail}, request=request)

    monkeypatch.setattr(local_stream_cli, "_renderer_base_url", lambda: "http://renderer")
    monkeypatch.setattr(local_stream_cli.httpx, "AsyncClient", FakeHTTPClient)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            local_stream_cli._prepare_nvidia_vsr_renderer(
                output_aspect="16:9",
                output_quality="balanced",
            )
        )

    assert error.value.status_code == 503
    assert error.value.detail == detail


def test_failed_rtx_preflight_does_not_consume_prepared_engine_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_calls: list[tuple[str, str]] = []
    close_calls: list[str] = []

    class FakeEngine:
        answer_sdp = None

        async def close(self) -> None:
            close_calls.append("closed")

    async def fake_build(*_args: object, **_kwargs: object):
        return object(), FakeEngine()

    async def fail_preflight(*, output_aspect: str, output_quality: str) -> None:
        preflight_calls.append((output_aspect, output_quality))
        raise HTTPException(
            status_code=503,
            detail={
                "code": "nvidia_vsr_model_load_failed",
                "message": "NVIDIA VSR model files could not be loaded.",
            },
        )

    monkeypatch.setattr(local_stream_cli, "build_engine", fake_build)
    monkeypatch.setattr(
        local_stream_cli,
        "detect_nvidia_video_effects",
        lambda: {"available": True, "reason": "nvvfx available"},
    )
    monkeypatch.setattr(
        local_stream_cli,
        "_prepare_nvidia_vsr_renderer",
        fail_preflight,
        raising=False,
    )
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )
    connected = client.post(
        "/engine-connections",
        json={"engine": {"type": "openai", "api_key": "test-key"}},
    )
    assert connected.status_code == 200
    connection_id = connected.json()["connection_id"]

    response = client.post(
        "/offer",
        json={
            "connection_id": connection_id,
            "sdp": "browser-offer",
            "type": "offer",
            "avatar_id": "avatar",
            "background_id": "background",
            "output_aspect": "16:9",
            "output_quality": "balanced",
            "rtx_super_resolution": True,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "nvidia_vsr_model_load_failed"
    assert preflight_calls == [("16:9", "balanced")]
    disconnected = client.delete(f"/engine-connections/{connection_id}")
    assert disconnected.status_code == 200
    assert close_calls == ["closed"]
