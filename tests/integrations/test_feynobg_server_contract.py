from __future__ import annotations

import gc
import io
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image


class _FakeFeyNoBg:
    model_id = "feyninc/FeyNobg"
    device = "cpu"
    loaded = True
    released = False
    active_requests = 0

    def cutout(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.new("L", rgba.size, 143))
        return rgba

    def release(self) -> dict[str, object]:
        self.loaded = False
        self.released = True
        return {
            "service": "feynobg",
            "status": "released",
            "released": True,
            "loaded": False,
            "active_requests": 0,
        }


def test_health_and_cutout_contract_return_rgba_png() -> None:
    from avaturn_live_streamer.integrations.feynobg_server import create_app

    client = TestClient(create_app(_FakeFeyNoBg()))
    source = io.BytesIO()
    Image.new("RGB", (37, 19), (40, 80, 120)).save(source, format="PNG")

    health = client.get("/health")
    response = client.post(
        "/v1/cutout",
        files={"file": ("person.png", source.getvalue(), "image/png")},
    )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "feynobg",
        "model": "feyninc/FeyNobg",
        "device": "cpu",
        "loaded": True,
        "released": False,
        "active_requests": 0,
    }
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    cutout = Image.open(io.BytesIO(response.content))
    assert cutout.mode == "RGBA"
    assert cutout.size == (37, 19)
    assert cutout.getchannel("A").getextrema() == (143, 143)


def test_invalid_upload_is_rejected_without_calling_model() -> None:
    from avaturn_live_streamer.integrations.feynobg_server import create_app

    response = TestClient(create_app(_FakeFeyNoBg())).post(
        "/v1/cutout",
        files={"file": ("bad.bin", b"not-an-image", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "图片" in response.json()["detail"]


def test_release_drops_model_and_processor_cleans_cuda_and_can_lazy_reload(
    monkeypatch,
) -> None:
    from avaturn_live_streamer.integrations import feynobg_server as module

    service = module.FeyNoBgService(
        model_id="local/FeyNoBg",
        revision="test-revision",
        device="cuda",
    )
    service._model = object()
    service._processor = object()
    cleanup_calls: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: cleanup_calls.append("gc") or 0)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: cleanup_calls.append("empty_cache"),
                ipc_collect=lambda: cleanup_calls.append("ipc_collect"),
            )
        ),
    )

    result = service.release()

    assert result == {
        "service": "feynobg",
        "status": "released",
        "released": True,
        "loaded": False,
        "active_requests": 0,
    }
    assert cleanup_calls == ["gc", "empty_cache", "ipc_collect"]
    assert service.loaded is False
    assert service.released is True

    class LoadedModel:
        def eval(self):
            return self

        def to(self, device):
            assert device == "cuda"
            return self

    loaded_model = LoadedModel()
    loaded_processor = object()
    monkeypatch.setitem(
        sys.modules,
        "nobg",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: loaded_model),
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: loaded_processor
            ),
        ),
    )

    service._ensure_loaded()

    assert service._model is loaded_model
    assert service._processor is loaded_processor
    assert service.released is False


def test_release_route_is_loopback_only() -> None:
    from avaturn_live_streamer.integrations.feynobg_server import create_app

    service = _FakeFeyNoBg()
    app = create_app(service)
    denied = TestClient(app, client=("203.0.113.10", 52000)).post("/release")
    allowed = TestClient(app, client=("::1", 52001)).post("/release")

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["service"] == "feynobg"
    assert allowed.json()["released"] is True


def test_release_refuses_to_run_while_cutout_owns_the_model_lock() -> None:
    from avaturn_live_streamer.integrations import feynobg_server as module

    service = module.FeyNoBgService(
        model_id="local/FeyNoBg",
        revision="test-revision",
        device="cpu",
    )
    lock_held = threading.Event()
    may_finish = threading.Event()

    def hold_active_request() -> None:
        with service._lock:
            service._active_requests += 1
            lock_held.set()
            assert may_finish.wait(timeout=2)
            service._active_requests -= 1

    worker = threading.Thread(target=hold_active_request)
    worker.start()
    assert lock_held.wait(timeout=2)
    try:
        with pytest.raises(module.ModelBusyError, match="active"):
            service.release()
    finally:
        may_finish.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
