from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from avtr1_renderer.api import app as subject


def test_avatar_preview_route_returns_png_from_registry(monkeypatch) -> None:
    avatar = SimpleNamespace(id="portrait")
    png = b"\x89PNG\r\n\x1a\npreview"
    calls = []

    def fake_encode(value):
        calls.append(value)
        return png

    monkeypatch.setattr(subject, "encode_avatar_preview_png", fake_encode, raising=False)
    monkeypatch.setattr(
        subject.app.state,
        "registry",
        {"portrait": avatar},
        raising=False,
    )

    response = TestClient(subject.app, raise_server_exceptions=False).get(
        "/avatars/portrait/preview"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == png
    assert calls == [avatar]


def test_avatar_preview_route_rejects_unknown_avatar(monkeypatch) -> None:
    monkeypatch.setattr(subject.app.state, "registry", {}, raising=False)

    response = TestClient(subject.app, raise_server_exceptions=False).get(
        "/avatars/missing/preview"
    )

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]
