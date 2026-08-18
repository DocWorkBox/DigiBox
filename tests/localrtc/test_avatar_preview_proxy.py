from __future__ import annotations

from typing import ClassVar

from fastapi.testclient import TestClient

from avaturn_live_streamer import local_stream_cli


def test_avatar_preview_proxy_quotes_id_and_preserves_png_response(monkeypatch) -> None:
    requested_urls = []
    png = b"\x89PNG\r\n\x1a\nproxy-preview"

    class FakeResponse:
        status_code = 200
        content = png
        headers: ClassVar[dict[str, str]] = {"content-type": "image/png; charset=binary"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            _ = args

        async def get(self, url):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(local_stream_cli, "_renderer_base_url", lambda: "http://renderer")
    monkeypatch.setattr(local_stream_cli.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        raise_server_exceptions=False,
    )

    response = client.get("/avatars/person%20one%3F%23/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == png
    assert requested_urls == ["http://renderer/avatars/person%20one%3F%23/preview"]


def test_avatar_preview_proxy_requires_renderer_url(monkeypatch) -> None:
    monkeypatch.setattr(local_stream_cli, "_renderer_base_url", lambda: None)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        raise_server_exceptions=False,
    )

    assert client.get("/avatars/portrait/preview").status_code == 503


def test_avatar_idle_loop_proxy_quotes_id_and_preserves_webp_response(monkeypatch) -> None:
    requested_urls = []
    webp = b"RIFF\x10\x00\x00\x00WEBPidle"

    class FakeResponse:
        status_code = 200
        content = webp
        headers: ClassVar[dict[str, str]] = {"content-type": "image/webp"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            _ = args

        async def get(self, url):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(local_stream_cli, "_renderer_base_url", lambda: "http://renderer")
    monkeypatch.setattr(local_stream_cli.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        raise_server_exceptions=False,
    )

    response = client.get("/avatars/person%20one%3F%23/idle-loop")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == webp
    assert requested_urls == ["http://renderer/avatars/person%20one%3F%23/idle-loop"]


def test_avatar_delete_proxy_quotes_id_and_forwards_delete(monkeypatch) -> None:
    requested_urls = []

    class FakeResponse:
        status_code = 200
        content = b'{"deleted":true}'
        headers: ClassVar[dict[str, str]] = {"content-type": "application/json"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            _ = args

        async def delete(self, url):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(local_stream_cli, "_renderer_base_url", lambda: "http://renderer")
    monkeypatch.setattr(local_stream_cli, "_is_loopback_client", lambda _host: True)
    monkeypatch.setattr(local_stream_cli.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        raise_server_exceptions=False,
    )

    response = client.delete("/assets/avatar/user_person%3F%23_123456789abc")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert requested_urls == [
        "http://renderer/assets/avatar/user_person%3F%23_123456789abc"
    ]
