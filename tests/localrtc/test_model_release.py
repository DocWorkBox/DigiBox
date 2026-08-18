from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from avaturn_live_streamer import local_stream_cli


class _FakeResponse:
    def __init__(self, service: str) -> None:
        self.status_code = 200
        self._service = service
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "service": self._service,
            "status": "released",
            "released": True,
            "loaded": False,
        }


def test_release_models_proxies_to_all_local_workers(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url: str):
            calls.append(url)
            service = (
                "renderer"
                if ":8000" in url
                else "cosyvoice"
                if ":8768" in url
                else "feynobg"
            )
            return _FakeResponse(service)

    monkeypatch.setattr(
        local_stream_cli,
        "_renderer_base_url",
        lambda: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(local_stream_cli.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("AVTR1_COSYVOICE_PORT", "8768")
    monkeypatch.setenv("AVTR1_FEYNOBG_PORT", "8767")
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
    )

    response = client.post("/system/release-models")

    assert response.status_code == 200
    assert response.json()["status"] == "released"
    assert set(response.json()["services"]) == {"renderer", "cosyvoice", "feynobg"}
    assert set(calls) == {
        "http://127.0.0.1:8000/release",
        "http://127.0.0.1:8768/release",
        "http://127.0.0.1:8767/release",
    }


def test_release_models_is_loopback_only() -> None:
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("192.0.2.15", 50000),
    )

    response = client.post("/system/release-models")

    assert response.status_code == 403


def test_release_waits_for_recently_closed_session_to_leave_the_slot() -> None:
    class ClosingSlot:
        def __init__(self) -> None:
            self.calls = 0

        async def is_active(self) -> bool:
            self.calls += 1
            return self.calls < 3

    slot = ClosingSlot()

    idle = asyncio.run(
        local_stream_cli._wait_for_session_idle(
            slot,
            timeout_seconds=0.2,
            poll_seconds=0,
        )
    )

    assert idle is True
    assert slot.calls == 3
