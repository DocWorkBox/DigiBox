from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from avaturn_live_streamer import local_stream_cli
from avaturn_live_streamer.conversation_engines.custom_api_client import (
    ComponentPreflight,
    CustomAPIPreflightReport,
)


def test_turn_metrics_endpoint_is_loopback_only_and_starts_empty() -> None:
    app = local_stream_cli._make_app(idle_timeout=60, max_duration=300)
    local = TestClient(
        app,
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )
    remote = TestClient(
        app,
        client=("192.0.2.10", 50000),
        raise_server_exceptions=False,
    )

    response = local.get("/turn-metrics")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": None,
        "active": False,
        "turns": [],
        "summary": {"fast_slo_complete_turns": 0, "stages_ms": {}},
    }
    assert remote.get("/turn-metrics").status_code == 403


def test_browser_first_audible_report_is_loopback_only_and_safe_without_a_turn() -> None:
    app = local_stream_cli._make_app(idle_timeout=60, max_duration=300)
    local = TestClient(
        app,
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )
    remote = TestClient(
        app,
        client=("192.0.2.10", 50000),
        raise_server_exceptions=False,
    )

    response = local.post("/turn-metrics/browser-first-audible")

    assert response.status_code == 200
    assert response.json() == {"accepted": False, "turn_id": None}
    assert remote.post("/turn-metrics/browser-first-audible").status_code == 403


def test_codex_connection_is_prepared_and_previewed_before_offer(monkeypatch) -> None:
    calls: list[object] = []

    class FakeCodexEngine:
        answer_sdp = "codex-answer"

        async def preview_speech(self, text: str) -> None:
            calls.append(("preview", text))

        async def close(self) -> None:
            calls.append("closed")

    fake_engine = FakeCodexEngine()

    async def fake_build(options, *, stream_id, codex_sdp=None):
        calls.append((options.type, stream_id, codex_sdp))
        return object(), fake_engine

    monkeypatch.setattr(local_stream_cli, "build_engine", fake_build)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    connected = client.post(
        "/engine-connections",
        json={
            "engine": {"type": "codex", "voice": "maple", "prompt": "简短回答。"},
            "codex_sdp": "codex-offer",
        },
    )

    assert connected.status_code == 200
    payload = connected.json()
    assert payload["status"] == "ready"
    assert payload["engine_type"] == "codex"
    assert payload["codex_sdp"] == "codex-answer"
    assert payload["connection_id"]
    assert calls == [("codex", "prepared", "codex-offer")]

    preview = client.post(
        f'/engine-connections/{payload["connection_id"]}/preview',
        json={"text": "你好，这是当前音色的试听。"},
    )
    assert preview.status_code == 200
    assert calls[-1] == ("preview", "你好，这是当前音色的试听。")

    disconnected = client.delete(f'/engine-connections/{payload["connection_id"]}')
    assert disconnected.status_code == 200
    assert calls[-1] == "closed"


def test_engine_connections_are_restricted_to_the_windows_host(monkeypatch) -> None:
    async def forbidden_build(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("remote request reached engine construction")

    monkeypatch.setattr(local_stream_cli, "build_engine", forbidden_build)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("192.0.2.10", 50000),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/engine-connections",
        json={"engine": {"type": "codex"}, "codex_sdp": "offer"},
    )

    assert response.status_code == 403


def test_offer_can_reference_a_prepared_connection_without_resending_secrets() -> None:
    fields = local_stream_cli._OfferBody.model_fields

    assert "connection_id" in fields
    body = local_stream_cli._OfferBody.model_validate(
        {
            "connection_id": "054a1911-f375-4c5b-a352-432f9e1b5432",
            "sdp": "avtr-offer",
            "type": "offer",
            "avatar_id": "avatar",
            "background_id": "background",
        }
    )

    assert body.engine is None
    assert str(body.connection_id) == "054a1911-f375-4c5b-a352-432f9e1b5432"


def test_disconnected_offer_releases_slot_before_next_offer(monkeypatch) -> None:
    peers = []
    engines = []
    disconnect_checks = iter((True, False))

    class FakePeerConnection:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs
            self.localDescription = SimpleNamespace(sdp="answer", type="answer")
            self.closed = False
            peers.append(self)

        async def setRemoteDescription(self, description) -> None:
            _ = description

        async def createAnswer(self):
            return object()

        async def setLocalDescription(self, description) -> None:
            _ = description

        async def close(self) -> None:
            self.closed = True

    class FakeLocalRTC:
        def __init__(self, pc, **kwargs) -> None:
            _ = kwargs
            self.pc = pc

        async def close(self) -> None:
            await self.pc.close()

    class FakeEngine:
        answer_sdp = None

        def __init__(self) -> None:
            self.closed = False
            engines.append(self)

        async def close(self) -> None:
            self.closed = True

    async def fake_build_engine(*args, **kwargs):
        _ = args, kwargs
        return object(), FakeEngine()

    async def fake_run_session(**kwargs) -> None:
        await kwargs["peer"].close()
        await local_stream_cli._close_built_engine(kwargs["built_engine"])

    async def fake_is_disconnected(_request) -> bool:
        return next(disconnect_checks)

    async def no_ice_servers():
        return []

    monkeypatch.setattr(local_stream_cli, "RTCPeerConnection", FakePeerConnection)
    monkeypatch.setattr(local_stream_cli, "LocalRTC", FakeLocalRTC)
    monkeypatch.setattr(local_stream_cli, "build_engine", fake_build_engine)
    monkeypatch.setattr(local_stream_cli, "_run_session", fake_run_session)
    monkeypatch.setattr(local_stream_cli, "resolve_ice_servers", no_ice_servers)
    monkeypatch.setattr(local_stream_cli.Request, "is_disconnected", fake_is_disconnected)

    offer = {
        "engine": {"type": "openai", "api_key": "test-key"},
        "sdp": "avtr-offer",
        "type": "offer",
        "avatar_id": "avatar",
        "background_id": "background",
    }
    with TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    ) as client:
        disconnected = client.post("/offer", json=offer)
        next_offer = client.post("/offer", json=offer)

        assert disconnected.status_code == 499
        assert next_offer.status_code == 200
        assert peers[0].closed is True
        assert engines[0].closed is True


def test_custom_api_preconnection_reuses_the_engine_that_passed_preflight(
    monkeypatch,
) -> None:
    calls: list[object] = []

    class FakeEngine:
        async def close(self) -> None:
            calls.append("closed")

    fake_engine = FakeEngine()

    async def fake_prepare(config):
        calls.append(config.provider.kind)
        return SimpleNamespace(
            report=CustomAPIPreflightReport(
                status="ready",
                components={
                    "realtime": ComponentPreflight(status="ready", latency_ms=4)
                },
            ),
            engine=fake_engine,
        )

    monkeypatch.setattr(local_stream_cli, "prepare_custom_api", fake_prepare)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    connected = client.post(
        "/engine-connections",
        json={
            "engine": {
                "type": "custom_api",
                "provider": {
                    "kind": "minimax",
                    "api_key": "minimax-key",
                    "realtime_model": "abab6.5s-chat",
                    "voice": "male-qn-qingse",
                },
            }
        },
    )

    assert connected.status_code == 200
    payload = connected.json()
    assert payload["components"] == {
        "realtime": {"status": "ready", "latency_ms": 4, "error": None}
    }
    assert calls == ["minimax"]
    disconnected = client.delete(f'/engine-connections/{payload["connection_id"]}')
    assert disconnected.status_code == 200
    assert calls == ["minimax", "closed"]


def test_engine_connection_runtime_errors_redact_openai_key_from_body_and_log(
    monkeypatch,
) -> None:
    secret = "openai-must-not-leak"
    logged: list[dict[str, object]] = []

    async def failing_build(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError(f"authentication failed for {secret}")

    class FakeLogger:
        def warning(self, _message: str, **kwargs: object) -> None:
            logged.append(kwargs)

    monkeypatch.setattr(local_stream_cli, "build_engine", failing_build)
    monkeypatch.setattr(local_stream_cli, "_LOGGER", FakeLogger())
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/engine-connections",
        json={"engine": {"type": "openai", "api_key": secret}},
    )

    assert response.status_code == 400
    assert secret not in response.text
    assert secret not in str(logged)


def test_direct_offer_runtime_errors_redact_openai_key_from_body_and_log(
    monkeypatch,
) -> None:
    secret = "offer-key-must-not-leak"
    logged: list[dict[str, object]] = []

    async def failing_build(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError(f"authentication failed for {secret}")

    class FakeLogger:
        def warning(self, _message: str, **kwargs: object) -> None:
            logged.append(kwargs)

    monkeypatch.setattr(local_stream_cli, "build_engine", failing_build)
    monkeypatch.setattr(local_stream_cli, "_LOGGER", FakeLogger())
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/offer",
        json={
            "engine": {"type": "openai", "api_key": secret},
            "sdp": "offer",
            "type": "offer",
            "avatar_id": "avatar",
            "background_id": "background",
        },
    )

    assert response.status_code == 400
    assert secret not in response.text
    assert secret not in str(logged)
