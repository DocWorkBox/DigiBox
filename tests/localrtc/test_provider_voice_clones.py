from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from avaturn_live_streamer import local_stream_cli


def _client(host: str = "127.0.0.1") -> TestClient:
    return TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=(host, 50000),
        raise_server_exceptions=False,
    )


def _aliyun_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": {
            "kind": "aliyun_bailian",
            "api_key": "dashscope-key",
            "tts_model": "cosyvoice-v3-flash",
            "tts_voice": "longxiaochun",
        },
        "audio_url": "https://cdn.example.com/reference.wav",
        "prefix": "Avatar1",
        "consent": True,
    }
    payload.update(overrides)
    return payload


def test_aliyun_voice_clone_requires_consent_and_uses_public_url(monkeypatch) -> None:
    calls: list[object] = []

    async def fake_clone(provider, *, audio_url, prefix):
        calls.append((provider.kind, audio_url, prefix))
        return SimpleNamespace(voice_id="cosy-clone-id", preview_url=None)

    monkeypatch.setattr(local_stream_cli, "clone_cosyvoice", fake_clone)
    response = _client().post("/provider-voice-clones", json=_aliyun_payload())

    assert response.status_code == 200
    assert response.json() == {
        "id": "cosy-clone-id",
        "type": "voice",
        "status": "deploying",
    }
    assert calls == [
        ("aliyun_bailian", "https://cdn.example.com/reference.wav", "Avatar1")
    ]

    rejected = _client().post(
        "/provider-voice-clones",
        json=_aliyun_payload(consent=False),
    )
    assert rejected.status_code == 422


@pytest.mark.parametrize(
    ("filename", "submitted_type", "expected_type"),
    [
        ("reference.wav", "application/octet-stream", "audio/wav"),
        ("reference.mp3", "audio/mpeg", "audio/mpeg"),
        ("reference.m4a", "audio/mp4", "audio/mp4"),
    ],
)
def test_aliyun_voice_clone_accepts_local_audio_uploads(
    monkeypatch,
    filename: str,
    submitted_type: str,
    expected_type: str,
) -> None:
    calls: list[object] = []

    async def fake_clone(
        provider,
        *,
        filename,
        content_type,
        audio,
        prefix,
    ):
        calls.append(
            (provider.kind, filename, content_type, audio, prefix)
        )
        return SimpleNamespace(voice_id="cosy-local-id", preview_url=None)

    monkeypatch.setattr(local_stream_cli, "clone_cosyvoice", fake_clone)
    response = _client().post(
        "/provider-voice-clones",
        data={
            "provider": json.dumps(
                {
                    "kind": "aliyun_bailian",
                    "api_key": "dashscope-key",
                    "tts_model": "cosyvoice-v3-flash",
                    "tts_voice": "longxiaochun",
                }
            ),
            "consent": "true",
            "prefix": "Avatar1",
        },
        files={
            "reference_audio": (
                filename,
                b"audio-bytes",
                submitted_type,
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "cosy-local-id",
        "type": "voice",
        "status": "deploying",
    }
    assert calls == [
        (
            "aliyun_bailian",
            filename,
            expected_type,
            b"audio-bytes",
            "Avatar1",
        )
    ]


def test_aliyun_voice_clone_rejects_oversized_or_wrong_format_files() -> None:
    data = {
        "provider": json.dumps(
            {"kind": "aliyun_bailian", "api_key": "dashscope-key"}
        ),
        "consent": "true",
        "prefix": "Avatar1",
    }

    wrong_format = _client().post(
        "/provider-voice-clones",
        data=data,
        files={"reference_audio": ("reference.txt", b"audio", "text/plain")},
    )
    assert wrong_format.status_code == 422

    oversized = _client().post(
        "/provider-voice-clones",
        data=data,
        files={
            "reference_audio": (
                "reference.wav",
                b"0" * (10 * 1024 * 1024 + 1),
                "audio/wav",
            )
        },
    )
    assert oversized.status_code == 413


@pytest.mark.parametrize(
    "audio_url",
    [
        "http://127.0.0.1/reference.wav",
        "http://169.254.169.254/reference.wav",
        "https://user:password@cdn.example.com/reference.wav",
        "https://localhost/reference.wav",
    ],
)
def test_aliyun_voice_clone_rejects_non_public_urls(audio_url: str) -> None:
    response = _client().post(
        "/provider-voice-clones",
        json=_aliyun_payload(audio_url=audio_url),
    )

    assert response.status_code == 422


def test_minimax_voice_clone_uploads_local_audio_only_from_loopback(monkeypatch) -> None:
    calls: list[object] = []

    async def fake_clone(
        provider,
        *,
        filename,
        content_type,
        audio,
        voice_id,
        preview_text,
    ):
        calls.append(
            (
                provider.kind,
                filename,
                content_type,
                audio,
                voice_id,
                preview_text,
            )
        )
        return SimpleNamespace(
            voice_id=voice_id,
            preview_url="https://cdn.minimaxi.com/preview.mp3",
        )

    monkeypatch.setattr(local_stream_cli, "clone_minimax_voice", fake_clone)
    data = {
        "provider": json.dumps(
            {
                "kind": "minimax",
                "api_key": "minimax-key",
                "realtime_model": "abab6.5s-chat",
                "voice": "male-qn-qingse",
            }
        ),
        "consent": "true",
        "voice_id": "AvatarVoice001",
        "preview_text": "This is a preview.",
    }
    files = {
        "reference_audio": (
            "reference.wav",
            b"audio-bytes",
            "application/octet-stream",
        )
    }

    response = _client().post(
        "/provider-voice-clones",
        data=data,
        files=files,
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "AvatarVoice001",
        "type": "voice",
        "status": "ready",
        "preview_url": "https://cdn.minimaxi.com/preview.mp3",
    }
    assert calls == [
        (
            "minimax",
            "reference.wav",
            "audio/wav",
            b"audio-bytes",
            "AvatarVoice001",
            "This is a preview.",
        )
    ]

    remote = _client("192.0.2.20").post(
        "/provider-voice-clones",
        data=data,
        files=files,
    )
    assert remote.status_code == 403


def test_minimax_voice_clone_rejects_oversized_or_wrong_format_files() -> None:
    data = {
        "provider": json.dumps({"kind": "minimax", "api_key": "minimax-key"}),
        "consent": "true",
        "voice_id": "AvatarVoice001",
    }

    wrong_format = _client().post(
        "/provider-voice-clones",
        data=data,
        files={"reference_audio": ("reference.txt", b"audio", "text/plain")},
    )
    assert wrong_format.status_code == 422

    oversized = _client().post(
        "/provider-voice-clones",
        data=data,
        files={
            "reference_audio": (
                "reference.wav",
                b"0" * (20 * 1024 * 1024 + 1),
                "audio/wav",
            )
        },
    )
    assert oversized.status_code == 413


def test_validation_errors_never_echo_provider_secrets() -> None:
    secret = "must-never-appear"
    response = _client().post(
        "/engine-connections",
        json={
            "engine": {
                "type": "custom_api",
                "provider": {"kind": "not-a-provider", "api_key": secret},
            }
        },
    )

    assert response.status_code == 422
    assert secret not in response.text


def test_malformed_clone_json_is_a_safe_client_error() -> None:
    response = _client().post(
        "/provider-voice-clones",
        content=b'{"provider":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code in {400, 422}
    assert "Internal Server Error" not in response.text


def test_aliyun_voice_inventory_query_returns_stable_ui_shape(monkeypatch) -> None:
    calls: list[object] = []

    async def fake_list(provider):
        calls.append(provider)
        return [
            SimpleNamespace(
                id="cosyvoice-v3-flash-avatar-a1",
                status="OK",
                compatible=True,
                created_at="2026-08-10 12:00:00",
                modified_at="2026-08-10 12:01:00",
            ),
            SimpleNamespace(
                id="cosyvoice-v2-archive-b2",
                status="UNDEPLOYED",
                compatible=False,
                created_at=None,
                modified_at=None,
            ),
        ]

    monkeypatch.setattr(
        local_stream_cli,
        "list_cosyvoice_voices",
        fake_list,
        raising=False,
    )
    response = _client().post(
        "/provider-voice-clones/query",
        json={
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
                "tts_model": "cosyvoice-v3-flash",
                "tts_voice": "longanhuan_v3",
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "target_model": "cosyvoice-v3-flash",
        "voices": [
            {
                "id": "cosyvoice-v3-flash-avatar-a1",
                "status": "OK",
                "compatible": True,
                "created_at": "2026-08-10 12:00:00",
                "modified_at": "2026-08-10 12:01:00",
            },
            {
                "id": "cosyvoice-v2-archive-b2",
                "status": "UNDEPLOYED",
                "compatible": False,
                "created_at": None,
                "modified_at": None,
            },
        ],
    }
    assert len(calls) == 1
    assert calls[0].tts_model == "cosyvoice-v3-flash"


def test_aliyun_voice_inventory_query_is_loopback_only(monkeypatch) -> None:
    called = False

    async def fake_list(_provider):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        local_stream_cli,
        "list_cosyvoice_voices",
        fake_list,
        raising=False,
    )
    response = _client("192.0.2.20").post(
        "/provider-voice-clones/query",
        json={
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": "dashscope-key",
            }
        },
    )

    assert response.status_code == 403
    assert called is False


def test_aliyun_voice_inventory_query_redacts_api_key_from_errors(monkeypatch) -> None:
    secret = "inventory-secret-must-not-leak"

    async def fake_list(_provider):
        raise RuntimeError(f"upstream rejected Bearer {secret}")

    monkeypatch.setattr(
        local_stream_cli,
        "list_cosyvoice_voices",
        fake_list,
        raising=False,
    )
    response = _client().post(
        "/provider-voice-clones/query",
        json={
            "provider": {
                "kind": "aliyun_bailian",
                "api_key": secret,
            }
        },
    )

    assert response.status_code == 400
    assert secret not in response.text
    assert "***" in response.text
