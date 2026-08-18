from __future__ import annotations

import asyncio
import json

import httpx

from avaturn_live_streamer.conversation_engines.aliyun_bailian_client import (
    clone_cosyvoice,
)
from avaturn_live_streamer.conversation_engines.builders import (
    AliyunBailianProvider,
    MiniMaxProvider,
)
from avaturn_live_streamer.conversation_engines.minimax_voice_client import (
    clone_minimax_voice,
)


def test_cosyvoice_clone_uses_public_url_and_returns_voice_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"output": {"voice_id": "cosy-clone-id"}})

    async def exercise():
        return await clone_cosyvoice(
            AliyunBailianProvider(
                api_key="dashscope-key",
                tts_model="cosyvoice-v3-flash",
                tts_voice="longxiaochun",
            ),
            audio_url="https://cdn.example.com/reference.wav",
            prefix="avatar1",
            transport=httpx.MockTransport(handler),
        )

    result = asyncio.run(exercise())

    assert result.voice_id == "cosy-clone-id"
    assert result.preview_url is None
    request = requests[0]
    assert request.url == (
        "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    )
    assert request.headers["Authorization"] == "Bearer dashscope-key"
    assert request.headers["X-DashScope-OssResourceResolve"] == "enable"
    assert json.loads(request.content) == {
        "model": "voice-enrollment",
        "input": {
            "action": "create_voice",
            "target_model": "cosyvoice-v3-flash",
            "prefix": "avatar1",
            "url": "https://cdn.example.com/reference.wav",
        },
    }


def test_cosyvoice_clone_uploads_local_audio_to_temporary_oss() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/uploads":
            assert request.method == "GET"
            assert request.url.host == "dashscope.aliyuncs.com"
            assert dict(request.url.params) == {
                "action": "getPolicy",
                "model": "voice-enrollment",
            }
            return httpx.Response(
                200,
                json={
                    "data": {
                        "policy": "encoded-policy",
                        "signature": "signed-policy",
                        "upload_dir": "dashscope-instant/account/request",
                        "upload_host": "https://dashscope-file.oss-cn-beijing.aliyuncs.com",
                        "oss_access_key_id": "temporary-access-key",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                    }
                },
            )
        if request.url.host == "dashscope-file.oss-cn-beijing.aliyuncs.com":
            assert request.method == "POST"
            assert "multipart/form-data" in request.headers["content-type"]
            assert b'name="OSSAccessKeyId"' in request.content
            assert b"temporary-access-key" in request.content
            assert b'name="policy"' in request.content
            assert b"encoded-policy" in request.content
            assert b'name="Signature"' in request.content
            assert b"signed-policy" in request.content
            assert b'name="key"' in request.content
            assert b"dashscope-instant/account/request/reference.wav" in request.content
            assert b'name="x-oss-object-acl"' in request.content
            assert b'name="x-oss-forbid-overwrite"' in request.content
            assert b'name="success_action_status"' in request.content
            assert b'name="file"; filename="reference.wav"' in request.content
            assert b"audio-bytes" in request.content
            return httpx.Response(200)
        if request.url.path == "/api/v1/services/audio/tts/customization":
            assert request.url.host == (
                "workspace-1.cn-beijing.maas.aliyuncs.com"
            )
            assert request.headers["X-DashScope-OssResourceResolve"] == "enable"
            assert json.loads(request.content)["input"]["url"] == (
                "oss://dashscope-instant/account/request/reference.wav"
            )
            return httpx.Response(200, json={"output": {"voice_id": "local-clone-id"}})
        return httpx.Response(404)

    async def exercise():
        return await clone_cosyvoice(
            AliyunBailianProvider(
                api_key="dashscope-key",
                workspace_id="workspace-1",
                tts_model="cosyvoice-v3-flash",
                tts_voice="longxiaochun",
            ),
            filename="reference.wav",
            content_type="audio/wav",
            audio=b"audio-bytes",
            prefix="avatar1",
            transport=httpx.MockTransport(handler),
        )

    result = asyncio.run(exercise())

    assert result.voice_id == "local-clone-id"
    assert [request.url.path for request in requests] == [
        "/api/v1/uploads",
        "/",
        "/api/v1/services/audio/tts/customization",
    ]


def test_minimax_clone_uploads_local_file_then_creates_voice() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/files/upload":
            assert "multipart/form-data" in request.headers["content-type"]
            assert b'name="purpose"' in request.content
            assert b"voice_clone" in request.content
            assert b'name="file"' in request.content
            assert b"reference.wav" in request.content
            assert b"audio-bytes" in request.content
            return httpx.Response(
                200,
                json={
                    "file": {"file_id": 123456},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                },
            )
        if request.url.path == "/v1/voice_clone":
            return httpx.Response(
                200,
                json={
                    "demo_audio": "https://cdn.minimaxi.com/preview.mp3",
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                },
            )
        return httpx.Response(404)

    async def exercise():
        return await clone_minimax_voice(
            MiniMaxProvider(api_key="minimax-key"),
            filename="reference.wav",
            content_type="audio/wav",
            audio=b"audio-bytes",
            voice_id="AvatarVoice001",
            preview_text="This is a preview.",
            transport=httpx.MockTransport(handler),
        )

    result = asyncio.run(exercise())

    assert result.voice_id == "AvatarVoice001"
    assert result.preview_url == "https://cdn.minimaxi.com/preview.mp3"
    assert len(requests) == 2
    assert {request.url.host for request in requests} == {"api.minimaxi.com"}
    assert {request.headers["Authorization"] for request in requests} == {
        "Bearer minimax-key"
    }
    assert json.loads(requests[1].content) == {
        "file_id": 123456,
        "voice_id": "AvatarVoice001",
        "text": "This is a preview.",
        "model": "speech-2.8-turbo",
    }
