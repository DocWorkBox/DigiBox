from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

MODULE_NAME = "avaturn_live_streamer.conversation_engines.codex_realtime_client"
_OFFER_SDP = "v=0\r\ns=avtr-offer\r\n"
_ANSWER_SDP = "v=0\r\ns=codex-answer\r\n"


def _subject() -> Any:
    """Load the wished-for production module while keeping RED failures explicit."""

    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != MODULE_NAME:
            raise
        pytest.fail(
            "Codex realtime adapter module is missing: "
            "src/avaturn_live_streamer/conversation_engines/codex_realtime_client.py",
            pytrace=False,
        )


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout=5.0))


_FAKE_APP_SERVER = r"""
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
sdp_mode = sys.argv[2]


def record(entry):
    with trace_path.open("a", encoding="utf-8") as trace:
        trace.write(json.dumps(entry, separators=(",", ":")) + "\n")


def respond(request_id, result):
    print(
        json.dumps({"id": request_id, "result": result}, separators=(",", ":")),
        flush=True,
    )


def notify(method, params):
    print(
        json.dumps({"method": method, "params": params}, separators=(",", ":")),
        flush=True,
    )


record({"event": "spawned"})
try:
    for raw_line in sys.stdin:
        message = json.loads(raw_line)
        record({"message": message})
        method = message.get("method")

        if method == "initialized":
            continue
        if method == "initialize":
            respond(
                message["id"],
                {
                    "codexHome": "C:\\fake-codex-home",
                    "platformFamily": "windows",
                    "platformOs": "windows",
                    "userAgent": "fake-codex-app-server/0.137.0",
                },
            )
            if sdp_mode == "server-request":
                print(
                    json.dumps(
                        {
                            "method": "account/chatgptAuthTokens/refresh",
                            "id": 900,
                            "params": {"reason": "test"},
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            continue
        if method == "thread/start":
            cwd = message["params"]["cwd"]
            respond(
                message["id"],
                {
                    "approvalPolicy": "never",
                    "approvalsReviewer": "user",
                    "cwd": cwd,
                    "instructionSources": [],
                    "model": "gpt-live",
                    "modelProvider": "openai",
                    "reasoningEffort": None,
                    "runtimeWorkspaceRoots": [],
                    "sandbox": {"type": "readOnly", "networkAccess": False},
                    "serviceTier": None,
                    "thread": {
                        "cliVersion": "0.137.0",
                        "createdAt": 1,
                        "cwd": cwd,
                        "ephemeral": True,
                        "id": "thread-test",
                        "modelProvider": "openai",
                        "preview": "",
                        "sessionId": "thread-test",
                        "source": "appServer",
                        "status": {"type": "idle"},
                        "turns": [],
                        "updatedAt": 1,
                    },
                },
            )
            continue
        if method == "thread/realtime/start":
            respond(message["id"], {})
            if sdp_mode == "noisy":
                notify(
                    "thread/realtime/sdp",
                    {"threadId": "thread-test"},
                )
                notify(
                    "thread/realtime/sdp",
                    {
                        "threadId": "thread-other",
                        "sdp": "v=0\r\ns=wrong-thread\r\n",
                    },
                )
            notify(
                "thread/realtime/sdp",
                {
                    "threadId": "thread-test",
                    "sdp": "v=0\r\ns=codex-answer\r\n",
                },
            )
            continue
        if method == "thread/realtime/stop":
            respond(message["id"], {})
            continue
        respond(
            message["id"],
            {"unhandledMethod": method},
        )
finally:
    record({"event": "exited"})
"""


def _fake_command(
    trace_path: Path,
    *,
    noisy_sdp: bool = False,
    server_request: bool = False,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        "-c",
        _FAKE_APP_SERVER,
        str(trace_path),
        "server-request" if server_request else "noisy" if noisy_sdp else "clean",
    )


def _trace_entries(trace_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def _sent_messages(trace_path: Path) -> list[dict[str, Any]]:
    return [entry["message"] for entry in _trace_entries(trace_path) if "message" in entry]


def test_client_starts_and_reaps_the_app_server_process(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> None:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        await client.close()
        await client.close()

    _run(scenario())

    lifecycle = [entry["event"] for entry in _trace_entries(trace_path) if "event" in entry]
    assert lifecycle == ["spawned", "exited"]


def test_start_initializes_experimental_api_before_starting_thread(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> None:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        await client.close()

    _run(scenario())

    messages = _sent_messages(trace_path)
    assert [message["method"] for message in messages[:3]] == [
        "initialize",
        "initialized",
        "thread/start",
    ]
    assert messages[0]["params"] == {
        "clientInfo": {
            "name": "avtr1_streamer",
            "title": "AVTR-1 Streamer",
            "version": "0.1.0",
        },
        "capabilities": {"experimentalApi": True},
    }
    assert "id" not in messages[1]
    assert messages[1]["params"] == {}


def test_start_creates_safe_ephemeral_thread_in_workspace(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> str:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        thread_id = client.thread_id
        await client.close()
        return thread_id

    thread_id = _run(scenario())

    thread_start = next(
        message for message in _sent_messages(trace_path) if message["method"] == "thread/start"
    )
    assert thread_id == "thread-test"
    assert thread_start["params"] == {
        "approvalPolicy": "never",
        "cwd": str(tmp_path.resolve()),
        "ephemeral": True,
        "sandbox": "read-only",
        "serviceName": "avtr1-codex-realtime",
    }


def test_realtime_start_exchanges_sdp_over_webrtc(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> str:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        answer_sdp = await client.start_realtime(
            sdp=_OFFER_SDP,
            prompt="Speak briefly.",
            voice="shimmer",
        )
        await client.close()
        return answer_sdp

    answer_sdp = _run(scenario())

    realtime_start = next(
        message
        for message in _sent_messages(trace_path)
        if message["method"] == "thread/realtime/start"
    )
    assert answer_sdp == _ANSWER_SDP
    assert realtime_start["params"] == {
        "threadId": "thread-test",
        "outputModality": "audio",
        "prompt": "Speak briefly.",
        "transport": {"type": "webrtc", "sdp": _OFFER_SDP},
        "version": "v3",
        "voice": "shimmer",
    }


def test_realtime_start_can_disable_cloud_audio_for_local_tts(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> str:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        answer_sdp = await client.start_realtime(
            sdp=_OFFER_SDP,
            prompt="Speak briefly.",
            voice="cove",
            output_modality="text",
        )
        await client.close()
        return answer_sdp

    assert _run(scenario()) == _ANSWER_SDP
    realtime_start = next(
        message
        for message in _sent_messages(trace_path)
        if message["method"] == "thread/realtime/start"
    )
    assert realtime_start["params"]["outputModality"] == "text"
    assert realtime_start["params"]["version"] == "v2"
    assert realtime_start["params"]["model"] == "gpt-realtime-1.5"
    assert "voice" not in realtime_start["params"]


def test_realtime_start_ignores_malformed_and_cross_thread_sdp(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> str:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path, noisy_sdp=True),
            workspace=tmp_path,
        )
        await client.start()
        answer_sdp = await client.start_realtime(
            sdp=_OFFER_SDP,
            prompt="",
            voice="shimmer",
        )
        await client.close()
        return answer_sdp

    assert _run(scenario()) == _ANSWER_SDP


def test_realtime_stop_is_idempotent(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> None:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        await client.start_realtime(sdp=_OFFER_SDP, prompt="", voice="shimmer")
        await client.stop_realtime()
        await client.stop_realtime()
        await client.close()

    _run(scenario())

    stops = [
        message
        for message in _sent_messages(trace_path)
        if message["method"] == "thread/realtime/stop"
    ]
    assert len(stops) == 1
    assert stops[0]["params"] == {"threadId": "thread-test"}


def test_voice_preview_uses_append_speech_without_starting_a_model_turn(
    tmp_path: Path,
) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> None:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path),
            workspace=tmp_path,
        )
        await client.start()
        await client.start_realtime(sdp=_OFFER_SDP, prompt="", voice="maple")
        await client.append_speech("你好，这是当前音色的试听。")
        await client.close()

    _run(scenario())

    preview = next(
        message
        for message in _sent_messages(trace_path)
        if message["method"] == "thread/realtime/appendSpeech"
    )
    assert preview["params"] == {
        "threadId": "thread-test",
        "text": "你好，这是当前音色的试听。",
    }


def test_server_requests_receive_an_explicit_unsupported_error(tmp_path: Path) -> None:
    subject = _subject()
    trace_path = tmp_path / "trace.jsonl"

    async def scenario() -> None:
        client = subject.CodexRealtimeClient(
            command=_fake_command(trace_path, server_request=True),
            workspace=tmp_path,
        )
        await client.start()
        await asyncio.sleep(0.05)
        await client.close()

    _run(scenario())

    server_request_response = next(
        message
        for message in _sent_messages(trace_path)
        if message.get("id") == 900 and "method" not in message
    )
    assert server_request_response["error"]["code"] == -32601


def test_transcript_notifications_preserve_role_and_text() -> None:
    subject = _subject()

    delta = subject.parse_realtime_notification(
        {
            "method": "thread/realtime/transcript/delta",
            "params": {
                "threadId": "thread-test",
                "role": "assistant",
                "delta": "Hel",
            },
        }
    )
    done = subject.parse_realtime_notification(
        {
            "method": "thread/realtime/transcript/done",
            "params": {
                "threadId": "thread-test",
                "role": "assistant",
                "text": "Hello",
            },
        }
    )

    assert delta == subject.CodexTranscriptDelta(
        thread_id="thread-test",
        role="assistant",
        delta="Hel",
    )
    assert done == subject.CodexTranscriptDone(
        thread_id="thread-test",
        role="assistant",
        text="Hello",
    )


def test_lifecycle_notifications_are_typed_and_unknown_messages_are_ignored() -> None:
    subject = _subject()

    started = subject.parse_realtime_notification(
        {
            "method": "thread/realtime/started",
            "params": {
                "threadId": "thread-test",
                "realtimeSessionId": "realtime-1",
                "version": "v2",
            },
        }
    )
    closed = subject.parse_realtime_notification(
        {
            "method": "thread/realtime/closed",
            "params": {"threadId": "thread-test", "reason": "client request"},
        }
    )
    closed_without_reason = subject.parse_realtime_notification(
        {
            "method": "thread/realtime/closed",
            "params": {"threadId": "thread-test", "reason": None},
        }
    )
    error = subject.parse_realtime_notification(
        {
            "method": "thread/realtime/error",
            "params": {"threadId": "thread-test", "message": "upstream failed"},
        }
    )
    unrelated = subject.parse_realtime_notification(
        {"method": "thread/name/updated", "params": {"threadId": "thread-test"}}
    )

    assert started == subject.CodexRealtimeStarted(
        thread_id="thread-test",
        realtime_session_id="realtime-1",
        version="v2",
    )
    assert closed == subject.CodexRealtimeClosed(
        thread_id="thread-test",
        reason="client request",
    )
    assert closed_without_reason == subject.CodexRealtimeClosed(
        thread_id="thread-test",
        reason=None,
    )
    assert error == subject.CodexRealtimeError(
        thread_id="thread-test",
        message="upstream failed",
    )
    assert unrelated is None


def test_realtime_closed_before_sdp_fails_immediately(tmp_path: Path) -> None:
    subject = _subject()
    client = subject.CodexRealtimeClient(command=("codex-test",), workspace=tmp_path)
    client._thread_id = "thread-test"

    with pytest.raises(subject.CodexAppServerError, match="closed before WebRTC answer.*upstream"):
        client._matching_sdp_or_error(
            {
                "method": "thread/realtime/closed",
                "params": {"threadId": "thread-test", "reason": "upstream"},
            }
        )


def test_runtime_realtime_closed_marks_session_inactive_before_yield(
    tmp_path: Path,
) -> None:
    subject = _subject()

    async def exercise() -> None:
        client = subject.CodexRealtimeClient(command=("codex-test",), workspace=tmp_path)
        client._thread_id = "thread-test"
        client._realtime_active = True
        await client._notifications.put(
            {
                "method": "thread/realtime/closed",
                "params": {"threadId": "thread-test", "reason": "remote close"},
            }
        )
        notifications = client.notifications()
        event = await anext(notifications)
        assert isinstance(event, subject.CodexRealtimeClosed)
        assert client._realtime_active is False
        await notifications.aclose()

    _run(exercise())
