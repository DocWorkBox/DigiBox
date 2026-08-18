# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Experimental Codex app-server WebRTC signalling client.

The ChatGPT-authenticated Codex realtime path only accepts WebRTC media.  This
module keeps app-server and its credentials on the local machine: the browser
provides an SDP offer, app-server creates the upstream call, and only the SDP
answer and typed lifecycle/transcript events leave this client.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.local_tts_bridge import (
    StreamingLocalTTSBridge,
    StreamingTTSBackend,
)
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    CodexAssistantAudioReceived,
    CodexAssistantControlReceived,
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TextEchoEnqueueText,
)
from avaturn_live_streamer.management.types import SegmentId
from avaturn_live_streamer.utils.datetime import tzutcnow

_LOGGER = get_logger()

_REQUEST_TIMEOUT_SECONDS = 30.0
_PROCESS_EXIT_TIMEOUT_SECONDS = 5.0
_TEXT_REALTIME_MODEL = "gpt-realtime-1.5"


class CodexAppServerError(RuntimeError):
    """Raised when app-server rejects a request or its realtime backend fails."""


@dataclass(frozen=True, slots=True)
class CodexTranscriptDelta:
    thread_id: str
    role: str
    delta: str


@dataclass(frozen=True, slots=True)
class CodexTranscriptDone:
    thread_id: str
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class CodexRealtimeStarted:
    thread_id: str
    realtime_session_id: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class CodexRealtimeClosed:
    thread_id: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class CodexRealtimeError:
    thread_id: str
    message: str


type CodexRealtimeNotification = (
    CodexTranscriptDelta
    | CodexTranscriptDone
    | CodexRealtimeStarted
    | CodexRealtimeClosed
    | CodexRealtimeError
)


def parse_realtime_notification(message: dict[str, Any]) -> CodexRealtimeNotification | None:
    """Parse the stable subset of experimental ``thread/realtime/*`` events."""

    method = message.get("method")
    params = message.get("params")
    if not isinstance(params, dict):
        return None

    thread_id = params.get("threadId")
    if not isinstance(thread_id, str) or not thread_id:
        return None

    match method:
        case "thread/realtime/transcript/delta":
            role = params.get("role")
            delta = params.get("delta")
            if isinstance(role, str) and isinstance(delta, str):
                return CodexTranscriptDelta(thread_id=thread_id, role=role, delta=delta)
        case "thread/realtime/transcript/done":
            role = params.get("role")
            text = params.get("text")
            if isinstance(role, str) and isinstance(text, str):
                return CodexTranscriptDone(thread_id=thread_id, role=role, text=text)
        case "thread/realtime/started":
            session_id = params.get("realtimeSessionId")
            version = params.get("version")
            if isinstance(session_id, str):
                return CodexRealtimeStarted(
                    thread_id=thread_id,
                    realtime_session_id=session_id,
                    version=version if isinstance(version, str) else None,
                )
        case "thread/realtime/closed":
            reason = params.get("reason")
            if reason is None or isinstance(reason, str):
                return CodexRealtimeClosed(thread_id=thread_id, reason=reason)
        case "thread/realtime/error":
            error_message = params.get("message")
            if isinstance(error_message, str):
                return CodexRealtimeError(thread_id=thread_id, message=error_message)
        case _:
            return None
    return None


def discover_codex_executable() -> Path:
    """Find a Codex binary, preferring the desktop app's authenticated copy."""

    override = os.environ.get("AVTR1_CODEX_EXECUTABLE")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(
            f"AVTR1_CODEX_EXECUTABLE does not point to a file: {candidate}"
        )

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        desktop_bin = Path(local_app_data) / "OpenAI" / "Codex" / "bin"
        if desktop_bin.is_dir():
            candidates = [path for path in desktop_bin.glob("*/codex.exe") if path.is_file()]
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0].resolve()

    for executable_name in ("codex.exe", "codex"):
        resolved = shutil.which(executable_name)
        if resolved:
            return Path(resolved).resolve()

    raise FileNotFoundError(
        "Codex executable was not found. Open/update the Codex desktop app or set "
        "AVTR1_CODEX_EXECUTABLE to its codex.exe path."
    )


def build_codex_app_server_command(executable: str | Path | None = None) -> tuple[str, ...]:
    codex = Path(executable).resolve() if executable is not None else discover_codex_executable()
    return (
        str(codex),
        "app-server",
        "--stdio",
        "--enable",
        "realtime_conversation",
        "--disable",
        "plugins",
        "--disable",
        "apps",
    )


class CodexRealtimeClient:
    """Small JSONL-RPC client for one ephemeral Codex realtime thread."""

    def __init__(self, *, command: Sequence[str], workspace: str | Path) -> None:
        if not command:
            raise ValueError("Codex app-server command cannot be empty")
        self._command = tuple(str(part) for part in command)
        self._workspace = Path(workspace).resolve()
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._notification_backlog: deque[dict[str, Any]] = deque()
        self._write_lock = asyncio.Lock()
        self._next_request_id = 1
        self._thread_id: str | None = None
        self._realtime_active = False
        self._closing = False

    @property
    def thread_id(self) -> str:
        if self._thread_id is None:
            raise RuntimeError("Codex app-server thread has not been started")
        return self._thread_id

    async def start(self) -> None:
        if self._process is not None:
            return
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            cwd=str(self._workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stdout_task = asyncio.create_task(
            self._read_stdout(), name="CodexRealtimeClient.stdout"
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr(), name="CodexRealtimeClient.stderr"
        )
        try:
            await self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "avtr1_streamer",
                        "title": "AVTR-1 Streamer",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._notify("initialized", {})
            result = await self._request(
                "thread/start",
                {
                    "approvalPolicy": "never",
                    "cwd": str(self._workspace),
                    "ephemeral": True,
                    "sandbox": "read-only",
                    "serviceName": "avtr1-codex-realtime",
                },
            )
            thread = result.get("thread")
            thread_id = thread.get("id") if isinstance(thread, dict) else None
            if not isinstance(thread_id, str) or not thread_id:
                raise CodexAppServerError("thread/start response did not contain a thread id")
            self._thread_id = thread_id
        except BaseException:
            await self.close()
            raise

    async def start_realtime(
        self,
        *,
        sdp: str,
        prompt: str,
        voice: str,
        output_modality: Literal["text", "audio"] = "audio",
    ) -> str:
        if not sdp.strip():
            raise ValueError("Codex WebRTC offer SDP cannot be empty")
        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "outputModality": output_modality,
            "prompt": prompt,
            "transport": {"type": "webrtc", "sdp": sdp},
            "version": "v2" if output_modality == "text" else "v3",
        }
        if output_modality == "audio":
            params["voice"] = voice
        else:
            params["model"] = _TEXT_REALTIME_MODEL
        await self._request("thread/realtime/start", params)
        answer_sdp = await self._wait_for_sdp_answer()
        self._realtime_active = True
        return answer_sdp

    async def stop_realtime(self) -> None:
        if not self._realtime_active:
            return
        self._realtime_active = False
        await self._request("thread/realtime/stop", {"threadId": self.thread_id})

    async def append_text(self, text: str) -> None:
        if not self._realtime_active:
            raise RuntimeError("Codex realtime session is not active")
        await self._request(
            "thread/realtime/appendText",
            {"threadId": self.thread_id, "text": text, "role": "user"},
        )

    async def append_speech(self, text: str) -> None:
        """Ask the active voice transport to speak text without a model turn."""

        if not self._realtime_active:
            raise RuntimeError("Codex realtime session is not active")
        await self._request(
            "thread/realtime/appendSpeech",
            {"threadId": self.thread_id, "text": text},
        )

    async def notifications(self) -> AsyncIterator[CodexRealtimeNotification]:
        while self._notification_backlog:
            raw = self._notification_backlog.popleft()
            parsed = self._parse_notification(raw)
            if parsed is not None:
                yield parsed
        while True:
            raw = await self._notifications.get()
            if raw is None:
                return
            parsed = self._parse_notification(raw)
            if parsed is not None:
                yield parsed

    def _parse_notification(
        self,
        raw: dict[str, Any],
    ) -> CodexRealtimeNotification | None:
        parsed = parse_realtime_notification(raw)
        if (
            isinstance(parsed, CodexRealtimeClosed)
            and parsed.thread_id == self._thread_id
        ):
            # The remote session is already gone. close() must not issue a
            # second stop RPC and then wait for its timeout.
            self._realtime_active = False
        return parsed

    async def close(self) -> None:
        if self._closing:
            return
        process = self._process
        if process is None:
            return
        self._closing = True
        try:
            if self._realtime_active and self._thread_id is not None:
                try:
                    await self.stop_realtime()
                except Exception as exc:
                    _LOGGER.debug("Codex realtime stop failed during close", error=str(exc))
            if process.stdin is not None:
                process.stdin.close()
                with suppress(AttributeError, BrokenPipeError, ConnectionResetError):
                    await process.stdin.wait_closed()
            try:
                await asyncio.wait_for(process.wait(), timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            except TimeoutError:
                process.terminate()
                await process.wait()
        finally:
            for task in (self._stdout_task, self._stderr_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (self._stdout_task, self._stderr_task):
                if task is not None:
                    with suppress(asyncio.CancelledError):
                        await task
            self._fail_pending(CodexAppServerError("Codex app-server closed"))
            self._process = None
            self._stdout_task = None
            self._stderr_task = None
            self._thread_id = None
            self._realtime_active = False
            self._closing = False

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self._write_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[request_id] = future
            try:
                await self._write_message({"method": method, "id": request_id, "params": params})
            except BaseException:
                self._pending.pop(request_id, None)
                raise
        try:
            return await asyncio.wait_for(future, timeout=_REQUEST_TIMEOUT_SECONDS)
        except BaseException:
            self._pending.pop(request_id, None)
            raise

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        async with self._write_lock:
            await self._write_message({"method": method, "params": params})

    async def _write_message(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise CodexAppServerError("Codex app-server is not running")
        encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        process.stdin.write(encoded)
        await process.stdin.drain()

    async def _read_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        failure: BaseException | None = None
        try:
            while line := await process.stdout.readline():
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    _LOGGER.warning("Ignoring invalid Codex app-server JSON", error=str(exc))
                    continue
                if not isinstance(message, dict):
                    continue
                response_id = message.get("id")
                method = message.get("method")
                if isinstance(method, str) and isinstance(response_id, int):
                    await self._reject_server_request(response_id, method)
                elif isinstance(response_id, int):
                    future = self._pending.pop(response_id, None)
                    if future is None or future.done():
                        continue
                    error = message.get("error")
                    if error is not None:
                        future.set_exception(CodexAppServerError(self._format_rpc_error(error)))
                    else:
                        result = message.get("result")
                        future.set_result(result if isinstance(result, dict) else {})
                elif isinstance(method, str):
                    await self._notifications.put(message)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            failure = exc
            _LOGGER.warning("Codex app-server stdout reader failed", error=str(exc))
        finally:
            if failure is None:
                failure = CodexAppServerError("Codex app-server stdout closed")
            self._fail_pending(failure)
            await self._notifications.put(None)

    async def _read_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        try:
            while line := await process.stderr.readline():
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    _LOGGER.debug("Codex app-server", message=text[:2000])
        except asyncio.CancelledError:
            raise

    async def _reject_server_request(self, request_id: int, method: str) -> None:
        async with self._write_lock:
            await self._write_message(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported app-server request: {method}",
                    },
                }
            )

    async def _wait_for_sdp_answer(self) -> str:
        backlog_count = len(self._notification_backlog)
        for _ in range(backlog_count):
            message = self._notification_backlog.popleft()
            answer = self._matching_sdp_or_error(message)
            if answer is not None:
                return answer
            self._notification_backlog.append(message)

        async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
            while True:
                message = await self._notifications.get()
                if message is None:
                    raise CodexAppServerError(
                        "Codex app-server closed before returning WebRTC answer SDP"
                    )
                answer = self._matching_sdp_or_error(message)
                if answer is not None:
                    return answer
                self._notification_backlog.append(message)

    def _matching_sdp_or_error(self, message: dict[str, Any]) -> str | None:
        params = message.get("params")
        if not isinstance(params, dict) or params.get("threadId") != self.thread_id:
            return None
        method = message.get("method")
        if method == "thread/realtime/sdp":
            sdp = params.get("sdp")
            return sdp if isinstance(sdp, str) and sdp.strip() else None
        if method == "thread/realtime/error":
            error_message = params.get("message")
            if isinstance(error_message, str):
                raise CodexAppServerError(error_message)
        if method == "thread/realtime/closed":
            reason = params.get("reason")
            detail = reason if isinstance(reason, str) and reason else "unknown reason"
            raise CodexAppServerError(
                f"Codex realtime closed before WebRTC answer: {detail}"
            )
        return None

    def _fail_pending(self, error: BaseException) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(error)

    @staticmethod
    def _format_rpc_error(error: object) -> str:
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            if isinstance(message, str):
                return f"Codex app-server RPC error {code}: {message}"
        return f"Codex app-server RPC error: {error}"


class CodexConversationEngine:
    """Map browser-bridged Codex media and app-server events onto AVTR's bus."""

    def __init__(
        self,
        *,
        client: CodexRealtimeClient,
        answer_sdp: str,
        stream_id: str,
        tts: StreamingTTSBackend | None = None,
    ) -> None:
        self._client = client
        self.answer_sdp = answer_sdp
        self._stream_id = stream_id
        self._turn_counter = 0
        self._current_segment_id: SegmentId | None = None
        self._user_speech_active = False
        self._local_tts = (
            StreamingLocalTTSBridge(
                tts,
                task_name="CodexConversationEngine.local_tts",
            )
            if tts is not None
            else None
        )
        self._assistant_turn_counter = 0
        self._assistant_response_id: str | None = None
        self._assistant_delta_seen = False
        self._assistant_text_parts: list[str] = []
        self._suppress_assistant_until_delta = False

    def _current_assistant_response(self) -> str:
        response_id = self._assistant_response_id
        if response_id is None:
            self._assistant_turn_counter += 1
            response_id = f"codex-{self._stream_id}-text-{self._assistant_turn_counter}"
            self._assistant_response_id = response_id
            self._assistant_delta_seen = False
            self._assistant_text_parts = []
        return response_id

    def _end_assistant_response(self) -> None:
        self._assistant_response_id = None
        self._assistant_delta_seen = False
        self._assistant_text_parts = []

    async def _open_segment(self, bus: EventBus) -> SegmentId:
        self._turn_counter += 1
        segment_id = SegmentId(f"codex-{self._stream_id}-turn-{self._turn_counter}")
        self._current_segment_id = segment_id
        await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
        return segment_id

    async def _close_segment(self, bus: EventBus) -> None:
        segment_id = self._current_segment_id
        if segment_id is None:
            return
        self._current_segment_id = None
        await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))

    async def _interrupt_avatar(self, bus: EventBus) -> None:
        assistant_was_active = self._assistant_response_id is not None
        self._end_assistant_response()
        if self._local_tts is not None:
            if assistant_was_active:
                self._suppress_assistant_until_delta = True
            await self._local_tts.interrupt(bus)
            return
        await self._close_segment(bus)
        await bus.publish(DiscardAvatarSpeechBuffer())

    async def _notification_loop(self, bus: EventBus) -> None:
        bus.ready()
        async for event in self._client.notifications():
            match event:
                case CodexTranscriptDelta(role=role, delta=delta):
                    if role == "user" and not self._user_speech_active:
                        self._user_speech_active = True
                        await self._interrupt_avatar(bus)
                    elif role in ("assistant", "agent") and self._local_tts is not None:
                        if self._user_speech_active or self._suppress_assistant_until_delta:
                            continue
                        response_id = self._current_assistant_response()
                        self._assistant_delta_seen = True
                        self._assistant_text_parts.append(delta)
                        await self._local_tts.feed_text(bus, response_id, delta)
                case CodexTranscriptDone(role=role, text=text):
                    timestamp = tzutcnow().timestamp()
                    if role == "user":
                        self._user_speech_active = False
                        await bus.publish(InputTranscript(transcript=text, timestamp=timestamp))
                    elif role in ("assistant", "agent"):
                        if self._local_tts is not None:
                            if self._suppress_assistant_until_delta:
                                # Terminal transcript of the response that the
                                # user's barge-in cancelled.
                                self._suppress_assistant_until_delta = False
                                self._end_assistant_response()
                                continue
                            if self._user_speech_active:
                                continue
                        await bus.publish(ResponseTranscript(transcript=text, timestamp=timestamp))
                        if self._local_tts is not None:
                            response_id = self._current_assistant_response()
                            streamed_text = "".join(self._assistant_text_parts)
                            if not self._assistant_delta_seen:
                                fallback = text
                            else:
                                fallback = None
                                if text.startswith(streamed_text):
                                    missing_suffix = text[len(streamed_text) :]
                                    if missing_suffix:
                                        await self._local_tts.feed_text(
                                            bus,
                                            response_id,
                                            missing_suffix,
                                        )
                            await self._local_tts.finish_text(
                                bus,
                                response_id,
                                fallback_text=fallback,
                            )
                            self._end_assistant_response()
                case CodexRealtimeError(message=message):
                    raise CodexAppServerError(message)
                case CodexRealtimeClosed():
                    await bus.publish(Shutdown(reason="agent_left"))
                    return
                case _:
                    pass
        await bus.publish(Shutdown(reason="agent_left"))

    async def _bus_loop(self, bus: EventBus) -> None:
        async with bus.subscribe(
            CodexAssistantAudioReceived,
            CodexAssistantControlReceived,
            TextEchoEnqueueText,
            Shutdown,
        ) as subscription:
            bus.ready()
            async for event in subscription:
                match event:
                    case CodexAssistantAudioReceived(buffer=buffer):
                        if self._local_tts is not None:
                            continue
                        segment_id = self._current_segment_id or await self._open_segment(bus)
                        await bus.publish(
                            SegmentChunkGenerated(segment_id=segment_id, buffer=buffer)
                        )
                    case CodexAssistantControlReceived(type="output_audio_done"):
                        if self._local_tts is None:
                            await self._close_segment(bus)
                    case CodexAssistantControlReceived(type="speech_started"):
                        self._user_speech_active = True
                        await self._interrupt_avatar(bus)
                    case TextEchoEnqueueText(text=text):
                        if self._local_tts is not None:
                            await self._interrupt_avatar(bus)
                        await self._client.append_text(text)
                    case Shutdown():
                        if self._local_tts is not None:
                            await self._local_tts.close(bus)
                        else:
                            await self._close_segment(bus)
                        await self._client.close()
                        return

    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        _ = clocks
        try:
            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(
                    self._notification_loop(bus.clone()),
                    name="CodexConversationEngine.notifications",
                )
                task_group.create_task(
                    self._bus_loop(bus),
                    name="CodexConversationEngine.bus",
                )
        finally:
            if self._local_tts is not None:
                await self._local_tts.close(bus)
            await self._client.close()

    async def close(self) -> None:
        if self._local_tts is not None:
            await self._local_tts.close()
        await self._client.close()

    async def preview_speech(self, text: str) -> None:
        if self._local_tts is not None:
            raise RuntimeError("本地 TTS 混合模式请使用本地音色试听按钮")
        await self._client.append_speech(text)

    async def __call__(self, bus: EventBus, clocks: StreamClocks) -> None:
        await self.run(bus, clocks)
