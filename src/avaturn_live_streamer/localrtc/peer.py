# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Local WebRTC peer-state object backed by aiortc.

Parallel role to `streamer/rtc.py`'s `RTC` (which wraps Daily): owns the
`RTCPeerConnection`, the outbound video/audio tracks, and the inbound audio
queue for one local browser peer. Constructed *after* SDP negotiation by
the local-stream CLI's signaling handler, then handed to `LocalRTCWorklet`.

PTS on outbound video frames is preserved as-set by the worklet (the worklet
applies the hybrid nominal/wall-clock rule); the track does not re-stamp.
"""

import asyncio
import fractions
import json
import struct
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCDataChannel, RTCPeerConnection, RTCRtpSender
from aiortc.codecs import h264
from aiortc.mediastreams import MediaStreamError
from av.frame import Frame as AvFrame

from avaturn_live_streamer import constant
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer

_LOGGER = get_logger()

_AUDIO_TIME_BASE = fractions.Fraction(1, constant.NATIVE_SPEECH_SAMPLE_RATE)
_AUDIO_SAMPLES_PER_FRAME = constant.NATIVE_SPEECH_SAMPLE_RATE // 50  # 20ms

type ConnectionState = Literal["new", "connecting", "connected", "disconnected", "failed", "closed"]
type CodexBridgeControlType = Literal["speech_started", "output_audio_done"]

_CODEX_AUDIO_CHANNEL_LABEL = "avtr-codex-audio"
_CODEX_AUDIO_HEADER_SIZE = 4
_MIN_AUDIO_SAMPLE_RATE = 8_000
_MAX_AUDIO_SAMPLE_RATE = 384_000


@dataclass(frozen=True, slots=True)
class CodexBridgeControl:
    type: CodexBridgeControlType
    item_id: str | None = None


type CodexBridgeMessage = SpeechBuffer | CodexBridgeControl


def decode_codex_audio_packet(packet: bytes) -> SpeechBuffer:
    """Decode ``uint32-le sample rate + mono PCM16-le`` from the browser."""

    if len(packet) < _CODEX_AUDIO_HEADER_SIZE + 2:
        raise ValueError("Codex audio packet is too short")
    pcm = packet[_CODEX_AUDIO_HEADER_SIZE:]
    if len(pcm) % 2:
        raise ValueError("Codex audio packet has an odd PCM16 byte count")
    (sample_rate,) = struct.unpack_from("<I", packet)
    if not _MIN_AUDIO_SAMPLE_RATE <= sample_rate <= _MAX_AUDIO_SAMPLE_RATE:
        raise ValueError(f"Invalid Codex audio sample rate: {sample_rate}")
    return SpeechBuffer.from_bytes(pcm, sample_rate)


def parse_codex_bridge_control(message: str) -> CodexBridgeControl:
    """Parse the small allow-listed control surface forwarded by the browser."""

    try:
        value = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid Codex bridge control JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Codex bridge control must be an object")
    control_type = value.get("type")
    if control_type not in ("speech_started", "output_audio_done"):
        raise ValueError(f"Unsupported Codex bridge control: {control_type!r}")
    item_id = value.get("item_id")
    if item_id is not None and not isinstance(item_id, str):
        raise ValueError("Codex bridge item_id must be a string or null")
    return CodexBridgeControl(type=control_type, item_id=item_id)


class _QueuedVideoTrack(MediaStreamTrack):
    """Outbound video track. recv() awaits frames enqueued by the worklet."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.VideoFrame | None] = asyncio.Queue(maxsize=2)

    async def push(self, frame: av.VideoFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(frame)

    async def recv(self) -> AvFrame:
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise ConnectionError("video track closed")
        return frame


class _QueuedAudioTrack(MediaStreamTrack):
    """Outbound audio track. recv() pops PCM samples written by the worklet."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.AudioFrame | None] = asyncio.Queue(maxsize=64)
        self._pts: int = 0

    async def push(self, samples: np.ndarray) -> None:
        """`samples` is int16 mono shaped (N,) or (1, N)."""
        if samples.ndim == 2:
            samples = samples.reshape(-1)
        # Re-chunk into AUDIO_SAMPLES_PER_FRAME segments so each av.AudioFrame is small.
        for start in range(0, samples.shape[0], _AUDIO_SAMPLES_PER_FRAME):
            chunk = samples[start : start + _AUDIO_SAMPLES_PER_FRAME]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = constant.NATIVE_SPEECH_SAMPLE_RATE
            frame.pts = self._pts
            frame.time_base = _AUDIO_TIME_BASE
            self._pts += chunk.shape[0]
            await self._queue.put(frame)

    async def recv(self) -> AvFrame:
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise ConnectionError("audio track closed")
        return frame


class LocalRTC:
    """Owns the peer connection, outbound tracks, and inbound audio buffer.

    Constructed once SDP negotiation is complete. Handed to `LocalRTCWorklet`,
    which drives `push_video_frame` / `push_audio_chunk` and consumes
    `recv_audio_chunk`. Lifecycle (close) is the responsibility of the CLI app.
    """

    def __init__(
        self,
        pc: RTCPeerConnection,
        *,
        video_bitrate_bps: int = 2_500_000,
    ) -> None:
        if video_bitrate_bps <= 0:
            raise ValueError("video_bitrate_bps must be positive")
        self.pc = pc
        self._video_track = _QueuedVideoTrack()
        self._audio_track = _QueuedAudioTrack()
        self._inbound_audio: asyncio.Queue[SpeechBuffer] = asyncio.Queue(maxsize=600)
        self._inbound_codex_bridge: asyncio.Queue[CodexBridgeMessage] = asyncio.Queue(maxsize=600)
        self._connection_state: ConnectionState = "new"
        self._state_waiters: list[asyncio.Future[ConnectionState]] = []

        # aiortc otherwise follows the browser's offer order, which commonly
        # selects VP8 (500 kbps by default, capped at 1.5 Mbps).  Prefer H.264
        # and seed its software encoder with the selected quality preset.
        h264.DEFAULT_BITRATE = video_bitrate_bps
        h264.MAX_BITRATE = max(h264.MAX_BITRATE, video_bitrate_bps)
        video_sender = pc.addTrack(self._video_track)
        pc.addTrack(self._audio_track)
        h264_codecs = [
            codec
            for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.casefold() == "video/h264"
        ]
        get_transceivers = getattr(pc, "getTransceivers", None)
        if callable(get_transceivers):
            for transceiver in get_transceivers():
                if transceiver.sender is video_sender and h264_codecs:
                    transceiver.setCodecPreferences(h264_codecs)
                    break
        pc.on("connectionstatechange", self._on_state_change)
        pc.on("track", self._on_track)
        pc.on("datachannel", self._on_datachannel)

    def _on_state_change(self) -> None:
        state: ConnectionState = self.pc.connectionState  # pyright: ignore [reportAssignmentType]
        _LOGGER.info("localrtc peer connection state", state=state)
        self._connection_state = state
        waiters = self._state_waiters
        self._state_waiters = []
        for w in waiters:
            if not w.done():
                w.set_result(state)

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "audio":
            _LOGGER.debug("localrtc ignoring inbound track", kind=track.kind)
            return
        _LOGGER.info("localrtc inbound audio track attached")
        asyncio.create_task(self._consume_inbound_audio(track))

    def _on_datachannel(self, channel: RTCDataChannel) -> None:
        if channel.label != _CODEX_AUDIO_CHANNEL_LABEL:
            _LOGGER.debug("localrtc ignoring data channel", label=channel.label)
            return
        _LOGGER.info("localrtc Codex audio bridge attached")

        @channel.on("message")
        def _on_message(message: str | bytes) -> None:
            try:
                decoded: CodexBridgeMessage
                if isinstance(message, bytes):
                    decoded = decode_codex_audio_packet(message)
                else:
                    decoded = parse_codex_bridge_control(message)
            except ValueError as exc:
                _LOGGER.warning("invalid Codex audio bridge message", error=str(exc))
                return
            if self._inbound_codex_bridge.full():
                _LOGGER.warning("Codex audio bridge queue full, dropping oldest")
                with suppress(asyncio.QueueEmpty):
                    self._inbound_codex_bridge.get_nowait()
            self._inbound_codex_bridge.put_nowait(decoded)

    async def _consume_inbound_audio(self, track: MediaStreamTrack) -> None:
        target_sr = constant.NATIVE_SPEECH_SAMPLE_RATE
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        try:
            while True:
                frame = await track.recv()
                if not isinstance(frame, av.AudioFrame):
                    continue
                for out_frame in resampler.resample(frame):
                    arr = out_frame.to_ndarray().reshape(-1).astype(np.int16, copy=False)
                    if arr.size == 0:
                        continue
                    buf = SpeechBuffer(arr, target_sr)
                    if self._inbound_audio.full():
                        _LOGGER.warning("localrtc inbound audio queue full, dropping oldest")
                        try:
                            self._inbound_audio.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self._inbound_audio.put(buf)
        except (asyncio.CancelledError, ConnectionError):
            raise
        except MediaStreamError:
            _LOGGER.info("localrtc inbound audio track ended")
        except Exception:
            _LOGGER.exception("localrtc inbound audio loop crashed")

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    async def wait_state_change(self) -> ConnectionState:
        fut: asyncio.Future[ConnectionState] = asyncio.get_event_loop().create_future()
        self._state_waiters.append(fut)
        return await fut

    async def wait_connected(self, timeout: float = 30.0) -> None:
        """Block until the peer connection reaches ``"connected"``.

        Raises ``TimeoutError`` if the connection doesn't establish within
        ``timeout`` seconds, and ``ConnectionError`` if the peer enters a
        terminal failure state ("closed"/"failed") first.
        """
        async with asyncio.timeout(timeout):
            while self._connection_state != "connected":
                if self._connection_state in ("closed", "failed"):
                    raise ConnectionError(
                        f"peer connection entered terminal state {self._connection_state!r} "
                        f"before reaching 'connected'"
                    )
                await self.wait_state_change()

    async def push_video_frame(self, frame: av.VideoFrame) -> None:
        await self._video_track.push(frame)

    async def push_audio_chunk(self, speech: SpeechBuffer) -> None:
        if speech.is_empty:
            return
        resampled = speech.resample(constant.NATIVE_SPEECH_SAMPLE_RATE)
        samples = np.frombuffer(resampled.to_bytes(), dtype=np.int16)
        await self._audio_track.push(samples)

    async def recv_audio_chunk(self) -> SpeechBuffer:
        return await self._inbound_audio.get()

    async def recv_codex_bridge_message(self) -> CodexBridgeMessage:
        return await self._inbound_codex_bridge.get()

    async def close(self) -> None:
        await self.pc.close()
