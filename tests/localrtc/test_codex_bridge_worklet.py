from __future__ import annotations

import asyncio
from contextlib import suppress

import numpy as np

from avaturn_live_streamer import events as event_module
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.localrtc.peer import CodexBridgeControl
from avaturn_live_streamer.localrtc.worklet import LocalRTCWorklet
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.types import PixelFormat


class _BridgePeer:
    def __init__(self) -> None:
        self.messages: asyncio.Queue[object] = asyncio.Queue()

    async def recv_codex_bridge_message(self) -> object:
        return await self.messages.get()


def test_localrtc_worklet_publishes_codex_audio_and_control_events() -> None:
    audio_event_type = getattr(event_module, "CodexAssistantAudioReceived", None)
    control_event_type = getattr(event_module, "CodexAssistantControlReceived", None)
    assert audio_event_type is not None, "Codex assistant audio event is not implemented"
    assert control_event_type is not None, "Codex assistant control event is not implemented"
    assert hasattr(LocalRTCWorklet, "_read_codex_bridge_loop"), (
        "LocalRTCWorklet does not forward the Codex data channel"
    )

    async def exercise() -> None:
        peer = _BridgePeer()
        worklet = LocalRTCWorklet(peer, PixelFormat.YUV_I420)  # type: ignore[arg-type]
        bus = EventBus()
        expected_audio = SpeechBuffer(np.array([12, -12], dtype=np.int16), 24_000)

        async with bus.subscribe(audio_event_type, control_event_type) as subscription:
            bus.ready()
            task = asyncio.create_task(worklet._read_codex_bridge_loop(bus))
            try:
                await peer.messages.put(expected_audio)
                await peer.messages.put(
                    CodexBridgeControl(type="output_audio_done", item_id="item-9")
                )

                audio_event = await subscription.get_next(timeout=0.2)
                control_event = await subscription.get_next(timeout=0.2)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        assert isinstance(audio_event, audio_event_type)
        assert audio_event.buffer is expected_audio
        assert isinstance(control_event, control_event_type)
        assert control_event.type == "output_audio_done"
        assert control_event.item_id == "item-9"

    asyncio.run(exercise())
