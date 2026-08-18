from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from fractions import Fraction

from avaturn_live_streamer.config import RendererConfig
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import TurnLatencyMilestone, VideoFrameGenerated
from avaturn_live_streamer.management.types import SegmentId
from avaturn_live_streamer.renderer.interface import RenderConfig, RenderResponse
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.speech.speech_scheduler import (
    SegmentStarted,
    SpeechScheduler,
    TimestampedEvent,
)
from avaturn_live_streamer.types import PixelFormat
from avaturn_live_streamer.worklets.rendering import RenderingWorklet


def _silence(duration: Fraction) -> SpeechBuffer:
    return SpeechBuffer.silence(duration, 24_000)


def test_rendering_worklet_marks_the_actual_first_frame_for_a_turn() -> None:
    class TwoFrameRenderer:
        @asynccontextmanager
        async def generate(self, request):
            _ = request

            async def frames():
                yield b"frame-0"
                yield b"frame-1"

            async def state():
                return b"state"

            yield RenderResponse(frames(), 2, state())

    async def exercise() -> list[object]:
        worklet = RenderingWorklet(
            object(),
            RendererConfig(
                avatar_id="avatar",
                background_id="transparent",
                height=2,
                width=2,
            ),
        )
        segment_id = SegmentId("segment-1")
        worklet._segment_metadata[segment_id] = {"turn_id": "turn-1"}
        step = SpeechScheduler._StepResult(
            present=_silence(Fraction(2, 25)),
            future=_silence(Fraction(1, 25)),
            events=[
                TimestampedEvent(
                    SegmentStarted(str(segment_id)),
                    Fraction(1, 25),
                )
            ],
        )
        user_step = SpeechScheduler._StepResult(
            present=_silence(Fraction(2, 25)),
            future=_silence(Fraction(1, 25)),
            events=[],
        )
        render_config = RenderConfig(
            avatar_id="avatar",
            background_id="transparent",
            pixel_format=PixelFormat.YUV_I420,
            height=2,
            width=2,
        )
        bus = EventBus()
        events: list[object] = []
        async with bus.subscribe(
            VideoFrameGenerated,
            TurnLatencyMilestone,
        ) as subscription:
            bus.ready()
            # A very short segment can start and end in one scheduler step;
            # its metadata is removed while scheduling the completion event.
            # The start marker captured before that removal must still drive
            # the first-frame measurement.
            worklet._segment_metadata.clear()
            await worklet._generate_frames(
                bus,
                TwoFrameRenderer(),
                None,
                Fraction(),
                render_config,
                step,
                user_step,
                turn_frame_starts=[("turn-1", Fraction(1, 25))],
            )
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
            events.append(await subscription.get_next(timeout=0.5))
        return events

    events = asyncio.run(exercise())

    assert [type(event) for event in events] == [
        VideoFrameGenerated,
        VideoFrameGenerated,
        TurnLatencyMilestone,
    ]
    milestone = events[-1]
    assert isinstance(milestone, TurnLatencyMilestone)
    assert milestone.turn_id == "turn-1"
    assert milestone.phase == "renderer_first_frame"
