from __future__ import annotations

import asyncio
import time
from fractions import Fraction

import pytest

from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.speech.speech_scheduler import SegmentStarted, SpeechScheduler


def _audio(duration_ms: int, *, sample_rate: int = 1_000) -> SpeechBuffer:
    return SpeechBuffer.full(1, Fraction(duration_ms, 1_000), sample_rate)


def test_first_segment_coalesces_pcm_packets_before_starting_playback() -> None:
    async def exercise() -> SpeechScheduler._StepResult:
        scheduler = SpeechScheduler(
            sample_rate=1_000,
            present_duration=Fraction(1, 5),
            future_duration=Fraction(1, 5),
            startup_coalesce_timeout=0.1,
        )
        await scheduler.start_segment("turn-1")
        await scheduler.append(_audio(40), "turn-1")

        step_task = asyncio.create_task(scheduler.do_step())
        await asyncio.sleep(0.01)
        assert not step_task.done()

        await scheduler.append(_audio(360), "turn-1")
        return await asyncio.wait_for(step_task, timeout=0.1)

    step = asyncio.run(exercise())

    assert not step.present.is_silent()
    starts = [event for event in step.events if isinstance(event.event, SegmentStarted)]
    assert len(starts) == 1
    assert starts[0].timestamp == 0


def test_first_segment_has_a_bounded_startup_wait() -> None:
    async def exercise() -> tuple[float, SpeechScheduler._StepResult]:
        scheduler = SpeechScheduler(
            sample_rate=1_000,
            present_duration=Fraction(1, 5),
            future_duration=Fraction(1, 5),
            startup_coalesce_timeout=0.08,
        )
        await scheduler.start_segment("turn-1")
        await scheduler.append(_audio(40), "turn-1")

        started_at = time.perf_counter()
        step = await asyncio.wait_for(scheduler.do_step(), timeout=0.25)
        return time.perf_counter() - started_at, step

    elapsed, step = asyncio.run(exercise())

    assert 0.06 <= elapsed < 0.2
    assert step.present.is_silent()


def test_startup_coalescing_cannot_be_configured_above_120ms() -> None:
    with pytest.raises(ValueError, match=r"at most 0\.12 seconds"):
        SpeechScheduler(
            sample_rate=1_000,
            present_duration=Fraction(1, 5),
            future_duration=Fraction(1, 5),
            startup_coalesce_timeout=0.121,
        )


def test_interrupt_during_startup_coalescing_never_starts_cancelled_audio() -> None:
    async def exercise() -> list[SpeechScheduler._StepResult]:
        scheduler = SpeechScheduler(
            sample_rate=1_000,
            present_duration=Fraction(1, 5),
            future_duration=Fraction(1, 5),
            startup_coalesce_timeout=0.1,
        )
        await scheduler.start_segment("turn-1")
        await scheduler.append(_audio(40), "turn-1")

        first_task = asyncio.create_task(scheduler.do_step())
        await asyncio.sleep(0.01)
        await scheduler.interrupt()
        first = await asyncio.wait_for(first_task, timeout=0.05)
        second = await asyncio.wait_for(scheduler.do_step(), timeout=0.05)
        return [first, second]

    steps = asyncio.run(exercise())

    assert all(step.present.is_silent() for step in steps)
    assert not any(
        isinstance(event.event, SegmentStarted) for step in steps for event in step.events
    )


def test_interrupt_before_scheduled_start_removes_left_padded_audio() -> None:
    async def exercise() -> list[SpeechScheduler._StepResult]:
        scheduler = SpeechScheduler(
            sample_rate=1_000,
            present_duration=Fraction(1, 5),
            future_duration=Fraction(1, 5),
            startup_coalesce_timeout=0,
        )
        await scheduler.start_segment("turn-1")
        await scheduler.append(_audio(40), "turn-1")

        first = await scheduler.do_step()
        await scheduler.interrupt()
        second = await scheduler.do_step()
        return [first, second]

    steps = asyncio.run(exercise())

    assert all(step.present.is_silent() for step in steps)
    assert not any(
        isinstance(event.event, SegmentStarted) for step in steps for event in step.events
    )
