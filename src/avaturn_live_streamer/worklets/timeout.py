# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Worklet for session timeout management (user absence and max duration)."""

from asyncio import TimerHandle, get_running_loop

from attrs import define

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    ParticipantJoined,
    ParticipantLeft,
    Shutdown,
    ShutdownReason,
    StreamStarted,
)
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()


@define
class TimeoutWorklet:
    """
    Monitors session timeouts and triggers shutdown when exceeded.

    Architecture:
    - Subscribes to EventBus for participant events
    - Uses call_later to schedule shutdowns
    - TimerHandle.cancel() cancels shutdowns if needed

    Timeout types:
    1. user_absent_timeout: No participants in room
       - Scheduled on StreamStarted if count=0, or when last participant leaves
       - Cancelled when any participant joins
    2. max_duration: Hard stream duration limit
       - Scheduled on StreamStarted, cancelled only on Shutdown (for cleanup)
    """

    user_absent_timeout: float
    max_duration: float

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:  # noqa: ARG002
        participant_count = 0
        max_duration_handle: TimerHandle | None = None
        user_absent_handle: TimerHandle | None = None

        loop = get_running_loop()

        def schedule_shutdown(delay: float, reason: ShutdownReason) -> TimerHandle:
            def callback() -> None:
                _LOGGER.info("Timeout triggered", reason=reason)
                loop.create_task(
                    bus.publish(Shutdown(reason=reason)), name=f"TimeoutWorklet.shutdown[{reason}]"
                )

            _LOGGER.debug("Scheduled shutdown", reason=reason, delay=delay)
            return loop.call_later(delay, callback)

        async with bus.subscribe(
            StreamStarted,
            ParticipantJoined,
            ParticipantLeft,
            Shutdown,
        ) as sub:
            bus.ready()

            async for event in sub:
                match event:
                    case StreamStarted():
                        max_duration_handle = schedule_shutdown(
                            self.max_duration, "max_duration_reached"
                        )
                        _LOGGER.info(
                            "Stream started",
                            participant_count=participant_count,
                        )
                        if participant_count == 0:
                            user_absent_handle = schedule_shutdown(
                                self.user_absent_timeout, "user_absent_timeout"
                            )

                    case ParticipantJoined(participant_id=pid):
                        participant_count += 1

                        if user_absent_handle is not None:
                            user_absent_handle.cancel()
                            user_absent_handle = None

                        _LOGGER.info(
                            "Participant joined",
                            participant_id=pid,
                            participant_count=participant_count,
                        )

                    case ParticipantLeft(participant_id=pid):
                        participant_count = max(0, participant_count - 1)
                        _LOGGER.info(
                            "Participant left",
                            participant_id=pid,
                            participant_count=participant_count,
                        )

                        if participant_count == 0:
                            if user_absent_handle is not None:
                                raise RuntimeError(
                                    "Last participant left but user absent timer already scheduled"
                                )
                            user_absent_handle = schedule_shutdown(
                                self.user_absent_timeout, "user_absent_timeout"
                            )

                    case Shutdown():
                        if user_absent_handle is not None:
                            user_absent_handle.cancel()
                        if max_duration_handle is not None:
                            max_duration_handle.cancel()
                        return
