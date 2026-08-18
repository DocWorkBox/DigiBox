# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Worklet for clock-synchronized emission of scheduled events."""

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import ScheduledEvent, Shutdown
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit


@async_log_entry_exit
async def run_delayed_event_worklet(bus: EventBus, clocks: StreamClocks) -> None:
    async with bus.subscribe(ScheduledEvent, Shutdown) as sub:
        bus.ready()
        async for scheduled in sub:
            if isinstance(scheduled, Shutdown):
                return
            await clocks.wakeup_at(float(scheduled.emit_at))
            await bus.publish(scheduled.event)
