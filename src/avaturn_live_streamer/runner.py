# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Stream runner that coordinates all worklets via event bus."""

from asyncio import TaskGroup
from collections.abc import Callable, Coroutine

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.event_bus import EventBus

_LOGGER = get_logger()


type WorkletCallable = Callable[[EventBus, StreamClocks], Coroutine[None, None, None]]


async def run_stream(*worklets: WorkletCallable) -> None:
    bus = EventBus()
    clocks = StreamClocks()

    async with TaskGroup() as tg:
        for worklet in worklets:
            tg.create_task(
                worklet(bus.clone(), clocks), name=getattr(worklet, "__qualname__", repr(worklet))
            )
        bus.ready()
