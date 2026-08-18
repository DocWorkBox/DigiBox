# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import asyncio
import logging
from asyncio import Event, Queue, QueueFull
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Generic, Self, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Subscription(Generic[T]):
    """
    Handle for consuming events from an event bus subscription.

    Supports two consumption patterns:
    1. Async iteration: `async for event in subscription:`
    2. Explicit get: `event = await subscription.get_next()`
    """

    def __init__(self, queue: Queue[T | None]):
        self._queue = queue
        self._closed = False

    async def get_next(self, timeout: float | None = None) -> T | None:
        """
        Get next event, blocking until available.

        Returns:
            Next event, or None if subscription is closed.

        Example:
            async with bus.subscribe(EventA) as sub:
                while (event := await sub.get_next()) is not None:
                    handle(event)
        """
        if self._closed:
            return None
        if timeout is not None:
            event = await asyncio.wait_for(self._queue.get(), timeout)
        else:
            event = await self._queue.get()

        if event is None:
            self._closed = True
        return event

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        """
        Get next event for async iteration.

        Raises:
            StopAsyncIteration: When subscription is closed.

        Example:
            async with bus.subscribe(EventA, EventB) as sub:
                async for event in sub:
                    match event:
                        case EventA(): ...
                        case EventB(): ...
        """
        event = await self.get_next()
        if event is None:
            raise StopAsyncIteration
        return event


class EventBus:
    """
    Type-based async event bus with worker synchronization.

    Features:
    - Class-based routing: events routed by runtime type
    - Multiple type subscription: subscribe to several event types at once
    - Automatic backpressure: blocks publisher when subscriber queue is full
    - Context manager: automatic cleanup on exit
    - Worker synchronization: publish() blocks until all workers are ready

    Worker Synchronization:
        The bus starts in an "unready" state. Publishing blocks until all workers
        signal they've completed their subscription setup. This prevents race
        conditions where events are published before subscribers exist.

        Semantics:
        - Bus starts with pending_count=1 (unready by default)
        - Each clone() increments pending_count
        - Each ready() decrements pending_count
        - When pending_count reaches 0, publish() unblocks

        Usage pattern:
            bus = EventBus()

            async with TaskGroup() as tg:
                tg.create_task(worker1.run(bus.clone()))
                tg.create_task(worker2.run(bus.clone()))
                bus.ready()  # Coordinator signals ready after spawning workers

            # In each worker:
            async def run(self, bus: EventBus):
                async with bus.subscribe(MyEvent) as sub:
                    bus.ready()  # Signal ready after subscribing
                    async for event in sub:
                        handle(event)
    """

    def __init__(self, default_buffer_size: int = 100, ready_timeout: float = 5.0):
        """
        Initialize event bus.

        Args:
            default_buffer_size: Default queue size for subscriptions.
            ready_timeout: Timeout in seconds for publish() to wait for workers
                to be ready. Prevents deadlock if a worker crashes before calling ready().
        """
        self._subscribers: dict[type, list[Queue]] = defaultdict(list)
        self._default_buffer_size = default_buffer_size
        self._ready_timeout = ready_timeout
        # Start with pending_count=1: bus is unready by default.
        # Coordinator must call ready() after spawning all workers.
        self._pending_ready_count = 1
        self._all_ready_event: Event = Event()
        # Event is NOT set initially - bus starts unready
        # Buffer for publish_nowait(..., allow_pre_ready=True) calls that
        # arrive before the bus is ready. Drained by ready() once
        # pending_ready_count reaches 0.
        self._pre_ready_events: list[Any] = []

    def clone(self) -> Self:
        """
        Mark that a worker will use this bus.

        Call this before passing the bus to a worker task. The worker must
        call ready() after it has completed setting up its subscriptions.

        Returns self for method chaining:
            tg.create_task(worker.run(bus.clone()))

        While any clone is pending (worker hasn't called ready()), publish() will block.
        """
        self._pending_ready_count += 1
        return self

    @property
    def is_ready(self) -> bool:
        """True once all clones + the coordinator have called ``ready()``.

        Use from event handlers that may fire before subscription setup
        completes (e.g. SDK callbacks dispatched during a ``room.connect()``)
        to gate ``publish_nowait`` and avoid the pre-ready RuntimeError.
        """
        return self._all_ready_event.is_set()

    def ready(self) -> None:
        """
        Signal that this worker has completed its subscription setup.

        Must be called exactly once for each clone() call, plus once by the
        coordinator (to account for the initial unready state).

        When all workers and the coordinator have called ready(), publish() unblocks.

        Raises:
            RuntimeError: If called more times than expected (more ready() than clone()+1).

        Example:
            async def run(self, bus: EventBus) -> None:
                async with bus.subscribe(MyEvent) as sub:
                    bus.ready()  # Call after subscribing, before processing
                    async for event in sub:
                        handle(event)
        """
        if self._pending_ready_count <= 0:
            raise RuntimeError(
                "EventBus.ready() called more times than expected. "
                "Ensure each ready() has a matching clone() call."
            )
        self._pending_ready_count -= 1
        if self._pending_ready_count == 0:
            self._all_ready_event.set()
            # Flush events buffered via publish_nowait(..., allow_pre_ready=True)
            # in arrival order, now that subscribers are guaranteed to exist.
            pending = self._pre_ready_events
            self._pre_ready_events = []
            for event in pending:
                self._dispatch_nowait(event)

    async def publish(self, event: Any) -> None:
        """
        Publish event to all subscribers interested in its type.

        Blocks until all workers have called ready(). This ensures events
        are not lost due to subscribers not yet being set up.

        Routes based on event's runtime type (type(event)).
        If subscriber queue is full, logs warning and blocks until space is available.

        Args:
            event: Event instance to publish.

        Raises:
            RuntimeError: If timeout waiting for workers to be ready.

        Example:
            await bus.publish(AvatarStartedSpeaking(phrase_id="123"))
            await bus.publish(AvatarEndedSpeaking(phrase_id="123"))
        """
        # Wait for all workers to be ready before publishing
        if not self._all_ready_event.is_set():
            try:
                await asyncio.wait_for(
                    self._all_ready_event.wait(),
                    timeout=self._ready_timeout,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"EventBus.publish() timed out after {self._ready_timeout}s waiting for "
                    f"{self._pending_ready_count} worker(s) to call ready(). "
                    f"Possible causes: worker crashed before ready(), or ready() call missing."
                ) from None

        event_type = type(event)

        for queue in self._subscribers.get(event_type, []):
            try:
                queue.put_nowait(event)
            except QueueFull:
                logger.warning(
                    "Subscriber queue full for event %s, blocking until space available",
                    type(event).__name__,
                )
                await queue.put(event)

    def publish_nowait(self, event: Any, *, allow_pre_ready: bool = False) -> None:
        """Non-blocking publish for use from sync callbacks (e.g. LiveKit
        event handlers running inline on the loop).

        Dispatch is inline, so callsite ordering is preserved across multiple
        ``publish_nowait`` calls — unlike ``asyncio.create_task(publish(...))``.

        Contract differences vs. ``publish``:
        - Raises ``RuntimeError`` if called before all workers are ready and
          ``allow_pre_ready`` is False (sync caller can't wait).
        - Raises ``QueueFull`` if any subscriber queue is full — overflow is
          a programming/sizing error, not a normal condition for the
          control-plane events this method is intended for.

        Args:
            event: Event instance to publish.
            allow_pre_ready: If True and the bus is not yet ready, buffer the
                event and replay it (in arrival order, ahead of any post-ready
                publishes) when ``ready()`` reaches pending=0. Use this for SDK
                handlers that may fire before all workers have subscribed.

        Use only for low-volume control-plane events. Never use for media
        frames.
        """
        if not self._all_ready_event.is_set():
            if allow_pre_ready:
                self._pre_ready_events.append(event)
                return
            raise RuntimeError(
                f"EventBus.publish_nowait({type(event).__name__}) called before all "
                f"workers are ready ({self._pending_ready_count} pending)"
            )

        self._dispatch_nowait(event)

    def _dispatch_nowait(self, event: Any) -> None:
        for queue in self._subscribers.get(type(event), []):
            queue.put_nowait(event)

    @asynccontextmanager
    async def subscribe(
        self,
        *event_types: type[T],
        buffer_size: int | None = None,
    ) -> AsyncIterator[Subscription[T]]:
        """
        Subscribe to one or more event types.

        Args:
            *event_types: Event classes to subscribe to.
            buffer_size: Queue buffer size (defaults to bus default).

        Yields:
            Subscription object for consuming events.

        Example:
            # Pattern 1: Async iteration with pattern matching
            async with bus.subscribe(
                AvatarStartedSpeaking,
                AvatarEndedSpeaking,
            ) as sub:
                async for event in sub:
                    match event:
                        case AvatarStartedSpeaking(phrase_id=pid):
                            logger.info(f"Started: {pid}")
                        case AvatarEndedSpeaking(phrase_id=pid):
                            logger.info(f"Ended: {pid}")

            # Pattern 2: Explicit get_next
            async with bus.subscribe(MetricsEvent) as sub:
                while (event := await sub.get_next()) is not None:
                    await process_metrics(event)

            # Pattern 3: Mixed usage
            async with bus.subscribe(CommandEvent) as sub:
                # Get first event
                first = await sub.get_next()
                if first is None:
                    return

                # Process remaining events
                async for event in sub:
                    handle(event)
        """
        if not event_types:
            raise ValueError("Must subscribe to at least one event type")

        queue: Queue[T | None] = Queue(buffer_size or self._default_buffer_size)

        for event_type in event_types:
            self._subscribers[event_type].append(queue)

        try:
            yield Subscription(queue)
        finally:
            for event_type in event_types:
                self._subscribers[event_type] = [
                    q for q in self._subscribers[event_type] if q is not queue
                ]

            while not queue.empty():
                try:
                    queue.get_nowait()
                except Exception:
                    break
