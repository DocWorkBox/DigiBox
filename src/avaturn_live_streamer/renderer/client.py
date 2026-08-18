# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
import math
import time
from asyncio.futures import Future
from contextlib import aclosing, asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator, TypedDict
from urllib.parse import urljoin

import anyio
import httpx
from anyio.streams.memory import MemoryObjectSendStream
from asyncstdlib import enumerate as aenumerate
from httpx import AsyncClient
from httpx import Client as SyncClient
from opentelemetry import baggage, metrics
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_before_delay,
    wait_exponential_jitter,
)

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.utils.async_utils import aiotime, run_in_thread

from .interface import AbstractRendererClient, RendererRequest, RenderResponse

_LOGGER = get_logger()
_METER = metrics.get_meter(__name__)

# OpenTelemetry metrics for renderer performance
# fmt: off
_METRIC_TTFF = _METER.create_histogram(
    name="renderer.call.ttff",
    unit="s",
    description="Time to first frame in renderer calls",
    explicit_bucket_boundaries_advisory=[0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.500],
)
_METRIC_MTBF = _METER.create_histogram(
    name="renderer.call.mtbf",
    unit="s",
    description="Mean time between frames in renderer calls",
    explicit_bucket_boundaries_advisory=[0.010, 0.020, 0.033, 0.040, 0.050, 0.067, 0.083, 0.100, 0.150],
)
_METRIC_DURATION = _METER.create_histogram(
    name="renderer.call.duration",
    unit="s",
    description="Total generation time in renderer calls",
    explicit_bucket_boundaries_advisory=[0.050, 0.100, 0.200, 0.300, 0.500, 0.750, 1.0, 1.5, 2.0, 3.0],
)
# fmt: on
_METRIC_STATE_SIZE = _METER.create_gauge(
    name="renderer.call.state_size",
    unit="By",
    description="State size in bytes for renderer calls",
)
# fmt: off
_METRIC_ACQUIRE_DURATION = _METER.create_histogram(
    name="renderer.acquire.duration",
    unit="s",
    description="Time to acquire a renderer worker",
    explicit_bucket_boundaries_advisory=[0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.0, 5.0, 10.0],
)
_METRIC_RELEASE_DURATION = _METER.create_histogram(
    name="renderer.release.duration",
    unit="s",
    description="Time to release a renderer worker",
    explicit_bucket_boundaries_advisory=[0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0],
)
_METRIC_TOTAL_CALL_DURATION = _METER.create_histogram(
    name="renderer.total_call.duration",
    unit="s",
    description="Total duration of renderer call including acquire, render, and release",
    explicit_bucket_boundaries_advisory=[0.100, 0.200, 0.300, 0.500, 0.750, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)
# fmt: on


# Native Windows' portable renderer can take about a second to return headers
# and several seconds between 720p frame chunks.  The former 500 ms read
# timeout was tuned for a warmed remote GPU worker and caused every local
# render to be abandoned before its first frame arrived.
_RENDERER_CALL_TIMEOUTS = httpx.Timeout(
    connect=1.0,
    read=60.0,
    write=30.0,
    pool=5.0,
)
_RENDERER_LB_CALL_TIMEOUTS = httpx.Timeout(12.0, connect=0.005)
_WORKER_ACQUIRE_TIMEOUT = 10


class RendererProtocolError(Exception):
    """Renderer response violated the wire contract (missing/invalid headers,
    truncated body, etc.). Retryable: the next attempt may pick a healthy worker."""


class _WorkerInfo(TypedDict, total=True):
    id: str
    url: str


class RendererClient(AbstractRendererClient):
    """
    SPEC: RendererClient
    ====================

    Drives a single render call against the renderer worker pool: acquire a
    worker from the load balancer (async), POST the audio chunks and stream
    the response body (sync, on a worker thread), then release the worker.

    Why the receive path is sync-on-a-thread
    ----------------------------------------
    The renderer streams ~120 TCP chunks per call (~7.5 MB: 635 KB state +
    5 × 1.4 MB frames). An async httpx receive loop turns every chunk
    boundary into an ``await`` point. On an idle benchmark this is fine
    (~1.6 ms MTBF); inside a live session the same event loop also runs
    ``RTCWorklet``, ``RealtimeApiClient._listener``, the event bus, and
    OpenTelemetry publishers, and every await becomes a scheduling slot
    the renderer task has to wait for -- pushing MTBF to ~13 ms/frame.

    Running ``iter_raw()`` on a sync httpx client inside a worker thread
    (via :func:`avaturn_live_streamer.utils.async_utils.start_in_thread`) keeps the
    receive loop off the event loop entirely. Frames are handed back to
    the asyncio side via ``loop.call_soon_threadsafe`` into an anyio
    memory object stream; the state is published via a ``Future``. The
    async wrapper does ~5 receives per render call (one per frame)
    instead of ~120 await hops on TCP chunks.

    Measured on a live OpenAI realtime session (same host, same
    renderer):

    +----------+--------------------+----------------------+
    | metric   | before (async)     | after (sync+thread)  |
    +==========+====================+======================+
    | ttff     | ~96 ms             | ~82 ms               |
    +----------+--------------------+----------------------+
    | mtbf     | ~13 ms             | ~3 ms                |
    +----------+--------------------+----------------------+
    | duration | ~148 ms            | ~94 ms               |
    +----------+--------------------+----------------------+

    Idle standalone benches are equivalent in both modes (~80 ms / 1.1 ms
    MTBF) -- the win is specifically under loop contention.

    Caveats
    -------
    - The thread cannot be cancelled mid-recv. On early exit (retry or
      consumer close) the wrapper awaits the runner once to drain the
      result. Acceptable because render calls are short (~100 ms wall).
    - Uses the default asyncio threadpool. Currently unbounded; not an
      issue at session-per-process scale, but a dedicated
      ``ThreadPoolExecutor`` would be appropriate for heavy fan-out.
    - ``sync_http_client`` is an injection seam for tests; production
      callers leave it unset and the client builds its own.
    """

    def __init__(
        self,
        model: str,
        lb_url: str | None,
        instance_url: str | None = None,
        *,
        http_client: AsyncClient | None = None,
        sync_http_client: SyncClient | None = None,
    ):
        self.model = model

        if lb_url is not None:
            self._use_lb = True
            self._url = lb_url
        elif instance_url is not None:
            self._use_lb = False
            self._url = instance_url
        else:
            raise ValueError("Either lb_url or instance_url must be provided")

        self._http = http_client or AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=100)
        )
        self._sync_http = sync_http_client or SyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=100)
        )

    @staticmethod
    @lru_cache(maxsize=100)
    def _urls_cached(prefix, suffix):
        return urljoin(prefix, suffix)

    def _get_metric_attributes(self, **extra: str) -> dict[str, str]:
        """Get common metric attributes including stream_id from baggage."""
        attributes: dict[str, str] = {"model": self.model, **extra}

        # Add stream_id from baggage if available
        stream_id = baggage.get_baggage("stream_id")
        if stream_id is not None:
            attributes["stream_id"] = str(stream_id)

        return attributes

    @retry(wait=wait_exponential_jitter(max=5), stop=stop_before_delay(30), reraise=True)
    async def _acquire(self, metadata=None) -> _WorkerInfo:
        start = aiotime()

        if not self._use_lb:
            return {"id": "local", "url": self._url}

        response = await self._http.post(
            url=self._urls_cached(self._url, "/user/acquire_worker"),
            json=metadata,
            timeout=_RENDERER_LB_CALL_TIMEOUTS,
            params={"timeout": _WORKER_ACQUIRE_TIMEOUT},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            _LOGGER.warning("Renderer acquire failed: %s, body=%s", e, response.text)
            raise
        worker_info = response.json()

        # Record acquire duration
        duration = aiotime() - start
        _METRIC_ACQUIRE_DURATION.record(duration, self._get_metric_attributes())

        return worker_info

    @retry(wait=wait_exponential_jitter(max=5), stop=stop_before_delay(30), reraise=True)
    async def _release(self, worker_id: str, metadata=None) -> None:
        start = aiotime()

        if not self._use_lb:
            return

        response = await self._http.post(
            url=self._urls_cached(self._url, "/user/release_worker"),
            json=metadata,
            params={"worker_id": worker_id},
            timeout=_RENDERER_LB_CALL_TIMEOUTS,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            _LOGGER.warning("Renderer release failed: %s, body=%s", e, response.text)
            raise

        # Record release duration
        duration = aiotime() - start
        _METRIC_RELEASE_DURATION.record(duration, self._get_metric_attributes())

    def _run_rendering_sync(
        self,
        params: dict,
        files: dict,
        worker_info: _WorkerInfo,
        loop: asyncio.AbstractEventLoop,
        frames_send: MemoryObjectSendStream[bytes],
    ) -> tuple[bytes | None, list[float]]:
        """Render attempt driven by a sync httpx.Client on a worker thread.

        Frames are handed to the asyncio side via
        ``loop.call_soon_threadsafe``. State is returned to the async
        wrapper only on full success — never published mid-stream — so a
        failed attempt cannot leave an awaiter holding stale state and
        a retry can publish cleanly. Closing ``frames_send`` marks
        end-of-stream (success or failure). The thread never touches
        asyncio primitives directly aside from the threadsafe scheduler.
        """
        start = time.perf_counter()
        try:
            with self._sync_http.stream(
                "POST",
                self._urls_cached(worker_info["url"], "process-audio-v3"),
                params=params,
                files=files,
                timeout=_RENDERER_CALL_TIMEOUTS,
            ) as rsp:
                try:
                    rsp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = rsp.read().decode(errors="replace")
                    _LOGGER.warning("Renderer process-audio failed: %s, body=%s", e, body)
                    raise

                try:
                    frame_size_in_bytes = int(rsp.headers["X-Frame-Length-Bytes"])
                    num_frames = int(rsp.headers["X-Num-Frames"])
                    state_length = int(rsp.headers.get("X-State-Length-Bytes", "0"))
                except (KeyError, ValueError) as e:
                    raise RendererProtocolError(
                        f"missing or invalid renderer response headers: {e}"
                    ) from e

                previous_frame_at = start
                frame_times: list[float] = []

                total = state_length + num_frames * frame_size_in_bytes
                buf = bytearray(total)
                write_pos = 0
                read_pos = state_length
                frames_yielded = 0

                for chunk in rsp.iter_raw():
                    chunk_len = len(chunk)
                    if write_pos + chunk_len > total:
                        raise RendererProtocolError(
                            f"renderer over-produced: header total={total}, "
                            f"got {write_pos + chunk_len}"
                        )
                    buf[write_pos : write_pos + chunk_len] = chunk
                    write_pos += chunk_len

                    if write_pos >= state_length:
                        while (
                            frames_yielded < num_frames
                            and write_pos - read_pos >= frame_size_in_bytes
                        ):
                            frame = bytes(
                                memoryview(buf)[read_pos : read_pos + frame_size_in_bytes]
                            )
                            read_pos += frame_size_in_bytes
                            now = time.perf_counter()
                            frame_times.append(now - previous_frame_at)
                            previous_frame_at = now
                            loop.call_soon_threadsafe(frames_send.send_nowait, frame)
                            frames_yielded += 1

                if write_pos != total:
                    raise RendererProtocolError(
                        f"renderer under-produced: expected {total}, got {write_pos}"
                    )

                state = bytes(memoryview(buf)[:state_length]) if state_length > 0 else None
                return state, frame_times
        finally:
            loop.call_soon_threadsafe(frames_send.close)

    async def _do_rendering_attempt(
        self,
        params: dict,
        files: dict,
        worker_info: _WorkerInfo,
        state_holder: list[bytes | None],
    ) -> AsyncGenerator[bytes, None]:
        """Thin async wrapper around :meth:`_run_rendering_sync`.

        Spawns the sync runner on a thread, drains the frame queue,
        then awaits the thread to surface metrics or exceptions. On
        full success, populates ``state_holder`` so the caller can
        publish state — state is never published from inside a failed
        attempt.
        """
        loop = asyncio.get_running_loop()
        frames_send, frames_recv = anyio.create_memory_object_stream[bytes](math.inf)

        runner = run_in_thread(
            self._run_rendering_sync,
            params,
            files,
            worker_info,
            loop,
            frames_send,
        )
        try:
            try:
                async for item in frames_recv:
                    yield item
                # Surface any thread-side error + capture state/metrics.
                state, frame_times = await runner
                state_holder.append(state)
                await self._record_renderer_metrics(
                    frame_times, len(state) if state is not None else 0, worker_info
                )
            except BaseException:
                # If we exit early (consumer closed, retry condition, etc.),
                # observe the future so a thread error doesn't vanish.
                if not runner.done():
                    # Sync runner can't be cancelled mid-recv; just await it.
                    try:
                        await runner
                    except BaseException:
                        pass
                else:
                    runner.exception()  # mark retrieved
                raise
        finally:
            frames_recv.close()

    async def _produce_frames(
        self, params, files, state_future: Future[bytes | None]
    ) -> AsyncGenerator[bytes, None]:
        start = aiotime()
        returned_frames = 0

        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential_jitter(max=10),
                stop=stop_before_delay(60),
                retry=retry_if_exception_type((httpx.HTTPError, RendererProtocolError)),
                reraise=True,
            ):
                with attempt:
                    worker_info = await self._acquire()
                    state_holder: list[bytes | None] = []
                    worker_id = worker_info["id"]
                    try:
                        async with aclosing(
                            self._do_rendering_attempt(params, files, worker_info, state_holder)
                        ) as frame_stream:
                            async for i_in_stream, frame in aenumerate(frame_stream):
                                if i_in_stream >= returned_frames:
                                    yield frame
                                    returned_frames += 1
                        state_future.set_result(state_holder[0])
                    except (httpx.HTTPError, RendererProtocolError):
                        _LOGGER.warning(
                            "Renderer call attempt %d to %s failed, retrying",
                            attempt.retry_state.attempt_number,
                            worker_id,
                            exc_info=True,
                        )
                        raise
                    finally:
                        try:
                            await self._release(worker_id)
                        except Exception:
                            _LOGGER.exception(
                                "Worker(id=%s) release failed after successful attempt, suppressing",
                                worker_id,
                            )
            # Record total call duration on success
            duration = aiotime() - start
            _METRIC_TOTAL_CALL_DURATION.record(duration, self._get_metric_attributes())
        finally:
            if not state_future.done():
                state_future.set_exception(ValueError("Renderer call failed, no state available"))

    async def _record_renderer_metrics(
        self, frame_times: list[float], state_num_bytes: int, worker_info: _WorkerInfo
    ):
        """Record OpenTelemetry metrics for renderer call performance."""
        ttff = frame_times[0]
        total_generation = sum(frame_times)
        num_frames = len(frame_times)
        mtbf = (total_generation - ttff) / (num_frames - 1)

        attributes = self._get_metric_attributes(worker_url=worker_info["url"])

        _METRIC_TTFF.record(ttff, attributes)
        _METRIC_MTBF.record(mtbf, attributes)
        _METRIC_DURATION.record(total_generation, attributes)
        _METRIC_STATE_SIZE.set(state_num_bytes, attributes)

    async def _build_request_params_files(
        self, request: RendererRequest
    ) -> tuple[dict[str, str | int | float | bool], dict[str, bytes]]:
        files = {
            "current_chunk": request.current_chunk,
            "future_chunk": request.future_chunk,
            "current_chunk_listen": request.current_chunk_listen,
            "future_chunk_listen": request.future_chunk_listen,
            # Hotfix because of the bug in renderer
            "past_chunk": b"",
            "past_chunk_listen": b"",
        }
        if request.state is not None:
            files["state"] = request.state
        else:
            files["state"] = b""

        config = request.config
        params = dict(config.extra_params)
        params.update(
            {
                "h": config.height,
                "w": config.width,
                "avatar_id": config.avatar_id,
                "bg_id": config.background_id,
                "pixel_format": config.pixel_format.value,
                "timestamp_global": request.timestamp_global,
            }
        )

        return params, files

    @asynccontextmanager
    async def generate(self, request: RendererRequest) -> AsyncGenerator[RenderResponse, None]:
        params, files = await self._build_request_params_files(request)

        state_future = asyncio.get_running_loop().create_future()
        async with aclosing(self._produce_frames(params, files, state_future)) as fg:
            yield RenderResponse(fg, 5, state_future)
