# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Load-balancer keep-alive worker.

Posts periodic keep-alives to ``$LOAD_BALANCER_URL`` while the FastAPI
app is up, and a ``kill`` message on shutdown so the LB drops this
worker from its pool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, NewType
from urllib.parse import urlencode

from httpx import AsyncClient
from tenacity import retry, stop_after_delay, wait_random

_BASE_URL = os.environ.get("LOAD_BALANCER_URL", "http://renderer-load-balancer")
WorkerId = NewType("WorkerId", str)

LOG = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    id: WorkerId
    url: str


class RendererLoadBalancerClient:
    def __init__(self) -> None:
        self._http_client = AsyncClient()

    @retry(wait=wait_random(min=0, max=1), stop=stop_after_delay(30))
    async def keep_alive(self, worker_info: WorkerInfo, metadata: Any = None) -> None:
        query = {"worker_id": worker_info.id, "worker_url": worker_info.url}
        url = f"{_BASE_URL}/worker/keep_alive?{urlencode(query)}"
        response = await self._http_client.post(url=url, data=metadata)
        response.raise_for_status()

    @retry(wait=wait_random(min=0, max=1), stop=stop_after_delay(30))
    async def kill(self, worker_id: WorkerId, metadata: Any = None) -> None:
        query = {"worker_id": worker_id}
        url = f"{_BASE_URL}/worker/kill?{urlencode(query)}"
        response = await self._http_client.post(url=url, data=metadata)
        response.raise_for_status()


def _get_worker_info() -> WorkerInfo:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return WorkerInfo(id=WorkerId(hostname), url=f"http://{ip}:8000")


@asynccontextmanager
async def keep_alive_worker():
    if _BASE_URL == "disabled":
        yield
        return

    client = RendererLoadBalancerClient()
    worker_info = _get_worker_info()

    async def run_keep_alive_loop() -> None:
        while True:
            try:
                await client.keep_alive(worker_info=worker_info)
            except Exception as e:
                LOG.error("Keep alive loop iteration failed.", exc_info=e)
            await asyncio.sleep(0.5 + random.random() * 0.5)

    keep_alive_loop_task = asyncio.create_task(run_keep_alive_loop())
    try:
        yield
    finally:
        keep_alive_loop_task.cancel()
        try:
            await client.kill(worker_id=worker_info.id)
        except Exception as e:
            LOG.error("kill on shutdown failed", exc_info=e)
