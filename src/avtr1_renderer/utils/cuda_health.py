# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Smart CUDA healthcheck.

A device-side assert (or ECC error, OOM-killed worker, etc.) poisons
the whole CUDA context, so any subsequent launch on any stream
surfaces it. We probe on a private stream so the check can't deadlock
behind pipeline work and pipeline work can't deadlock behind it.
"""

from __future__ import annotations

import torch

from avtr1_renderer.utils.asyncio import run_in_thread


class CudaHealthChecker:
    def __init__(self) -> None:
        self._stream = torch.cuda.Stream()

    def _probe(self) -> None:
        with torch.cuda.stream(self._stream):
            x = torch.empty(1, device="cuda")
            x.fill_(1.0)
            x.add_(1.0)
        self._stream.synchronize()

    async def check(self) -> None:
        await run_in_thread(self._probe)
