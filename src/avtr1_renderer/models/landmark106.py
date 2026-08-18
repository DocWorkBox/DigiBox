# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""InsightFace landmark106 I/O contract.

Engine I/O:
    input  ``data``  FLOAT (1, 3, 192, 192)
    output ``fc1``   FLOAT (1, 212)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class Lm106Input:
    data: torch.Tensor  # (1, 3, 192, 192) float32 CUDA


@dataclass(slots=True)
class Lm106Output:
    fc1: torch.Tensor  # (1, 212) float32 CUDA


Lm106Engine = InferenceEngine[Lm106Input, Lm106Output]
