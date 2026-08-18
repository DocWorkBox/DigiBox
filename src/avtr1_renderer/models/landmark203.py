# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""LivePortrait landmark203 I/O contract.

Engine I/O:
    input  ``input``      FLOAT (1, 3, 224, 224)
    output ``landmarks``  FLOAT (1, 406)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class Lm203Input:
    input: torch.Tensor  # (1, 3, 224, 224) float32 CUDA


@dataclass(slots=True)
class Lm203Output:
    landmarks: torch.Tensor  # (1, 406) float32 CUDA


Lm203Engine = InferenceEngine[Lm203Input, Lm203Output]
