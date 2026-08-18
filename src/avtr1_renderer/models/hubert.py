# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""HuBERT speech-feature extractor I/O contract.

Engine I/O (verified by ``engine.get_tensor_*`` introspection):
    input  ``input_values``      FLOAT (B, -1)
    output ``last_hidden_state`` FLOAT (B, -1, 1024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class HubertInput:
    input_values: torch.Tensor  # (B, N) float32 CUDA, audio in [-1, 1]

    def __post_init__(self) -> None:
        assert self.input_values.ndim == 2, (
            f"input_values must be (B, N), got {tuple(self.input_values.shape)}"
        )


@dataclass(slots=True)
class HubertOutput:
    last_hidden_state: torch.Tensor  # (B, frames, 1024) float32 CUDA


HubertEngine = InferenceEngine[HubertInput, HubertOutput]
