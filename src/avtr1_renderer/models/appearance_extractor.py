# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""LivePortrait appearance extractor I/O contract.

Engine I/O:
    input  ``image``  FLOAT (1, 3, 256, 256)
    output ``pred``   FLOAT (1, 32, 16, 64, 64)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class AEInput:
    image: torch.Tensor  # (1, 3, 256, 256) float32 CUDA, [0, 1]

    def __post_init__(self) -> None:
        assert self.image.shape == (1, 3, 256, 256), (
            f"image must be (1, 3, 256, 256), got {tuple(self.image.shape)}"
        )


@dataclass(slots=True)
class AEOutput:
    pred: torch.Tensor  # (1, 32, 16, 64, 64) float32 CUDA


AppearanceExtractorEngine = InferenceEngine[AEInput, AEOutput]
