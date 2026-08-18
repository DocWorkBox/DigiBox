# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""SPADE decoder I/O contract.

The legacy build is fixed-batch ``B=1``; the batched build under
``renderer_engines_b5/decoder_b5_fp16.engine`` takes ``B`` in ``[1, 5]``.
The dataclass accepts both.

Engine I/O:
    input  ``feature`` FLOAT (B, 256, 64, 64)
    output ``output``  FLOAT (B, 3, 512, 512)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class DecoderInput:
    feature: torch.Tensor  # (B, 256, 64, 64) float32 CUDA

    def __post_init__(self) -> None:
        b = self.feature.shape[0]
        assert self.feature.shape == (b, 256, 64, 64), (
            f"feature must be (B, 256, 64, 64), got {tuple(self.feature.shape)}"
        )


@dataclass(slots=True)
class DecoderOutput:
    output: torch.Tensor  # (B, 3, 512, 512) float32 CUDA


DecoderEngine = InferenceEngine[DecoderInput, DecoderOutput]
