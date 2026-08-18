# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""MODNet portrait-matting engine I/O contract.

The engine expects the input downsampled to 288x512 and normalised to
``[-1, 1]``; the alpha output is in ``[0, 1]``. The resize +
normalise + alpha upsample steps live in the ``components/matting.py``
caller, not here.

The legacy build is fixed-batch ``B=1``; the batched build under
``renderer_engines_b5/modnet_b5_fp16.engine`` takes ``B`` in ``[1, 5]``.
The dataclass accepts both.

Engine I/O:
    input  ``input``   FLOAT (B, 3, 288, 512)
    output ``output``  FLOAT (B, 1, 288, 512)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine

# Engine input resolution. Baked into the engine -- the build profile is
# (B, 3, 288, 512) for min/opt/max alike on the spatial dims.
MODNET_INPUT_H = 288
MODNET_INPUT_W = 512


@dataclass(slots=True)
class MODNetInput:
    input: torch.Tensor  # (B, 3, 288, 512) float32 CUDA, normalised to [-1, 1]

    def __post_init__(self) -> None:
        b = self.input.shape[0]
        assert self.input.shape == (b, 3, MODNET_INPUT_H, MODNET_INPUT_W), (
            f"input must be (B, 3, {MODNET_INPUT_H}, {MODNET_INPUT_W}), "
            f"got {tuple(self.input.shape)}"
        )


@dataclass(slots=True)
class MODNetOutput:
    output: torch.Tensor  # (B, 1, 288, 512) float32 CUDA, alpha in [0, 1]


MODNetEngine = InferenceEngine[MODNetInput, MODNetOutput]


__all__ = [
    "MODNET_INPUT_H",
    "MODNET_INPUT_W",
    "MODNetEngine",
    "MODNetInput",
    "MODNetOutput",
]
