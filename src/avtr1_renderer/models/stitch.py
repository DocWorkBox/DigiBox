# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Stitch network -- keypoint refinement I/O contract.

The legacy build is fixed-batch ``B=1``; the batched build under
``renderer_engines_b5/stitch_network_b5_fp16.engine`` takes ``B`` in
``[1, 5]`` (Reshape + ScatterND surgery; see
``scripts/build_renderer_engines.py``). The dataclass accepts both.

Engine I/O:
    input  ``kp_source``   FLOAT (B, 21, 3)
    input  ``kp_driving``  FLOAT (B, 21, 3)
    output ``out``         FLOAT (B, 21, 3)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class StitchInput:
    kp_source: torch.Tensor  # (B, 21, 3)
    kp_driving: torch.Tensor  # (B, 21, 3)

    def __post_init__(self) -> None:
        b = self.kp_source.shape[0]
        assert self.kp_source.shape == self.kp_driving.shape == (b, 21, 3), (
            f"kp shapes must be ({b}, 21, 3), got "
            f"source={tuple(self.kp_source.shape)}, "
            f"driving={tuple(self.kp_driving.shape)}"
        )


@dataclass(slots=True)
class StitchOutput:
    out: torch.Tensor  # (B, 21, 3)


StitchEngine = InferenceEngine[StitchInput, StitchOutput]
