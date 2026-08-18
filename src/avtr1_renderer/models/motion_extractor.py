# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""LivePortrait motion extractor I/O contract.

The raw engine emits pitch/yaw/roll as 66-bin classification logits; the
``KPInfo`` returned by :func:`extract_motion` is a single scalar angle (in
degrees) per axis. The softmax+expectation post-process lives in the
component module, not here.

Engine I/O:
    input  ``image``  FLOAT (1, 3, 256, 256)
    outputs:
        ``pitch``  (1, 66), ``yaw`` (1, 66), ``roll`` (1, 66)
        ``t``      (1, 3),  ``exp`` (1, 63),
        ``scale``  (1, 1),  ``kp``  (1, 63)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class MotionInput:
    image: torch.Tensor  # (1, 3, 256, 256) float32 CUDA, [0, 1]

    def __post_init__(self) -> None:
        assert self.image.shape == (1, 3, 256, 256), (
            f"image must be (1, 3, 256, 256), got {tuple(self.image.shape)}"
        )


@dataclass(slots=True)
class MotionOutput:
    pitch: torch.Tensor  # (1, 66)
    yaw: torch.Tensor  # (1, 66)
    roll: torch.Tensor  # (1, 66)
    t: torch.Tensor  # (1, 3)
    exp: torch.Tensor  # (1, 63)
    scale: torch.Tensor  # (1, 1)
    kp: torch.Tensor  # (1, 63)


MotionExtractorEngine = InferenceEngine[MotionInput, MotionOutput]
