# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""InsightFace SCRFD face detector I/O contract.

Engine I/O:
    input  ``image``  FLOAT (1, 3, 512, 512)
    outputs:
        ``scores1`` (8192, 1), ``scores2`` (2048, 1), ``scores3`` (512, 1)
        ``boxes1``  (8192, 4), ``boxes2``  (2048, 4), ``boxes3``  (512, 4)
        ``kps1``    (8192,10), ``kps2``    (2048,10), ``kps3``    (512,10)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class FaceDetInput:
    image: torch.Tensor  # (1, 3, 512, 512) float32 CUDA


@dataclass(slots=True)
class FaceDetOutput:
    scores1: torch.Tensor  # (8192, 1)
    scores2: torch.Tensor  # (2048, 1)
    scores3: torch.Tensor  # (512, 1)
    boxes1: torch.Tensor  # (8192, 4)
    boxes2: torch.Tensor  # (2048, 4)
    boxes3: torch.Tensor  # (512, 4)
    kps1: torch.Tensor  # (8192, 10)
    kps2: torch.Tensor  # (2048, 10)
    kps3: torch.Tensor  # (512, 10)


FaceDetEngine = InferenceEngine[FaceDetInput, FaceDetOutput]
