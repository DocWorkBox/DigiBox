# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""MODNet portrait-matting wrapper.

Hides the fixed 288x512 engine resolution: callers feed a frame at any
size and get an alpha matte back at the same size. Inner pre/post-
processing matches the reference's ``PutBack.generate_alpha_channel`` so
the two are bit-comparable within fp16 precision.

I/O contracts live in ``avtr1_renderer.models.matting``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from avtr1_renderer.models.matting import (
    MODNET_INPUT_H,
    MODNET_INPUT_W,
    MODNetEngine,
    MODNetInput,
    MODNetOutput,
)


def matte_chunk(
    chunk_rgb: torch.Tensor,
    *,
    modnet: MODNetEngine,
) -> torch.Tensor:
    """Predict alpha mattes for a whole chunk in one engine call.

    Args:
        chunk_rgb: (N, 3, H, W) float in ``[0, 1]`` on CUDA.
        modnet: MODNet engine (e.g. built via
            ``load_engine(path, MODNetInput, MODNetOutput)``).

    Returns:
        (N, 1, H, W) alpha in ``[0, 1]`` on CUDA, float32.
    """
    assert chunk_rgb.ndim == 4 and chunk_rgb.shape[1] == 3, (
        f"chunk_rgb must be (N, 3, H, W), got {tuple(chunk_rgb.shape)}"
    )
    h, w = chunk_rgb.shape[-2:]
    downsampled = F.interpolate(
        chunk_rgb, size=(MODNET_INPUT_H, MODNET_INPUT_W), mode="bilinear"
    )
    normalised = ((downsampled - 0.5) / 0.5).contiguous()
    output = MODNetOutput(
        output=torch.empty(
            (chunk_rgb.shape[0], 1, MODNET_INPUT_H, MODNET_INPUT_W),
            dtype=torch.float32,
            device=chunk_rgb.device,
        )
    )
    alpha_lr = modnet(MODNetInput(input=normalised), out=output).output
    return F.interpolate(alpha_lr, size=(h, w), mode="bilinear")
