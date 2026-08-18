# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Pack a GPU RGB(A) chunk into host ``Frame``s.

The render pipeline produces a per-chunk RGB or RGBA float tensor on CUDA
(``(N, 3, H, W)`` for ``yuv_i420`` output, ``(N, 4, H, W)`` for
``yuv_i420_stacked_alpha``). :func:`pack_frames` is the last stage:
pixel-format pack, then a single batched H2D copy to numpy.

One batched ``assemble_frame_as_uint8`` + one ``.cpu().numpy()`` H2D per
chunk -- the bandwidth win documented in ``scratchbook/speedup.md`` over
five small per-frame copies.
"""

from __future__ import annotations

import torch

from avtr1_renderer.components.pixel_format import (
    PIXEL_FORMATS_WITH_ALPHA,
    PixelFormat,
    assemble_frame_as_uint8,
)
from avtr1_renderer.types import Frame


@torch.no_grad()
def pack_frames(
    rgb: torch.Tensor,
    alpha: torch.Tensor | None,
    *,
    pixel_format: PixelFormat,
) -> list[Frame]:
    """Pack a renderer-produced chunk into a list of host ``Frame``s.

    Args:
        rgb: ``(N, 3, H, W)`` float CUDA in ``[0, 1]`` -- the composited
            render.
        alpha: ``(N, 1, H, W)`` float CUDA in ``[0, 1]`` -- the MODNet
            matte. Required for ``yuv_i420_stacked_alpha``, ignored (and
            may be ``None``) for ``yuv_i420``.
        pixel_format: output frame layout.

    Returns:
        ``list[Frame]`` of length ``N``. One batched
        ``assemble_frame_as_uint8`` call + one ``.cpu().numpy()`` H2D.
    """
    needs_alpha = pixel_format in PIXEL_FORMATS_WITH_ALPHA
    assert rgb.ndim == 4 and rgb.shape[1] == 3, (
        f"rgb must be (N, 3, H, W), got {tuple(rgb.shape)}"
    )
    if needs_alpha:
        assert alpha is not None and alpha.shape[:2] == (rgb.shape[0], 1), (
            f"{pixel_format} requires alpha of shape (N, 1, H, W); "
            f"got {None if alpha is None else tuple(alpha.shape)}"
        )

    h, w = rgb.shape[-2:]
    rgb_clamped = rgb.clamp(0.0, 1.0)
    alpha_clamped = alpha.clamp(0.0, 1.0) if needs_alpha else None

    packed = assemble_frame_as_uint8(rgb_clamped, alpha_clamped, pixel_format)
    host = packed.contiguous().cpu().numpy()
    return [
        Frame(data=host[i], format=pixel_format, height=h, width=w)
        for i in range(host.shape[0])
    ]


__all__ = ["pack_frames"]
