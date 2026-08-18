# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""GPU-side RGB(A) -> packed-uint8 pixel format conversion.

Two output formats are supported, both BT.601 limited-range YUV with 4:2:0
chroma subsampling, both assembled on CUDA so the H2D transfer at the end
of the renderer ships compact bytes:

- ``yuv_i420``: plain I420, packed as ``(1.5 H, W)`` (Y on top, U + V each
  ``(H/4, W)`` stacked below). 1.5 bpp. The default. The renderer will
  have already composited the rendered head over the avatar's bg using
  the predicted alpha matte, so there is no separate alpha plane to ship.
- ``yuv_i420_stacked_alpha``: I420 plus a *second* full I420 carrying the
  alpha luma (alpha as Y, constant 128 for U/V). Layout is six vertically
  stacked planes ``(Y, AY, U, AU, V, AV)`` packed as ``(3 H, W)`` -- 3
  bpp. ffmpeg can encode this as a single ``grey`` stream at ``3 H``
  height, which keeps the alpha channel through any H.264 / VP9 / AV1
  pipeline. The colour planes still hold the bg-composited render; the
  alpha planes let downstream callers replace the bg if they want.

Both conversions are batchable: pass ``(B, 3|4, H, W)`` and get
``(B, *out_shape)`` back. Single-frame inputs ``(3|4, H, W)`` return the
unbatched layout.

Mirrors ``avtr1-live-delta-renderer/src/utils/image/image_transforms.py``;
the alpha-luma LUT was generated from ``cv2.cvtColor(.., COLOR_RGB2YUV_I420)``
on a single-channel image and captures the exact non-linear mapping
ffmpeg's ``grey -> yuv420p`` reads back.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from kornia.color import rgb_to_yuv420

PixelFormat = Literal["yuv_i420", "yuv_i420_stacked_alpha"]

PIXEL_FORMATS_WITH_ALPHA: frozenset[str] = frozenset({"yuv_i420_stacked_alpha"})

_LUT_PATH = Path(__file__).parent / "alpha_to_luma_lut.npy"


def get_bytes_per_frame(h: int, w: int, fmt: PixelFormat) -> int:
    wh = h * w
    if fmt == "yuv_i420":
        return wh * 3 // 2
    if fmt == "yuv_i420_stacked_alpha":
        return wh * 3
    raise ValueError(f"unknown pixel format: {fmt!r}")


@lru_cache(maxsize=2)
def _luma_uv_scale_bias(device: torch.device) -> tuple[torch.Tensor, ...]:
    """BT.601 limited-range scale / bias for Y and U/V planes.

    Maps kornia's float Y in [0, 1] to [16, 235] uint8, and U/V in
    (-0.436, 0.436) / (-0.615, 0.615) to [16, 240] uint8. ``addcmul``-friendly:
    ``out = bias + value * scale``.
    """
    y_scale = (235.0 - 16.0) / (1.0 - 0.0)
    u_scale = (240.0 - 16.0) / (0.436 - (-0.436))
    v_scale = (240.0 - 16.0) / (0.615 - (-0.615))
    y_bias = 16.0 - 0.0 * y_scale
    u_bias = 16.0 - (-0.436) * u_scale
    v_bias = 16.0 - (-0.615) * v_scale

    def _t(values: float | tuple[float, ...]) -> torch.Tensor:
        arr = (values,) if isinstance(values, float) else values
        return torch.tensor(arr, dtype=torch.float32, device=device).view(-1, 1, 1)

    return _t(y_scale), _t(y_bias), _t((u_scale, v_scale)), _t((u_bias, v_bias))


@lru_cache(maxsize=2)
def _alpha_to_luma_lut(device: torch.device) -> torch.Tensor:
    """LUT mapping uint8 alpha -> the Y value ffmpeg yields when round-
    tripping a single-channel image through ``yuv420p``. 256 entries."""
    return torch.as_tensor(np.load(_LUT_PATH), device=device)


def _alpha_chroma(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Constant 128 chroma plane for the alpha "channel" in stacked_alpha."""
    return torch.full((h // 4, w), 128, dtype=torch.uint8, device=device)


def assemble_frame_as_uint8(
    rgb: torch.Tensor,
    alpha: torch.Tensor | None,
    pixel_format: PixelFormat,
) -> torch.Tensor:
    """Pack ``rgb`` (+ ``alpha``) into the requested layout as uint8.

    Args:
        rgb: ``(3, H, W)`` or ``(B, 3, H, W)`` float in [0, 1] on CUDA.
        alpha: matching shape ``(1, H, W)`` / ``(B, 1, H, W)`` float in
            [0, 1] on CUDA. Required for ``yuv_i420_stacked_alpha``;
            ignored (may be ``None``) for ``yuv_i420``.
        pixel_format: see :data:`PixelFormat`.

    Returns:
        uint8 tensor on the same device, contiguous. Shapes:
        - yuv_i420:               ``(3 H // 2, W)`` (or batched ``(B, 3 H // 2, W)``)
        - yuv_i420_stacked_alpha: ``(3 H, W)``      (or batched ``(B, 3 H, W)``)
    """
    if pixel_format in PIXEL_FORMATS_WITH_ALPHA:
        assert alpha is not None, f"{pixel_format} requires an alpha tensor"

    batched = rgb.ndim == 4
    work = rgb if batched else rgb.unsqueeze(0)
    B, _, h, w = work.shape
    assert h % 2 == 0 and w % 2 == 0, "I420 needs even H and W"

    y, uv = rgb_to_yuv420(work)  # y: (B, 1, H, W), uv: (B, 2, H/2, W/2)
    y_scale, y_bias, uv_scale, uv_bias = _luma_uv_scale_bias(work.device)
    y = torch.addcmul(y_bias, y[:, 0], y_scale)  # (B, H, W)
    uv = torch.addcmul(uv_bias, uv, uv_scale)    # (B, 2, H/2, W/2)
    # Pack U + V vertically into (H/2, W) -- U on top, V below. View-only.
    uv = uv.reshape(B, h // 2, w)

    if pixel_format == "yuv_i420":
        out = torch.cat([y, uv], dim=1)  # (B, 1.5 H, W)
        out = out.round_().clamp_(0, 255).to(torch.uint8)
        return out if batched else out[0].contiguous()

    if pixel_format == "yuv_i420_stacked_alpha":
        if alpha is None:
            raise ValueError("Alpha must be provided for yuv_i420_stacked_alpha")
        # Round-trip alpha through the ffmpeg-grey-yuv420p LUT so a downstream
        # H.264 / VP9 / AV1 encoder reproduces the alpha exactly.
        a = alpha if batched else alpha.unsqueeze(0)
        a_idx = (a.clamp(0.0, 1.0) * 255.0).round().to(torch.int64).view(B, h, w)
        ay = _alpha_to_luma_lut(work.device)[a_idx]                            # (B, H, W)
        # Static 128 chroma for the alpha plane, broadcast across batch.
        a_chroma = _alpha_chroma(h, w, work.device).expand(B, -1, -1)          # (B, H/4, W)

        y_u8 = y.round().clamp(0, 255).to(torch.uint8)                          # (B, H, W)
        u_u8, v_u8 = torch.split(
            uv.round().clamp(0, 255).to(torch.uint8), h // 4, dim=1
        )                                                                       # each (B, H/4, W)
        # Layout: Y, AY, U, AU, V, AV vertically -- (B, 3 H, W).
        out = torch.cat([y_u8, ay.to(torch.uint8), u_u8, a_chroma, v_u8, a_chroma], dim=1)
        return out.contiguous() if batched else out[0].contiguous()

    raise ValueError(f"unknown pixel format: {pixel_format!r}")


__all__ = [
    "PIXEL_FORMATS_WITH_ALPHA",
    "PixelFormat",
    "assemble_frame_as_uint8",
    "get_bytes_per_frame",
]
