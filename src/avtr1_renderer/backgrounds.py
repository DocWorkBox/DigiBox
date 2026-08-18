# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Background image loading for the render pipeline.

Backgrounds are a render-time choice -- they have nothing to do with the
avatar's identity -- so they live in their own registry on the
``Pipeline``, separate from ``Avatar``. ``load_background`` reads a PNG
from disk, resizes to ``(H, W)``, and uploads to CUDA as a
``(1, 3, H, W)`` float tensor in ``[0, 1]``.

The fallback (``path is None``) is a flat white tensor at the requested
resolution, which is what the previous in-Avatar ``_load_bg`` produced
when no background was configured.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def load_background(path: str | Path | None, h: int, w: int) -> torch.Tensor:
    """Load a background image as a ``(1, 3, H, W)`` float CUDA tensor.

    ``path=None`` returns flat white. Any non-None path must point to a
    readable image; missing files raise ``FileNotFoundError``.
    """
    if path is None:
        return torch.ones((1, 3, h, w), dtype=torch.float32, device="cuda")
    path = Path(path)
    bg_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bg_img is None:
        raise FileNotFoundError(f"Background not found: {path}")
    if bg_img.shape[:2] != (h, w):
        bg_img = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_AREA)
    bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (
        torch.from_numpy(np.ascontiguousarray(bg_rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .cuda()
    )


__all__ = ["load_background"]
