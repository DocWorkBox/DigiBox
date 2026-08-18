# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Face landmark wrappers used during avatar registration.

Two pure functions:
- ``landmark106`` (insightface) -- 106-point landmarks from a bounding
  box, used to seed the cascade.
- ``landmark203`` (LivePortrait) -- 203-point landmarks from a 224x224
  crop; these drive the final 512x512 source crop.

I/O contracts live in ``avtr1_renderer.models.{landmark106,landmark203}``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from skimage import transform as trans

from avtr1_renderer.models.landmark106 import Lm106Engine, Lm106Input
from avtr1_renderer.models.landmark203 import Lm203Engine, Lm203Input

# ---- landmark106 ----

LM106_INPUT_SIZE = 192
LM106_LMK_NUM = 106


def _similarity_transform(
    img: np.ndarray, center: tuple[float, float], output_size: int, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact replica of insightface's ``transform`` helper."""
    t1 = trans.SimilarityTransform(scale=scale)
    cx = center[0] * scale
    cy = center[1] * scale
    t2 = trans.SimilarityTransform(translation=(-cx, -cy))
    t3 = trans.SimilarityTransform(rotation=0)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(img, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def _trans_points2d(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    new_pts = np.zeros(pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = np.array([pts[i, 0], pts[i, 1], 1.0], dtype=np.float32)
        new_pts[i] = (M @ pt)[0:2]
    return new_pts


def landmark106(
    img_rgb: np.ndarray,
    bbox: np.ndarray,
    *,
    lm106: Lm106Engine,
) -> np.ndarray:
    """Predict 106 2D landmarks from a face bounding box.

    Args:
        img_rgb: HWC uint8 image.
        bbox: ``(4,)`` ``[x1, y1, x2, y2]`` (or any longer; only the
            first 4 entries are read).
        lm106: insightface landmark106 engine.

    Returns:
        ``(106, 2)`` landmarks already mapped back to the original image
        coordinate system.
    """
    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
    scale = LM106_INPUT_SIZE / (max(w, h) * 1.5)

    aimg, M = _similarity_transform(img_rgb, center, LM106_INPUT_SIZE, scale)

    # Same blobFromImage as the reference: mean=0, std=1, swapRB=True.
    blob = cv2.dnn.blobFromImage(
        aimg, 1.0, (LM106_INPUT_SIZE, LM106_INPUT_SIZE), (0.0, 0.0, 0.0), swapRB=True
    )
    inp = torch.from_numpy(blob).cuda().contiguous()
    out = lm106(Lm106Input(data=inp))
    torch.cuda.synchronize()
    pred = out.fc1.cpu().numpy()

    pred = pred.reshape((-1, 2))
    if pred.shape[0] > LM106_LMK_NUM:
        pred = pred[-LM106_LMK_NUM:, :]
    pred[:, 0:2] += 1
    pred[:, 0:2] *= LM106_INPUT_SIZE // 2

    IM = cv2.invertAffineTransform(M)
    return _trans_points2d(pred, IM)


# ---- landmark203 ----

LM203_DSIZE = 224


def _transform_pts(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    return pts @ M[:2, :2].T + M[:2, 2]


def landmark203(
    img_crop_rgb: np.ndarray,
    M_c2o: np.ndarray | None = None,
    *,
    lm203: Lm203Engine,
) -> np.ndarray:
    """Predict 203 2D landmarks from a pre-cropped 224x224 image.

    Args:
        img_crop_rgb: 224x224x3 uint8. Must already be aligned (typically
            the result of a 1.5x scale crop seeded from the 106-point
            detector).
        M_c2o: optional crop->original affine; when supplied the
            returned landmarks are mapped back to the original frame
            coordinates.
        lm203: LivePortrait landmark203 engine.

    Returns:
        ``(203, 2)`` landmarks.
    """
    inp_np = (img_crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    inp = torch.from_numpy(np.ascontiguousarray(inp_np)).cuda()
    out = lm203(Lm203Input(input=inp))
    torch.cuda.synchronize()
    out_pts = out.landmarks.cpu().numpy()
    lmk = out_pts[0].reshape(-1, 2) * LM203_DSIZE
    if M_c2o is not None:
        lmk = _transform_pts(lmk, M_c2o)
    return lmk
