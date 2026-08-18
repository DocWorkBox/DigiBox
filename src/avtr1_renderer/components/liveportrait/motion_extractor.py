# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Motion extractor wrapper.

Calls a ``MotionExtractorEngine`` and runs the bin66->degree softmax-
expectation post-process plus the rotation-matrix derivation, returning a
typed ``KPInfo``.

I/O contracts live in ``avtr1_renderer.models.motion_extractor``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from avtr1_renderer.models.motion_extractor import MotionExtractorEngine, MotionInput
from avtr1_renderer.types import KPInfo


def _bin66_to_degree(pred: torch.Tensor) -> torch.Tensor:
    """Convert a (1, 66) logit row to a (1, 1) degree value.

    Same formula as the reference's ``headpose_pred_to_degree``: softmax,
    take the expectation against the bin index, scale by 3, subtract 97.5.
    """
    assert pred.ndim == 2 and pred.shape[1] == 66, f"expected (1, 66), got {tuple(pred.shape)}"
    idx = torch.arange(66, device=pred.device, dtype=pred.dtype)
    p = F.softmax(pred, dim=1)
    deg = (p * idx).sum(dim=1) * 3 - 97.5
    return deg.view(1, -1)


def _get_rotation_matrix(
    pitch: torch.Tensor, yaw: torch.Tensor, roll: torch.Tensor
) -> torch.Tensor:
    """``pitch/yaw/roll`` are in degrees, shape (1, 1). Returns (1, 3, 3).

    Matches the reference exactly: ``R = (Rz(roll) @ Ry(yaw) @ Rx(pitch)).T``.
    """
    pi = float(np.pi)
    p = pitch / 180.0 * pi
    y = yaw / 180.0 * pi
    r = roll / 180.0 * pi

    bs = p.shape[0]
    ones = torch.ones((bs, 1), device=p.device, dtype=p.dtype)
    zeros = torch.zeros_like(ones)

    rot_x = torch.cat(
        [ones, zeros, zeros, zeros, p.cos(), -p.sin(), zeros, p.sin(), p.cos()],
        dim=1,
    ).reshape(bs, 3, 3)
    rot_y = torch.cat(
        [y.cos(), zeros, y.sin(), zeros, ones, zeros, -y.sin(), zeros, y.cos()],
        dim=1,
    ).reshape(bs, 3, 3)
    rot_z = torch.cat(
        [r.cos(), -r.sin(), zeros, r.sin(), r.cos(), zeros, zeros, zeros, ones],
        dim=1,
    ).reshape(bs, 3, 3)

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)


def extract_motion(
    image_bchw: torch.Tensor,
    *,
    motion: MotionExtractorEngine,
) -> KPInfo:
    """Encode a 256x256 source crop into ``KPInfo``.

    Args:
        image_bchw: ``(1, 3, 256, 256)`` float32 CUDA in ``[0, 1]``.
        motion: motion extractor engine.

    Returns:
        Typed ``KPInfo`` with degree-scalar pose angles and a derived
        rotation matrix.
    """
    out = motion(MotionInput(image=image_bchw.contiguous()))

    pitch = _bin66_to_degree(out.pitch)
    yaw = _bin66_to_degree(out.yaw)
    roll = _bin66_to_degree(out.roll)
    R = _get_rotation_matrix(pitch, yaw, roll)

    return KPInfo(
        kp=out.kp.reshape(1, 21, 3).contiguous(),
        exp=out.exp.reshape(1, 21, 3).contiguous(),
        scale=out.scale.contiguous(),
        t=out.t.contiguous(),
        pitch=pitch.contiguous(),
        yaw=yaw.contiguous(),
        roll=roll.contiguous(),
        R=R.contiguous(),
    )
