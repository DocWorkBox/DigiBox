# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Motion stitching: blend predicted motion deltas with source identity.

The motion model (AVTR1) emits per-frame ``(R, exp)`` over a small subset
of the 21x3 expression coordinates (``LIPSYNC_COORDS``); everything else
stays at the source portrait's value. This module:

1. Builds the full 63-d expression delta by overlaying the predicted
   lipsync coords on top of the source ``exp``.
2. Transforms the source canonical keypoints by ``scale * (kp + delta) @ R + t``.
3. Optionally refines the result through the stitch network (mouth-region
   seam removal).
4. Returns ``(x_s, x_d)`` -- ``x_s`` is ``(1, 21, 3)`` (per-avatar), ``x_d``
   is ``(N, 21, 3)`` (one per motion frame in the batch) -- which the warp
   network then consumes per frame.

Batched along the leading frame dim. ``MotionFrame`` carries N motions
stacked as ``(N, 3, 3)`` rotation + ``(N, len(LIPSYNC_COORDS))`` exp; the
math (delta build, driving transform, source transform) is one op for any
N. The stitch TRT engine takes batch=1 inputs, so it's iterated inside
``motion_stitch``; the warp / decoder / matting downstream of it iterate
the same way.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.models.stitch import StitchEngine, StitchInput, StitchOutput
from avtr1_renderer.types import KPInfo

_LIPSYNC_INDEX = list(LIPSYNC_COORDS)


@dataclass(slots=True, frozen=True)
class MotionFrame:
    """One or more motion-model predictions stacked along a leading frame dim.

    The leading dim ``N`` (>= 1) lets a single instance hold either a
    chunk's worth of predictions (``N=5`` from AVTR1) or an entire intro
    recording (``N=120``). All math (retarget, splice into source exp,
    transform driving keypoints) is one batched op for any N.

    Iteration yields ``N=1`` views of individual frames *without copy*,
    so consumers that drive batch=1 TRT engines (stitch / warp / decoder)
    can still iterate naturally::

        for frame in motions:           # frame.R: (1, 3, 3), frame.exp: (1, n_lip)
            x_s, x_d = motion_stitch(kp_info, frame, stitch=...)

    Indexing follows the same shape contract: ``motions[i]`` and
    ``motions[a:b]`` return ``MotionFrame`` instances; the int form
    returns a single-frame batch (``N=1``), the slice form returns the
    natural slice length.

    Despite the singular name, *N can be > 1*. Renamed-by-not-renaming:
    ``MotionFrame`` is the type of "a frame's worth of motion data" and a
    stacked batch is just N of those, so reusing the name keeps every
    call site readable. ``len(motions)`` is the canonical way to ask "how
    many frames are in here".
    """

    R: torch.Tensor  # (N, 3, 3) rotation matrices
    exp: torch.Tensor  # (N, len(LIPSYNC_COORDS)) lipsync expression deltas

    def __len__(self) -> int:
        return self.R.shape[0]

    def __iter__(self) -> Iterator[MotionFrame]:
        for i in range(self.R.shape[0]):
            yield MotionFrame(R=self.R[i : i + 1], exp=self.exp[i : i + 1])

    def __getitem__(self, idx: int | slice) -> MotionFrame:
        if isinstance(idx, int):
            if idx < 0:
                idx += self.R.shape[0]
            return MotionFrame(R=self.R[idx : idx + 1], exp=self.exp[idx : idx + 1])
        return MotionFrame(R=self.R[idx], exp=self.exp[idx])


def _transform_source(kp_info: KPInfo) -> torch.Tensor:
    """Transform the source canonical keypoints into pixel space.

    Replicates the reference's ``transform_keypoint_xs``::

        kp_transformed = scale * ((kp @ R) + exp) + t_xy

    Returns ``(1, 21, 3)`` -- per-avatar, no frame dim.
    """
    bs, num_kp, _ = kp_info.kp.shape
    kp_t = kp_info.kp.view(bs, num_kp, 3) @ kp_info.R + kp_info.exp.view(bs, num_kp, 3)
    kp_t = kp_t * kp_info.scale[..., None]
    kp_t[:, :, 0:2] = kp_t[:, :, 0:2] + kp_info.t[:, None, 0:2]
    return kp_t


def _build_x_d_delta(motion: MotionFrame, kp_info: KPInfo) -> torch.Tensor:
    """Splice predicted lipsync coords into the source's full expression.

    Batched along the frame dim: ``motion`` carries ``N >= 1`` frames, the
    output is ``(N, 21, 3)``. Mirrors the reference's
    ``delta_new[:, LIPSYNC_COORDS] = motion.exp`` per frame -- and yes,
    ``LIPSYNC_COORDS`` contains a duplicate (54 appears at index 30 and
    32) so the second write wins for that slot. The duplicate-write is
    nondeterministic on CUDA (and stays nondeterministic in batched form,
    independently per row), which is why parity tests mask out keypoint
    18 (= flat index 54).
    """
    n = motion.R.shape[0]
    delta = kp_info.exp.flatten(1).expand(n, -1).clone()  # (N, 63)
    delta[:, _LIPSYNC_INDEX] = motion.exp
    return delta.view(n, 21, 3)


def _transform_driving(
    x_d_delta: torch.Tensor,
    R_new: torch.Tensor,
    kp_info: KPInfo,
) -> torch.Tensor:
    """Replicates ``transform_keypoint_xd`` with ``R_new`` non-None, batched.

    ``x_d_delta`` is ``(N, 21, 3)``, ``R_new`` is ``(N, 3, 3)``. Output is
    ``(N, 21, 3)``::

        x_d_new = scale * ((kp_canonical + x_d_delta) @ R_new) + t

    The avatar's per-call constants ``kp.kp``/``scale``/``t`` broadcast
    across the frame dim.
    """
    x_c = kp_info.kp.view(1, 21, 3)
    x_d = kp_info.scale * ((x_c + x_d_delta) @ R_new) + kp_info.t.view(1, 1, 3)
    return x_d


def motion_stitch(
    kp_info: KPInfo,
    motions: MotionFrame,
    *,
    stitch: StitchEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_s, x_d)`` keypoint pairs ready for the warp network.

    Args:
        kp_info: source-side keypoint bundle from the avatar (per-avatar,
            no frame dim).
        motions: ``N`` motion frames stacked. ``N=1`` is a normal case --
            a single-frame caller passes ``MotionFrame`` with leading dim
            1 and gets ``(1, 21, 3)`` outputs back.
        stitch: stitch-network engine that refines the driving keypoints
            (closes the mouth-region seam). The engine takes batch=1
            inputs, so it's iterated inside this function.

    Returns:
        Tuple ``(x_s, x_d)``:
            - ``x_s``: ``(1, 21, 3)`` source keypoints, on CUDA float32.
            - ``x_d``: ``(N, 21, 3)`` driving keypoints, one per input
              frame, post-stitch.

    The stitch engine is built batch-dynamic (b=1..5), so this is a
    single call -- no per-frame Python loop, no per-call kernel launch
    overhead. ``kp_source`` is broadcast across the batch via
    ``expand``; ``kp_driving`` is the (N, 21, 3) tensor of pre-stitch
    driving keypoints.

    Tests that want the pre-stitch math can call ``_transform_source`` /
    ``_build_x_d_delta`` / ``_transform_driving`` directly.
    """
    x_d_delta = _build_x_d_delta(motions, kp_info)
    x_d_raw = _transform_driving(x_d_delta, motions.R, kp_info)
    x_s = _transform_source(kp_info)

    n = x_d_raw.shape[0]
    x_s_b = x_s.expand(n, -1, -1).contiguous()
    output = StitchOutput(out=torch.empty_like(x_d_raw))
    x_d = stitch(
        StitchInput(kp_source=x_s_b, kp_driving=x_d_raw.contiguous()),
        out=output,
    ).out
    return x_s, x_d


__all__ = ["MotionFrame", "motion_stitch"]
