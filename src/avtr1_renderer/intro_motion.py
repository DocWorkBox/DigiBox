# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Intro animation: a fixed pre-recorded motion played at session start.

A small process-global library: load both ``with_smile`` / ``without_smile``
recordings once (each as a stacked ``MotionFrame``), pick one per call by
avatar id, and retarget + normalise frames into ``past_cond`` shape on
demand. The library is parameterised by the consumer's lipsync coord set
and motion-normaliser stats, so the same code works with both the legacy
``AVTR1`` and the guided generator.

Mirrors the reference's ``_load_intro_motion`` / ``get_intro_motion`` /
``retarget_relative_motion`` / ``transform_motion_to_predictions`` from
``avtr1-live-delta-renderer``.

Layout on disk (``intro_motion``):

- ``09_27_v15.pkl`` -- "with smile". Used for avatars whose id contains
  ``"woman"`` (case-insensitive) or starts with ``"anya"`` (literal prefix).
- ``09_27_v23.pkl`` -- "without smile". Everyone else.

The reference's exact predicate is preserved verbatim, including its
case-insensitivity asymmetry: only the ``"woman"`` check is lowercased.
The recording length must be a multiple of ``chunksize_present`` so the
stream-time slice never spans recordings; the loader asserts this.

Whether to actually play the intro is a per-call decision (the
``play_intro`` kwarg on ``generate_chunk`` / ``process_chunk``). The
library always loads -- only the call site decides what to emit.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import roma
import torch

from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
from avtr1_renderer.types import KPInfo

WITH_SMILE_PKL = "09_27_v15.pkl"
WITHOUT_SMILE_PKL = "09_27_v23.pkl"


def _select_pkl_for_avatar(avatar_id: str) -> str:
    """Pick the recording for an avatar id.

    Reproduces the reference's predicate verbatim -- ``"woman"`` is
    matched case-insensitively but ``"anya"`` is matched as a literal
    prefix on the raw id. Don't refactor without checking the avatar
    registry: there are ids like ``"AnyaPro_v2"`` that fall on the
    ``without_smile`` branch today and we'd silently flip them.
    """
    if "woman" in avatar_id.lower() or avatar_id.startswith("anya"):
        return WITH_SMILE_PKL
    return WITHOUT_SMILE_PKL


class IntroMotion:
    """Process-global intro animation library.

    Loads both pkls once (as stacked ``MotionFrame``s, one per recording)
    and exposes per-avatar selection + per-chunk retarget + past_cond
    normalisation. Construct with the consumer generator's normaliser
    stats; methods are pure and stateless.
    """

    def __init__(
        self,
        intro_dir: str | Path,
        *,
        lipsync_coords: tuple[int, ...],
        so3_offset: torch.Tensor,
        so3_scale: torch.Tensor,
        exp_offset: torch.Tensor,
        exp_scale: torch.Tensor,
        chunksize_present: int = 5,
    ) -> None:
        intro_dir = Path(intro_dir)
        self._lipsync_coords = lipsync_coords
        self._lipsync_index = list(lipsync_coords)
        # Stats live on whatever device the caller provided them on
        # (typically CUDA). Pre-slice the exp stats to lipsync coords so
        # the per-chunk normalisation is one elementwise op.
        device = so3_offset.device
        idx = torch.tensor(self._lipsync_index, dtype=torch.long, device=device)
        self._so3_offset = so3_offset
        self._so3_scale = so3_scale
        self._exp_lipsync_offset = exp_offset.flatten().index_select(0, idx)
        self._exp_lipsync_scale = exp_scale.flatten().index_select(0, idx)

        self._with_smile = self._load_pkl(intro_dir / WITH_SMILE_PKL, chunksize_present)
        self._without_smile = self._load_pkl(
            intro_dir / WITHOUT_SMILE_PKL, chunksize_present
        )

    def _load_pkl(self, path: Path, chunksize_present: int) -> MotionFrame:
        """Load a recording into a stacked ``MotionFrame`` (length = recording size)."""
        with open(path, "rb") as f:
            motion = pickle.load(f)["motion"]
        n = len(motion)
        assert n % chunksize_present == 0, (
            f"Intro recording '{path.name}' has {n} frames, not a "
            f"multiple of chunksize_present={chunksize_present}"
        )
        # Stack into single tensors so all downstream math is one batched op.
        Rs = torch.stack(
            [torch.from_numpy(frame["R"]).reshape(3, 3) for frame in motion],
            dim=0,
        ).cuda().float()  # (N, 3, 3)
        exps = torch.stack(
            [torch.from_numpy(frame["exp"].ravel()[self._lipsync_index]) for frame in motion],
            dim=0,
        ).cuda().float()  # (N, n_lipsync)
        return MotionFrame(R=Rs, exp=exps)

    def for_avatar(self, avatar_id: str) -> MotionFrame:
        """Return the raw recording for ``avatar_id`` (no retarget)."""
        if _select_pkl_for_avatar(avatar_id) == WITH_SMILE_PKL:
            return self._with_smile
        return self._without_smile

    def retarget(
        self,
        window: MotionFrame,
        *,
        anchor: MotionFrame,
        kp_info: KPInfo,
    ) -> MotionFrame:
        """Relative-driving retarget into the avatar's source.

        Replicates ``Audio2Motion.retarget_relative_motion``::

            R_out   = window.R
            exp_out = window.exp - anchor.exp + source.exp[lipsync]

        ``anchor`` is the recording's *first* frame (``recording[0]``,
        a ``MotionFrame`` with ``len == 1``) -- not the slice's first
        frame -- so subsequent windows from the same recording stay
        relative to the same anchor pose. Caller passes both explicitly.

        Output has ``len(window)`` frames.
        """
        x_s_exp = kp_info.exp.view(1, -1)[:, self._lipsync_index]  # (1, n_lip)
        return MotionFrame(R=window.R, exp=window.exp - anchor.exp + x_s_exp)

    def predictions(
        self,
        motions: MotionFrame,
        kp_info: KPInfo,
    ) -> torch.Tensor:
        """Convert retargeted motions into normalised ``past_cond`` rows.

        Mirrors ``Audio2Motion.transform_motion_to_predictions`` in a
        single batched op: splice each frame's lipsync exp into the
        source's full 63-d expression, rotate by the frame's R, re-extract
        the lipsync coords, then z-score normalise. So3 path is
        rotmat -> rotvec -> z-score.

        Returns ``(1, N, 3 + len(lipsync_coords))``. Caller rolls these
        into its ``past_cond``.
        """
        n = len(motions)
        n_lip = len(self._lipsync_index)
        device = kp_info.exp.device
        if n == 0:
            return torch.empty(1, 0, 3 + n_lip, device=device, dtype=torch.float32)

        # Splice lipsync exp into the source's full expression for all N frames.
        src_exp_full = kp_info.exp.view(1, -1).expand(n, -1).clone()  # (N, 63)
        src_exp_full[:, self._lipsync_index] = motions.exp
        # Rotate the full expression by each frame's R, then re-extract lipsync.
        exp_rotated = (src_exp_full.view(n, 21, 3) @ motions.R).view(n, -1)[
            :, self._lipsync_index
        ]  # (N, n_lip)
        so3 = roma.rotmat_to_rotvec(motions.R).view(n, 3)  # (N, 3)

        exp_norm = (exp_rotated - self._exp_lipsync_offset) / self._exp_lipsync_scale
        so3_norm = (so3 - self._so3_offset) / self._so3_scale
        return torch.cat((so3_norm, exp_norm), dim=1).unsqueeze(0)


__all__ = [
    "WITHOUT_SMILE_PKL",
    "WITH_SMILE_PKL",
    "IntroMotion",
]
