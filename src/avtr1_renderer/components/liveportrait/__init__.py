# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""LivePortrait-family components.

The closed face-animation pipeline from the LivePortrait paper:
``MotionExtractor`` (kp / pose / expression bundle) and
``AppearanceFeatureExtractor`` (3D feature volume) precompute per-portrait
state; ``WarpNetwork`` warps the volume given absolute source / driving
keypoints (``x_s`` and ``x_d``, not a delta); ``StitchNetwork`` refines
the driving keypoints to remove mouth-region seams; ``Decoder`` (SPADE)
renders the warped volume to RGB. ``motion_stitch`` is the per-frame
keypoint-transform math that produces ``x_s`` and ``x_d`` for
``WarpNetwork``.
"""
