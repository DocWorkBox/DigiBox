# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Minimal subset of upstream persona_api types.

Only the aliases and primitives used by the localrtc streamer slice are kept
here; the full SessionConfig / TypeID helpers are intentionally omitted because
they pulled in the session + db machinery that we don't vendor.
"""

from enum import StrEnum
from numbers import Rational
from typing import Literal, NewType

from avaturn_live_streamer.sdk_events import AvatarEvent

RendererAvatarId = NewType("RendererAvatarId", str)
BackgroundId = NewType("BackgroundId", str)
PhraseID = NewType("PhraseID", str)

type StreamWorkerEvent = AvatarEvent

type Duration = float | Rational
type OutputAspect = Literal["16:9", "9:16", "1:1", "4:3", "3:4"]
type OutputQuality = Literal["smooth", "balanced", "ultra"]


class PixelFormat(StrEnum):
    YUV_I420 = "yuv_i420"
    YUV_I420_STACKED_ALPHA = "yuv_i420_stacked_alpha"

    @property
    def has_alpha(self) -> bool:
        """Check if the pixel format includes an alpha channel"""
        return self == self.YUV_I420_STACKED_ALPHA

    @property
    def is_yuv(self) -> bool:
        """Check if the pixel format is YUV-based"""
        return True
