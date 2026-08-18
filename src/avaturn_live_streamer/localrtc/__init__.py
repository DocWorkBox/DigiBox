# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Local-dev WebRTC transport (aiortc-backed).

Parallel to `streamer/rtc.py` (Daily) but for direct browser-peer connections
on localhost / Docker / remote dev VMs. Used by the `local-stream` CLI; not
part of the production stream pipeline.
"""

from avaturn_live_streamer.localrtc.ice import (
    has_turn,
    resolve_ice_servers,
    serialize_ice_servers,
)
from avaturn_live_streamer.localrtc.peer import LocalRTC
from avaturn_live_streamer.localrtc.worklet import LocalRTCWorklet

__all__ = [
    "LocalRTC",
    "LocalRTCWorklet",
    "has_turn",
    "resolve_ice_servers",
    "serialize_ice_servers",
]
