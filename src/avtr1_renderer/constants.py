# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Constants used across pipeline components.

The motion model only predicts these 39 indices into the flattened
21*3 = 63 expression vector; the remaining 24 stay at the
source-portrait values.
"""

# fmt: off
LIPSYNC_COORDS: tuple[int, ...] = (
    3, 4, 5,
    6, 7, 8,
    18, 19, 20,
    33, 34, 35,
    36, 37, 38,
    39, 40, 41,
    42, 43, 44,
    45, 46, 47,
    48, 49, 50,
    51, 52, 53,
    54, 55, 56,
    57, 58, 59,
    60, 61, 62,
)
# fmt: on


__all__ = ["LIPSYNC_COORDS"]
