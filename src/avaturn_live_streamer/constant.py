# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from fractions import Fraction

RENDERER_SPEECH_SAMPLE_RATE = 16_000

NATIVE_SPEECH_SAMPLE_RATE = 24_000
OPENAI_SPEECH_SAMPLE_RATE = 24_000

VIDEO_RESOLUTION = (720, 1280)
VIDEO_FPS = 25
FRAMES_PER_RENDER_CHUNK = 5
FRAME_DURATION = Fraction(1, VIDEO_FPS)

MAX_STREAM_DURATION = 24 * 60 * 60 - 5  # 24H - 5 seconds to leave time for proper shutdown
