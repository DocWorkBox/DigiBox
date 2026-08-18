# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Generate a reusable transparent listening loop for an avatar.

The live renderer already models two independent audio conditions: speech from
the avatar and speech heard by the avatar.  A short all-silence pass through
both conditions produces the model's neutral idle/listening motion while the
flow sampler still contributes subtle movement.  We render that motion against
the reserved transparent background, recover straight RGBA from AVTR's packed
I420 colour/alpha transport, and encode a ping-pong animated WebP.

Animated WebP is used instead of a theme-specific MP4 so one cached asset can
sit over every CSS theme without regenerating when the user changes themes.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from io import BytesIO

import numpy as np
from PIL import Image

from avtr1_renderer.pipeline import Pipeline
from avtr1_renderer.types import Chunk, RenderOptions

IDLE_LOOP_FPS = 20
IDLE_LOOP_SOURCE_FRAMES = 40
IDLE_LOOP_LOSSLESS = True
IDLE_LOOP_METHOD = 0


def _ping_pong(frames: list[Image.Image]) -> list[Image.Image]:
    if not frames:
        raise ValueError("cannot encode an empty idle loop")
    if len(frames) < 3:
        return list(frames)
    # Do not duplicate either turning point. The last displayed frame flows
    # back through the same poses and the final frame is adjacent to frame 0.
    return [*frames, *frames[-2:0:-1]]


def encode_idle_loop_webp(
    frames: list[Image.Image],
    *,
    fps: int = IDLE_LOOP_FPS,
) -> bytes:
    """Encode RGBA source frames as a forever-looping ping-pong WebP."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    sequence = _ping_pong([frame.convert("RGBA") for frame in frames])
    first_size = sequence[0].size
    if any(frame.size != first_size for frame in sequence):
        raise ValueError("all idle-loop frames must have the same dimensions")
    output = BytesIO()
    sequence[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=sequence[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        lossless=IDLE_LOOP_LOSSLESS,
        quality=100,
        method=IDLE_LOOP_METHOD,
        exact=True,
    )
    return output.getvalue()


def generate_avatar_idle_loop(
    pipeline: Pipeline,
    avatar: object,
    *,
    source_frames: int = IDLE_LOOP_SOURCE_FRAMES,
    fps: int = IDLE_LOOP_FPS,
    inference_guard: Callable[[], AbstractContextManager[None]] | None = None,
) -> bytes:
    """Render and encode one transparent neutral listening loop."""

    if source_frames <= 0:
        raise ValueError("source_frames must be positive")
    motion_generator = pipeline._motion_generator
    window_samples = (
        (motion_generator.chunk_size + motion_generator.future_size)
        * motion_generator.frame_len
        + motion_generator.audio_shift
    )
    silence = np.zeros(window_samples, dtype=np.float32)
    options = RenderOptions(
        cfg_self_audio=0.0,
        cfg_other_audio=2.0,
        cfg_kp=4.0,
    )

    state = None
    decoded: list[Image.Image] = []
    guard = inference_guard or nullcontext
    # The first 200 ms starts from a cold autoregressive state. Advance it but
    # do not cache those frames, so the visible loop begins after settling.
    warmup = Chunk(audio_speech=silence.copy(), audio_listen=silence.copy())
    with guard():
        state, warmup_frames = pipeline.process_transparent_rgba_chunk(
            avatar, warmup, state, options
        )
        for _ in warmup_frames:
            pass
    while len(decoded) < source_frames:
        chunk = Chunk(audio_speech=silence.copy(), audio_listen=silence.copy())
        with guard():
            state, rendered = pipeline.process_transparent_rgba_chunk(
                avatar, chunk, state, options
            )
            for rgba in rendered:
                decoded.append(Image.fromarray(rgba, mode="RGBA"))
                if len(decoded) >= source_frames:
                    break

    return encode_idle_loop_webp(decoded, fps=fps)


__all__ = [
    "IDLE_LOOP_FPS",
    "IDLE_LOOP_LOSSLESS",
    "IDLE_LOOP_METHOD",
    "IDLE_LOOP_SOURCE_FRAMES",
    "encode_idle_loop_webp",
    "generate_avatar_idle_loop",
]
