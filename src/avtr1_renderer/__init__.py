# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""avtr1-live-talkinghead — real-time talking-head renderer.

Quickstart::

    from avtr1_renderer import Pipeline, RenderOptions, Chunk

    pipeline, registry = Pipeline.from_artifacts(avatar_ids=["anya_03_studio"])
    avatar = registry["anya_03_studio"]

    state = None
    for chunk in audio_chunks:
        state, frames = pipeline.process_chunk(avatar, chunk, state)
        for frame in frames:
            yield frame  # 25 fps YUV frames
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from avtr1_renderer.avatar_loader import Avatar
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.types import Chunk, Frame, FrameIterator, RenderOptions


def __getattr__(name: str) -> Any:
    """Import the public API lazily so lightweight submodules stay optional."""
    if name == "Avatar":
        from avtr1_renderer.avatar_loader import Avatar

        return Avatar
    if name == "Pipeline":
        from avtr1_renderer.pipeline import Pipeline

        return Pipeline
    if name in {"Chunk", "Frame", "FrameIterator", "RenderOptions"}:
        from avtr1_renderer import types

        return getattr(types, name)
    raise AttributeError(name)

__all__ = ["Avatar", "Chunk", "Frame", "FrameIterator", "Pipeline", "RenderOptions"]
