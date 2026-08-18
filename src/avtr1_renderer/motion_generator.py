# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Motion generation Protocol.

The ``MotionGenerator`` Protocol isolates the speech->motion half of the
streaming orchestrator behind a swappable interface. The ``StateT`` type
variable lets a generator pick its own per-session state shape -- the
renderer never inspects it, so different motion models can carry whatever
they need.

Today there's one concrete impl: :class:`AVTR1MotionGenerator` (in
``avtr1_motion_generator``), which wraps the AVTR1 flow-matching
encode + decode TRT engines.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
from avtr1_renderer.types import Chunk, RenderOptions

StateT = TypeVar("StateT")


class MotionGenerator(Protocol[StateT]):
    """Speech -> motion contract.

    Implementations own whatever engines + samplers they need. The state
    blob is opaque to callers (and to the renderer); only ``initial_state``
    constructs it and only ``generate_chunk`` advances it.

    Per-request knobs come in via ``options`` (a ``RenderOptions``).
    Implementations read only the fields they understand; extra fields
    (CFG weights, noise params) are read by the concrete generator and
    silently ignored by others.

    ``generate_chunk`` returns a stacked ``MotionFrame`` (``len ==
    chunk_size`` in normal operation); the renderer iterates it per frame.
    """

    def initial_state(self, avatar: Avatar) -> StateT: ...

    def generate_chunk(
        self,
        audio_chunk: Chunk,
        avatar: Avatar,
        state: StateT,
        options: RenderOptions | None = None,
    ) -> tuple[MotionFrame, StateT]: ...


__all__ = ["MotionGenerator"]
