# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Renderer: stacked ``MotionFrame`` -> per-chunk RGB(A) tensor on GPU.

The motion->pixel half of the streaming orchestrator. Stateless: every
call takes the avatar (immutable, per-portrait) and a stacked
``MotionFrame`` (``len == N``), and either returns one batched ``(N, C,
H, W)`` tensor or yields ``(1, C, H, W)`` slices one at a time.

The warp engine runs once on the full chunk (the b=5 batched path
documented in ``scratchbook/speedup.md`` -- warp shares work across
the chunk and the win there is real). Everything downstream of the
warp -- decoder, pasteback, matting, bg composite -- runs per frame
in :func:`render_chunk_streaming` so the pipeline can yield each
finished frame before the next one starts.

``motion_stitch`` still iterates the b=1 stitch engine internally
since its compute is small.

Pixel-format conversion + H2D copy live in :func:`pack_frames` so the
renderer can be paired with arbitrary output formats without touching
this body.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame, motion_stitch
from avtr1_renderer.components.putback import putback_chunk
from avtr1_renderer.models.decoder import DecoderEngine, DecoderInput, DecoderOutput
from avtr1_renderer.models.matting import MODNetEngine
from avtr1_renderer.models.stitch import StitchEngine
from avtr1_renderer.models.warp import WarpEngine, WarpInput, WarpOutput


def _run_warp(warp: WarpEngine, inputs: WarpInput) -> torch.Tensor:
    """Run warp with an explicit output buffer for dynamic ONNX batches."""
    batch = inputs.feature_3d.shape[0]
    output = WarpOutput(
        out=torch.empty(
            (batch, 256, 64, 64),
            dtype=torch.float32,
            device=inputs.feature_3d.device,
        )
    )
    return warp(inputs, out=output).out


def render_chunk_streaming(
    motions: MotionFrame,
    avatar: Avatar,
    bg: torch.Tensor,
    *,
    stitch: StitchEngine,
    warp: WarpEngine,
    decoder: DecoderEngine,
    matting: MODNetEngine,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(rgb_1, alpha_1)`` per frame; warp once, everything else per frame.

    Same inputs as :func:`render_chunk`. The warp engine still runs on
    the full chunk because its per-element work overlaps; once warped
    features exist, the decoder + putback + matting + bg composite for
    frame ``i`` runs and yields before frame ``i+1`` starts. Callers
    that want each frame on the host as soon as it's ready pair this
    with a per-frame ``pack_frames`` + H2D copy.
    """
    n = len(motions)
    x_s, x_d_all = motion_stitch(avatar.kp_info, motions, stitch=stitch)
    f_s_b = avatar.f_s.expand(n, -1, -1, -1, -1).contiguous()
    x_s_b = x_s.expand_as(x_d_all).contiguous()
    warped = _run_warp(
        warp,
        WarpInput(feature_3d=f_s_b, kp_source=x_s_b, kp_driving=x_d_all.contiguous()),
    )
    for i in range(n):
        face = torch.empty((1, 3, 512, 512), dtype=torch.float32, device="cuda")
        decoder(DecoderInput(feature=warped[i : i + 1]), out=DecoderOutput(output=face))
        face.clamp_(0.0, 1.0)
        rgb, alpha = putback_chunk(face, avatar, bg, matting=matting)
        yield rgb, alpha


def render_chunk(
    motions: MotionFrame,
    avatar: Avatar,
    bg: torch.Tensor,
    *,
    stitch: StitchEngine,
    warp: WarpEngine,
    decoder: DecoderEngine,
    matting: MODNetEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render an entire chunk in one batched pass.

    Args:
        motions: stacked ``MotionFrame`` (``len == N``).
        avatar: the registered ``Avatar``.
        bg: ``(1, 3, H, W)`` float CUDA background to composite the head
            over. The pipeline picks this per-request from its bg
            registry; the renderer is bg-agnostic.
        stitch: stitch network (b=1 or b=1..5).
        warp: warp network (b=1 or b=1..5).
        decoder: SPADE decoder (b=1 or b=1..5).
        matting: MODNet matting engine (b=1 or b=1..5).

    Returns:
        ``(rgb, alpha)`` -- ``rgb`` is ``(N, 3, H, W)`` float CUDA in
        ``[0, 1]`` with the head composited over ``bg``; ``alpha`` is
        ``(N, 1, H, W)`` MODNet matte in ``[0, 1]``. Returned as
        separate tensors so ``pack_frames`` (which packs them into
        independent YUV planes) doesn't pay for a cat/slice round-trip;
        alpha-less callers just drop it.

    The decoder runs per-frame at b=1 even though the engine supports
    b>1 -- batched calls go ~2 ms slower in our setup (the SPADE stack
    doesn't share work across batch elements) and b>1 is the surface
    where the SPADE Resize bug previously hid. We pre-allocate the (N,
    3, 512, 512) face buffer once and pass each slice as the engine's
    ``out=`` so per-frame results land directly in the contiguous chunk
    layout -- no ``cat`` after the loop.
    """
    n = len(motions)
    # Batched: x_s is (1, 21, 3), x_d_all is (N, 21, 3).
    x_s, x_d_all = motion_stitch(avatar.kp_info, motions, stitch=stitch)
    f_s_b = avatar.f_s.expand(n, -1, -1, -1, -1).contiguous()
    x_s_b = x_s.expand_as(x_d_all).contiguous()
    warped = _run_warp(
        warp,
        WarpInput(feature_3d=f_s_b, kp_source=x_s_b, kp_driving=x_d_all.contiguous()),
    )
    faces = torch.empty((n, 3, 512, 512), dtype=torch.float32, device="cuda")
    face_slot = DecoderOutput(output=faces[0:1])
    for i in range(n):
        face_slot.output = faces[i : i + 1]
        decoder(DecoderInput(feature=warped[i : i + 1]), out=face_slot)
    faces.clamp_(0.0, 1.0)
    return putback_chunk(faces, avatar, bg, matting=matting)


__all__ = ["render_chunk", "render_chunk_streaming"]
