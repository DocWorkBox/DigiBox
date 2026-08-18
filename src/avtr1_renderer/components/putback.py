# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Composite the rendered face crop back onto the avatar and its background.

Calls the MODNet matting engine via ``matte_chunk``. The reference's
variant (see ``putback.py:37`` in the old repo) bundles body-motion
simulation, alpha matting, and per-avatar custom mask paths; the
simplified pipeline keeps just the ``warp_affine + mask blend + matte +
bg-composite`` core.

``putback_chunk`` is the only entry point: it processes all N frames at
once in a single batched warp + blend + matting + composite. Single-
frame callers wrap their face in ``unsqueeze(0)`` and read ``[0]`` from
each output -- there is no per-frame variant, because every op in the
pipeline broadcasts cleanly across the leading dim and the engines
support ``B >= 1``.

Conventions:
- Tensors are NCHW float in ``[0, 1]`` on CUDA.
- ``avatar.mask`` is pre-warped to original-frame coordinates at registration
  time, so the inner loop only does a single warp + blend.
- ``avatar.source`` is straight portrait RGB. RGBA portraits carry a trusted
  source-frame matte plus a narrow contour band where per-frame MODNet may
  update the moving outline; legacy portraits use full-frame MODNet.
- ``bg`` is passed in by the caller -- the final background the rendered
  head is composited onto using the predicted alpha matte. Backgrounds
  are a render-time choice (the pipeline keeps a registry keyed by id),
  not part of the avatar's identity.
- The output is ``(rgb, alpha)`` -- separate tensors so consumers can
  drop alpha without paying a slice / cat round-trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from avtr1_renderer.components.matting import matte_chunk

if TYPE_CHECKING:
    from avtr1_renderer.avatar_loader import Avatar
    from avtr1_renderer.models.matting import MODNetEngine


def _dynamic_matte_confidence(
    dynamic_alpha: torch.Tensor,
    static_alpha: torch.Tensor,
    evaluation_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a continuous per-frame confidence for a dynamic matte.

    MODNet can collapse during the first generated frames.  Comparing its
    foreground mass with the trusted source matte rejects those frames without
    a CPU synchronisation or a hard temporal state transition.
    """

    dims = (-2, -1)
    base_mass = (static_alpha * evaluation_mask).sum(dim=dims, keepdim=True)
    dynamic_mass = (dynamic_alpha * evaluation_mask).sum(dim=dims, keepdim=True)
    eps = torch.finfo(dynamic_alpha.dtype).eps
    ratio = dynamic_mass / base_mass.clamp_min(eps)
    lower = ((ratio - 0.35) / 0.30).clamp(0.0, 1.0)
    upper = ((1.80 - ratio) / 0.40).clamp(0.0, 1.0)
    confidence = lower * upper
    return torch.where(base_mass > eps, confidence, torch.ones_like(confidence))


def putback_chunk(
    face_crops: torch.Tensor,
    avatar: Avatar,
    bg: torch.Tensor,
    *,
    matting: MODNetEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched composite: ``(N, 3, H_crop, W_crop)`` faces -> bg-composited frames.

    Args:
        face_crops: (N, 3, H_crop, W_crop) float in [0, 1] -- decoder
            outputs for one chunk.
        avatar: provides ``M_grid`` (precomputed F.affine_grid input),
            ``mask`` (pre-warped pasteback mask), straight-RGB ``source``,
            and optional source-alpha priors for stable dynamic matting.
        bg: ``(1, 3, H, W)`` float in ``[0, 1]`` -- the background the
            rendered head is composited over. Broadcasts against the
            ``(N, 3, H, W)`` chunk.
        matting: MODNet alpha-matting engine; called once per chunk.

    Returns:
        ``(rgb, alpha)`` -- ``rgb`` is ``(N, 3, H, W)`` bg-composited
        float in [0, 1]; ``alpha`` is ``(N, 1, H, W)`` MODNet matte in
        [0, 1]. Both on CUDA. Callers that don't need alpha just drop
        it -- no separate code path. We keep them as separate tensors
        so consumers (e.g. ``pack_frames``) don't pay for a concat /
        slice round-trip when they handle the two planes independently
        (the YUV stacked-alpha format does, plain YUV ignores alpha).
    """
    assert face_crops.ndim == 4 and face_crops.shape[1] == 3, (
        f"face_crops must be (N, 3, H, W), got {tuple(face_crops.shape)}"
    )
    n = face_crops.shape[0]
    h, w = avatar.source.shape[-2:]

    # Hand-rolled affine warp: ``F.affine_grid`` builds the sampling
    # grid from the precomputed normalised inverse on ``avatar``, and
    # ``F.grid_sample`` does the bilinear lookup. Two kernels total.
    # ``kornia.warp_affine`` would do the same end-to-end but ~40
    # kernels (it inverts ``M`` on every call via cuBLAS LU, with two
    # D2H sync points to check for singularity); we cache the inverse
    # on Avatar so we never pay that cost at render time.
    M_b = avatar.M_grid.unsqueeze(0).expand(n, -1, -1)
    grid = F.affine_grid(M_b, [n, 3, h, w], align_corners=False)
    face_warped = F.grid_sample(
        face_crops,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    # blended = mask * face + (1 - mask) * source -- one fused kernel via lerp.
    # avatar.mask: (1, H, W); avatar.source: (3, H, W); both broadcast to (N, 3, H, W).
    blended = torch.lerp(avatar.source.unsqueeze(0), face_warped, avatar.mask.unsqueeze(0))
    source_alpha = avatar.source_alpha
    if source_alpha is not None:
        # The uploaded alpha is valid only for the source pose. The decoder
        # changes the head inside the pasteback region, so keeping that alpha
        # fixed exposes its neutral registration fill when the head moves.
        # Predict a per-frame outline on the same full-frame composition used
        # upstream, then let it replace only a narrow band around the original
        # contour. Deep body pixels and distant background remain static.
        static_alpha = source_alpha.unsqueeze(0).expand(n, -1, -1, -1)
        static_rgb = avatar.source.unsqueeze(0).expand(n, -1, -1, -1)
        neutral_level = 200.0 / 255.0
        matting_source = avatar.matting_source
        if matting_source is None:
            neutral = torch.full_like(static_rgb, neutral_level)
            matting_source_b = torch.lerp(neutral, static_rgb, static_alpha)
        else:
            matting_source_b = matting_source.unsqueeze(0).expand(n, -1, -1, -1)
        pasteback_mask = avatar.mask.unsqueeze(0)
        matting_blended = torch.lerp(matting_source_b, face_warped, pasteback_mask)
        dynamic_alpha = matte_chunk(matting_blended, modnet=matting)

        # MODNet sometimes interprets the neutral registration fixture as a
        # faint foreground extension.  Reject only those newly-created pixels
        # whose RGB is still indistinguishable from the fixture.  Existing
        # source-alpha hair is deliberately untouched, while genuinely coloured
        # moving strands pass through the smooth guard.
        neutral_distance = (matting_blended - neutral_level).abs().amax(
            dim=1, keepdim=True
        )
        neutral_guard = ((neutral_distance - (6.0 / 255.0)) / (18.0 / 255.0)).clamp(
            0.0, 1.0
        )
        neutral_guard = neutral_guard * neutral_guard * (3.0 - 2.0 * neutral_guard)
        outside_source = static_alpha <= (1.0 / 255.0)
        dynamic_alpha = dynamic_alpha * torch.where(
            outside_source,
            neutral_guard,
            torch.ones_like(neutral_guard),
        )

        confidence = _dynamic_matte_confidence(
            dynamic_alpha,
            static_alpha,
            pasteback_mask,
        )
        motion_mask = avatar.alpha_motion_mask
        if motion_mask is None:
            motion_mask_b = pasteback_mask
        else:
            motion_mask_b = motion_mask.unsqueeze(0)
        candidate_alpha = torch.lerp(static_alpha, dynamic_alpha, motion_mask_b)
        alpha = torch.lerp(static_alpha, candidate_alpha, confidence).clamp_(0.0, 1.0)

        # The decoder crop is generated against the same neutral registration
        # colour used by MODNet. Convert it directly to premultiplied colour:
        # ``C - neutral * (1 - a) == F * a``. Keeping this in premultiplied
        # form avoids an unstable divide at fine semi-transparent hair while
        # preventing the neutral fixture from becoming a bright fringe.
        base_pm = blended * static_alpha
        dynamic_pm = torch.minimum(
            (
                matting_blended - neutral_level * (1.0 - dynamic_alpha)
            ).clamp_min(0.0),
            dynamic_alpha,
        )
        candidate_pm = torch.lerp(base_pm, dynamic_pm, motion_mask_b)

        # A rejected matte falls back to the trusted source only inside the
        # moving contour band. Deep generated head pixels remain animated, so
        # one bad MODNet frame cannot flash the entire head back to its source
        # pose while RGB and alpha still agree along the rejected outline.
        static_pm = static_rgb * static_alpha
        fallback_pm = torch.lerp(base_pm, static_pm, motion_mask_b)
        foreground_pm = torch.lerp(fallback_pm, candidate_pm, confidence)
        composited = bg * (1.0 - alpha) + foreground_pm
        return composited, alpha
    if avatar.no_matting:
        # Portrait carries its own background — skip MODNet and return as-is.
        alpha = torch.ones(n, 1, h, w, device=blended.device, dtype=blended.dtype)
        return blended, alpha
    alpha = matte_chunk(blended, modnet=matting)  # (N, 1, H, W)
    # composited = alpha * blended + (1 - alpha) * bg -- another fused lerp.
    composited = torch.lerp(bg, blended, alpha)
    return composited, alpha


