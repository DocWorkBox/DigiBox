# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""AVTR1MotionGenerator: speech -> motion for the AVTR1 model.

Wraps the encode + decode TRT engines built by
``scripts/build_avtr1_engines.py`` from the AVTR1 flow-matching
checkpoint. The only motion generator in the runtime today.

Highlights:

- **Two-stage TRT inference per chunk.** ``encode`` runs once and emits
  five attention-ready condition tensors; ``decode`` runs ``n_ode_steps``
  times for Euler integration of the flow field.
- **CFG inside the engine.** The four-pass classifier-free-guidance batch
  (past / self-audio / other-audio / kp) is internal to the decode TRT
  engine, with weights baked at build time. The runtime sees a single
  prediction per ODE step, not multiple condition modes.
- **Long audio context, but cheap to extend.** The model wants
  ``past + chunk + future = 85`` audio features per call, but we only
  HuBERT a ``(3, 5, 5)`` window every chunk -- the past 75 features are
  carried in ``State.audio_features`` and shifted forward each call.
- **One HuBERT call per chunk for both tracks.** Speech and listen are
  stacked along the batch dim and fed in a single ``HubertEngine`` call;
  the engine must be built with batch >= 2.
- **Normalised motion in ``State.past_cond``.** The model's input
  projection is trained on z-score-normalised motion vectors (so3 +
  lipsync exp); we keep the same normalisation in the autoregressive
  memory and de-normalise only when emitting ``MotionFrame``s.

Per-chunk flow:

1. Concatenate the 3-frame audio tails with the new ``Chunk``'s
   ``5 + 5`` frames; stack speech || listen as a batch of two and run
   ``HubertEngine`` once -> ``(2, 10, 1024)``.
2. Reshape that to ``(1, 10, 2048)`` (concat speech+listen along feat
   dim) and prepend the 75-frame ``audio_features`` history ->
   ``audio_cond`` of shape ``(1, 85, 2048)``.
3. Build ``kp_cond`` from the avatar's normalised pose (so3 + kp + exp);
   ``past_times`` is all-ones (every past frame is "clean" at inference).
4. ``encode`` -> ``Condition``-shaped 5 tensors.
5. Sample initial noise ``z`` (progressive AR(1), correlated with the
   previous chunk via ``state.noise_shared``).
6. Euler-step ``n_ode_steps`` times: ``v = decode(x, *cond, t)``;
   ``x += dt * v``.
7. De-normalise each of the ``chunk_size`` predictions into
   ``MotionFrame(R, exp)``.
8. Update ``State``: append the *normalised* predictions to ``past_cond``
   (drop the oldest ``chunk_size`` frames); shift ``audio_features``
   forward by ``chunk_size`` (drop oldest 5, append the 5 "current"
   features from this call's HuBERT output -- the future-5 are dropped,
   they'll re-enter as the *current* portion of the next chunk's audio);
   slice raw audio tails forward by ``chunk_size`` frames; carry the
   last frame of ``z`` as the next ``noise_shared``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import roma
import torch

from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.components.hubert import run_hubert
from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeEngine,
    Avtr1DecodeInput,
    Avtr1EncodeEngine,
    Avtr1EncodeInput,
)
from avtr1_renderer.models.hubert import HubertEngine
from avtr1_renderer.types import Chunk, RenderOptions

N_LIPSYNC = len(LIPSYNC_COORDS)

Z_SCORE_CLIP_VALUE = 5.0


# ---------------------------------------------------------------------------
# State.


def state_to_safetensors(state: AVTR1State) -> bytes:
    """Serialise a state for cross-request transport.

    The runtime keeps state on CUDA so the per-chunk path is sync-free.
    Persistence (e.g. an HTTP API saving state across requests) needs
    host bytes; this helper is the *only* place state should ever
    cross the device boundary.
    """
    from safetensors.torch import save

    tensors = {
        "audio_prev_speech": state.audio_prev_speech,
        "audio_prev_listen": state.audio_prev_listen,
        "audio_features": state.audio_features,
        "past_cond": state.past_cond,
    }
    if state.noise_shared is not None:
        tensors["noise_shared"] = state.noise_shared
    return save(tensors)


def state_from_safetensors(
    blob: bytes, *, device: str | torch.device = "cuda"
) -> AVTR1State:
    """Inverse of :func:`state_to_safetensors`. Loads each tensor
    directly onto ``device`` (default CUDA)."""
    from safetensors.torch import load

    tensors = load(blob)
    moved = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
    return AVTR1State(
        audio_prev_speech=moved["audio_prev_speech"],
        audio_prev_listen=moved["audio_prev_listen"],
        audio_features=moved["audio_features"],
        past_cond=moved["past_cond"],
        noise_shared=moved.get("noise_shared"),
    )


@dataclass(slots=True, frozen=True)
class AVTR1State:
    """Per-session state for ``AVTR1MotionGenerator``.

    All large fields live on CUDA between calls so that ``generate_chunk``
    never crosses the host boundary in its hot path -- no D2H syncs, no
    H2D re-uploads, no numpy round-trips.

    The audio buffers carry only HuBERT's 3-frame left-context. The big
    ~3 s audio context the model wants per call is carried as
    ``audio_features``, the already-extracted HuBERT outputs for the
    last ``past_size`` motion frames at ``2 * 1024`` channels (speech ||
    listen).

    Persist / restore via ``safetensors`` (see :func:`state_to_safetensors`
    / :func:`state_from_safetensors`); that explicit codec is the only
    place state ever touches the host.
    """

    audio_prev_speech: torch.Tensor       # (chunksize[0] * frame_len,) float32 CUDA
    audio_prev_listen: torch.Tensor       # parallel to speech
    audio_features: torch.Tensor          # (1, past_size, 2 * 1024) float32 CUDA -- speech || listen HuBERT history
    past_cond: torch.Tensor               # (1, past_size, nfeats) float32 CUDA, normalised motion
    noise_shared: torch.Tensor | None     # (1, 1, nfeats) CUDA AR(1) carry, or None on cold start


# ---------------------------------------------------------------------------
# Helpers.


def _truncnorm_conditional(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    lo: torch.Tensor | None = None,
    hi: torch.Tensor | None = None,
    z: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``N(0, 1)`` truncated to ``[lo, hi]`` (default ``[-z, z]``)."""
    rsqrt2 = 0.7071067811865476
    sqrt2 = 1.4142135623730951
    if lo is None and hi is None:
        Phi_z = 0.5 * (1.0 + math.erf(z * rsqrt2))
        Phi_mz = 1.0 - Phi_z
        u = torch.rand(shape, device=device, dtype=torch.float32)
        u = u * (Phi_z - Phi_mz) + Phi_mz
    else:
        cdf_lo = 0.5 * (1.0 + torch.erf(lo * rsqrt2))
        cdf_hi = 0.5 * (1.0 + torch.erf(hi * rsqrt2))
        delta = (cdf_hi - cdf_lo).clamp_min(1e-8)
        u = cdf_lo + torch.rand(shape, device=device, dtype=torch.float32) * delta
    u = u.clamp(eps, 1.0 - eps)
    return sqrt2 * torch.erfinv(2.0 * u - 1.0)


def _progressive_noise(
    n_frames: int,
    n_feats: int,
    *,
    device: torch.device,
    alpha: float,
    trunc_z: float,
    noise_shared: torch.Tensor | None,
) -> torch.Tensor:
    """AR(1) progressive noise. Mirrors the AVTR1 ``_progressive_noise``."""
    coef = alpha / math.sqrt(1.0 + alpha**2)
    std = 1.0 / math.sqrt(1.0 + alpha**2)

    def eps(prev: torch.Tensor | None) -> torch.Tensor:
        if prev is not None:
            lo = (-trunc_z - coef * prev) / std
            hi = (trunc_z - coef * prev) / std
            return _truncnorm_conditional((1, n_feats), device, lo=lo, hi=hi, z=trunc_z)
        return _truncnorm_conditional((1, n_feats), device, z=trunc_z)

    noise = torch.empty((1, n_frames, n_feats), device=device, dtype=torch.float32)
    if noise_shared is None:
        noise[:, 0] = eps(None)
    else:
        prev = noise_shared.to(torch.float32).view(1, n_feats)
        noise[:, 0] = coef * prev + std * eps(prev)
    for t in range(1, n_frames):
        prev = noise[:, t - 1]
        noise[:, t] = coef * prev + std * eps(prev)
    return noise


# ---------------------------------------------------------------------------
# Normaliser.


@dataclass(slots=True)
class Normalizer:
    """Subset of the AVTR1 ``Normalizer`` we need at runtime.

    Built from the ``<prefix>_normalizer.safetensors`` sidecar that
    ``scripts/build_avtr1_engines.py`` writes next to the engines -- the
    sidecar is a flat dump of the scripted wrapper's normalization
    buffers, so the runtime no longer depends on the eager AVTR1
    checkpoint. Public so callers (e.g. ``Pipeline.from_artifacts``) can
    build it once and share it between the motion generator and the
    intro library.
    """

    offset_so3: torch.Tensor       # (3,)
    scale_so3: torch.Tensor        # (3,)
    offset_kp: torch.Tensor        # (21, 3)
    scale_kp: torch.Tensor         # (21, 3)
    offset_exp: torch.Tensor       # (21, 3)
    scale_exp: torch.Tensor        # (21, 3)
    exp_lipsync_offset: torch.Tensor   # (N_LIPSYNC,) -- pre-sliced for de-norm
    exp_lipsync_scale: torch.Tensor    # (N_LIPSYNC,)

    @classmethod
    def from_scripted(
        cls,
        scripted: Any,
        *,
        device: str | torch.device = "cuda",
    ) -> Normalizer:
        """Read normalization buffers directly from the released checkpoint.

        TensorRT builds persist these values as a safetensors sidecar.  The
        portable TorchScript backend already owns the same buffers, so copying
        them to the target device avoids a platform-specific build step.
        """
        names = (
            "motion_offset",
            "motion_scale",
            "offset_so3",
            "scale_so3",
            "offset_kp",
            "scale_kp",
            "offset_exp",
            "scale_exp",
            "lipsync_coords",
        )
        missing = [name for name in names if not hasattr(scripted, name)]
        if missing:
            raise RuntimeError(
                "scripted AVTR-1 checkpoint is missing required normalizer "
                f"buffer(s): {', '.join(missing)}"
            )

        target = torch.device(device)

        def moved(name: str) -> torch.Tensor:
            return getattr(scripted, name).detach().to(target).contiguous()

        motion_offset = moved("motion_offset")
        motion_scale = moved("motion_scale")
        return cls(
            offset_so3=moved("offset_so3"),
            scale_so3=moved("scale_so3"),
            offset_kp=moved("offset_kp"),
            scale_kp=moved("scale_kp"),
            offset_exp=moved("offset_exp"),
            scale_exp=moved("scale_exp"),
            exp_lipsync_offset=motion_offset[3:].contiguous(),
            exp_lipsync_scale=motion_scale[3:].contiguous(),
        )

    @classmethod
    def from_safetensors(
        cls, path: str | Path, *, device: str | torch.device = "cuda"
    ) -> Normalizer:
        """Load from the sidecar written alongside the AVTR1 engines.

        ``motion_offset`` / ``motion_scale`` are the combined
        ``[so3 (3) | exp_lipsync (39)]`` flat tensors, so we take
        ``exp_lipsync`` straight from their tail slice rather than
        re-indexing ``offset_exp`` -- one fewer place where the lipsync
        coord set can drift between trainer and runtime.
        """
        from safetensors.torch import load_file

        tensors = load_file(str(path), device=str(torch.device(device)))
        motion_offset = tensors["motion_offset"]
        motion_scale = tensors["motion_scale"]
        return cls(
            offset_so3=tensors["offset_so3"],
            scale_so3=tensors["scale_so3"],
            offset_kp=tensors["offset_kp"],
            scale_kp=tensors["scale_kp"],
            offset_exp=tensors["offset_exp"],
            scale_exp=tensors["scale_exp"],
            exp_lipsync_offset=motion_offset[3:].contiguous(),
            exp_lipsync_scale=motion_scale[3:].contiguous(),
        )


# ---------------------------------------------------------------------------
# Generator.


class AVTR1MotionGenerator:
    """``MotionGenerator[AVTR1State]`` for the AVTR1 model.

    See module docstring for the per-chunk flow and conventions.
    """

    # HuBERT operates with a (past, current, future) windowing convention.
    # We keep the same conventional past=3 left context (matches the shipped
    # ``hubert-lbs-avtr1-3-5-5.onnx`` training recipe). The model's *motion*
    # past_size and the HuBERT past are independent.
    HUBERT_PAST_FRAMES = 3
    AUDIO_DIM = 1024  # per-track HuBERT feature dim

    def __init__(
        self,
        *,
        hubert: HubertEngine,
        encode_engine: Avtr1EncodeEngine,
        decode_engine: Avtr1DecodeEngine,
        normalizer: Normalizer,
        chunk_size: int = 5,
        past_size: int = 75,
        future_size: int = 5,
        nfeats: int = 42,
        latent_dim: int = 512,
        frame_len: int = 640,
        audio_shift: int = 80,
        n_ode_steps: int = 5,
    ) -> None:
        assert past_size % chunk_size == 0, (
            f"past_size ({past_size}) must be divisible by chunk_size ({chunk_size})"
        )
        self._hubert = hubert
        self._encode = encode_engine
        self._decode = decode_engine
        self._normalizer = normalizer
        self.chunk_size = chunk_size
        self.past_size = past_size
        self.future_size = future_size
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.frame_len = frame_len
        self.audio_shift = audio_shift
        self.n_ode_steps = n_ode_steps

        self._lipsync_index = list(LIPSYNC_COORDS)
        # ``past_times`` is constant at inference: every past chunk is
        # marked "clean" (timestep = 1.0).
        n_past_chunks = past_size // chunk_size
        self._past_times = torch.ones(
            (1, n_past_chunks, 1), dtype=torch.float32, device="cuda"
        )
        # HuBERT chunksize convention: (past, current, future) motion frames.
        self._hubert_chunksize: tuple[int, int, int] = (
            self.HUBERT_PAST_FRAMES,
            chunk_size,
            future_size,
        )
        # Per-call HuBERT output count: current + future. We never need the
        # past portion -- those features were already collected on a prior
        # call and live in ``State.audio_features``.
        self._hubert_n_motion = chunk_size + future_size


    # -- State ---------------------------------------------------------------

    def initial_state(self, avatar: Avatar) -> AVTR1State:
        prev_len = self.HUBERT_PAST_FRAMES * self.frame_len
        device = torch.device("cuda")
        # Tile the avatar's normalised neutral motion across past_size frames.
        with torch.no_grad():
            so3 = roma.rotmat_to_rotvec(avatar.kp_info.R)  # (1, 3)
            so3_n = (so3 - self._normalizer.offset_so3) / self._normalizer.scale_so3

            exp_in_R = avatar.kp_info.exp @ avatar.kp_info.R.transpose(-1, -2)  # (1, 21, 3)
            exp_n = (exp_in_R - self._normalizer.offset_exp) / self._normalizer.scale_exp
            exp_n = exp_n.clamp(-Z_SCORE_CLIP_VALUE, Z_SCORE_CLIP_VALUE)
            exp_lipsync_n = exp_n.view(1, -1)[:, self._lipsync_index]  # (1, N_LIPSYNC)

            frame = torch.cat([so3_n, exp_lipsync_n], dim=-1)  # (1, nfeats)
            past_cond = frame.unsqueeze(1).repeat(1, self.past_size, 1)  # (1, P, nfeats)

        return AVTR1State(
            audio_prev_speech=torch.zeros(prev_len, dtype=torch.float32, device=device),
            audio_prev_listen=torch.zeros(prev_len, dtype=torch.float32, device=device),
            audio_features=torch.zeros(
                (1, self.past_size, 2 * self.AUDIO_DIM),
                dtype=torch.float32,
                device=device,
            ),
            past_cond=past_cond.contiguous(),
            noise_shared=None,
        )

    # -- Per-chunk -----------------------------------------------------------

    @torch.no_grad()
    def generate_chunk(
        self,
        audio_chunk: Chunk,
        avatar: Avatar,
        state: AVTR1State,
        options: RenderOptions | None = None,
    ) -> tuple[MotionFrame, AVTR1State]:
        if options is None:
            options = RenderOptions()
        device = torch.device("cuda")

        # ---- HuBERT: 3-frame left-context tail || this chunk's audio,
        # one batched call for both tracks --------------------------------
        # Upload incoming chunk audio once, then do all concat / slice on GPU.
        chunk_speech = torch.from_numpy(audio_chunk.audio_speech).to(
            device, dtype=torch.float32, non_blocking=True
        )
        chunk_listen = torch.from_numpy(audio_chunk.audio_listen).to(
            device, dtype=torch.float32, non_blocking=True
        )
        full_speech = torch.cat([state.audio_prev_speech, chunk_speech], dim=0)
        full_listen = torch.cat([state.audio_prev_listen, chunk_listen], dim=0)

        # Tails always roll forward, both for the speech path and for the
        # intro path -- so when intro ends the speech path's HuBERT
        # window is still contiguous.
        slice_start = self.chunk_size * self.frame_len
        slice_end = slice_start + self.HUBERT_PAST_FRAMES * self.frame_len
        next_audio_prev_speech = full_speech[slice_start:slice_end].contiguous()
        next_audio_prev_listen = full_listen[slice_start:slice_end].contiguous()

        past_cond = state.past_cond  # already on CUDA

        audio_batch = torch.stack([full_speech, full_listen], dim=0).contiguous()  # (2, 8400)
        feats = run_hubert(
            audio_batch,
            n_motion_frames=self._hubert_n_motion,
            hubert=self._hubert,
        )  # (2, chunk_size + future_size, 1024)
        # Concat speech || listen along feat dim, add the leading batch dim.
        new_audio_feats = torch.cat([feats[0:1], feats[1:2]], dim=-1)  # (1, 10, 2048)

        # ---- Build full audio_cond by prepending the 75-frame history ---
        history = state.audio_features  # already on CUDA
        audio_cond = torch.cat([history, new_audio_feats], dim=1)  # (1, past+chunk+fut, 2*AUDIO_DIM)

        # ---- kp_cond ----------------------------------------------------
        kp_cond = self._build_kp_cond(avatar)

        # ---- encode (run once) ------------------------------------------
        # Pre-allocate the encode output once per chunk and pass it via
        # ``out=``; same trick we use for the decode loop below. The
        # engines are static-shape so ``allocate_outputs()`` reads the
        # shapes straight from the engine without an inference call.
        enc = self._encode.allocate_outputs()
        self._encode(
            Avtr1EncodeInput(
                past_cond=past_cond.contiguous(),
                audio_cond=audio_cond.contiguous(),
                kp_cond=kp_cond.contiguous(),
                past_times=self._past_times,
            ),
            out=enc,
        )

        # ---- initial noise (progressive AR(1) carry) --------------------
        noise_shared_t = state.noise_shared  # already on CUDA, or None
        x = _progressive_noise(
            self.chunk_size,
            self.nfeats,
            device=device,
            alpha=options.noise_alpha,
            trunc_z=options.noise_trunc_z,
            noise_shared=noise_shared_t,
        )
        next_noise_shared = x[:, -1:].clone()  # (1, 1, nfeats)

        # ---- ODE Euler integration over [0, 1] --------------------------
        # Allocate the v (decode output) and t (scalar) buffers once for
        # this chunk and reuse them across all ``n_ode_steps - 1`` Euler
        # steps; ``x`` is updated in-place. Saves ~3 allocations per step
        # vs the obvious ``v = decode(...).output; x = x + dt * v``.
        # The ``.contiguous()`` calls are dropped because every source
        # tensor is already contiguous: ``x`` came from
        # ``_progressive_noise`` (in-place add preserves layout),
        # ``enc.*`` are TRT-allocated outputs, ``t_buf`` is fresh.
        v_out = self._decode.allocate_outputs()
        v_buf = v_out.output
        # CFG weights as per-coord tensors, built once per chunk and
        # reused across every ODE step. Each option is a single scalar
        # broadcast across all ``latent_dim`` coords -- if per-coord
        # tuning is ever needed, change ``torch.full`` to a per-coord
        # tensor here.
        w_self = torch.full(
            (self.latent_dim,), options.cfg_self_audio, device=device, dtype=torch.float32
        )
        w_other = torch.full(
            (self.latent_dim,), options.cfg_other_audio, device=device, dtype=torch.float32
        )
        w_kp = torch.full(
            (self.latent_dim,), options.cfg_kp, device=device, dtype=torch.float32
        )
        # Pre-build the full ``t`` schedule on GPU as one contiguous
        # tensor (n_ode_steps, 1, 1); slice ``times[i]`` per step to get
        # a (1, 1) view -- no D2H. Linspace is uniform so ``dt`` is a
        # constant host scalar, dodging the second sync. Previous code
        # did ``t_buf.fill_(float(time[i]))`` and
        # ``(time[i + 1] - time[i]).item()`` -- both CUDA->host syncs
        # inside the hot loop, ~9 syncs per chunk at n_ode_steps=10.
        times = torch.linspace(
            0.0, 1.0, self.n_ode_steps, device=device, dtype=torch.float32
        ).view(-1, 1, 1)
        dt = 1.0 / (self.n_ode_steps - 1)
        for i in range(self.n_ode_steps - 1):
            self._decode(
                Avtr1DecodeInput(
                    x=x,
                    kp_tokens=enc.kp_tokens,
                    past_context=enc.past_context,
                    past_last=enc.past_last,
                    audio_self=enc.audio_self,
                    audio_other=enc.audio_other,
                    w_self=w_self,
                    w_other=w_other,
                    w_kp=w_kp,
                    t=times[i],
                ),
                out=v_out,
            )
            x.add_(v_buf, alpha=dt)
        # ``x`` is the normalised motion prediction for this chunk.
        motions = self._motion_to_frames(x)

        # ---- next state -------------------------------------------------
        # past_cond: append this chunk's normalised motion, drop oldest.
        new_past_cond = torch.cat([past_cond, x], dim=1)[:, -self.past_size :]
        # audio_features: drop oldest 5, append the 5 "current" features
        # from this call (the future-5 are dropped -- they'll come back
        # as the *current* portion of the next chunk's HuBERT output).
        new_audio_features = torch.cat(
            [history[:, self.chunk_size :], new_audio_feats[:, : self.chunk_size]], dim=1
        )
        next_state = AVTR1State(
            audio_prev_speech=next_audio_prev_speech,
            audio_prev_listen=next_audio_prev_listen,
            audio_features=new_audio_features.contiguous(),
            past_cond=new_past_cond.contiguous(),
            noise_shared=next_noise_shared,
        )
        return motions, next_state

    # -- Internal -----------------------------------------------------------

    def _build_kp_cond(self, avatar: Avatar) -> torch.Tensor:
        """Build the avatar's normalised kp_cond ``(1, 1, 129)`` tensor."""
        R = avatar.kp_info.R
        so3 = roma.rotmat_to_rotvec(R)
        so3_n = (so3 - self._normalizer.offset_so3) / self._normalizer.scale_so3

        kp_n = (avatar.kp_info.kp - self._normalizer.offset_kp) / self._normalizer.scale_kp
        kp_n = kp_n.clamp(-Z_SCORE_CLIP_VALUE, Z_SCORE_CLIP_VALUE)

        exp_in_R = avatar.kp_info.exp @ R.transpose(-1, -2)
        exp_n = (exp_in_R - self._normalizer.offset_exp) / self._normalizer.scale_exp
        exp_n = exp_n.clamp(-Z_SCORE_CLIP_VALUE, Z_SCORE_CLIP_VALUE)

        return torch.cat(
            [so3_n, kp_n.view(1, -1), exp_n.view(1, -1)], dim=-1
        )[:, None]  # (1, 1, 129)

    def _motion_to_frames(self, x: torch.Tensor) -> MotionFrame:
        """De-normalise the ODE result into a stacked ``MotionFrame``.

        ``x`` is ``(1, T, 42)``. Drops the leading batch dim, runs
        rotvec->rotmat batched, and returns ``MotionFrame`` of length T.
        """
        so3_n = x[0, :, :3]  # (T, 3)
        exp_lipsync_n = x[0, :, 3 : 3 + N_LIPSYNC]  # (T, N_LIPSYNC)

        so3 = so3_n * self._normalizer.scale_so3 + self._normalizer.offset_so3  # (T, 3)
        exp_lipsync = (
            exp_lipsync_n * self._normalizer.exp_lipsync_scale
            + self._normalizer.exp_lipsync_offset
        )  # (T, N_LIPSYNC)
        R = roma.rotvec_to_rotmat(so3)  # (T, 3, 3)
        return MotionFrame(R=R, exp=exp_lipsync)


__all__ = [
    "AVTR1MotionGenerator",
    "AVTR1State",
    "Normalizer",
]
