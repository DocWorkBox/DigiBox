# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""AVTR1 encode + decode I/O contracts.

Two engines on the speech->motion path:

- ``Avtr1Encode`` runs once per chunk and produces five attention-ready
  condition tensors (kp tokens + past context split into bulk and last-N
  + speech/listen audio embeddings).
- ``Avtr1Decode`` runs ``n_ode_steps`` times per chunk to integrate the
  flow-matching ODE; the decoder output is the velocity ``v(x, t)``.

CFG (classifier-free guidance) is internal to the decode TRT engine. The
per-condition weight tensors (``w_self`` / ``w_other`` / ``w_kp``, each
shape ``(latent_dim,)`` = ``(512,)``) are engine inputs so the caller
can retune guidance strength per request without rebuilding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avtr1_renderer.runtime import InferenceEngine


@dataclass(slots=True)
class Avtr1EncodeInput:
    past_cond: torch.Tensor      # (1, past_size, nfeats=42)   normalised
    audio_cond: torch.Tensor     # (1, 85, 2048)               speech || listen HuBERT
    kp_cond: torch.Tensor        # (1, 1, 129)                 normalised
    past_times: torch.Tensor     # (1, past_size//chunk, 1)    all 1.0 at inference


@dataclass(slots=True)
class Avtr1EncodeOutput:
    kp_tokens: torch.Tensor      # (1, 1, latent_dim=512)
    past_context: torch.Tensor   # (1, past_size - chunk_size, 512)
    past_last: torch.Tensor      # (1, chunk_size, 512)
    audio_self: torch.Tensor     # (1, 85, 512)
    audio_other: torch.Tensor    # (1, 85, 512)


@dataclass(slots=True)
class Avtr1DecodeInput:
    x: torch.Tensor              # (1, chunk_size, nfeats)
    kp_tokens: torch.Tensor
    past_context: torch.Tensor
    past_last: torch.Tensor
    audio_self: torch.Tensor
    audio_other: torch.Tensor
    w_self: torch.Tensor         # (latent_dim,) per-coord CFG weight (self audio)
    w_other: torch.Tensor        # (latent_dim,) per-coord CFG weight (other audio)
    w_kp: torch.Tensor           # (latent_dim,) per-coord CFG weight (kp / median)
    t: torch.Tensor              # (1, 1) ODE step in [0, 1]


@dataclass(slots=True)
class Avtr1DecodeOutput:
    output: torch.Tensor         # (1, chunk_size, nfeats) -- flow-field velocity v(x, t)


Avtr1EncodeEngine = InferenceEngine[Avtr1EncodeInput, Avtr1EncodeOutput]
Avtr1DecodeEngine = InferenceEngine[Avtr1DecodeInput, Avtr1DecodeOutput]
