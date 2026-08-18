# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""HuBERT speech-feature streaming-chunk policy.

Calls a ``HubertEngine`` and folds in the reference's
``Wav2FeatHubert.__call__`` post-processing: take the last
``2 * n_motion_frames`` HuBERT frames, reshape to
``(B, n_motion_frames, 2, 1024)``, and average each adjacent pair to
downsample 50 Hz -> 25 Hz.

I/O contracts live in ``avtr1_renderer.models.hubert``.
"""

from __future__ import annotations

import torch

from avtr1_renderer.models.hubert import HubertEngine, HubertInput, HubertOutput

_HUBERT_RECEPTIVE_FIELD = 400
_HUBERT_STRIDE = 320


def _hubert_frame_count(samples: int) -> int:
    if samples < _HUBERT_RECEPTIVE_FIELD:
        raise ValueError(
            f"HuBERT needs at least {_HUBERT_RECEPTIVE_FIELD} audio samples, "
            f"got {samples}"
        )
    return (samples - _HUBERT_RECEPTIVE_FIELD) // _HUBERT_STRIDE + 1


def run_hubert(
    audio_batch: torch.Tensor,
    *,
    n_motion_frames: int,
    hubert: HubertEngine,
) -> torch.Tensor:
    """Run HuBERT on a batch of tracks and return chunked features.

    HuBERT's effective stride is 320 samples (50 Hz at 16 kHz input) with
    a 400-sample feature-extractor receptive field.

    Args:
        audio_batch: ``(B, N)`` float32 CUDA. ``B`` must satisfy the
            engine's optimisation profile.
        n_motion_frames: number of trailing 25 Hz motion frames to
            return per batch element.
        hubert: HuBERT engine (e.g. built via
            ``load_engine(path, HubertInput, HubertOutput)``).

    Returns:
        ``(B, n_motion_frames, 1024)`` float32 CUDA.
    """
    batch, samples = audio_batch.shape
    output = HubertOutput(
        last_hidden_state=torch.empty(
            (batch, _hubert_frame_count(samples), 1024),
            dtype=torch.float32,
            device=audio_batch.device,
        )
    )
    out = hubert(
        HubertInput(input_values=audio_batch.contiguous()),
        out=output,
    )
    encoding = out.last_hidden_state  # (B, frames_50hz, 1024)
    valid = encoding[:, -n_motion_frames * 2 :]  # (B, 2 * n_motion, 1024)
    batch = valid.shape[0]
    return valid.reshape(batch, n_motion_frames, 2, 1024).mean(dim=2)
