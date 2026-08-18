# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Portable CUDA runtime for the released AVTR-1 TorchScript checkpoint.

The upstream runtime exports the scripted module to TensorRT before inference.
That export is optional here: the released module already exposes the same
``encode_conditions`` and guided ``forward`` methods, so these adapters present
them through the existing ``InferenceEngine`` protocol.  This is the native
Windows path and also a useful correctness fallback on Linux.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeInput,
    Avtr1DecodeOutput,
    Avtr1EncodeInput,
    Avtr1EncodeOutput,
)


@dataclass(frozen=True, slots=True)
class _ModelShapes:
    chunk_size: int
    past_size: int
    future_size: int
    nfeats: int
    cond_feature_dim: int
    latent_dim: int

    @classmethod
    def from_scripted(cls, scripted: Any) -> _ModelShapes:
        model = scripted.model
        return cls(
            chunk_size=int(model.chunk_size),
            past_size=int(model.past_size),
            future_size=int(model.future_size),
            nfeats=int(model.nfeats),
            cond_feature_dim=int(model.cond_feature_dim),
            latent_dim=int(model.latent_dim),
        )

    @property
    def audio_seq_len(self) -> int:
        return self.past_size + self.chunk_size + self.future_size


def _copy_tensor(destination: torch.Tensor, source: torch.Tensor) -> None:
    if destination.shape != source.shape:
        raise ValueError(
            f"TorchScript output shape changed: allocated {tuple(destination.shape)}, "
            f"received {tuple(source.shape)}"
        )
    if destination.dtype != source.dtype:
        raise ValueError(
            f"TorchScript output dtype changed: allocated {destination.dtype}, "
            f"received {source.dtype}"
        )
    if destination.device != source.device:
        raise ValueError(
            f"TorchScript output device changed: allocated {destination.device}, "
            f"received {source.device}"
        )
    if not destination.is_contiguous():
        raise ValueError("TorchScript output buffer must be contiguous")
    destination.copy_(source)


class _TorchScriptEncodeEngine:
    def __init__(self, scripted: Any, shapes: _ModelShapes, device: torch.device) -> None:
        self._scripted = scripted
        self._shapes = shapes
        self._device = device

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> Avtr1EncodeOutput:
        default = {
            "kp_tokens": (1, 1, self._shapes.latent_dim),
            "past_context": (
                1,
                self._shapes.past_size - self._shapes.chunk_size,
                self._shapes.latent_dim,
            ),
            "past_last": (1, self._shapes.chunk_size, self._shapes.latent_dim),
            "audio_self": (1, self._shapes.audio_seq_len, self._shapes.latent_dim),
            "audio_other": (1, self._shapes.audio_seq_len, self._shapes.latent_dim),
        }
        if shapes:
            default.update(shapes)
        return Avtr1EncodeOutput(
            **{
                name: torch.empty(shape, dtype=torch.float32, device=self._device)
                for name, shape in default.items()
            }
        )

    @torch.inference_mode()
    def __call__(
        self,
        inputs: Avtr1EncodeInput,
        out: Avtr1EncodeOutput | None = None,
    ) -> Avtr1EncodeOutput:
        values = self._scripted.encode_conditions(
            past_cond=inputs.past_cond,
            audio_cond=inputs.audio_cond,
            kp_cond=inputs.kp_cond,
            past_times=inputs.past_times,
        )
        result = Avtr1EncodeOutput(
            kp_tokens=values[0],
            past_context=values[1],
            past_last=values[2],
            audio_self=values[3],
            audio_other=values[4],
        )
        if out is None:
            return result
        for name in (
            "kp_tokens",
            "past_context",
            "past_last",
            "audio_self",
            "audio_other",
        ):
            _copy_tensor(getattr(out, name), getattr(result, name))
        return out


class _TorchScriptDecodeEngine:
    def __init__(self, scripted: Any, shapes: _ModelShapes, device: torch.device) -> None:
        self._scripted = scripted
        self._shapes = shapes
        self._device = device

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> Avtr1DecodeOutput:
        output_shape = (1, self._shapes.chunk_size, self._shapes.nfeats)
        if shapes and "output" in shapes:
            output_shape = shapes["output"]
        return Avtr1DecodeOutput(
            output=torch.empty(output_shape, dtype=torch.float32, device=self._device)
        )

    @torch.inference_mode()
    def __call__(
        self,
        inputs: Avtr1DecodeInput,
        out: Avtr1DecodeOutput | None = None,
    ) -> Avtr1DecodeOutput:
        value = self._scripted(
            x=inputs.x,
            kp=inputs.kp_tokens,
            past_context=inputs.past_context,
            past_last=inputs.past_last,
            audio_self=inputs.audio_self,
            audio_other=inputs.audio_other,
            t=inputs.t,
            self_weights=inputs.w_self,
            other_weights=inputs.w_other,
            kp_weights=inputs.w_kp,
        )
        if out is None:
            return Avtr1DecodeOutput(output=value)
        _copy_tensor(out.output, value)
        return out


class TorchScriptAVTR1Backend:
    """One shared scripted module exposed as encode and decode engines."""

    def __init__(
        self,
        scripted: Any,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.scripted = scripted.eval().to(self.device)
        self.shapes = _ModelShapes.from_scripted(self.scripted)
        self.encode = _TorchScriptEncodeEngine(self.scripted, self.shapes, self.device)
        self.decode = _TorchScriptDecodeEngine(self.scripted, self.shapes, self.device)

    @classmethod
    def from_file(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> TorchScriptAVTR1Backend:
        target = torch.device(device)
        scripted = torch.jit.load(str(checkpoint), map_location=target)
        return cls(scripted, device=target)


__all__ = ["TorchScriptAVTR1Backend"]
