# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

from typing import Protocol


class InferenceEngine[InputT, OutputT](Protocol):
    """Run a single model.

    `InputT` and `OutputT` are dataclasses whose field names match the
    underlying engine's tensor names. Every field — on both input and output
    — is a `torch.Tensor` on a CUDA device. CPU tensors, numpy arrays, and
    cross-device transfers are not part of this interface; callers move data
    onto the GPU before calling and consume outputs there.

    Known implementations:
    - `TRTEngine` — TensorRT engines, zero host↔device copies, runs on the
      caller's current torch CUDA stream.
    - `OnnxRTEngine` — ONNX Runtime via IOBinding, same torch-CUDA-tensor
      contract. Useful for validating ONNX exports before TRT compilation.

    Both implementations also support `out=` for output reuse and
    `allocate_outputs()` for one-shot output buffer construction; the hot
    path of the orchestrator uses both to avoid per-call allocations.
    """

    def __call__(self, inputs: InputT, out: OutputT | None = None) -> OutputT: ...

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> OutputT: ...
