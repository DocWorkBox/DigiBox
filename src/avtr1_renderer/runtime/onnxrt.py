# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""ONNX Runtime implementation of the ``InferenceEngine`` protocol.

Runs an ONNX model on torch CUDA tensors with zero host↔device copies via
ORT's ``IOBinding``: input tensors are bound by ``data_ptr()`` and outputs
are written into pre-allocated CUDA torch tensors. Drop-in alternative to
``TRTEngine`` -- same dataclass-field-to-tensor-name mapping, same call
shape -- useful for validating ONNX exports before TRT compilation.

ORT can't switch the CUDA EP's stream after the session is built, so
``from_file`` captures the caller's current torch CUDA stream and wires
it in as ``user_compute_stream``. As long as subsequent calls keep that
stream current there is zero per-call coordination -- ORT's compute
queues onto the same stream as surrounding torch ops. If the caller
later switches streams, ``__call__`` falls back to CUDA-event fencing
between the EP stream and the new current stream (still device-side, no
host blocking, no host↔device copies). From the caller's perspective
the contract matches ``TRTEngine``: enqueue work on the current stream
and read outputs from the same stream without explicit synchronisation.

Limitations vs ``TRTEngine``:
- Output shapes must either be static in the ONNX graph, or the caller
  passes ``out=...`` (or ``shapes=...`` to ``allocate_outputs``). ORT
  doesn't expose a "given these input shapes, what are the output
  shapes" API short of running the model, so we can't auto-resolve
  dynamic outputs the way TRT can via ``ctx.get_tensor_shape``.
"""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

_ORT_TYPE_TO_TORCH: dict[str, torch.dtype] = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(int16)": torch.int16,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(bool)": torch.bool,
}

_TORCH_TO_NUMPY: dict[torch.dtype, type] = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}


def _windows_preload_onnxruntime_dlls(
    runtime: Any = ort,
    *,
    platform: str = sys.platform,
) -> None:
    if platform == "win32" and hasattr(runtime, "preload_dlls"):
        runtime.preload_dlls()


def _require_cuda_provider(providers: list[str]) -> None:
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "ONNX Runtime CUDAExecutionProvider is unavailable. "
            f"Available providers: {providers}. Install onnxruntime-gpu and "
            "a CUDA-compatible PyTorch/ONNX Runtime build."
        )


def _ort_to_torch_dtype(ort_type: str) -> torch.dtype:
    try:
        return _ORT_TYPE_TO_TORCH[ort_type]
    except KeyError as e:
        raise ValueError(f"Unsupported ORT element type: {ort_type!r}") from e


def _torch_to_numpy_dtype(dtype: torch.dtype) -> Any:
    try:
        return _TORCH_TO_NUMPY[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from e


def _static_shape(shape: list[Any]) -> tuple[int, ...] | None:
    """Return ``shape`` as a tuple of ints if every dim is concrete; else None.

    ONNX graphs encode dynamic dims as strings (named symbols) or ``None``;
    only fully-concrete shapes can be used to pre-allocate outputs.
    """
    return tuple(shape) if all(isinstance(d, int) for d in shape) else None


class OnnxRTEngine[InputT, OutputT]:
    """
    Run an ONNX model on torch CUDA tensors with zero host↔device copies.

    Inputs and outputs are dataclasses whose field names must match the
    model's tensor names. Outputs are written into pre-allocated CUDA
    torch tensors -- either auto-allocated from the model's static
    output shapes, or supplied by the caller via ``out=...`` (required
    when any output dim is symbolic).

    The CUDA EP runs on the stream that was current at ``from_file``
    time. Calls made on that same stream incur no extra coordination;
    calls on a different stream are fenced against the EP via CUDA
    events. Either way the caller need only enqueue work on the current
    torch stream and read outputs from it as usual.

    Construct from an existing ``InferenceSession`` (you must also pass
    the ``ep_stream`` you wired in via ``user_compute_stream``) or load
    from disk via the ``from_file`` classmethod, which sets one up
    automatically.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        input_cls: type[InputT],
        output_cls: type[OutputT],
        ep_stream: torch.cuda.Stream,
    ) -> None:
        assert is_dataclass(input_cls), "input_cls must be a dataclass"
        assert is_dataclass(output_cls), "output_cls must be a dataclass"

        self.session = session
        self.input_cls = input_cls
        self.output_cls = output_cls
        self._ep_stream = ep_stream
        self._device = ep_stream.device
        self._input_names = [f.name for f in fields(input_cls)]
        self._output_names = [f.name for f in fields(output_cls)]

        sess_inputs = {n.name: n for n in session.get_inputs()}
        sess_outputs = {n.name: n for n in session.get_outputs()}

        assert set(self._input_names) == set(sess_inputs), (
            f"Input dataclass fields {self._input_names} do not match "
            f"session inputs {list(sess_inputs)}"
        )
        assert set(self._output_names) == set(sess_outputs), (
            f"Output dataclass fields {self._output_names} do not match "
            f"session outputs {list(sess_outputs)}"
        )

        self._input_torch_dtypes: dict[str, torch.dtype] = {
            name: _ort_to_torch_dtype(sess_inputs[name].type)
            for name in self._input_names
        }
        self._input_numpy_dtypes: dict[str, Any] = {
            name: _torch_to_numpy_dtype(self._input_torch_dtypes[name])
            for name in self._input_names
        }
        self._output_torch_dtypes: dict[str, torch.dtype] = {
            name: _ort_to_torch_dtype(sess_outputs[name].type) for name in self._output_names
        }
        self._output_numpy_dtypes: dict[str, Any] = {
            name: _torch_to_numpy_dtype(self._output_torch_dtypes[name])
            for name in self._output_names
        }
        self._output_static_shapes: dict[str, tuple[int, ...] | None] = {
            name: _static_shape(sess_outputs[name].shape) for name in self._output_names
        }

    @classmethod
    def from_file(
        cls,
        onnx_file: str,
        input_cls: type[InputT],
        output_cls: type[OutputT],
        device_id: int = 0,
    ) -> OnnxRTEngine[InputT, OutputT]:
        """Load an ONNX model and wrap it with the CUDA EP.

        Captures the caller's current torch CUDA stream and wires it in as
        the EP's ``user_compute_stream``. As long as the caller keeps that
        stream current, ``__call__`` runs ORT on it directly with no
        per-call event coordination; on any other stream we fall back to
        device-side fencing.
        """
        _windows_preload_onnxruntime_dlls()
        _require_cuda_provider(ort.get_available_providers())
        ep_stream = torch.cuda.current_stream(device=device_id)
        cuda_options: dict[str, Any] = {
            "device_id": device_id,
            "user_compute_stream": str(ep_stream.cuda_stream),
        }
        session = ort.InferenceSession(
            onnx_file,
            providers=[("CUDAExecutionProvider", cuda_options)],
        )
        _require_cuda_provider(session.get_providers())
        return cls(session, input_cls, output_cls, ep_stream=ep_stream)

    def allocate_outputs(self, shapes: dict[str, tuple[int, ...]] | None = None) -> OutputT:
        """Pre-allocate output tensors on CUDA.

        ``shapes`` overrides the model's declared static output shapes and
        is required when any output has a symbolic dim.
        """
        provided = shapes or {}
        out_kwargs: dict[str, torch.Tensor] = {}
        for name in self._output_names:
            shape = provided.get(name) or self._output_static_shapes[name]
            if shape is None:
                raise ValueError(
                    f"Output {name!r} has a dynamic shape; pass it via "
                    f"`shapes={{ {name!r}: ... }}` or supply `out=...` to `__call__`"
                )
            out_kwargs[name] = torch.empty(
                shape,
                dtype=self._output_torch_dtypes[name],
                device=self._device,
            )
        return self.output_cls(**out_kwargs)

    @staticmethod
    def _validate_tensors(
        tensors: dict[str, torch.Tensor],
        expected_dtypes: dict[str, torch.dtype] | None = None,
    ) -> None:
        for name, tensor in tensors.items():
            assert tensor.is_cuda, f"{name!r}: must be a CUDA tensor, got {tensor.device}"
            assert tensor.is_contiguous(), f"{name!r}: must be contiguous"
            if expected_dtypes is not None:
                expected = expected_dtypes[name]
                assert tensor.dtype == expected, (
                    f"{name!r}: expected dtype {expected}, got {tensor.dtype}"
                )

    def __call__(self, inputs: InputT, out: OutputT | None = None) -> OutputT:
        """Run the model.

        Fences against ``torch.cuda.current_stream()`` via CUDA events --
        the caller need only enqueue work on the current stream as usual.
        """
        in_dict = {name: getattr(inputs, name) for name in self._input_names}
        self._validate_tensors(in_dict, expected_dtypes=self._input_torch_dtypes)

        if out is None:
            out = self.allocate_outputs()
        else:
            self._validate_tensors(
                {name: getattr(out, name) for name in self._output_names},
                expected_dtypes=self._output_torch_dtypes,
            )

        binding = self.session.io_binding()
        for name, tensor in in_dict.items():
            binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=tensor.device.index,
                element_type=self._input_numpy_dtypes[name],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        for name in self._output_names:
            tensor = getattr(out, name)
            binding.bind_output(
                name=name,
                device_type="cuda",
                device_id=tensor.device.index,
                element_type=self._output_numpy_dtypes[name],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        ep_stream = self._ep_stream
        caller_stream = torch.cuda.current_stream()
        cross_stream = caller_stream.cuda_stream != ep_stream.cuda_stream

        if cross_stream:
            ev_in = torch.cuda.Event()
            ev_in.record(caller_stream)
            ep_stream.wait_event(ev_in)  # pyright: ignore[reportArgumentType]

        self.session.run_with_iobinding(binding)

        if cross_stream:
            ev_out = torch.cuda.Event()
            ev_out.record(ep_stream)
            caller_stream.wait_event(ev_out)  # pyright: ignore[reportArgumentType]

        return out


__all__ = ["OnnxRTEngine"]
