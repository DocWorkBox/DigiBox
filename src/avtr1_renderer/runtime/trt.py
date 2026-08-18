# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

import ctypes
import os
import sys
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import tensorrt as trt
import torch

_logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(_logger, "")

_DLL_DIRECTORY_HANDLES: list[Any] = []
_PLUGIN_HANDLES: list[Any] = []


def _load_plugin_library(
    plugin: str | Path,
    *,
    platform: str = sys.platform,
) -> None:
    """Load a TensorRT plugin and retain Windows DLL search/load handles."""
    plugin_path = Path(plugin)
    if platform == "win32" and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(plugin_path.parent)))
    _PLUGIN_HANDLES.append(ctypes.CDLL(str(plugin_path)))


_TRT_TO_TORCH_DTYPE: dict[trt.DataType, torch.dtype] = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int8: torch.int8,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
    trt.uint8: torch.uint8,
}
if hasattr(trt, "bfloat16"):
    _TRT_TO_TORCH_DTYPE[trt.bfloat16] = torch.bfloat16


def _trt_to_torch_dtype(dtype: trt.DataType) -> torch.dtype:
    try:
        return _TRT_TO_TORCH_DTYPE[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported TRT dtype: {dtype}") from e


class TRTEngine[InputT, OutputT]:
    """
    Run a TensorRT engine on torch CUDA tensors with zero host↔device copies,
    stream-ordered against the caller's current PyTorch stream.

    Inputs and outputs are dataclasses whose field names must match the
    engine's tensor names. A fresh OutputT is allocated per call (or the
    caller can pass `out=...` to write into pre-allocated tensors), so
    multiple results can be kept live without cloning a shared buffer.

    The engine runs on the caller's current torch stream. No
    `torch.cuda.synchronize()` is performed — same-stream torch consumers
    see results in order; cross-stream / non-torch consumers must fence
    themselves (events or `stream.synchronize()`).

    Construct from a deserialised engine (`TRTEngine(engine, ...)`) or load
    from disk via the `from_file` classmethod.
    """

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_cls: type[InputT],
        output_cls: type[OutputT],
    ) -> None:
        assert is_dataclass(input_cls), "input_cls must be a dataclass"
        assert is_dataclass(output_cls), "output_cls must be a dataclass"
        assert engine is not None, "engine must be a deserialised ICudaEngine"

        self.engine: trt.ICudaEngine = engine
        self.input_cls = input_cls
        self.output_cls = output_cls
        self._input_names = [f.name for f in fields(input_cls)]
        self._output_names = [f.name for f in fields(output_cls)]

        engine_inputs: list[str] = []
        engine_outputs: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                engine_inputs.append(name)
            else:
                engine_outputs.append(name)
        assert set(self._input_names) == set(engine_inputs), (
            f"Input dataclass fields {self._input_names} do not match engine inputs {engine_inputs}"
        )
        assert set(self._output_names) == set(engine_outputs), (
            f"Output dataclass fields {self._output_names} do not match "
            f"engine outputs {engine_outputs}"
        )

        self._output_dtypes: dict[str, torch.dtype] = {
            name: _trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            for name in self._output_names
        }

        self.context = self.engine.create_execution_context()

    @classmethod
    def from_file(
        cls,
        trt_file: str,
        input_cls: type[InputT],
        output_cls: type[OutputT],
        plugin_files: Iterable[str] = (),
    ) -> "TRTEngine[InputT, OutputT]":
        """Load and deserialise a TRT engine from disk, then wrap it.

        ``plugin_files`` are dlopen'd before deserialisation so engines that
        reference custom plugins resolve their symbols.
        """
        for plugin in plugin_files:
            _load_plugin_library(plugin)

        runtime = trt.Runtime(_logger)
        with open(trt_file, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine is not None, f"Failed to deserialize {trt_file}"
        return cls(engine, input_cls, output_cls)

    @staticmethod
    def _validate_tensors(
        tensors: dict[str, torch.Tensor],
        expected_shapes: dict[str, tuple[int, ...]] | None = None,
        expected_dtypes: dict[str, torch.dtype] | None = None,
    ) -> None:
        for name, tensor in tensors.items():
            assert tensor.is_cuda, f"{name!r}: must be a CUDA tensor, got {tensor.device}"
            assert tensor.is_contiguous(), f"{name!r}: must be contiguous"
            if expected_shapes is not None:
                expected = expected_shapes[name]
                assert tuple(tensor.shape) == expected, (
                    f"{name!r}: expected shape {expected}, got {tuple(tensor.shape)}"
                )
            if expected_dtypes is not None:
                expected = expected_dtypes[name]
                assert tensor.dtype == expected, (
                    f"{name!r}: expected dtype {expected}, got {tensor.dtype}"
                )

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> OutputT:
        """Allocate a fresh ``OutputT`` of CUDA tensors sized for the engine.

        ``shapes=None`` (the default) introspects the engine's static
        binding shapes -- correct for engines with no dynamic output
        dims. Pass an explicit dict for engines whose outputs depend on
        runtime input shape (queried from the execution context after
        ``set_input_shape``).
        """
        if shapes is None:
            shapes = {
                name: tuple(self.engine.get_tensor_shape(name))
                for name in self._output_names
            }
            for name, shape in shapes.items():
                if any(d < 0 for d in shape):
                    raise ValueError(
                        f"output {name!r} has dynamic shape {shape}; "
                        "pass an explicit shapes dict to allocate_outputs"
                    )
        return self.output_cls(
            **{
                name: torch.empty(
                    shapes[name],
                    dtype=self._output_dtypes[name],
                    device="cuda",
                )
                for name in self._output_names
            }
        )

    def __call__(self, inputs: InputT, out: OutputT | None = None) -> OutputT:
        in_dict = {name: getattr(inputs, name) for name in self._input_names}
        ctx = self.context

        self._validate_tensors(in_dict)
        for name, tensor in in_dict.items():
            ctx.set_input_shape(name, tuple(tensor.shape))
            ctx.set_tensor_address(name, tensor.data_ptr())

        resolved_shapes = {name: tuple(ctx.get_tensor_shape(name)) for name in self._output_names}

        if out is None:
            out = self.allocate_outputs(resolved_shapes)
        else:
            self._validate_tensors(
                {name: getattr(out, name) for name in self._output_names},
                expected_shapes=resolved_shapes,
                expected_dtypes=self._output_dtypes,
            )

        for name in self._output_names:
            ctx.set_tensor_address(name, getattr(out, name).data_ptr())

        ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

        return out
