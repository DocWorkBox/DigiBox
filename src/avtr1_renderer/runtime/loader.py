# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Path-based engine loader that picks the implementation by file extension.

Components don't care which ``InferenceEngine`` they run on, so the
constructor pattern is::

    self._engine = load_engine(path, InputCls, OutputCls)

and the file's suffix decides whether ``TRTEngine`` or ``OnnxRTEngine``
takes over. This lets the same component class work with a serialized
TRT engine in production and a portable ``.onnx`` in dev / setup-time
paths -- without dragging both wrappers into every constructor.

Recognised suffixes:
    ``.engine``, ``.trt``, ``.plan``  → ``TRTEngine``
    ``.onnx``                          → ``OnnxRTEngine``
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from avtr1_renderer.runtime.inference_engine import InferenceEngine
from avtr1_renderer.runtime.onnxrt import OnnxRTEngine

_TRT_SUFFIXES = frozenset({".engine", ".trt", ".plan"})
_ONNX_SUFFIXES = frozenset({".onnx"})


def load_engine[InputT, OutputT](
    path: str | Path,
    input_cls: type[InputT],
    output_cls: type[OutputT],
    *,
    plugin_files: Iterable[str] = (),
) -> InferenceEngine[InputT, OutputT]:
    """Pick a concrete ``InferenceEngine`` based on ``path``'s suffix.

    ``plugin_files`` are dlopen'd before TRT deserialisation so engines
    that reference custom plugins resolve their symbols. Ignored for
    ONNX paths (they don't need TRT plugins).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _TRT_SUFFIXES:
        try:
            from avtr1_renderer.runtime.trt import TRTEngine
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TensorRT is required only for .engine/.trt/.plan files. "
                "Install TensorRT or select an ONNX/TorchScript backend."
            ) from exc
        return TRTEngine.from_file(str(p), input_cls, output_cls, plugin_files=plugin_files)
    if suffix in _ONNX_SUFFIXES:
        return OnnxRTEngine.from_file(str(p), input_cls, output_cls)
    raise ValueError(
        f"Unrecognised engine file suffix {suffix!r} for {p}; "
        f"expected one of {sorted(_TRT_SUFFIXES | _ONNX_SUFFIXES)}"
    )


__all__ = ["load_engine"]
