# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

from typing import TYPE_CHECKING, Any

from avtr1_renderer.runtime.inference_engine import InferenceEngine
from avtr1_renderer.runtime.loader import load_engine
from avtr1_renderer.runtime.onnxrt import OnnxRTEngine
from avtr1_renderer.runtime.torchscript_avtr1 import TorchScriptAVTR1Backend

if TYPE_CHECKING:
    from avtr1_renderer.runtime.trt import TRTEngine


def __getattr__(name: str) -> Any:
    """Load the optional TensorRT wrapper only when explicitly requested."""
    if name != "TRTEngine":
        raise AttributeError(name)
    try:
        from avtr1_renderer.runtime.trt import TRTEngine
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorRT is required only for .engine/.trt/.plan files. "
            "Install TensorRT or select an ONNX/TorchScript backend."
        ) from exc
    return TRTEngine

__all__ = [
    "InferenceEngine",
    "OnnxRTEngine",
    "TRTEngine",
    "TorchScriptAVTR1Backend",
    "load_engine",
]
