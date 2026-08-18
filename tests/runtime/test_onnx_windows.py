from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from avtr1_renderer.runtime.onnxrt import (
    OnnxRTEngine,
    _require_cuda_provider,
    _windows_preload_onnxruntime_dlls,
)


def test_windows_preloads_onnxruntime_cuda_dlls_when_supported() -> None:
    calls: list[str] = []
    runtime = SimpleNamespace(preload_dlls=lambda: calls.append("preloaded"))

    _windows_preload_onnxruntime_dlls(runtime, platform="win32")

    assert calls == ["preloaded"]


def test_non_windows_does_not_preload_windows_dlls() -> None:
    calls: list[str] = []
    runtime = SimpleNamespace(preload_dlls=lambda: calls.append("preloaded"))

    _windows_preload_onnxruntime_dlls(runtime, platform="linux")

    assert calls == []


def test_missing_cuda_provider_has_an_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        _require_cuda_provider(["CPUExecutionProvider"])


def test_call_validates_input_dtype_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ValidationReachedError(Exception):
        pass

    expected = {"input": torch.float32}

    def validate(tensors: object, expected_dtypes: object = None) -> None:
        assert expected_dtypes == expected
        raise ValidationReachedError

    monkeypatch.setattr(OnnxRTEngine, "_validate_tensors", staticmethod(validate))
    engine = object.__new__(OnnxRTEngine)
    engine._input_names = ["input"]
    engine._input_torch_dtypes = expected

    with pytest.raises(ValidationReachedError):
        engine(SimpleNamespace(input=object()))


def test_output_allocation_uses_the_execution_provider_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class Input:
        input: torch.Tensor

    @dataclass
    class Output:
        output: torch.Tensor

    metadata = SimpleNamespace(name="input", type="tensor(float)", shape=[1])
    output_metadata = SimpleNamespace(name="output", type="tensor(float)", shape=[1])
    session = SimpleNamespace(
        get_inputs=lambda: [metadata],
        get_outputs=lambda: [output_metadata],
    )
    stream = SimpleNamespace(device=torch.device("cuda", 3))
    devices: list[torch.device | str] = []
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, dtype, device: devices.append(device) or object(),
    )

    engine = OnnxRTEngine(session, Input, Output, ep_stream=stream)
    engine.allocate_outputs()

    assert devices == [torch.device("cuda", 3)]
