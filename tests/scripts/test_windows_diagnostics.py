from __future__ import annotations

from pathlib import Path

from scripts.windows_diagnostics import collect_status


def test_current_windows_runtime_has_cuda_ort_and_ffmpeg() -> None:
    status = collect_status(check_hf_access=False)

    assert status["torch"]["cuda_available"] is True
    assert status["torch"]["device_name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert status["torch"]["compute_capability"] == [12, 0]
    assert "CUDAExecutionProvider" in status["onnxruntime"]["providers"]
    assert status["onnxruntime"]["cuda_smoke_passed"] is True
    assert status["onnxruntime"]["cuda_smoke_error"] is None
    assert status["tensorrt"]["version"] == "10.11.0.33"
    assert status["tensorrt"]["builder_available"] is True
    assert Path(status["ffmpeg"]["executable"]).is_file()
