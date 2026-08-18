# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Validate the native Windows CUDA runtime without exposing credentials."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any


def _smoke_test_ort_cuda(ort: Any) -> tuple[bool, str | None]:
    """Create and execute a tiny CUDA-only graph to catch missing runtime DLLs."""
    try:
        import numpy as np
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Relu", ["input"], ["output"])],
            "avtr1_windows_cuda_smoke",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = min(model.ir_version, 10)
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )
        actual = session.run(None, {"input": np.asarray([-1.0, 2.0], np.float32)})[0]
        if not np.array_equal(actual, np.asarray([0.0, 2.0], np.float32)):
            raise RuntimeError(f"unexpected CUDA smoke output: {actual}")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def collect_status(*, check_hf_access: bool = False) -> dict[str, Any]:
    import imageio_ffmpeg
    import onnxruntime as ort
    import torch
    from huggingface_hub import (
        get_hf_file_metadata,
        get_token,
        hf_hub_url,
    )

    if sys.platform == "win32" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    ort_cuda_smoke_passed, ort_cuda_smoke_error = _smoke_test_ort_cuda(ort)

    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        tensorrt_status: dict[str, Any] = {
            "available": builder is not None,
            "version": trt.__version__,
            "builder_available": builder is not None,
            "error": None,
        }
    except Exception as exc:
        tensorrt_status = {
            "available": False,
            "version": None,
            "builder_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cuda_available = torch.cuda.is_available()
    torch_status: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": None,
        "compute_capability": None,
        "vram_gib": None,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        torch_status.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "vram_gib": round(properties.total_memory / 1024**3, 2),
            }
        )

    token = get_token()
    hf_status: dict[str, Any] = {
        "token_present": token is not None,
        "model_access": None,
        "error": None,
    }
    if check_hf_access:
        if token is None:
            hf_status["model_access"] = False
            hf_status["error"] = "No Hugging Face token is configured"
        else:
            try:
                url = hf_hub_url(
                    "avaturn-live/avtr-1",
                    "build_artifacts/avtr1.scripted.pt",
                )
                get_hf_file_metadata(url, token=token)
                hf_status["model_access"] = True
            except Exception as exc:
                hf_status["model_access"] = False
                hf_status["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "torch": torch_status,
        "onnxruntime": {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
            "cuda_smoke_passed": ort_cuda_smoke_passed,
            "cuda_smoke_error": ort_cuda_smoke_error,
        },
        "tensorrt": tensorrt_status,
        "ffmpeg": {"executable": imageio_ffmpeg.get_ffmpeg_exe()},
        "huggingface": hf_status,
    }


def _failures(
    status: dict[str, Any],
    *,
    require_hf_access: bool,
    require_tensorrt: bool,
) -> list[str]:
    failures: list[str] = []
    if status["platform"]["system"] != "Windows":
        failures.append("This diagnostic is for native Windows")
    if not status["torch"]["cuda_available"]:
        failures.append("PyTorch CUDA is unavailable")
    if "CUDAExecutionProvider" not in status["onnxruntime"]["providers"]:
        failures.append("ONNX Runtime CUDAExecutionProvider is unavailable")
    elif not status["onnxruntime"]["cuda_smoke_passed"]:
        failures.append(
            "ONNX Runtime CUDA execution failed: "
            f"{status['onnxruntime']['cuda_smoke_error']}"
        )
    if require_hf_access and status["huggingface"]["model_access"] is not True:
        failures.append("Hugging Face gated model access is unavailable")
    if require_tensorrt and not status["tensorrt"]["available"]:
        failures.append("TensorRT is unavailable")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-hf-access",
        action="store_true",
        help="Perform a metadata request against the gated AVTR-1 weight.",
    )
    parser.add_argument(
        "--require-tensorrt",
        action="store_true",
        help="Fail when the optional TensorRT 10.11 runtime cannot be imported.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    status = collect_status(check_hf_access=args.check_hf_access)
    failures = _failures(
        status,
        require_hf_access=args.check_hf_access,
        require_tensorrt=args.require_tensorrt,
    )
    if args.json:
        print(json.dumps({"status": status, "failures": failures}, indent=2))
    else:
        print(f"Windows: {status['platform']['version']}")
        print(
            "PyTorch: "
            f"{status['torch']['version']} / CUDA {status['torch']['cuda_runtime']}"
        )
        print(
            "GPU: "
            f"{status['torch']['device_name']} / cc {status['torch']['compute_capability']} "
            f"/ {status['torch']['vram_gib']} GiB"
        )
        print(
            f"ONNX Runtime: {status['onnxruntime']['version']} / "
            f"{status['onnxruntime']['providers']}"
        )
        print(f"ONNX Runtime CUDA smoke: {status['onnxruntime']['cuda_smoke_passed']}")
        if status["onnxruntime"]["cuda_smoke_error"]:
            print(f"ORT CUDA error: {status['onnxruntime']['cuda_smoke_error']}")
        print(
            "TensorRT: "
            f"{status['tensorrt']['version'] or status['tensorrt']['error']}"
        )
        print(f"FFmpeg: {status['ffmpeg']['executable']}")
        print(
            "Hugging Face token: "
            f"{'present' if status['huggingface']['token_present'] else 'missing'}"
        )
        if args.check_hf_access:
            print(f"AVTR-1 gated access: {status['huggingface']['model_access']}")
            if status["huggingface"]["error"]:
                print(f"HF error: {status['huggingface']['error']}")
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
