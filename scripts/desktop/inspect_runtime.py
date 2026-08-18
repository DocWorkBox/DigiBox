# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Machine-readable Windows runtime and TensorRT compatibility inspection."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _version(module_name: str) -> tuple[str | None, str | None]:
    try:
        module = __import__(module_name)
    except Exception as exc:  # diagnostics must remain usable on partial installs
        return None, f"{type(exc).__name__}: {exc}"
    return str(getattr(module, "__version__", "unknown")), None


def _gpu_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {
            "available": False,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "error": "torch.cuda.is_available() is false",
        }
    capability = torch.cuda.get_device_capability()
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "total_memory_bytes": int(properties.total_memory),
        "torch_cuda": getattr(torch.version, "cuda", None),
    }


def _engine_files(runtime_root: Path) -> list[Path]:
    storage = runtime_root / "artifacts" / "main"
    if not storage.is_dir():
        return []
    return sorted(storage.glob("*_runtime_artifacts_cc_win64/*.engine"))


def _runtime_manifest_identity(runtime_root: Path) -> dict[str, Any] | None:
    path = runtime_root / "runtime-manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    schema_version = manifest.get("schemaVersion")
    layout = manifest.get("layout")
    if not isinstance(schema_version, int) or not isinstance(layout, str):
        return None
    return {"schema_version": schema_version, "layout": layout}


def _probe_engines(paths: list[Path]) -> dict[str, dict[str, Any]]:
    if not paths:
        return {}
    result: dict[str, dict[str, Any]] = {}
    try:
        import tensorrt as trt
    except Exception as exc:
        error = f"TensorRT import failed: {type(exc).__name__}: {exc}"
        return {str(path): {"ok": False, "error": error} for path in paths}

    plugin_paths = sorted({path.with_name("grid_sample_3d_plugin.dll") for path in paths})
    for plugin in plugin_paths:
        if plugin.is_file():
            with contextlib.suppress(OSError):
                ctypes.WinDLL(str(plugin))
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    for path in paths:
        try:
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None")
            tensors = []
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                tensors.append(
                    {
                        "name": name,
                        "shape": list(engine.get_tensor_shape(name)),
                        "dtype": str(engine.get_tensor_dtype(name)),
                        "mode": str(engine.get_tensor_mode(name)),
                    }
                )
            result[str(path)] = {
                "ok": True,
                "size": path.stat().st_size,
                "io": tensors,
            }
        except Exception as exc:
            result[str(path)] = {
                "ok": False,
                "size": path.stat().st_size if path.is_file() else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def inspect(
    runtime_root: Path,
    *,
    probe_engines: bool,
    engine_paths: list[Path] | None = None,
) -> dict[str, Any]:
    runtime_root = runtime_root.resolve()
    torch_version, torch_error = _version("torch")
    tensorrt_version, tensorrt_error = _version("tensorrt")
    if engine_paths is None:
        engines = _engine_files(runtime_root)
    else:
        engines = [path.resolve() for path in engine_paths]
        missing = [path for path in engines if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "explicit TensorRT engine path does not exist: "
                + ", ".join(str(path) for path in missing)
            )
    return {
        "schema_version": 1,
        "runtime_root": str(runtime_root),
        "platform": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "runtime_manifest": _runtime_manifest_identity(runtime_root),
        "torch": {"version": torch_version, "error": torch_error},
        "tensorrt": {"version": tensorrt_version, "error": tensorrt_error},
        "gpu": _gpu_status(),
        "artifacts_ready": (runtime_root / "artifacts" / "main").is_dir(),
        "models_ready": (runtime_root / "models").is_dir(),
        "engine_files": [str(path) for path in engines],
        "engine_probe": _probe_engines(engines) if probe_engines else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--probe-engines", action="store_true")
    parser.add_argument(
        "--engine-path",
        action="append",
        type=Path,
        dest="engine_paths",
        help="Inspect this exact staged engine instead of active runtime engines; repeatable.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = inspect(
        args.runtime_root,
        probe_engines=args.probe_engines,
        engine_paths=args.engine_paths,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
