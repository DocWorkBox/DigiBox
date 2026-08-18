from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_without_tensorrt(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    inherited = env.get("AVTR1_MAIN_PYTHONPATH") or env.get("PYTHONPATH", "")
    python_paths = [str(PROJECT_ROOT / "src")]
    python_paths.extend(path for path in inherited.split(os.pathsep) if path)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    blocker = """
import importlib.abc
import sys

class _BlockTensorRT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorrt" or fullname.startswith("tensorrt."):
            raise ModuleNotFoundError("tensorrt intentionally blocked by test")
        return None

sys.meta_path.insert(0, _BlockTensorRT())
"""
    return subprocess.run(
        [sys.executable, "-c", blocker + code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_runtime_import_does_not_require_tensorrt() -> None:
    result = _run_without_tensorrt(
        "from avtr1_renderer.runtime import OnnxRTEngine, load_engine\n"
        "print(OnnxRTEngine.__name__, load_engine.__name__)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OnnxRTEngine load_engine"


def test_loading_a_tensorrt_plan_reports_the_optional_dependency() -> None:
    result = _run_without_tensorrt(
        """
from dataclasses import dataclass
from avtr1_renderer.runtime import load_engine

@dataclass
class Input:
    value: object

@dataclass
class Output:
    value: object

try:
    load_engine("missing.engine", Input, Output)
except ModuleNotFoundError as exc:
    print(str(exc))
else:
    raise AssertionError("expected an optional TensorRT dependency error")
"""
    )

    assert result.returncode == 0, result.stderr
    assert "TensorRT is required only for .engine/.trt/.plan files" in result.stdout
