"""Detect optional NVIDIA video-enhancement runtimes on Windows.

Detection is deliberately side-effect free: importing this module never loads
an NVIDIA DLL.  The renderer can expose the result in its settings UI before a
future SDK adapter is enabled, without making the SDK a hard dependency.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from importlib import util as importlib_util
from pathlib import Path
from typing import TypedDict


class NvidiaVideoEffectsCapability(TypedDict):
    available: bool
    backend: str | None
    reason: str


_RTX_ENV = "NVIDIA_RTX_VIDEO_SDK_PATH"
_MAXINE_ENV = "NV_VIDEO_EFFECTS_PATH"


def _runtime_candidates(environment: Mapping[str, str]) -> tuple[tuple[str, Path], ...]:
    candidates: list[tuple[str, Path]] = []

    rtx_root = environment.get(_RTX_ENV, "").strip()
    if rtx_root:
        root = Path(rtx_root)
        candidates.extend(
            (
                ("rtx_video_sdk", root / "bin" / "x64" / "NVVideoEffects.dll"),
                ("rtx_video_sdk", root / "NVVideoEffects.dll"),
            )
        )

    maxine_root = environment.get(_MAXINE_ENV, "").strip()
    if maxine_root:
        root = Path(maxine_root)
        candidates.extend(
            (
                ("maxine_vfx", root / "NVVideoEffects.dll"),
                ("maxine_vfx", root / "bin" / "x64" / "NVVideoEffects.dll"),
            )
        )

    program_files = environment.get("ProgramFiles", r"C:\Program Files")
    pf = Path(program_files)
    candidates.extend(
        (
            (
                "rtx_video_sdk",
                pf / "NVIDIA Corporation" / "RTX Video SDK" / "bin" / "x64" / "NVVideoEffects.dll",
            ),
            (
                "maxine_vfx",
                pf
                / "NVIDIA Corporation"
                / "NVIDIA Video Effects"
                / "NVIDIA Video Effects SDK"
                / "bin"
                / "x64"
                / "NVVideoEffects.dll",
            ),
        )
    )
    return tuple(candidates)


def detect_nvidia_video_effects(
    environment: Mapping[str, str] | None = None,
    exists: Callable[[str | Path], bool] | None = None,
    find_spec: Callable[[str], object | None] | None = None,
) -> NvidiaVideoEffectsCapability:
    """Return whether the renderer's official ``nvvfx`` adapter is installed.

    A discovered ``NVVideoEffects.dll`` is diagnostic only: this application
    implements NVIDIA's official Python/DLPack path, so a DLL by itself must
    never be advertised as an active RTX backend. The SDK is not imported and
    no GPU/model resources are loaded by this capability check.
    """

    env = os.environ if environment is None else environment
    path_exists = Path.is_file if exists is None else exists
    spec_finder = importlib_util.find_spec if find_spec is None else find_spec
    try:
        binding_spec = spec_finder("nvvfx")
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        binding_spec = None
    if binding_spec is not None:
        return {
            "available": True,
            "backend": "nvidia_vfx_python",
            "reason": "Official nvvfx Python binding found; SDK feature/model loading "
            "is validated lazily on first VSR use.",
        }

    checked: list[str] = []
    discovered_runtime: Path | None = None
    for backend, runtime in _runtime_candidates(env):
        _ = backend
        runtime_text = str(runtime)
        if runtime_text in checked:
            continue
        checked.append(runtime_text)
        if path_exists(runtime):
            discovered_runtime = runtime
            break

    if discovered_runtime is not None:
        reason = (
            f"NVVideoEffects DLL found at {discovered_runtime}, but the official "
            "nvvfx Python binding is not installed; the DLL alone is not a usable "
            "renderer backend."
        )
    else:
        reason = (
            "Official nvvfx Python binding is not installed. Install NVIDIA's "
            "Windows VFX SDK core, Python binding, and nvvfxvideosuperres feature "
            "before enabling RTX VSR."
        )

    return {
        "available": False,
        "backend": None,
        "reason": reason,
    }


__all__ = ["NvidiaVideoEffectsCapability", "detect_nvidia_video_effects"]
