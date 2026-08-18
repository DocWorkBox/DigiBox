# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Artifact manager for avtr1-live-talkinghead.

A single :class:`ArtifactManager` covers all HF-downloadable artifacts.
Locally-built TRT engines are resolved via :func:`get_trt_engine_path`.

Storage root
------------
Set ``AVTR1_LOCAL_STORAGE`` to a directory that will hold all artifacts and
built TRT engines.  When the env var is absent, ``<project_root>/artifacts/``
is used (the project root is the repository checkout this file lives in,
not the caller's current working directory).

Both downloads from HuggingFace and outputs from ``scripts/build_*.py`` go to
the same root, so one directory covers everything::

    {AVTR1_LOCAL_STORAGE}/{HF_REVISION}/
        renderer_runtime_artifacts/{name}.onnx
        speech2motion_runtime_artifacts_cc/avtr1_normalizer.safetensors
        build_artifacts/{filename}
        avatars_artifacts/backgrounds/
        avatars_artifacts/reference_frames/
        renderer_runtime_artifacts_cc/{engine_filename}
        speech2motion_runtime_artifacts_cc/{engine_filename}

TRT engines
-----------
``find_engine_or_onnx()`` uses a TRT engine when present and falls back to the
ONNX model so OnnxRT can run without TRT. The AVTR-1 speech-to-motion model
also has a direct TorchScript CUDA backend. Native Windows prefers locally
built TensorRT engines and falls back to that portable path when they are
absent; released Linux engines/plugins are never reused on Windows.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from avtr1_renderer.artifact_manage_basic import ArtifactManager
from avtr1_renderer.artifact_configs import latest as _latest

_managers: dict[tuple, ArtifactManager] = {}


def _cfg(version: ModuleType | None) -> ModuleType:
    return version if version is not None else _latest


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_storage_root(version: ModuleType | None = None) -> Path:
    """Root directory for all artifacts and built TRT engines.

    Returns ``{AVTR1_LOCAL_STORAGE}/{HF_REVISION}`` when the env var is set,
    otherwise ``<project_root>/artifacts/{HF_REVISION}`` — resolved against
    the repository root (two levels above this file), so the location is
    stable regardless of the caller's current working directory.
    """
    cfg = _cfg(version)
    base = os.environ.get("AVTR1_LOCAL_STORAGE")
    if base:
        return Path(base) / cfg.HF_REVISION
    return _PROJECT_ROOT / "artifacts" / cfg.HF_REVISION


def get_artifact_manager(version: ModuleType | None = None) -> ArtifactManager:
    """Single manager for all HF-downloadable artifacts."""
    cfg = _cfg(version)
    storage_root = get_storage_root(version)
    key = (cfg, storage_root)
    if key not in _managers:
        _managers[key] = ArtifactManager(
            repo_id=cfg.HF_REPO_ID,
            revision=cfg.HF_REVISION,
            artifacts=cfg.HF_ARTIFACTS,
            storage_root=storage_root,
        )
    return _managers[key]


def get_trt_engine_path(
    engine_name: str,
    *,
    version: ModuleType | None = None,
) -> Path:
    """Full path for a named TRT engine, whether built or not.

    Returns the path even if the engine hasn't been built yet — callers should
    check ``.is_file()`` before loading.
    """
    cfg = _cfg(version)
    return get_storage_root(version) / cfg.TRT_ENGINES[engine_name]


def find_engine_or_onnx(
    engine_name: str,
    onnx_artifact: str,
    *,
    version: ModuleType | None = None,
) -> Path:
    """Return the TRT engine path if built, otherwise download and return the ONNX.

    Args:
        engine_name:   Key in ``TRT_ENGINES`` (e.g. ``"decoder"``).
        onnx_artifact: Key in ``HF_ARTIFACTS`` for the ONNX fallback.
        version:       Config module; defaults to ``latest``.
    """
    engine_path = get_trt_engine_path(engine_name, version=version)
    if engine_path.is_file():
        return engine_path
    return get_artifact_manager(version).get_artifact_path(onnx_artifact)
