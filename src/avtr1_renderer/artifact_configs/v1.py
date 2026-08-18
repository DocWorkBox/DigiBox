# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Artifact definitions for avtr1-live-talkinghead v1.

Storage layout under ``{AVTR1_LOCAL_STORAGE}/{HF_REVISION}/`` (or
``<project_root>/artifacts/{HF_REVISION}/``)::

    renderer_runtime_artifacts/
        appearance_extractor.onnx
        motion_extractor.onnx
        landmark106.onnx
        landmark203.onnx
        insightface_det.onnx
        blaze_face.onnx
        face_mesh.onnx
        libgrid_sample_3d_plugin.so  # locally built TRT plugin (not CC-specific)
    renderer_runtime_artifacts_cc/   # locally built, CC-specific TRT engines
        decoder_b5_fp16.engine
        warp_network_b5_fp16.engine
        stitch_network_b5_fp16.engine
        modnet_b5_fp16.engine
    speech2motion_runtime_artifacts_cc/
        avtr1_normalizer.safetensors
        hubert_lbs_fp16.engine
        avtr1_encode_fp16.engine
        avtr1_decode_fp16.engine
    build_artifacts/                 # ONNX + TorchScript sources for building TRT engines
        decoder.onnx
        warp_network.onnx
        warp_network_ori.onnx
        stitch_network.onnx
        modnet.onnx
        hubert-lbs-avtr1.onnx
        avtr1.scripted.pt
    avatars_artifacts/
        backgrounds/
        reference_frames/

The HuggingFace repo uses the same directory layout (minus ``*_cc/`` dirs,
which are locally built and never uploaded).
"""

from __future__ import annotations

import sys

from avtr1_renderer.artifact_manage_basic import ArtifactSource

HF_REPO_ID = "avaturn-live/avtr-1"
HF_REVISION = "main"

# ---------------------------------------------------------------------------
# Ditto ONNX models — loaded from digital-avatar/ditto-talkinghead on HF.
# Stored locally under renderer_runtime_artifacts/ and build_artifacts/ as usual.
# ---------------------------------------------------------------------------

_DITTO_REPO_ID = "digital-avatar/ditto-talkinghead"


def _ditto(local_path: str, filename: str) -> ArtifactSource:
    return ArtifactSource(local_path, repo_id=_DITTO_REPO_ID, repo_path=f"ditto_onnx/{filename}", revision="main")


# ---------------------------------------------------------------------------
# HF-downloadable artifacts — path_in_repo equals the local relative path.
# ---------------------------------------------------------------------------

_RUNTIME_ONNX_NAMES = [
    "appearance_extractor",
    "motion_extractor",
    "landmark106",
    "landmark203",
    "insightface_det",
    "blaze_face",
    "face_mesh",
]
RENDERER_RUNTIME_ARTIFACTS: dict[str, ArtifactSource] = {
    name: _ditto(f"renderer_runtime_artifacts/{name}.onnx", f"{name}.onnx")
    for name in _RUNTIME_ONNX_NAMES
}
if sys.platform != "win32":
    RENDERER_RUNTIME_ARTIFACTS["warp_plugin"] = ArtifactSource(
        "renderer_runtime_artifacts/libgrid_sample_3d_plugin.so"
    )


AVATAR_ARTIFACTS: dict[str, ArtifactSource] = {
    "backgrounds":      ArtifactSource("avatars_artifacts/backgrounds",      is_dir=True),
    "reference_frames": ArtifactSource("avatars_artifacts/reference_frames", is_dir=True),
    "pasteback_mask":   ArtifactSource("avatars_artifacts/pasteback_mask.png"),
}

BUILD_ARTIFACTS: dict[str, ArtifactSource] = {
    # From Ditto — shared LP/renderer architecture models
    "decoder_onnx":          _ditto("build_artifacts/decoder.onnx",          "decoder.onnx"),
    "warp_network_onnx":     _ditto("build_artifacts/warp_network.onnx",     "warp_network.onnx"),
    "warp_network_ori_onnx": _ditto("build_artifacts/warp_network_ori.onnx", "warp_network_ori.onnx"),
    "stitch_network_onnx":   _ditto("build_artifacts/stitch_network.onnx",   "stitch_network.onnx"),
    "modnet_onnx":           ArtifactSource("build_artifacts/modnet.onnx"),
    # From avtr1 HF repo — avtr1-specific models
    "hubert_onnx":           ArtifactSource("build_artifacts/hubert-lbs-avtr1.onnx"),
    "avtr1_scripted":         ArtifactSource("build_artifacts/avtr1.scripted.pt"),
}

# All HF-downloadable artifacts in one flat namespace for the single manager.
HF_ARTIFACTS: dict[str, ArtifactSource] = {
    **RENDERER_RUNTIME_ARTIFACTS,
    **AVATAR_ARTIFACTS,
    **BUILD_ARTIFACTS,
}

# ---------------------------------------------------------------------------
# Locally-built TRT engines — never on HF, paths relative to storage root.
# ---------------------------------------------------------------------------

_RENDERER_ENGINE_DIR = (
    "renderer_runtime_artifacts_cc_win64"
    if sys.platform == "win32"
    else "renderer_runtime_artifacts_cc"
)
_SPEECH_ENGINE_DIR = (
    "speech2motion_runtime_artifacts_cc_win64"
    if sys.platform == "win32"
    else "speech2motion_runtime_artifacts_cc"
)

TRT_ENGINES: dict[str, str] = {
    "decoder":        f"{_RENDERER_ENGINE_DIR}/decoder_b5_fp16.engine",
    "warp_network":   f"{_RENDERER_ENGINE_DIR}/warp_network_b5_fp16.engine",
    "stitch_network": f"{_RENDERER_ENGINE_DIR}/stitch_network_b5_fp16.engine",
    "modnet":         f"{_RENDERER_ENGINE_DIR}/modnet_b5_fp16.engine",
    "hubert_lbs":     f"{_SPEECH_ENGINE_DIR}/hubert_lbs_fp16.engine",
    "avtr1_encode":   f"{_SPEECH_ENGINE_DIR}/avtr1_encode_fp16.engine",
    "avtr1_decode":   f"{_SPEECH_ENGINE_DIR}/avtr1_decode_fp16.engine",
}
