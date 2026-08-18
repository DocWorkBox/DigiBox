from __future__ import annotations

import sys

import pytest

from avtr1_renderer.artifact_configs.v1 import HF_ARTIFACTS, TRT_ENGINES


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path policy")
def test_windows_tensorrt_engines_use_platform_specific_directories() -> None:
    assert "renderer_runtime_artifacts_cc_win64" in TRT_ENGINES["warp_network"]
    assert "speech2motion_runtime_artifacts_cc_win64" in TRT_ENGINES["avtr1_encode"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows artifact policy")
def test_windows_download_set_excludes_linux_warp_plugin() -> None:
    assert "warp_plugin" not in HF_ARTIFACTS
