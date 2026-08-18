from __future__ import annotations

from pathlib import Path

import pytest

import avtr1_renderer.runtime.trt as trt_runtime


def test_windows_plugin_loader_registers_dll_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories: list[str] = []
    libraries: list[str] = []
    directory_handle = object()
    library_handle = object()

    monkeypatch.setattr(
        trt_runtime.os,
        "add_dll_directory",
        lambda path: directories.append(path) or directory_handle,
    )
    monkeypatch.setattr(
        trt_runtime.ctypes,
        "CDLL",
        lambda path: libraries.append(path) or library_handle,
    )
    trt_runtime._DLL_DIRECTORY_HANDLES.clear()
    trt_runtime._PLUGIN_HANDLES.clear()
    plugin = Path(r"F:\plugins\grid_sample_3d_plugin.dll")

    trt_runtime._load_plugin_library(plugin, platform="win32")

    assert directories == [str(plugin.parent)]
    assert libraries == [str(plugin)]
    assert [directory_handle] == trt_runtime._DLL_DIRECTORY_HANDLES
    assert [library_handle] == trt_runtime._PLUGIN_HANDLES
