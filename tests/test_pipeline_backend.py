from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import avtr1_renderer.pipeline as pipeline_module
from avtr1_renderer.pipeline import (
    _load_avtr1_engines,
    _resolve_component_path,
    _resolve_dynamic_component_path,
    _resolve_warp_path,
    _resolve_warp_plugin,
    resolve_avtr1_backend,
)


def test_auto_uses_tensorrt_on_windows_when_engine_files_exist() -> None:
    assert (
        resolve_avtr1_backend(
            "auto",
            platform="win32",
            encode_path=Path("encode.engine"),
            decode_path=Path("decode.engine"),
            engines_exist=True,
        )
        == "tensorrt"
    )


def test_auto_preserves_tensorrt_on_linux_when_both_engines_exist() -> None:
    assert (
        resolve_avtr1_backend(
            "auto",
            platform="linux",
            encode_path=Path("encode.engine"),
            decode_path=Path("decode.engine"),
            engines_exist=True,
        )
        == "tensorrt"
    )


def test_auto_falls_back_to_torchscript_when_engines_are_missing() -> None:
    assert (
        resolve_avtr1_backend(
            "auto",
            platform="linux",
            encode_path=Path("missing-encode.engine"),
            decode_path=Path("missing-decode.engine"),
            engines_exist=False,
        )
        == "torchscript"
    )


def test_explicit_tensorrt_reports_missing_engine_paths() -> None:
    with pytest.raises(RuntimeError, match=r"missing-encode\.engine"):
        resolve_avtr1_backend(
            "tensorrt",
            platform="win32",
            encode_path=Path("missing-encode.engine"),
            decode_path=Path("missing-decode.engine"),
            engines_exist=False,
        )


def test_torchscript_backend_loads_one_checkpoint_and_uses_its_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = Path("avtr1.scripted.pt")
    manager = SimpleNamespace(get_artifact_path=lambda name: checkpoint)
    fake_backend = SimpleNamespace(
        encode=object(),
        decode=object(),
        scripted=object(),
    )
    loaded: list[Path] = []

    monkeypatch.setattr(
        pipeline_module.TorchScriptAVTR1Backend,
        "from_file",
        lambda path: loaded.append(path) or fake_backend,
    )
    monkeypatch.setattr(
        pipeline_module.Normalizer,
        "from_scripted",
        lambda scripted: ("normalizer", scripted),
    )

    encode, decode, normalizer = _load_avtr1_engines(
        "torchscript",
        manager=manager,
        encode_path=Path("unused-encode.engine"),
        decode_path=Path("unused-decode.engine"),
    )

    assert loaded == [checkpoint]
    assert encode is fake_backend.encode
    assert decode is fake_backend.decode
    assert normalizer == ("normalizer", fake_backend.scripted)


def test_portable_backend_forces_component_onnx_even_if_an_engine_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx_path = Path("warp_network.onnx")
    manager = SimpleNamespace(get_artifact_path=lambda name: onnx_path)
    monkeypatch.setattr(
        pipeline_module,
        "find_engine_or_onnx",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("TensorRT path must not be considered")
        ),
    )

    resolved = _resolve_component_path(
        "torchscript",
        manager=manager,
        engine_name="warp_network",
        onnx_artifact="warp_network_onnx",
    )

    assert resolved == onnx_path


def test_tensorrt_core_falls_back_to_standard_portable_warp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_engine = tmp_path / "missing-warp.engine"
    original = tmp_path / "warp_network_ori.onnx"
    converted = tmp_path / "warp_network_ori_ort_opset20_b5.onnx"
    manager = SimpleNamespace(get_artifact_path=lambda name: original)

    monkeypatch.setattr(pipeline_module, "get_trt_engine_path", lambda name: missing_engine)
    monkeypatch.setattr(
        pipeline_module,
        "prepare_portable_warp_onnx",
        lambda path: converted if path == original else None,
    )

    assert _resolve_warp_path("tensorrt", manager=manager) == converted


def test_dynamic_renderer_onnx_is_prepared_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("stitch_network.onnx")
    converted = Path("stitch_network_ort_dynamic.onnx")
    manager = SimpleNamespace(get_artifact_path=lambda name: source)
    calls: list[Path] = []

    resolved = _resolve_dynamic_component_path(
        "torchscript",
        manager=manager,
        engine_name="stitch_network",
        onnx_artifact="stitch_network_onnx",
        prepare=lambda path: calls.append(path) or converted,
    )

    assert resolved == converted
    assert calls == [source]


def test_windows_warp_plugin_accepts_explicit_dll_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "grid_sample_3d_plugin.dll"
    plugin.touch()
    monkeypatch.setenv("AVTR1_WARP_PLUGIN", str(plugin))

    assert _resolve_warp_plugin(platform="win32", manager=object()) == plugin


def test_windows_warp_plugin_error_names_required_dll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AVTR1_WARP_PLUGIN", raising=False)
    monkeypatch.setattr(
        pipeline_module,
        "get_trt_engine_path",
        lambda name: tmp_path / "warp_network_b5_fp16.engine",
    )

    with pytest.raises(RuntimeError, match=r"grid_sample_3d_plugin\.dll"):
        _resolve_warp_plugin(platform="win32", manager=object())
