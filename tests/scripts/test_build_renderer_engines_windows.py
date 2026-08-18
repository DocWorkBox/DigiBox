from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import scripts.build_renderer_engines as renderer_builder
from scripts.build_renderer_engines import _persist_windows_warp_plugin


def test_tensorrt_logger_does_not_hide_cuda_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()

    logger.log(renderer_builder.trt.Logger.ERROR, "CUDA error 720")

    assert "CUDA error 720" in capsys.readouterr().err


def test_tensorrt_logger_suppresses_recoverable_cuda_720_tactic_flood(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()
    message = (
        "Error Code: 9: Skipping tactic 0x123 due to exception CUDA error 720 launching kernel"
    )

    logger.log(renderer_builder.trt.Logger.ERROR, message)
    logger.log(renderer_builder.trt.Logger.ERROR, message)

    stderr = capsys.readouterr().err
    assert stderr.count("[TRT] [INFO]") == 1
    assert "build is continuing to search" in stderr
    assert "[TRT] [ERROR]" not in stderr
    assert message not in stderr
    assert logger.recoverable_tactic_skip_count == 2


@pytest.mark.parametrize(
    ("severity", "message"),
    [
        (renderer_builder.trt.Logger.ERROR, "CUDA error 720"),
        (
            renderer_builder.trt.Logger.ERROR,
            "Skipping tactic 0x123 after CUDA error 719",
        ),
        (renderer_builder.trt.Logger.ERROR, "unrelated TensorRT failure"),
        (
            renderer_builder.trt.Logger.WARNING,
            "Skipping tactic 0x123 after CUDA error 720",
        ),
        (
            renderer_builder.trt.Logger.ERROR,
            "Skipping tactic 0x123 after CUDA error 7200",
        ),
        (
            renderer_builder.trt.Logger.ERROR,
            "Skipping tactic 0x123 after CUDA error 1720",
        ),
    ],
)
def test_tensorrt_logger_preserves_nonrecoverable_messages(
    severity: renderer_builder.trt.ILogger.Severity,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()

    logger.log(severity, message)

    stderr = capsys.readouterr().err
    assert f"[TRT] [{severity.name}] {message}" in stderr
    assert logger.recoverable_tactic_skip_count == 0


def test_tensorrt_logger_reports_recoverable_tactic_count_after_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()
    message = "Skipping tactic 0x123 after CUDA error 720"
    logger.log(renderer_builder.trt.Logger.ERROR, message)
    logger.log(renderer_builder.trt.Logger.ERROR, message)
    capsys.readouterr()

    logger.report_recoverable_tactic_skips("decoder", success=True)

    stderr = capsys.readouterr().err
    assert "[decoder] [INFO]" in stderr
    assert "2 tactic candidate(s)" in stderr
    assert "serialized engine plan was built successfully" in stderr


def test_tensorrt_logger_reports_filtered_tactics_when_build_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()
    logger.log(
        renderer_builder.trt.Logger.ERROR,
        "Skipping tactic 0x123 after CUDA error 720",
    )
    capsys.readouterr()

    logger.report_recoverable_tactic_skips("decoder", success=False)

    stderr = capsys.readouterr().err
    assert "[decoder] [INFO]" in stderr
    assert "1 tactic candidate(s)" in stderr
    assert "ultimately failed or raised an exception" in stderr
    assert "built successfully" not in stderr


def test_serialized_plan_none_reports_filtered_tactics_before_failing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()

    class NonePlanBuilder:
        def build_serialized_network(self, network: object, config: object) -> None:
            logger.log(
                renderer_builder.trt.Logger.ERROR,
                "Skipping tactic 0x123 after CUDA error 720",
            )
            return None

    with pytest.raises(RuntimeError, match=r"\[decoder\] TRT build failed"):
        renderer_builder._build_serialized_engine_plan(
            NonePlanBuilder(),
            object(),
            object(),
            logger,
            engine_name="decoder",
        )

    stderr = capsys.readouterr().err
    assert "1 tactic candidate(s)" in stderr
    assert "ultimately failed or raised an exception" in stderr
    assert "built successfully" not in stderr


def test_serialized_plan_exception_reports_filtered_tactics_before_reraising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = renderer_builder._FilteredLogger()

    class RaisingBuilder:
        def build_serialized_network(self, network: object, config: object) -> None:
            logger.log(
                renderer_builder.trt.Logger.ERROR,
                "Skipping tactic 0x123 after CUDA error 720",
            )
            raise ValueError("builder exploded")

    with pytest.raises(ValueError, match="builder exploded"):
        renderer_builder._build_serialized_engine_plan(
            RaisingBuilder(),
            object(),
            object(),
            logger,
            engine_name="warp",
        )

    stderr = capsys.readouterr().err
    assert "1 tactic candidate(s)" in stderr
    assert "ultimately failed or raised an exception" in stderr
    assert "built successfully" not in stderr


@pytest.mark.parametrize(
    "builder_name",
    ["_build_decoder", "_build_warp", "_build_modnet", "_build_stitch"],
)
def test_engine_builders_route_plan_build_through_outcome_reporting(
    builder_name: str,
) -> None:
    source = inspect.getsource(getattr(renderer_builder, builder_name))

    assert source.count("_build_serialized_engine_plan(") == 1
    assert "builder.build_serialized_network" not in source


def test_warp_plugin_smoke_rechecks_batch_one_after_max_batch() -> None:
    sequence = getattr(renderer_builder, "_warp_plugin_smoke_batches", None)

    assert sequence is not None
    assert sequence(5) == (1, 5, 1)


def test_windows_warp_plugin_is_copied_next_to_engine(tmp_path: Path) -> None:
    plugin = tmp_path / "sdk" / "custom_grid_sample.dll"
    engine = tmp_path / "engines" / "warp_network_b5_fp16.engine"
    plugin.parent.mkdir()
    engine.parent.mkdir()
    plugin.write_bytes(b"native-plugin")

    installed = _persist_windows_warp_plugin(
        plugin,
        engine,
        platform="win32",
    )

    assert installed == engine.with_name("grid_sample_3d_plugin.dll")
    assert installed.read_bytes() == b"native-plugin"


def test_non_windows_warp_plugin_is_not_copied(tmp_path: Path) -> None:
    plugin = tmp_path / "libgrid_sample_3d_plugin.so"
    engine = tmp_path / "warp_network_b5_fp16.engine"
    plugin.write_bytes(b"native-plugin")

    installed = _persist_windows_warp_plugin(
        plugin,
        engine,
        platform="linux",
    )

    assert installed == plugin
    assert not engine.with_name("grid_sample_3d_plugin.dll").exists()


def test_conflicting_windows_plugin_stops_before_engine_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "sdk" / "new_grid_sample.dll"
    onnx = tmp_path / "warp.onnx"
    out_dir = tmp_path / "engines"
    target = out_dir / "grid_sample_3d_plugin.dll"
    plugin.parent.mkdir()
    out_dir.mkdir()
    plugin.write_bytes(b"new-plugin")
    target.write_bytes(b"old-plugin")
    onnx.write_bytes(b"placeholder")
    build_called = False

    def fail_if_built(*args: object, **kwargs: object) -> None:
        nonlocal build_called
        build_called = True

    monkeypatch.setattr(renderer_builder, "_build_warp", fail_if_built)
    monkeypatch.setattr(renderer_builder.torch.cuda, "is_available", lambda: False)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        renderer_builder.main(
            [
                "warp",
                "--warp-onnx",
                str(onnx),
                "--warp-plugin",
                str(plugin),
                "--out-dir",
                str(out_dir),
            ]
        )

    assert build_called is False
    assert target.read_bytes() == b"old-plugin"
