from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch

import avtr1_renderer.api.app as renderer_api
import avtr1_renderer.pipeline as pipeline_module
from avtr1_renderer.nvidia_vsr import NvidiaVSRUnavailableError
from avtr1_renderer.pipeline import Pipeline
from avtr1_renderer.types import Chunk, Frame, RenderOptions


def test_renderer_launcher_preinitializes_nvidia_vsr_before_importing_uvicorn() -> None:
    launcher_path = Path(renderer_api.__file__).with_name("launcher.py")
    launcher_source = launcher_path.read_text(encoding="utf-8")

    assert launcher_source.index("preinitialize_nvidia_vsr()") < launcher_source.index(
        "from uvicorn.main import main"
    )


def test_center_crop_output_preserves_the_middle_of_rgb_and_alpha() -> None:
    rgb = torch.arange(1 * 3 * 4 * 8, dtype=torch.float32).reshape(1, 3, 4, 8)
    alpha = torch.arange(1 * 1 * 4 * 8, dtype=torch.float32).reshape(1, 1, 4, 8)

    cropped_rgb, cropped_alpha = pipeline_module._center_crop_output(
        rgb,
        alpha,
        output_height=4,
        output_width=4,
    )

    assert torch.equal(cropped_rgb, rgb[:, :, :, 2:6])
    assert torch.equal(cropped_alpha, alpha[:, :, :, 2:6])


@pytest.mark.parametrize("stream_frames", [True, False])
def test_pipeline_crops_before_packing_in_both_render_modes(
    monkeypatch: pytest.MonkeyPatch,
    stream_frames: bool,
) -> None:
    rgb = torch.arange(1 * 3 * 4 * 8, dtype=torch.float32).reshape(1, 3, 4, 8)
    alpha = torch.arange(1 * 1 * 4 * 8, dtype=torch.float32).reshape(1, 1, 4, 8)
    packed_inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class MotionGenerator:
        chunk_size = 1
        future_size = 0
        frame_len = 2
        audio_shift = 0

        def initial_state(self, _avatar: object) -> str:
            return "initial"

        def generate_chunk(
            self,
            _chunk: Chunk,
            _avatar: object,
            _state: str,
            _options: RenderOptions,
        ) -> tuple[object, str]:
            return object(), "next"

    def fake_stream(*_args: object, **_kwargs: object):
        yield rgb, alpha

    def fake_batch(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return rgb, alpha

    def fake_pack(
        packed_rgb: torch.Tensor,
        packed_alpha: torch.Tensor | None,
        *,
        pixel_format: str,
    ) -> list[Frame]:
        _ = pixel_format
        packed_inputs.append(
            (packed_rgb.clone(), packed_alpha.clone() if packed_alpha is not None else None)
        )
        h, w = packed_rgb.shape[-2:]
        return [Frame(data=np.zeros((h * 3 // 2, w), dtype=np.uint8), height=h, width=w)]

    monkeypatch.setattr(pipeline_module, "render_chunk_streaming", fake_stream)
    monkeypatch.setattr(pipeline_module, "render_chunk", fake_batch)
    monkeypatch.setattr(pipeline_module, "pack_frames", fake_pack)

    pipeline = Pipeline(
        motion_generator=MotionGenerator(),
        stitch=object(),
        warp=object(),
        decoder=object(),
        matting=object(),
        backgrounds={"bg": torch.zeros(1, 3, 4, 8)},
        out_size=(4, 8),
    )
    chunk = Chunk(
        audio_speech=np.zeros(2, dtype=np.float32),
        audio_listen=np.zeros(2, dtype=np.float32),
    )
    options = RenderOptions(
        bg_id="bg",
        output_height=4,
        output_width=4,
        stream_frames=stream_frames,
    )

    _state, frames = pipeline.process_chunk(object(), chunk, "state", options)
    assert [(frame.height, frame.width) for frame in frames] == [(4, 4)]
    assert len(packed_inputs) == 1
    packed_rgb, packed_alpha = packed_inputs[0]
    assert torch.equal(packed_rgb, rgb[:, :, :, 2:6])
    assert packed_alpha is not None
    assert torch.equal(packed_alpha, alpha[:, :, :, 2:6])


@pytest.mark.parametrize("stream_frames", [True, False])
def test_pipeline_runs_nvidia_vsr_after_crop_and_before_packing(
    monkeypatch: pytest.MonkeyPatch,
    stream_frames: bool,
) -> None:
    rgb = torch.ones((1, 3, 4, 8), dtype=torch.float32)
    alpha = torch.linspace(0, 1, 32, dtype=torch.float32).reshape(1, 1, 4, 8)
    enhanced_calls: list[tuple[torch.Tensor, dict[str, object]]] = []
    packed_inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class MotionGenerator:
        chunk_size = 1
        future_size = 0
        frame_len = 2
        audio_shift = 0

        def initial_state(self, _avatar: object) -> str:
            return "initial"

        def generate_chunk(
            self,
            _chunk: Chunk,
            _avatar: object,
            _state: str,
            _options: RenderOptions,
        ) -> tuple[object, str]:
            return object(), "next"

    class FakeNvidiaVSR:
        def enhance(self, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
            enhanced_calls.append((value.clone(), kwargs))
            return torch.full((value.shape[0], 3, 8, 16), 0.75, dtype=value.dtype)

        def close(self) -> None:
            pass

    def fake_stream(*_args: object, **_kwargs: object):
        yield rgb, alpha

    def fake_batch(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return rgb, alpha

    def fake_pack(
        packed_rgb: torch.Tensor,
        packed_alpha: torch.Tensor | None,
        *,
        pixel_format: str,
    ) -> list[Frame]:
        _ = pixel_format
        packed_inputs.append(
            (
                packed_rgb.clone(),
                packed_alpha.clone() if packed_alpha is not None else None,
            )
        )
        height, width = packed_rgb.shape[-2:]
        return [
            Frame(
                data=np.zeros((height * 3 // 2, width), dtype=np.uint8),
                height=height,
                width=width,
            )
        ]

    monkeypatch.setattr(pipeline_module, "render_chunk_streaming", fake_stream)
    monkeypatch.setattr(pipeline_module, "render_chunk", fake_batch)
    monkeypatch.setattr(pipeline_module, "pack_frames", fake_pack)

    pipeline = Pipeline(
        motion_generator=MotionGenerator(),
        stitch=object(),
        warp=object(),
        decoder=object(),
        matting=object(),
        backgrounds={"bg": torch.zeros(1, 3, 4, 8)},
        nvidia_vsr=FakeNvidiaVSR(),
        out_size=(4, 8),
    )
    chunk = Chunk(
        audio_speech=np.zeros(2, dtype=np.float32),
        audio_listen=np.zeros(2, dtype=np.float32),
    )
    options = RenderOptions(
        bg_id="bg",
        output_height=4,
        output_width=8,
        rtx_super_resolution=True,
        rtx_input_height=2,
        rtx_input_width=4,
        rtx_output_height=8,
        rtx_output_width=16,
        rtx_quality="high_bitrate_medium",
        stream_frames=stream_frames,
    )

    _state, frames = pipeline.process_chunk(object(), chunk, "state", options)

    assert [(frame.height, frame.width) for frame in frames] == [(8, 16)]
    assert len(enhanced_calls) == 1
    enhanced_rgb, kwargs = enhanced_calls[0]
    assert torch.equal(enhanced_rgb, rgb)
    assert kwargs == {
        "input_height": 2,
        "input_width": 4,
        "output_height": 8,
        "output_width": 16,
        "quality": "high_bitrate_medium",
    }
    packed_rgb, packed_alpha = packed_inputs[0]
    assert tuple(packed_rgb.shape) == (1, 3, 8, 16)
    assert torch.all(packed_rgb == 0.75)
    assert packed_alpha is not None
    assert tuple(packed_alpha.shape) == (1, 1, 8, 16)


def _request_process_audio(
    monkeypatch: pytest.MonkeyPatch,
    *,
    height: int,
    width: int,
    extra_params: dict[str, object] | None = None,
    prepare_nvidia_vsr: Callable[..., None] | None = None,
) -> tuple[httpx.Response, RenderOptions | None]:
    captured: dict[str, RenderOptions] = {}
    pipeline = SimpleNamespace(
        _backgrounds={"bg": torch.zeros(1, 3, 720, 1280)},
        _motion_generator=SimpleNamespace(chunk_size=1),
        prepare_nvidia_vsr=(prepare_nvidia_vsr or (lambda **_kwargs: None)),
    )
    monkeypatch.setattr(renderer_api.app.state, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(renderer_api.app.state, "registry", {"avatar": object()}, raising=False)
    monkeypatch.setattr(renderer_api.app.state, "current_samples", 1, raising=False)
    monkeypatch.setattr(renderer_api.app.state, "future_samples", 1, raising=False)

    def fake_run_in_thread(target: object, *args: object, **kwargs: object):
        if target is renderer_api._run_live_preflight:

            async def complete_preflight() -> None:
                target(*args, **kwargs)  # type: ignore[operator]

            return complete_preflight()

        activity = args[0]
        state_future = args[3]
        send = args[4]
        options = args[9]

        def complete_request() -> None:
            assert isinstance(options, RenderOptions)
            captured["options"] = options
            assert isinstance(state_future, asyncio.Future)
            state_future.set_result(b"state")
            send.close()  # type: ignore[union-attr]

        target(activity, complete_request)  # type: ignore[operator]

    monkeypatch.setattr(renderer_api, "run_in_thread", fake_run_in_thread)

    async def request() -> httpx.Response:
        params: dict[str, object] = {
            "avatar_id": "avatar",
            "bg_id": "bg",
            "h": height,
            "w": width,
        }
        if extra_params is not None:
            params.update(extra_params)
        transport = httpx.ASGITransport(app=renderer_api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/process-audio-v3",
                params=params,
                files={
                    "current_chunk": ("current.pcm", b"\0\0"),
                    "future_chunk": ("future.pcm", b"\0\0"),
                    "current_chunk_listen": ("current-listen.pcm", b"\0\0"),
                    "future_chunk_listen": ("future-listen.pcm", b"\0\0"),
                },
            )

    response = asyncio.run(request())
    return response, captured.get("options")


def test_process_audio_uses_requested_output_size_in_options_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, options = _request_process_audio(monkeypatch, height=720, width=720)

    assert response.status_code == 200
    assert options is not None
    assert (options.output_height, options.output_width) == (720, 720)
    assert response.headers["x-frame-height"] == "720"
    assert response.headers["x-frame-width"] == "720"
    assert response.headers["x-frame-length-bytes"] == str(720 * 720 * 3 // 2)


def test_process_audio_passes_rtx_dimensions_and_reports_final_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, options = _request_process_audio(
        monkeypatch,
        height=720,
        width=1280,
        extra_params={
            "rtx_super_resolution": True,
            "rtx_input_h": 540,
            "rtx_input_w": 960,
            "rtx_output_h": 1080,
            "rtx_output_w": 1920,
            "rtx_quality": "high_bitrate_medium",
        },
    )

    assert response.status_code == 200
    assert options is not None
    assert options.rtx_super_resolution is True
    assert (options.rtx_input_height, options.rtx_input_width) == (540, 960)
    assert (options.rtx_output_height, options.rtx_output_width) == (1080, 1920)
    assert options.rtx_quality == "high_bitrate_medium"
    assert response.headers["x-frame-height"] == "1080"
    assert response.headers["x-frame-width"] == "1920"
    assert response.headers["x-frame-length-bytes"] == str(1080 * 1920 * 3 // 2)


def test_process_audio_rejects_failed_rtx_prepare_before_starting_a_200_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_failure = (
        r"NVIDIA RTX Video Super Resolution unavailable: "
        r"model C:\Users\private\secret-model.bin is missing"
    )

    def fail_prepare(**_kwargs: object) -> None:
        raise NvidiaVSRUnavailableError(private_failure)

    response, options = _request_process_audio(
        monkeypatch,
        height=720,
        width=1280,
        extra_params={
            "rtx_super_resolution": True,
            "rtx_input_h": 540,
            "rtx_input_w": 960,
            "rtx_output_h": 1080,
            "rtx_output_w": 1920,
            "rtx_quality": "high_bitrate_medium",
        },
        prepare_nvidia_vsr=fail_prepare,
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "nvidia_vsr_model_load_failed",
        "message": "NVIDIA VSR model files could not be loaded.",
    }
    assert options is None
    assert "private" not in response.text.casefold()
    assert "secret-model.bin" not in response.text.casefold()


def test_process_audio_reports_transient_native_rtx_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prepare(**_kwargs: object) -> None:
        raise NvidiaVSRUnavailableError(
            "NVIDIA RTX Video Super Resolution unavailable: "
            "NvVFX_CreateEffect failed: "
            "The requested feature is not yet implemented (code -2)"
        )

    response, options = _request_process_audio(
        monkeypatch,
        height=720,
        width=1280,
        extra_params={
            "rtx_super_resolution": True,
            "rtx_input_h": 720,
            "rtx_input_w": 1280,
            "rtx_output_h": 1080,
            "rtx_output_w": 1920,
            "rtx_quality": "high_bitrate_high",
        },
        prepare_nvidia_vsr=fail_prepare,
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "nvidia_vsr_temporary_initialization_failed",
        "message": (
            "NVIDIA VSR was temporarily unavailable during native initialization. "
            "Retry the session."
        ),
    }
    assert options is None
    assert "nvvfx_createeffect" not in response.text.casefold()
    assert "code -2" not in response.text.casefold()


def test_nvidia_vsr_prepare_endpoint_loads_the_requested_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    pipeline = SimpleNamespace(
        prepare_nvidia_vsr=lambda **kwargs: calls.append(kwargs),
    )
    activity = renderer_api._RendererActivityGate()
    monkeypatch.setattr(renderer_api.app.state, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(renderer_api.app.state, "registry", {}, raising=False)
    monkeypatch.setattr(
        renderer_api.app.state,
        "renderer_activity",
        activity,
        raising=False,
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=renderer_api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/nvidia-vsr/prepare",
                params={
                    "input_h": 540,
                    "input_w": 960,
                    "output_h": 1080,
                    "output_w": 1920,
                    "quality": "high_bitrate_medium",
                },
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "input_height": 540,
        "input_width": 960,
        "output_height": 1080,
        "output_width": 1920,
        "quality": "high_bitrate_medium",
    }
    assert calls == [
        {
            "input_height": 540,
            "input_width": 960,
            "output_height": 1080,
            "output_width": 1920,
            "quality": "high_bitrate_medium",
        }
    ]
    assert activity.has_active_live is False


def test_process_audio_rejects_incomplete_rtx_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, _options = _request_process_audio(
        monkeypatch,
        height=720,
        width=1280,
        extra_params={
            "rtx_super_resolution": True,
            "rtx_input_h": 540,
            "rtx_input_w": 960,
            "rtx_output_h": 1080,
        },
    )

    assert response.status_code == 422


def test_releasing_renderer_runtime_closes_the_pipeline_before_dropping_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls: list[str] = []
    pipeline = SimpleNamespace(close=lambda: close_calls.append("closed"))
    state = SimpleNamespace(
        pipeline=pipeline,
        registry={},
        current_samples=1,
        future_samples=1,
        health=object(),
        runtime_status="ready",
        runtime_error=None,
        renderer_activity=renderer_api._RendererActivityGate(),
    )
    monkeypatch.setattr(renderer_api, "_release_accelerator_memory", lambda: None)

    renderer_api._release_renderer_runtime(SimpleNamespace(state=state))

    assert close_calls == ["closed"]
    assert state.pipeline is None


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (0, 1280),
        (719, 1280),
        (722, 1280),
        (720, 0),
        (720, 1279),
        (720, 1282),
    ],
)
def test_process_audio_rejects_non_positive_odd_or_oversized_output(
    monkeypatch: pytest.MonkeyPatch,
    height: int,
    width: int,
) -> None:
    response, _options = _request_process_audio(
        monkeypatch,
        height=height,
        width=width,
    )

    assert response.status_code == 422
