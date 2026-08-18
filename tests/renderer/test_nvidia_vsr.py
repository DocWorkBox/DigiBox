from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from avtr1_renderer.nvidia_vsr import (
    NvidiaVSR,
    NvidiaVSRUnavailableError,
    preinitialize_nvidia_vsr,
)


class _FakeEffect:
    def __init__(self) -> None:
        self.output_height: int | None = None
        self.output_width: int | None = None
        self.load_calls = 0
        self.close_calls = 0
        self.input_shapes: list[tuple[int, ...]] = []
        self._output: torch.Tensor | None = None

    def load(self) -> None:
        self.load_calls += 1

    def run(self, frame: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        self.input_shapes.append(tuple(frame.shape))
        assert self.output_height is not None
        assert self.output_width is not None
        shape = (3, self.output_height, self.output_width)
        if self._output is None or tuple(self._output.shape) != shape:
            self._output = torch.empty(shape, dtype=frame.dtype, device=frame.device)
        self._output.fill_(float(frame.mean()))
        return SimpleNamespace(image=self._output)

    def close(self) -> None:
        self.close_calls += 1


def test_nvidia_vsr_adapter_is_available_to_the_renderer() -> None:
    try:
        module = importlib.import_module("avtr1_renderer.nvidia_vsr")
    except ModuleNotFoundError:
        pytest.fail("avtr1_renderer.nvidia_vsr is not implemented", pytrace=False)

    assert hasattr(module, "NvidiaVSR")
    assert hasattr(module, "NvidiaVSRUnavailableError")


def test_nvidia_vsr_lazily_loads_and_reuses_one_effect_for_a_configuration() -> None:
    created: list[tuple[str, _FakeEffect]] = []

    def factory(quality: str) -> _FakeEffect:
        effect = _FakeEffect()
        created.append((quality, effect))
        return effect

    adapter = NvidiaVSR(effect_factory=factory)
    assert created == []

    frame = torch.ones((1, 3, 4, 8), dtype=torch.float32)
    first = adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )
    second = adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )

    assert tuple(first.shape) == (1, 3, 4, 8)
    assert tuple(second.shape) == (1, 3, 4, 8)
    assert len(created) == 1
    quality, effect = created[0]
    assert quality == "high_bitrate_low"
    assert effect.load_calls == 1
    assert effect.input_shapes == [(3, 2, 4), (3, 2, 4)]


def test_nvidia_vsr_prepare_really_loads_and_enhance_reuses_the_effect() -> None:
    created: list[tuple[str, _FakeEffect]] = []

    def factory(quality: str) -> _FakeEffect:
        effect = _FakeEffect()
        created.append((quality, effect))
        return effect

    adapter = NvidiaVSR(effect_factory=factory)

    adapter.prepare(
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )

    assert len(created) == 1
    quality, effect = created[0]
    assert quality == "high_bitrate_low"
    assert effect.load_calls == 1
    assert effect.input_shapes == []

    enhanced = adapter.enhance(
        torch.ones((1, 3, 2, 4), dtype=torch.float32),
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )

    assert tuple(enhanced.shape) == (1, 3, 4, 8)
    assert len(created) == 1
    assert effect.load_calls == 1
    assert effect.input_shapes == [(3, 2, 4)]


def test_nvidia_vsr_clones_each_dlpack_result_before_the_next_run() -> None:
    effect = _FakeEffect()
    adapter = NvidiaVSR(effect_factory=lambda _quality: effect)
    frames = torch.stack(
        [
            torch.ones((3, 2, 4), dtype=torch.float32),
            torch.full((3, 2, 4), 2.0, dtype=torch.float32),
        ]
    )

    enhanced = adapter.enhance(
        frames,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )

    assert torch.all(enhanced[0] == 1.0)
    assert torch.all(enhanced[1] == 2.0)


def test_nvidia_vsr_makes_a_cropped_tensor_contiguous_for_dlpack() -> None:
    contiguous_inputs: list[bool] = []

    class ContiguityEffect(_FakeEffect):
        def run(self, frame: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            contiguous_inputs.append(frame.is_contiguous())
            return super().run(frame, **kwargs)

    effect = ContiguityEffect()
    adapter = NvidiaVSR(effect_factory=lambda _quality: effect)
    cropped = torch.ones((1, 3, 2, 6), dtype=torch.float32)[:, :, :, 1:5]
    assert not cropped.is_contiguous()

    adapter.enhance(
        cropped,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )

    assert contiguous_inputs == [True]


def test_nvidia_vsr_default_factory_uses_the_official_nvvfx_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_level = object()
    created: list[tuple[object, int, _FakeEffect]] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = SimpleNamespace(HIGHBITRATE_HIGH=quality_level)

        def __init__(self, *, quality: object, device: int = 0) -> None:
            super().__init__()
            created.append((quality, device, self))

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )
    adapter = NvidiaVSR()

    enhanced = adapter.enhance(
        torch.ones((1, 3, 2, 4), dtype=torch.float32),
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_high",
    )

    assert tuple(enhanced.shape) == (1, 3, 4, 8)
    assert len(created) == 1
    assert created[0][:2] == (quality_level, 0)


def test_official_nvvfx_retries_transient_create_effect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_level = object()
    attempts = 0
    created: list[_FakeEffect] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = SimpleNamespace(HIGHBITRATE_HIGH=quality_level)

        def __init__(self, *, quality: object, device: int = 0) -> None:
            nonlocal attempts
            attempts += 1
            assert quality is quality_level
            assert device == 0
            if attempts == 1:
                raise RuntimeError(
                    "NvVFX_CreateEffect failed: "
                    "The requested feature is not yet implemented (code -2)"
                )
            super().__init__()
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )
    monkeypatch.setattr("avtr1_renderer.nvidia_vsr.time.sleep", lambda _seconds: None)

    NvidiaVSR().prepare(
        output_height=1080,
        output_width=1920,
        quality="high_bitrate_high",
    )

    assert attempts == 2
    assert len(created) == 1
    assert created[0].load_calls == 1


def test_official_nvvfx_preinitializes_one_unloaded_effect_and_prepare_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_levels = SimpleNamespace(
        HIGHBITRATE_HIGH=object(),
        HIGHBITRATE_MEDIUM=object(),
    )
    created: list[_FakeEffect] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = quality_levels

        def __init__(self, *, quality: object, device: int = 0) -> None:
            super().__init__()
            self.quality = quality
            assert device == 0
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )

    assert preinitialize_nvidia_vsr() is True
    assert preinitialize_nvidia_vsr() is True
    assert len(created) == 1
    assert created[0].load_calls == 0

    NvidiaVSR().prepare(
        output_height=1080,
        output_width=1920,
        quality="high_bitrate_medium",
    )

    assert len(created) == 1
    assert created[0].quality is quality_levels.HIGHBITRATE_MEDIUM
    assert created[0].output_height == 1080
    assert created[0].output_width == 1920
    assert created[0].load_calls == 1


def test_official_nvvfx_preinitialize_failure_is_best_effort_and_not_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_level = object()
    attempts = 0
    created: list[_FakeEffect] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = SimpleNamespace(HIGHBITRATE_HIGH=quality_level)

        def __init__(self, *, quality: object, device: int = 0) -> None:
            nonlocal attempts
            attempts += 1
            assert quality is quality_level
            assert device == 0
            if attempts <= 2:
                raise RuntimeError(
                    "NvVFX_CreateEffect failed: "
                    "The requested feature is not yet implemented (code -2)"
                )
            super().__init__()
            self.quality = quality
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )
    monkeypatch.setattr("avtr1_renderer.nvidia_vsr.time.sleep", lambda _seconds: None)

    assert preinitialize_nvidia_vsr() is False
    NvidiaVSR().prepare(
        output_height=1080,
        output_width=1920,
        quality="high_bitrate_high",
    )

    assert attempts == 3
    assert len(created) == 1
    assert created[0].load_calls == 1


def test_transient_create_effect_failure_does_not_poison_later_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_level = object()
    attempts = 0
    created: list[_FakeEffect] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = SimpleNamespace(HIGHBITRATE_HIGH=quality_level)

        def __init__(self, *, quality: object, device: int = 0) -> None:
            nonlocal attempts
            attempts += 1
            assert quality is quality_level
            assert device == 0
            if attempts <= 2:
                raise RuntimeError(
                    "NvVFX_CreateEffect failed: "
                    "The requested feature is not yet implemented (code -2)"
                )
            super().__init__()
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )
    monkeypatch.setattr("avtr1_renderer.nvidia_vsr.time.sleep", lambda _seconds: None)
    adapter = NvidiaVSR()
    kwargs = {
        "output_height": 1080,
        "output_width": 1920,
        "quality": "high_bitrate_high",
    }

    with pytest.raises(NvidiaVSRUnavailableError, match="code -2"):
        adapter.prepare(**kwargs)
    adapter.prepare(**kwargs)

    assert attempts == 3
    assert len(created) == 1
    assert created[0].load_calls == 1


def test_official_nvvfx_effect_is_reused_across_adapters_without_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_level = object()
    created: list[_FakeEffect] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = SimpleNamespace(HIGHBITRATE_LOW=quality_level)

        def __init__(self, *, quality: object, device: int = 0) -> None:
            super().__init__()
            assert quality is quality_level
            assert device == 0
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )
    kwargs = {
        "input_height": 2,
        "input_width": 4,
        "output_height": 4,
        "output_width": 8,
        "quality": "high_bitrate_low",
    }

    first_adapter = NvidiaVSR()
    first_adapter.enhance(torch.ones((1, 3, 2, 4)), **kwargs)
    first_adapter.close()

    second_adapter = NvidiaVSR()
    second_adapter.enhance(torch.ones((1, 3, 2, 4)), **kwargs)
    second_adapter.close()

    assert len(created) == 1
    assert created[0].load_calls == 1
    assert created[0].close_calls == 0


def test_official_nvvfx_reconfigures_one_loaded_effect_between_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_levels = SimpleNamespace(
        HIGHBITRATE_LOW=object(),
        HIGHBITRATE_MEDIUM=object(),
    )
    created: list[_FakeEffect] = []

    class FakeVideoSuperRes(_FakeEffect):
        QualityLevel = quality_levels

        def __init__(self, *, quality: object, device: int = 0) -> None:
            super().__init__()
            assert quality in {
                quality_levels.HIGHBITRATE_LOW,
                quality_levels.HIGHBITRATE_MEDIUM,
            }
            assert device == 0
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "nvvfx",
        SimpleNamespace(VideoSuperRes=FakeVideoSuperRes),
    )
    first_adapter = NvidiaVSR()
    second_adapter = NvidiaVSR()
    frame = torch.ones((1, 3, 2, 4))

    first_adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )
    second_adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=6,
        output_width=12,
        quality="high_bitrate_medium",
    )
    first_adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )
    first_adapter.close()
    second_adapter.close()

    assert len(created) == 1
    assert created[0].load_calls == 1
    assert created[0].close_calls == 0
    assert created[0].output_height == 4
    assert created[0].output_width == 8


def test_nvidia_vsr_latches_a_clear_unavailable_error_after_load_failure() -> None:
    factory_calls = 0

    class BrokenEffect(_FakeEffect):
        def load(self) -> None:
            raise RuntimeError("VSR model files are missing")

    def factory(_quality: str) -> BrokenEffect:
        nonlocal factory_calls
        factory_calls += 1
        return BrokenEffect()

    adapter = NvidiaVSR(effect_factory=factory)
    kwargs = {
        "input_height": 2,
        "input_width": 4,
        "output_height": 4,
        "output_width": 8,
        "quality": "high_bitrate_low",
    }

    with pytest.raises(NvidiaVSRUnavailableError, match="VSR model files are missing"):
        adapter.enhance(torch.ones((1, 3, 2, 4)), **kwargs)
    with pytest.raises(NvidiaVSRUnavailableError, match="VSR model files are missing"):
        adapter.enhance(torch.ones((1, 3, 2, 4)), **kwargs)

    assert factory_calls == 1


def test_nvidia_vsr_does_not_retry_unrelated_code_minus_two_failure() -> None:
    factory_calls = 0

    class BrokenEffect(_FakeEffect):
        def load(self) -> None:
            raise RuntimeError("another native operation failed with code -2")

    def factory(_quality: str) -> BrokenEffect:
        nonlocal factory_calls
        factory_calls += 1
        return BrokenEffect()

    adapter = NvidiaVSR(effect_factory=factory)
    kwargs = {
        "output_height": 4,
        "output_width": 8,
        "quality": "high_bitrate_low",
    }

    with pytest.raises(NvidiaVSRUnavailableError, match="code -2"):
        adapter.prepare(**kwargs)
    with pytest.raises(NvidiaVSRUnavailableError, match="code -2"):
        adapter.prepare(**kwargs)

    assert factory_calls == 1


def test_nvidia_vsr_releases_the_effect_when_inference_fails() -> None:
    class BrokenRunEffect(_FakeEffect):
        def run(self, frame: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            _ = frame
            raise RuntimeError("VSR inference failed")

    effect = BrokenRunEffect()
    adapter = NvidiaVSR(effect_factory=lambda _quality: effect)

    with pytest.raises(NvidiaVSRUnavailableError, match="VSR inference failed"):
        adapter.enhance(
            torch.ones((1, 3, 2, 4)),
            input_height=2,
            input_width=4,
            output_height=4,
            output_width=8,
            quality="high_bitrate_low",
        )

    assert effect.close_calls == 1


def test_nvidia_vsr_closes_the_previous_effect_on_reconfigure_and_shutdown() -> None:
    created: list[_FakeEffect] = []

    def factory(_quality: str) -> _FakeEffect:
        effect = _FakeEffect()
        created.append(effect)
        return effect

    adapter = NvidiaVSR(effect_factory=factory)
    frame = torch.ones((1, 3, 2, 4), dtype=torch.float32)

    adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=4,
        output_width=8,
        quality="high_bitrate_low",
    )
    adapter.enhance(
        frame,
        input_height=2,
        input_width=4,
        output_height=6,
        output_width=12,
        quality="high_bitrate_medium",
    )

    assert len(created) == 2
    assert created[0].close_calls == 1
    assert created[1].close_calls == 0

    adapter.close()
    assert created[1].close_calls == 1

    adapter.close()
    assert created[1].close_calls == 1
