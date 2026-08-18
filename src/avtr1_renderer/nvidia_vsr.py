"""Optional NVIDIA Video Super Resolution adapter for renderer CUDA frames."""

from __future__ import annotations

import ctypes
import importlib
import logging
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as torch_functional

LOG = logging.getLogger(__name__)

_QUALITY_ENUM_NAMES = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "ultra": "ULTRA",
    "high_bitrate_low": "HIGHBITRATE_LOW",
    "high_bitrate_medium": "HIGHBITRATE_MEDIUM",
    "high_bitrate_high": "HIGHBITRATE_HIGH",
    "high_bitrate_ultra": "HIGHBITRATE_ULTRA",
}
_TRANSIENT_CREATE_RETRY_DELAY_SECONDS = 0.25
_PREINITIALIZE_QUALITY = "high_bitrate_high"
_OFFICIAL_DEVICE = 0


@dataclass(slots=True)
class _OfficialEffectEntry:
    effect: Any
    run_lock: threading.Lock
    loaded: bool = False
    configuration: tuple[str, int, int] | None = None


class _TransientOfficialEffectCreationError(RuntimeError):
    """A native effect-creation failure that may recover on a later attempt."""


_OFFICIAL_EFFECT_CACHE: dict[tuple[type[Any], int], _OfficialEffectEntry] = {}
_OFFICIAL_EFFECT_CACHE_LOCK = threading.Lock()


def _pin_for_process_lifetime(effect: Any) -> None:
    """Keep an official binding alive until the OS reclaims the process.

    ``nvidia-vfx==0.1.0.1`` on Windows can block indefinitely in both
    ``VideoSuperRes.close()`` and the C++ destructor.  The shared cache avoids
    per-session loads; the deliberate extra CPython reference prevents module
    teardown from entering that destructor.  Process termination still
    reclaims the CUDA/native resources.
    """

    python_incref = ctypes.pythonapi.Py_IncRef
    python_incref.argtypes = (ctypes.py_object,)
    python_incref.restype = None
    python_incref(effect)


def _create_official_effect(quality: str, video_super_res: type[Any]) -> Any:
    try:
        enum_name = _QUALITY_ENUM_NAMES[quality]
    except KeyError as exc:
        raise ValueError(f"Unsupported NVIDIA VSR quality: {quality}") from exc
    return video_super_res(
        quality=getattr(video_super_res.QualityLevel, enum_name),
        device=_OFFICIAL_DEVICE,
    )


def _is_transient_official_effect_creation_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "nvvfx_createeffect" in message and (
        "code -2" in message or "nvcv_err_unimplemented" in message
    )


def _get_official_effect(quality: str) -> _OfficialEffectEntry:
    from nvvfx import VideoSuperRes

    key = (VideoSuperRes, _OFFICIAL_DEVICE)
    with _OFFICIAL_EFFECT_CACHE_LOCK:
        cached = _OFFICIAL_EFFECT_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            effect = _create_official_effect(quality, VideoSuperRes)
        except Exception as first_exc:
            if not _is_transient_official_effect_creation_error(first_exc):
                raise
            LOG.warning(
                "NVIDIA VSR native effect creation failed transiently; retrying once: %s",
                first_exc,
            )
            time.sleep(_TRANSIENT_CREATE_RETRY_DELAY_SECONDS)
            try:
                effect = _create_official_effect(quality, VideoSuperRes)
            except Exception as second_exc:
                if _is_transient_official_effect_creation_error(second_exc):
                    raise _TransientOfficialEffectCreationError(str(second_exc)) from second_exc
                raise
        # Pin before any model load: even a partially initialised 0.1.0.1 wrapper can
        # enter the blocking native destructor when an exception unwinds.
        _pin_for_process_lifetime(effect)
        entry = _OfficialEffectEntry(effect=effect, run_lock=threading.Lock())
        _OFFICIAL_EFFECT_CACHE[key] = entry
        return entry


def _configure_official_effect(
    entry: _OfficialEffectEntry,
    *,
    quality: str,
    output_height: int,
    output_width: int,
) -> None:
    """Configure a shared official effect while its ``run_lock`` is held."""

    configuration = (quality, output_height, output_width)
    if entry.configuration == configuration:
        return

    from nvvfx import VideoSuperRes

    try:
        enum_name = _QUALITY_ENUM_NAMES[quality]
    except KeyError as exc:
        raise ValueError(f"Unsupported NVIDIA VSR quality: {quality}") from exc
    entry.effect.quality = getattr(VideoSuperRes.QualityLevel, enum_name)
    entry.effect.output_height = output_height
    entry.effect.output_width = output_width
    if not entry.loaded:
        entry.effect.load()
        entry.loaded = True
    entry.configuration = configuration


def preinitialize_nvidia_vsr() -> bool:
    """Best-effort native-handle creation before renderer engine loading.

    The Full Runtime contains the renderer TensorRT binding as well as native
    libraries bundled by ``nvidia-vfx``. Loading the main binding first and
    creating one process-wide VSR handle before deserialising renderer engines
    avoids a later native effect-creation ordering failure. The expensive model
    ``load()`` remains lazy until a session enables RTX VSR.
    """

    try:
        importlib.import_module("tensorrt")
    except Exception as exc:
        LOG.info(
            "Main TensorRT binding is unavailable before NVIDIA VSR preinitialization: %s",
            exc,
        )
    try:
        _get_official_effect(_PREINITIALIZE_QUALITY)
    except Exception as exc:
        LOG.warning("NVIDIA VSR native handle preinitialization was skipped: %s", exc)
        return False
    LOG.info("NVIDIA VSR native handle preinitialized; model load remains lazy")
    return True


class NvidiaVSRUnavailableError(RuntimeError):
    """Raised when the requested NVIDIA VSR backend cannot be used."""


class NvidiaVSR:
    """Lazy holder for NVIDIA's optional VideoSuperRes effect."""

    def __init__(self, *, effect_factory: Callable[[str], Any] | None = None) -> None:
        self._effect_factory = effect_factory
        self._owns_effect = effect_factory is not None
        self._effect: Any | None = None
        self._effect_run_lock: threading.Lock | None = None
        self._configuration: tuple[str, int, int] | None = None
        self._unavailable_reason: str | None = None

    def _close_effect(self) -> None:
        effect = self._effect
        self._effect = None
        self._effect_run_lock = None
        self._configuration = None
        if effect is not None and self._owns_effect:
            close = getattr(effect, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        """Release this adapter's reference; safe to call more than once.

        Injected effects are closed normally. Official effects stay in the
        process cache because the Windows 0.1.0.1 native teardown path blocks.
        """

        self._close_effect()

    def _ensure_injected_effect(
        self,
        *,
        output_height: int,
        output_width: int,
        quality: str,
    ) -> None:
        configuration = (quality, output_height, output_width)
        if self._effect is not None and self._configuration == configuration:
            return

        assert self._effect_factory is not None
        self._close_effect()
        effect = self._effect_factory(quality)
        effect.output_height = output_height
        effect.output_width = output_width
        try:
            effect.load()
        except Exception:
            close = getattr(effect, "close", None)
            if callable(close):
                close()
            raise
        self._effect = effect
        self._configuration = configuration

    def _bind_official_effect(
        self,
        entry: _OfficialEffectEntry,
        configuration: tuple[str, int, int],
    ) -> None:
        self._effect = entry.effect
        self._effect_run_lock = entry.run_lock
        self._configuration = configuration

    def _latch_unavailable(self, exc: Exception) -> NvidiaVSRUnavailableError:
        with suppress(Exception):
            self._close_effect()
        reason = f"NVIDIA RTX Video Super Resolution unavailable: {exc}"
        if not isinstance(exc, _TransientOfficialEffectCreationError):
            self._unavailable_reason = reason
        return NvidiaVSRUnavailableError(reason)

    def prepare(
        self,
        *,
        output_height: int,
        output_width: int,
        quality: str,
    ) -> None:
        """Load and configure the requested effect without consuming a frame."""

        if self._unavailable_reason is not None:
            raise NvidiaVSRUnavailableError(self._unavailable_reason)
        try:
            configuration = (quality, output_height, output_width)
            if self._effect_factory is None:
                entry = _get_official_effect(quality)
                with entry.run_lock:
                    _configure_official_effect(
                        entry,
                        output_height=output_height,
                        output_width=output_width,
                        quality=quality,
                    )
                self._bind_official_effect(entry, configuration)
            else:
                self._ensure_injected_effect(
                    output_height=output_height,
                    output_width=output_width,
                    quality=quality,
                )
        except NvidiaVSRUnavailableError:
            raise
        except Exception as exc:
            raise self._latch_unavailable(exc) from exc

    @torch.no_grad()
    def enhance(
        self,
        rgb: torch.Tensor,
        *,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        quality: str,
    ) -> torch.Tensor:
        """Downscale to the selected tier and run VSR before host packing."""

        if self._unavailable_reason is not None:
            raise NvidiaVSRUnavailableError(self._unavailable_reason)
        if tuple(rgb.shape[-2:]) != (input_height, input_width):
            rgb = torch_functional.interpolate(
                rgb,
                size=(input_height, input_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        rgb = rgb.contiguous()

        try:
            enhanced: list[torch.Tensor] = []
            if self._effect_factory is None:
                entry = _get_official_effect(quality)
                run_guard = entry.run_lock
            else:
                self._ensure_injected_effect(
                    output_height=output_height,
                    output_width=output_width,
                    quality=quality,
                )
                entry = None
                run_guard = nullcontext()
            with run_guard:
                if entry is not None:
                    _configure_official_effect(
                        entry,
                        output_height=output_height,
                        output_width=output_width,
                        quality=quality,
                    )
                    self._bind_official_effect(
                        entry,
                        (quality, output_height, output_width),
                    )
                for frame in rgb:
                    assert self._effect is not None
                    result = self._effect.run(frame)
                    # NVIDIA owns the DLPack buffer and may reuse it on the next run.
                    # Clone immediately, as required by the official Python binding.
                    enhanced.append(torch.from_dlpack(result.image).clone())
            return torch.stack(enhanced, dim=0)
        except NvidiaVSRUnavailableError:
            raise
        except Exception as exc:
            raise self._latch_unavailable(exc) from exc


__all__ = [
    "NvidiaVSR",
    "NvidiaVSRUnavailableError",
    "preinitialize_nvidia_vsr",
]
