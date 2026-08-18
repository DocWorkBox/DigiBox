# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""FastAPI HTTP front-end for ``Pipeline.process_chunk``.

One streaming endpoint plus health. Avatars are loaded at startup by
scanning the ``reference_frames`` artifact directory.

The response body is the new safetensors state blob first, followed by
rendered frames concatenated. ``X-State-Length-Bytes`` gives the state-blob
length so the client can split the body without buffering the whole response.

Audio format: raw int16 PCM at 16 kHz mono (same convention as the old
renderer). The server converts to float32 [-1, 1] internally before
constructing the ``Chunk``.

Run locally::

    AVTR1_LOCAL_STORAGE=/var/lib/avtr1/artifacts \\
    pixi run python -m avtr1_renderer.api.app
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

import anyio
import cv2
import numpy as np
import uvicorn
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from avtr1_renderer.api.load_balancing import keep_alive_worker
from avtr1_renderer.avatar_loader import encode_avatar_preview_png
from avtr1_renderer.avtr1_motion_generator import state_from_safetensors, state_to_safetensors
from avtr1_renderer.components.pixel_format import PixelFormat, get_bytes_per_frame
from avtr1_renderer.idle_loop import generate_avatar_idle_loop
from avtr1_renderer.nvidia_vsr import NvidiaVSRUnavailableError
from avtr1_renderer.pipeline import TRANSPARENT_BG_ID, Pipeline
from avtr1_renderer.types import Chunk, NvidiaVSRQuality, RenderOptions
from avtr1_renderer.utils.asyncio import run_in_thread
from avtr1_renderer.utils.cuda_health import CudaHealthChecker

LOG = logging.getLogger(__name__)

_INT16_MAX = 32768.0
_MAX_USER_ASSET_BYTES = 12 * 1024 * 1024
_USER_IMAGE_HEIGHT = 720
_USER_IMAGE_WIDTH = 1280
_USER_ASSET_HASH_LENGTH = 12
_USER_AVATAR_ID_RE = re.compile(r"^user_[a-z0-9_]{1,40}_[0-9a-f]{12}$")
PRESET_BACKGROUND_ID = "tech_particles_dark"
_IDLE_LOOP_RECIPE = "v3"
_USER_ASSET_MIGRATION_MARKER = ".avtr1-user-assets-migrated-v1"
_THEME_BACKGROUND_FILENAMES = {
    "theme_aurora": "theme_aurora.png",
    "theme_winter_hearth": "theme_winter_hearth.png",
    "theme_romantic": "theme_romantic.png",
    "theme_cozy_cabin": "theme_cozy_cabin.png",
    "theme_pearl": "theme_pearl.png",
    "theme_cyberspace": "theme_cyberspace.png",
    "theme_rainforest": "theme_rainforest.png",
}

UserAssetKind = Literal["avatar", "background"]


@dataclass(frozen=True, slots=True)
class _NormalisedUserAsset:
    kind: UserAssetKind
    asset_id: str
    png_bytes: bytes


@dataclass(frozen=True, slots=True)
class _LoadedRendererRuntime:
    """All GPU-backed objects that are installed and released as one unit."""

    pipeline: Pipeline
    registry: dict[str, object]
    current_samples: int
    future_samples: int
    health: CudaHealthChecker


class _RendererBusyError(RuntimeError):
    """Raised when offline asset inference would contend with live rendering."""


class _RendererActivityGate:
    """Serialize shared inference contexts while giving live chunks priority.

    Live requests register before their worker thread is submitted. Idle-loop
    generation acquires the inference lock one 200 ms model chunk at a time;
    once a live request is registered, the next idle chunk fails fast instead
    of making the realtime request wait for the full animation.
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._active_live = 0

    @property
    def has_active_live(self) -> bool:
        with self._state_lock:
            return self._active_live > 0

    def begin_live(self) -> None:
        with self._state_lock:
            self._active_live += 1

    def end_live(self) -> None:
        with self._state_lock:
            if self._active_live <= 0:
                raise RuntimeError("live renderer activity counter underflow")
            self._active_live -= 1

    @contextmanager
    def live_inference(self):
        with self._inference_lock:
            yield

    @contextmanager
    def idle_inference(self):
        if self.has_active_live:
            raise _RendererBusyError("live rendering is active")
        if not self._inference_lock.acquire(blocking=False):
            raise _RendererBusyError("renderer inference is busy")
        try:
            # A live request can register between the first check and lock
            # acquisition. It has priority, so do not start another idle chunk.
            if self.has_active_live:
                raise _RendererBusyError("live rendering is active")
            yield
        finally:
            self._inference_lock.release()

    @contextmanager
    def release_inference(self):
        """Reserve the renderer exclusively for dropping its model runtime."""

        if self.has_active_live:
            raise _RendererBusyError("live rendering is active")
        if not self._inference_lock.acquire(blocking=False):
            raise _RendererBusyError("renderer inference is busy")
        try:
            if self.has_active_live:
                raise _RendererBusyError("live rendering is active")
            yield
        finally:
            self._inference_lock.release()


def _renderer_activity(app: FastAPI) -> _RendererActivityGate:
    activity: _RendererActivityGate | None = getattr(app.state, "renderer_activity", None)
    if activity is None:
        activity = _RendererActivityGate()
        app.state.renderer_activity = activity
    return activity


def _renderer_busy_http_exception() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="renderer is busy with a live conversation; retry the idle loop later",
        headers={"Retry-After": "1", "Cache-Control": "no-store"},
    )


def _run_reserved_live(
    activity: _RendererActivityGate,
    operation: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run one pre-registered live operation and always release its reservation."""

    try:
        with activity.live_inference():
            return operation(*args, **kwargs)
    finally:
        activity.end_live()


def _run_live_preflight(
    activity: _RendererActivityGate,
    operation: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run setup under the renderer inference lock without ending its reservation."""

    with activity.live_inference():
        return operation(*args, **kwargs)


def _public_nvidia_vsr_failure(exc: NvidiaVSRUnavailableError) -> dict[str, str]:
    """Classify a native load error without exposing paths or driver internals."""

    message = str(exc).casefold()
    if "nvvfx_createeffect" in message and (
        "code -2" in message or "nvcv_err_unimplemented" in message
    ):
        return {
            "code": "nvidia_vsr_temporary_initialization_failed",
            "message": (
                "NVIDIA VSR was temporarily unavailable during native initialization. "
                "Retry the session."
            ),
        }
    if "nvvfx" in message and (
        "no module" in message or "not installed" in message or "import" in message
    ):
        return {
            "code": "nvidia_vsr_runtime_missing",
            "message": "NVIDIA VFX Python runtime (nvvfx) is not installed.",
        }
    if "model" in message or "weight" in message or "file" in message:
        return {
            "code": "nvidia_vsr_model_load_failed",
            "message": "NVIDIA VSR model files could not be loaded.",
        }
    if "out of memory" in message or "cuda" in message or "device" in message:
        return {
            "code": "nvidia_vsr_cuda_initialization_failed",
            "message": "NVIDIA VSR could not initialize the CUDA device.",
        }
    if "driver" in message:
        return {
            "code": "nvidia_vsr_driver_incompatible",
            "message": "The NVIDIA driver is incompatible with NVIDIA VSR.",
        }
    return {
        "code": "nvidia_vsr_initialization_failed",
        "message": "NVIDIA VSR could not initialize the requested quality profile.",
    }


def _nvidia_vsr_http_exception(
    exc: NvidiaVSRUnavailableError,
) -> HTTPException:
    LOG.error("NVIDIA VSR preflight failed: %s", exc, exc_info=True)
    return HTTPException(
        status_code=503,
        detail=_public_nvidia_vsr_failure(exc),
        headers={"Cache-Control": "no-store"},
    )


def _bad_upload(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _safe_asset_stem(filename: str, kind: UserAssetKind) -> str:
    # Treat both slash styles as separators regardless of the host platform,
    # then keep a short portable ASCII stem. A non-Latin-only name falls back
    # to the asset kind; the content hash still makes the id unique.
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    raw_stem = Path(basename).stem
    ascii_stem = unicodedata.normalize("NFKD", raw_stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^a-z0-9]+", "_", ascii_stem.casefold()).strip("_")
    return (stem or kind)[:40]


def _contain_image(image: np.ndarray, *, channels: int) -> np.ndarray:
    """Fit an image inside 1280x720 without distortion and centre-pad it."""
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise _bad_upload("图片尺寸无效")

    scale = min(_USER_IMAGE_WIDTH / src_w, _USER_IMAGE_HEIGHT / src_h)
    dst_w = max(1, min(_USER_IMAGE_WIDTH, round(src_w * scale)))
    dst_h = max(1, min(_USER_IMAGE_HEIGHT, round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    if channels == 4 and image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        premultiplied = image[:, :, :3].astype(np.float32) * alpha[:, :, None]
        resized_alpha = cv2.resize(
            alpha,
            (dst_w, dst_h),
            interpolation=interpolation,
        )
        resized_premultiplied = cv2.resize(
            premultiplied,
            (dst_w, dst_h),
            interpolation=interpolation,
        )
        safe_alpha = np.maximum(resized_alpha[:, :, None], 1.0 / 255.0)
        resized_rgb = np.where(
            resized_alpha[:, :, None] > 0.0,
            resized_premultiplied / safe_alpha,
            0.0,
        )
        resized = np.concatenate(
            [
                np.rint(np.clip(resized_rgb, 0.0, 255.0)).astype(np.uint8),
                np.rint(np.clip(resized_alpha * 255.0, 0.0, 255.0))
                .astype(np.uint8)[:, :, None],
            ],
            axis=2,
        )
    else:
        resized = cv2.resize(image, (dst_w, dst_h), interpolation=interpolation)

    canvas = np.zeros(
        (_USER_IMAGE_HEIGHT, _USER_IMAGE_WIDTH, channels),
        dtype=np.uint8,
    )
    top = (_USER_IMAGE_HEIGHT - dst_h) // 2
    left = (_USER_IMAGE_WIDTH - dst_w) // 2
    canvas[top : top + dst_h, left : left + dst_w] = resized
    return canvas


def _resolve_preset_background_path(background_id: str = PRESET_BACKGROUND_ID) -> Path:
    """Locate the single project-bundled background used for person uploads."""

    if background_id != PRESET_BACKGROUND_ID:
        raise ValueError(f"unknown preset background: {background_id}")
    return Path(__file__).resolve().parents[1] / "assets" / f"{background_id}.png"


def _bundled_background_paths() -> dict[str, Path]:
    """Return every background that the browser theme registry can request."""

    assets = Path(__file__).resolve().parents[1] / "assets"
    return {
        PRESET_BACKGROUND_ID: assets / f"{PRESET_BACKGROUND_ID}.png",
        **{
            background_id: assets / filename
            for background_id, filename in _THEME_BACKGROUND_FILENAMES.items()
        },
    }


def _feather_cutout_alpha(image: np.ndarray) -> np.ndarray:
    """Soften only the cutout boundary enough to avoid a pasted-on edge."""

    if image.ndim != 3 or image.shape[2] != 4:
        raise _bad_upload("FeyNoBg 必须返回 RGBA 图片")
    feathered = image.copy()
    alpha = feathered[:, :, 3]
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.25, sigmaY=1.25)
    # Feather inward only. Expanding alpha outside FeyNoBg's support exposes
    # the original background RGB and creates the dark/grey fringe this step
    # is meant to remove.
    feathered[:, :, 3] = np.minimum(alpha, blurred)
    return feathered


def _refine_cutout_edges(image: np.ndarray) -> np.ndarray:
    """Remove light-background colour spill and lightly choke the matte.

    FeyNoBg returns a useful alpha matte, but its straight RGB still comes from
    the uploaded photograph.  Hair pixels antialiased against a white source
    background therefore carry white RGB even after that background becomes
    transparent.  Detect the portion of an edge colour that lies on the vector
    from its nearest trusted interior colour toward the old background, then
    blend only that high-confidence spill back toward the interior colour.
    Unlike direct alpha unmatting, this cannot turn a clean dark strand black
    merely because the segmentation alpha is uncertain.  The operation is
    deliberately gated to a light, nearly neutral background estimate so a
    colourful or non-uniform photograph is not globally recoloured.
    """

    if image.ndim != 3 or image.shape[2] != 4:
        raise _bad_upload("FeyNoBg must return an RGBA image")

    refined = _feather_cutout_alpha(image)
    raw_alpha_u8 = image[:, :, 3]
    raw_alpha = raw_alpha_u8.astype(np.float32) / 255.0
    raw_rgb = image[:, :, :3].astype(np.float32)
    non_black_rgb = np.max(raw_rgb, axis=2) > 16.0
    background_candidates = (raw_alpha_u8 <= 32) & non_black_rgb

    if np.count_nonzero(background_candidates) >= 64:
        samples = raw_rgb[background_candidates]
        background = np.median(samples, axis=0).astype(np.float32)
        median_deviation = float(np.median(np.abs(samples - background)))
        is_light_neutral = (
            float(np.min(background)) >= 180.0
            and float(np.max(background) - np.min(background)) <= 50.0
            and median_deviation <= 30.0
        )
        if is_light_neutral:
            support = (raw_alpha > 0.03).astype(np.uint8)
            distance_inside = cv2.distanceTransform(support, cv2.DIST_L2, 5)
            interior = (raw_alpha >= (240.0 / 255.0)) & (distance_inside > 5.0)
            if np.any(interior):
                distance_source = (~interior).astype(np.uint8)
                distance_to_interior, labels = cv2.distanceTransformWithLabels(
                    distance_source,
                    cv2.DIST_L2,
                    5,
                    labelType=cv2.DIST_LABEL_PIXEL,
                )
                interior_labels = labels[interior]
                colour_lut = np.zeros(
                    (int(labels.max()) + 1, 3),
                    dtype=np.float32,
                )
                colour_lut[interior_labels] = raw_rgb[interior]
                nearest_interior = colour_lut[labels]
                background_vector = background[None, None, :] - nearest_interior
                projection_denominator = np.sum(
                    background_vector * background_vector,
                    axis=2,
                )
                projection = np.sum(
                    (raw_rgb - nearest_interior) * background_vector,
                    axis=2,
                ) / np.maximum(projection_denominator, 1.0)
                projection = np.clip(projection, 0.0, 1.0)
                smooth = np.clip((projection - 0.15) / 0.65, 0.0, 1.0)
                spill_weight = smooth * smooth * (3.0 - 2.0 * smooth)
                eligible = (
                    (raw_alpha > 0.0)
                    & (raw_alpha < 0.985)
                    & (distance_to_interior <= 24.0)
                )
                spill_weight *= eligible.astype(np.float32)
                clean_rgb = (
                    raw_rgb * (1.0 - spill_weight[:, :, None])
                    + nearest_interior * spill_weight[:, :, None]
                )
                refined[:, :, :3] = np.rint(
                    np.clip(clean_rgb, 0.0, 255.0)
                ).astype(np.uint8)

    alpha = refined[:, :, 3].astype(np.float32)
    choked = np.clip(
        (alpha - 8.0) * (255.0 / (255.0 - 8.0)),
        0.0,
        255.0,
    )
    refined[:, :, 3] = np.rint(choked).astype(np.uint8)
    return refined


def _normalise_user_image(
    payload: bytes,
    *,
    filename: str,
    kind: UserAssetKind,
    preserve_background: bool = False,
    cutout: Callable[[np.ndarray], np.ndarray] | None = None,
) -> _NormalisedUserAsset:
    """Validate an upload and return a deterministic, renderer-sized PNG."""
    if kind not in {"avatar", "background"}:
        raise _bad_upload("未知的资源类型")
    if not payload:
        raise _bad_upload("上传文件为空")
    if len(payload) > _MAX_USER_ASSET_BYTES:
        raise _bad_upload("图片不能超过 12 MiB")

    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise _bad_upload("无法识别该图片")

    if kind == "avatar":
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise _bad_upload("人物图片必须是 RGB 或 RGBA 彩色图片")
        if preserve_background:
            if image.shape[2] == 4:
                alpha = image[:, :, 3:4].astype(np.float32) / 255.0
                image = np.rint(
                    image[:, :, :3].astype(np.float32) * alpha
                ).astype(np.uint8)
            # RGB is intentional: AvatarLoader treats it as a portrait whose
            # original background is baked in and skips frame-time matting.
            normalised = _contain_image(image, channels=3)
        else:
            source_alpha: np.ndarray | None = None
            source_rgb: np.ndarray | None = None
            cutout_input = image
            if image.shape[2] == 4:
                # An alpha channel alone does not prove that the portrait has
                # already been cut out: ordinary PNG exports are often fully
                # opaque RGBA files. Always honour "remove background" while
                # retaining any transparency already supplied by the source.
                source_rgb = image[:, :, :3].copy()
                source_alpha = image[:, :, 3].copy()
                alpha = source_alpha[:, :, None].astype(np.float32) / 255.0
                cutout_input = np.rint(
                    source_rgb.astype(np.float32) * alpha + 200.0 * (1.0 - alpha)
                ).astype(np.uint8)
            if cutout is None:
                from avtr1_renderer.feynobg import feynobg_cutout

                cutout = feynobg_cutout
            image = cutout(cutout_input)
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise _bad_upload("FeyNoBg 返回了无效图片")
            if image.shape[:2] != cutout_input.shape[:2]:
                raise _bad_upload("FeyNoBg returned an image with the wrong dimensions")
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            if source_rgb is not None:
                # The segmentation fixture may be composited for inference,
                # but the saved RGBA must retain straight RGB. Otherwise the
                # source alpha is applied a second time during rendering and
                # semi-transparent edges turn dark.
                image[:, :, :3] = source_rgb
            if source_alpha is not None:
                image[:, :, 3] = np.rint(
                    image[:, :, 3].astype(np.float32)
                    * (source_alpha.astype(np.float32) / 255.0)
                ).astype(np.uint8)
            if not np.any(image[:, :, 3] < 255):
                raise _bad_upload("FeyNoBg 未能生成有效的人物遮罩，请确认抠图服务可用")
            if source_alpha is None or bool(np.all(source_alpha == 255)):
                image = _refine_cutout_edges(image)
            else:
                # A genuinely transparent upload already contains intentional
                # straight edge colour; only soften its trusted source alpha.
                image = _feather_cutout_alpha(image)
            normalised = _contain_image(image, channels=4)
    else:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim != 3 or image.shape[2] not in (3, 4):
            raise _bad_upload("背景图片必须是灰度、RGB 或 RGBA 图片")
        elif image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = np.rint(image[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        normalised = _contain_image(image, channels=3)

    ok, png = cv2.imencode(".png", normalised)
    if not ok:
        raise _bad_upload("图片转换为 PNG 失败")
    png_bytes = png.tobytes()
    digest = hashlib.sha256(png_bytes).hexdigest()[:_USER_ASSET_HASH_LENGTH]
    safe_stem = _safe_asset_stem(filename, kind)
    return _NormalisedUserAsset(
        kind=kind,
        asset_id=f"user_{safe_stem}_{digest}",
        png_bytes=png_bytes,
    )


def _asset_directory(user_root: Path, kind: UserAssetKind) -> Path:
    return user_root / ("reference_frames" if kind == "avatar" else "backgrounds")


def _publish_prepared_asset(
    asset: _NormalisedUserAsset,
    prepared: object,
    *,
    pipeline: Pipeline,
    registry: dict[str, object],
) -> None:
    if asset.kind == "avatar":
        registry[asset.asset_id] = prepared
    else:
        pipeline.register_background(asset.asset_id, prepared)  # type: ignore[arg-type]


def _prepare_user_asset(
    asset: _NormalisedUserAsset,
    path: Path,
    *,
    pipeline: Pipeline,
) -> object:
    if asset.kind == "avatar":
        return pipeline.prepare_avatar(path, avatar_id=asset.asset_id)
    return pipeline.prepare_background(path)


def _is_registered(
    asset: _NormalisedUserAsset,
    *,
    pipeline: Pipeline,
    registry: dict[str, object],
) -> bool:
    if asset.kind == "avatar":
        return asset.asset_id in registry
    return asset.asset_id in pipeline._backgrounds


def _install_user_asset(
    asset: _NormalisedUserAsset,
    *,
    pipeline: Pipeline,
    registry: dict[str, object],
    user_root: Path,
) -> dict[str, str | bool]:
    """Prepare, atomically persist, then publish one user image.

    A failed GPU preprocessing pass leaves neither a final file nor a registry
    entry. Existing content-addressed files are reused and re-registered after
    a process restart without rewriting them.
    """
    asset_dir = _asset_directory(user_root, asset.kind)
    asset_dir.mkdir(parents=True, exist_ok=True)
    final_path = asset_dir / f"{asset.asset_id}.png"

    if final_path.is_file():
        if final_path.read_bytes() != asset.png_bytes:
            raise HTTPException(status_code=409, detail="资源 ID 冲突")
        if not _is_registered(asset, pipeline=pipeline, registry=registry):
            prepared = _prepare_user_asset(asset, final_path, pipeline=pipeline)
            _publish_prepared_asset(asset, prepared, pipeline=pipeline, registry=registry)
        result: dict[str, str | bool] = {
            "id": asset.asset_id,
            "asset_id": asset.asset_id,
            "kind": asset.kind,
            "reused": True,
        }
        if asset.kind == "avatar":
            result["background_id"] = PRESET_BACKGROUND_ID
        return result

    temp_path: Path | None = None
    published_final = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=asset_dir,
            prefix=f".{asset.asset_id}.",
            suffix=".tmp.png",
            delete=False,
        ) as temp:
            temp.write(asset.png_bytes)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)

        prepared = _prepare_user_asset(asset, temp_path, pipeline=pipeline)
        os.replace(temp_path, final_path)
        temp_path = None
        published_final = True
        _publish_prepared_asset(asset, prepared, pipeline=pipeline, registry=registry)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if published_final:
            if asset.kind == "avatar":
                registry.pop(asset.asset_id, None)
            else:
                pipeline._backgrounds.pop(asset.asset_id, None)
            final_path.unlink(missing_ok=True)
        raise

    result = {
        "id": asset.asset_id,
        "asset_id": asset.asset_id,
        "kind": asset.kind,
        "reused": False,
    }
    if asset.kind == "avatar":
        result["background_id"] = PRESET_BACKGROUND_ID
    return result


def _trash_user_avatar(
    avatar_id: str,
    *,
    registry: dict[str, object],
    user_root: Path,
) -> dict[str, str | bool]:
    """Remove one uploaded avatar from service while retaining a recovery copy."""

    if not _USER_AVATAR_ID_RE.fullmatch(avatar_id):
        raise HTTPException(status_code=404, detail="只能删除本机上传的人物")
    live_path = user_root / "reference_frames" / f"{avatar_id}.png"
    if not live_path.is_file():
        raise HTTPException(status_code=404, detail=f"未找到可删除的人物: {avatar_id}")

    stamp = time.time_ns()
    trash_dir = user_root / ".trash" / "reference_frames"
    trash_dir.mkdir(parents=True, exist_ok=True)
    trash_path = trash_dir / f"{avatar_id}.{stamp}.png"
    idle_live_path = _avatar_idle_loop_path(user_root, avatar_id)
    idle_trash_path: Path | None = None
    os.replace(live_path, trash_path)
    try:
        if idle_live_path.is_file():
            idle_trash_dir = (
                user_root / ".trash" / "idle_loops" / _IDLE_LOOP_RECIPE
            )
            idle_trash_dir.mkdir(parents=True, exist_ok=True)
            idle_trash_path = idle_trash_dir / f"{avatar_id}.{stamp}.webp"
            os.replace(idle_live_path, idle_trash_path)
        registry.pop(avatar_id, None)
    except BaseException:
        if idle_trash_path is not None and idle_trash_path.is_file():
            idle_live_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(idle_trash_path, idle_live_path)
        os.replace(trash_path, live_path)
        raise
    return {
        "id": avatar_id,
        "kind": "avatar",
        "deleted": True,
        "recoverable": True,
    }


def _bytes_to_float32(buf: bytes, expected_samples: int, field: str) -> np.ndarray:
    expected_bytes = expected_samples * 2
    if len(buf) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field}: expected exactly {expected_bytes} bytes "
                f"({expected_samples} int16 samples), got {len(buf)} bytes"
            ),
        )
    return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / _INT16_MAX


def _build_chunk(
    audio_bytes: tuple[bytes, bytes, bytes, bytes],
    cur_n: int,
    fut_n: int,
) -> Chunk:
    cur, fut, curl, futl = audio_bytes
    speech = np.concatenate([
        _bytes_to_float32(cur, cur_n, "speech_current"),
        _bytes_to_float32(fut, fut_n, "speech_future"),
    ])
    listen = np.concatenate([
        _bytes_to_float32(curl, cur_n, "listen_current"),
        _bytes_to_float32(futl, fut_n, "listen_future"),
    ])
    return Chunk(audio_speech=speech, audio_listen=listen)


def run_chunk(
    loop: asyncio.AbstractEventLoop,
    state_fut: asyncio.Future[bytes],
    send: MemoryObjectSendStream[bytes],
    pipeline: Pipeline,
    avatar,
    audio_bytes: tuple[bytes, bytes, bytes, bytes],
    state_blob_in: bytes | None,
    options: RenderOptions,
    cur_n: int,
    fut_n: int,
) -> None:
    """Drive one chunk end-to-end on a worker thread."""
    try:
        chunk = _build_chunk(audio_bytes, cur_n, fut_n)
        if state_blob_in is not None:
            try:
                prev_state = state_from_safetensors(state_blob_in, device="cuda")
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"failed to decode state safetensors blob: {exc}",
                ) from exc
        else:
            prev_state = None

        next_state, frames_iter = pipeline.process_chunk(avatar, chunk, prev_state, options)
        loop.call_soon_threadsafe(state_fut.set_result, state_to_safetensors(next_state))

        for f in frames_iter:
            loop.call_soon_threadsafe(send.send_nowait, f.data.tobytes())

    except BaseException as exc:
        if not state_fut.done():
            loop.call_soon_threadsafe(state_fut.set_exception, exc)
        else:
            LOG.exception("post-state worker failure; closing connection")
    finally:
        loop.call_soon_threadsafe(send.close)


def _user_assets_root() -> Path:
    from avtr1_renderer.avtr1_artifact_manager import get_storage_root

    legacy_root = get_storage_root() / "user_assets"
    configured_root = os.environ.get("AVTR1_USER_ASSETS_ROOT")
    if not configured_root:
        return legacy_root

    writable_root = Path(configured_root).expanduser()
    if writable_root.resolve(strict=False) == legacy_root.resolve(strict=False):
        return legacy_root
    _migrate_user_assets_once(source=legacy_root, target=writable_root)
    return writable_root


def _migrate_user_assets_once(*, source: Path, target: Path) -> None:
    """Copy a legacy user-asset tree once without making it a live fallback.

    The marker is deliberately stored outside the legacy tree.  After it is
    written, deleting an avatar or background from the writable store cannot
    cause the packaged copy to reappear on the next renderer start.
    """

    target.mkdir(parents=True, exist_ok=True)
    marker = target / _USER_ASSET_MIGRATION_MARKER
    if marker.is_file():
        return

    if source.is_dir():
        for source_path in sorted(source.rglob("*")):
            if source_path.is_symlink() or not source_path.is_file():
                continue
            relative = source_path.relative_to(source)
            target_path = target / relative
            if target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target_path.parent,
                    prefix=f".{target_path.name}.",
                    suffix=".migrating",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    with source_path.open("rb") as source_file:
                        shutil.copyfileobj(source_file, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                # A concurrent first start may have installed the same file.
                # Never replace a writable value with the legacy value.
                if not target_path.exists():
                    os.replace(temp_path, target_path)
                    temp_path = None
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    marker_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target,
            prefix=f".{_USER_ASSET_MIGRATION_MARKER}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            marker_temp = Path(temp_file.name)
            temp_file.write("legacy user assets migrated\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(marker_temp, marker)
        marker_temp = None
    finally:
        if marker_temp is not None:
            marker_temp.unlink(missing_ok=True)


def _avatar_idle_loop_path(user_root: Path, avatar_id: str) -> Path:
    return user_root / "idle_loops" / _IDLE_LOOP_RECIPE / f"{avatar_id}.webp"


def _ensure_avatar_idle_loop(
    *,
    avatar_id: str,
    pipeline: Pipeline,
    registry: dict[str, object],
    user_root: Path,
    inference_guard: Callable[[], AbstractContextManager[None]] | None = None,
) -> Path:
    """Generate one versioned idle loop atomically, or reuse the cache."""

    avatar = registry.get(avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"unknown avatar: {avatar_id}")
    final_path = _avatar_idle_loop_path(user_root, avatar_id)
    if final_path.is_file() and final_path.stat().st_size > 12:
        return final_path

    payload = generate_avatar_idle_loop(
        pipeline,
        avatar,
        inference_guard=inference_guard,
    )
    if not (
        len(payload) > 12
        and payload.startswith(b"RIFF")
        and payload[8:12] == b"WEBP"
    ):
        raise RuntimeError("idle-loop encoder returned an invalid WebP")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=final_path.parent,
            prefix=f".{avatar_id}.",
            suffix=".tmp.webp",
            delete=False,
        ) as temp:
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)
        os.replace(temp_path, final_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return final_path


def _restore_user_assets(
    *,
    pipeline: Pipeline,
    registry: dict[str, object],
    user_root: Path,
) -> None:
    """Reload previously validated user assets into their GPU registries."""
    avatars_dir = user_root / "reference_frames"
    for path in sorted(avatars_dir.glob("user_*.png")):
        if path.stem in registry:
            continue
        try:
            registry[path.stem] = pipeline.prepare_avatar(path, avatar_id=path.stem)
        except Exception:
            LOG.exception("Failed to restore user avatar %s", path)

    backgrounds_dir = user_root / "backgrounds"
    for path in sorted(backgrounds_dir.glob("user_*.png")):
        if path.stem in pipeline._backgrounds:
            continue
        try:
            background = pipeline.prepare_background(path)
            pipeline.register_background(path.stem, background)
        except Exception:
            LOG.exception("Failed to restore user background %s", path)


def _load_renderer_runtime(
    *,
    portraits_dir: Path,
    user_root: Path,
) -> _LoadedRendererRuntime:
    """Construct the complete renderer runtime for startup or lazy reload."""

    started = time.monotonic()
    avatar_ids = [p.stem for p in sorted(portraits_dir.glob("*.png"))]
    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=avatar_ids,
        portraits_dir=portraits_dir,
        background_paths=_bundled_background_paths(),
    )
    _restore_user_assets(pipeline=pipeline, registry=registry, user_root=user_root)
    motion_generator = pipeline._motion_generator
    runtime = _LoadedRendererRuntime(
        pipeline=pipeline,
        registry=registry,
        current_samples=motion_generator.chunk_size * motion_generator.frame_len,
        future_samples=(
            motion_generator.future_size * motion_generator.frame_len
            + motion_generator.audio_shift
        ),
        health=CudaHealthChecker(),
    )
    LOG.info("Avatars: %s", ", ".join(sorted(registry)))
    LOG.info(
        "Loaded engines + %d avatars + %d backgrounds in %.1fs",
        len(registry),
        len(pipeline._backgrounds),
        time.monotonic() - started,
    )
    return runtime


def _install_renderer_runtime(app: FastAPI, runtime: _LoadedRendererRuntime) -> None:
    app.state.pipeline = runtime.pipeline
    app.state.registry = runtime.registry
    app.state.current_samples = runtime.current_samples
    app.state.future_samples = runtime.future_samples
    app.state.health = runtime.health
    app.state.runtime_status = "ready"
    app.state.runtime_error = None


async def _ensure_renderer_runtime_loaded(
    app: FastAPI,
) -> tuple[Pipeline, dict[str, object]]:
    """Return a loaded runtime, reconstructing it exactly once when released."""

    pipeline: Pipeline | None = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        return pipeline, getattr(app.state, "registry", None) or {}

    lock: asyncio.Lock | None = getattr(app.state, "runtime_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.runtime_lock = lock
    async with lock:
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None:
            return pipeline, getattr(app.state, "registry", None) or {}

        portraits_dir: Path | None = getattr(app.state, "portraits_dir", None)
        if portraits_dir is None:
            from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager

            portraits_dir = Path(
                get_artifact_manager().get_artifact_path("reference_frames")
            )
            app.state.portraits_dir = portraits_dir
        user_root: Path | None = getattr(app.state, "user_assets_root", None)
        if user_root is None:
            user_root = _user_assets_root()
            app.state.user_assets_root = user_root

        app.state.runtime_status = "loading"
        app.state.runtime_error = None
        try:
            runtime = await run_in_thread(
                _load_renderer_runtime,
                portraits_dir=portraits_dir,
                user_root=user_root,
            )
        except BaseException as exc:
            app.state.runtime_status = "error"
            app.state.runtime_error = str(exc)
            raise
        _install_renderer_runtime(app, runtime)
        return runtime.pipeline, runtime.registry


def _release_accelerator_memory() -> None:
    """Collect model objects and return allocator-owned CUDA memory to the OS."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Runtime release must remain usable on CPU-only builds and when CUDA
        # is already being torn down. The removed model references still stay
        # removed; the next load will create a fresh runtime.
        LOG.warning("CUDA allocator cleanup failed", exc_info=True)
    gc.collect()


def _release_renderer_runtime(app: FastAPI) -> dict[str, str | bool]:
    """Drop every GPU-backed renderer reference while keeping the API alive."""

    activity = _renderer_activity(app)
    with activity.release_inference():
        pipeline = getattr(app.state, "pipeline", None)
        close = getattr(pipeline, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOG.warning("Renderer pipeline close failed during release", exc_info=True)
        app.state.pipeline = None
        registry = getattr(app.state, "registry", None)
        if isinstance(registry, dict):
            registry.clear()
        app.state.registry = {}
        app.state.current_samples = None
        app.state.future_samples = None
        app.state.health = None
        app.state.runtime_status = "released"
        app.state.runtime_error = None
        _release_accelerator_memory()
    return {
        "status": "released",
        "released": True,
        "loaded": False,
        "active_requests": 0,
    }


def _available_avatar_ids(app: FastAPI) -> list[str]:
    ids = set((getattr(app.state, "registry", None) or {}).keys())
    portraits_dir: Path | None = getattr(app.state, "portraits_dir", None)
    if portraits_dir is not None:
        ids.update(path.stem for path in portraits_dir.glob("*.png"))
    user_root: Path | None = getattr(app.state, "user_assets_root", None)
    if user_root is not None:
        ids.update(
            path.stem
            for path in (user_root / "reference_frames").glob("user_*.png")
        )
    return sorted(ids)


def _available_background_ids(app: FastAPI) -> list[str]:
    ids = set(_bundled_background_paths())
    pipeline: Pipeline | None = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        ids.update(
            background_id
            for background_id in pipeline._backgrounds
            if background_id != TRANSPARENT_BG_ID
        )
    user_root: Path | None = getattr(app.state, "user_assets_root", None)
    if user_root is not None:
        ids.update(
            path.stem for path in (user_root / "backgrounds").glob("user_*.png")
        )
    ids.discard(TRANSPARENT_BG_ID)
    return sorted(ids)


def _avatar_source_path(app: FastAPI, avatar_id: str) -> Path | None:
    user_root: Path | None = getattr(app.state, "user_assets_root", None)
    if user_root is not None:
        candidate = user_root / "reference_frames" / f"{avatar_id}.png"
        if candidate.is_file():
            return candidate
    portraits_dir: Path | None = getattr(app.state, "portraits_dir", None)
    if portraits_dir is not None:
        candidate = portraits_dir / f"{avatar_id}.png"
        if candidate.is_file():
            return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()

    # Auto-discover all portrait PNGs in reference_frames.
    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
    mgr = get_artifact_manager()
    portraits_dir = Path(mgr.get_artifact_path("reference_frames"))
    avatar_ids = [p.stem for p in sorted(portraits_dir.glob("*.png"))]

    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=avatar_ids,
        portraits_dir=portraits_dir,
        background_paths=_bundled_background_paths(),
    )
    user_root = _user_assets_root()
    _restore_user_assets(pipeline=pipeline, registry=registry, user_root=user_root)
    LOG.info("Avatars: %s", ", ".join(sorted(registry)))
    LOG.info(
        "Loaded engines + %d avatars + %d backgrounds in %.1fs",
        len(registry),
        len(pipeline._backgrounds),
        time.monotonic() - t0,
    )
    app.state.pipeline = pipeline
    app.state.registry = registry
    app.state.portraits_dir = portraits_dir
    app.state.user_assets_root = user_root
    app.state.user_asset_lock = asyncio.Lock()
    app.state.runtime_lock = asyncio.Lock()
    app.state.runtime_status = "ready"
    app.state.runtime_error = None
    app.state.renderer_activity = _RendererActivityGate()
    mg = pipeline._motion_generator
    app.state.current_samples = mg.chunk_size * mg.frame_len
    app.state.future_samples = mg.future_size * mg.frame_len + mg.audio_shift
    app.state.health = CudaHealthChecker()
    # The async lifespan frame survives across ``yield``. Do not let its local
    # variables retain a second, invisible reference after /release clears
    # app.state; otherwise TensorRT/ONNX objects and avatar tensors stay alive.
    del pipeline, registry, mg
    async with keep_alive_worker():
        yield


app = FastAPI(lifespan=lifespan)


def _is_loopback_request(request: Request) -> bool:
    if request.client is None:
        return False
    host = request.client.host.split("%", 1)[0]
    if host.casefold() == "localhost":
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


async def _upload_user_asset(
    request: Request,
    file: UploadFile,
    *,
    kind: UserAssetKind,
    preserve_background: bool = False,
) -> dict[str, str | bool]:
    if not _is_loopback_request(request):
        raise HTTPException(status_code=403, detail="用户资源上传仅允许本机访问")

    payload = await file.read(_MAX_USER_ASSET_BYTES + 1)
    asset = await run_in_thread(
        _normalise_user_image,
        payload,
        filename=file.filename or f"{kind}.png",
        kind=kind,
        preserve_background=preserve_background,
    )
    user_root: Path | None = getattr(request.app.state, "user_assets_root", None)
    if user_root is None:
        user_root = _user_assets_root()
    activity = _renderer_activity(request.app)
    lock: asyncio.Lock | None = getattr(request.app.state, "user_asset_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.user_asset_lock = lock

    async with lock:
        pipeline, registry = await _ensure_renderer_runtime_loaded(request.app)
        try:
            result = await run_in_thread(
                _install_user_asset,
                asset,
                pipeline=pipeline,
                registry=registry,
                user_root=user_root,
            )
        except HTTPException:
            raise
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"资源预处理失败: {exc}") from exc

        if asset.kind == "avatar":
            result["idle_loop_url"] = (
                f"/avatars/{asset.asset_id}/idle-loop?recipe={_IDLE_LOOP_RECIPE}"
            )
            if activity.has_active_live:
                result["idle_loop_ready"] = False
                return result
            try:
                await run_in_thread(
                    _ensure_avatar_idle_loop,
                    avatar_id=asset.asset_id,
                    pipeline=pipeline,
                    registry=registry,
                    user_root=user_root,
                    inference_guard=activity.idle_inference,
                )
            except _RendererBusyError:
                result["idle_loop_ready"] = False
            except Exception:
                # The portrait is already valid and installed. Keep it usable;
                # the browser will show its static poster and the GET route can
                # retry generation later.
                LOG.exception("Failed to generate idle loop for %s", asset.asset_id)
                result["idle_loop_ready"] = False
            else:
                result["idle_loop_ready"] = True
        return result


@app.post("/assets/avatar")
async def upload_avatar(
    request: Request,
    file: UploadFile,
    preserve_background: bool = Form(False),
) -> dict[str, str | bool]:
    return await _upload_user_asset(
        request,
        file,
        kind="avatar",
        preserve_background=preserve_background,
    )


@app.delete("/assets/avatar/{avatar_id}")
async def delete_avatar(avatar_id: str, request: Request) -> dict[str, str | bool]:
    if not _is_loopback_request(request):
        raise HTTPException(status_code=403, detail="用户资源删除仅允许本机访问")
    registry: dict[str, object] = request.app.state.registry
    user_root: Path | None = getattr(request.app.state, "user_assets_root", None)
    if user_root is None:
        user_root = _user_assets_root()
    lock: asyncio.Lock | None = getattr(request.app.state, "user_asset_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.user_asset_lock = lock
    async with lock:
        return await run_in_thread(
            _trash_user_avatar,
            avatar_id,
            registry=registry,
            user_root=user_root,
        )


@app.post("/assets/background")
async def upload_background(request: Request, file: UploadFile) -> dict[str, str | bool]:
    return await _upload_user_asset(request, file, kind="background")


@app.post("/process-audio-v3")
async def process_audio_v3(
    request: Request,
    current_chunk: UploadFile,
    future_chunk: UploadFile,
    current_chunk_listen: UploadFile,
    future_chunk_listen: UploadFile,
    state: UploadFile | None = None,
    avatar_id: str = "anya_03_studio",
    bg_id: str = Query(..., description="Background id; must match an entry from /avatars"),
    pixel_format: PixelFormat = "yuv_i420",
    h: int = Query(_USER_IMAGE_HEIGHT, gt=0, le=_USER_IMAGE_HEIGHT, multiple_of=2),
    w: int = Query(_USER_IMAGE_WIDTH, gt=0, le=_USER_IMAGE_WIDTH, multiple_of=2),
    rtx_super_resolution: bool = False,
    rtx_input_h: int | None = Query(None, gt=0, le=720, multiple_of=2),
    rtx_input_w: int | None = Query(None, gt=0, le=1280, multiple_of=2),
    rtx_output_h: int | None = Query(None, gt=0, le=1080, multiple_of=2),
    rtx_output_w: int | None = Query(None, gt=0, le=1920, multiple_of=2),
    rtx_quality: NvidiaVSRQuality = "high_bitrate_high",
    cfg_self_audio: float = 2.0,
    cfg_other_audio: float = 2.0,
    cfg_kp: float = 4.0,
    noise_alpha: float = 2.0,
    noise_trunc_z: float = 1.2,
) -> StreamingResponse:
    rtx_dimensions = (rtx_input_h, rtx_input_w, rtx_output_h, rtx_output_w)
    if rtx_super_resolution:
        if any(value is None for value in rtx_dimensions):
            raise HTTPException(
                status_code=422,
                detail="RTX super resolution requires input and output height/width",
            )
        assert rtx_input_h is not None
        assert rtx_input_w is not None
        assert rtx_output_h is not None
        assert rtx_output_w is not None
        if rtx_input_h > h or rtx_input_w > w:
            raise HTTPException(
                status_code=422,
                detail="RTX input dimensions cannot exceed the renderer crop",
            )
        if rtx_output_h < rtx_input_h or rtx_output_w < rtx_input_w:
            raise HTTPException(
                status_code=422,
                detail="RTX output dimensions cannot be smaller than its input",
            )

    activity = _renderer_activity(request.app)
    # Reserve before lazy loading so a concurrent release cannot clear the
    # newly loaded runtime in the gap before the first inference begins.
    activity.begin_live()
    try:
        pipeline, registry = await _ensure_renderer_runtime_loaded(request.app)
        if rtx_super_resolution:
            assert rtx_input_h is not None
            assert rtx_input_w is not None
            assert rtx_output_h is not None
            assert rtx_output_w is not None
            try:
                await run_in_thread(
                    _run_live_preflight,
                    activity,
                    pipeline.prepare_nvidia_vsr,
                    input_height=rtx_input_h,
                    input_width=rtx_input_w,
                    output_height=rtx_output_h,
                    output_width=rtx_output_w,
                    quality=rtx_quality,
                )
            except NvidiaVSRUnavailableError as exc:
                raise _nvidia_vsr_http_exception(exc) from exc
        if avatar_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown avatar_id {avatar_id!r}; available: {sorted(registry)}",
            )
        avatar = registry[avatar_id]
        if bg_id.endswith(".png"):
            bg_id = bg_id[:-4]

        audio_bytes = (
            await current_chunk.read(),
            await future_chunk.read(),
            await current_chunk_listen.read(),
            await future_chunk_listen.read(),
        )
        state_blob_in = await state.read() if state is not None else None
        if state_blob_in is not None and len(state_blob_in) == 0:
            state_blob_in = None

        options = RenderOptions(
            pixel_format=pixel_format,
            bg_id=bg_id,
            output_height=h,
            output_width=w,
            rtx_super_resolution=rtx_super_resolution,
            rtx_input_height=rtx_input_h,
            rtx_input_width=rtx_input_w,
            rtx_output_height=rtx_output_h,
            rtx_output_width=rtx_output_w,
            rtx_quality=rtx_quality,
            cfg_self_audio=cfg_self_audio,
            cfg_other_audio=cfg_other_audio,
            cfg_kp=cfg_kp,
            noise_alpha=noise_alpha,
            noise_trunc_z=noise_trunc_z,
        )
        loop = asyncio.get_running_loop()
        state_fut: asyncio.Future[bytes] = loop.create_future()
        send, recv = anyio.create_memory_object_stream[bytes](max_buffer_size=math.inf)
        run_in_thread(
            _run_reserved_live,
            activity,
            run_chunk,
            loop,
            state_fut,
            send,
            pipeline,
            avatar,
            audio_bytes,
            state_blob_in,
            options,
            request.app.state.current_samples,
            request.app.state.future_samples,
        )
    except BaseException:
        activity.end_live()
        raise

    state_blob = await state_fut

    out_h = rtx_output_h if rtx_super_resolution else h
    out_w = rtx_output_w if rtx_super_resolution else w
    assert out_h is not None
    assert out_w is not None
    frame_bytes = get_bytes_per_frame(out_h, out_w, options.pixel_format)
    n_frames = pipeline._motion_generator.chunk_size

    async def body():
        try:
            yield state_blob
            async with recv:
                async for frame_chunk in recv:
                    yield frame_chunk
        except BaseException:
            logging.warning("Body streaming ended abruptly", exc_info=True)

    return StreamingResponse(
        body(),
        headers={
            "X-Num-Frames": str(n_frames),
            "X-Frame-Height": str(out_h),
            "X-Frame-Width": str(out_w),
            "X-Frame-Length-Bytes": str(frame_bytes),
            "X-Has-State": "yes",
            "X-State-Format": "safetensors",
            "X-State-Length-Bytes": str(len(state_blob)),
        },
    )


@app.get("/avatars")
async def avatars(request: Request) -> dict[str, object]:
    """List available disk assets without forcing released models to reload.

    Backgrounds excludes the reserved ``transparent`` sentinel since callers
    pick it via ``pixel_format`` rather than as a real background image.
    """
    pipeline: Pipeline | None = getattr(request.app.state, "pipeline", None)
    return {
        "avatars": _available_avatar_ids(request.app),
        "backgrounds": _available_background_ids(request.app),
        "loaded": pipeline is not None,
    }


@app.get("/avatars/{avatar_id}/preview")
async def avatar_preview(avatar_id: str, request: Request) -> Response:
    """Return the selected avatar's immutable source portrait as PNG."""

    source_path = _avatar_source_path(request.app, avatar_id)
    if source_path is not None:
        return FileResponse(
            source_path,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )
    registry: dict = getattr(request.app.state, "registry", None) or {}
    avatar = registry.get(avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"unknown avatar: {avatar_id}")
    try:
        png = await asyncio.to_thread(encode_avatar_preview_png, avatar)
    except Exception as exc:
        LOG.exception("avatar preview encoding failed", extra={"avatar_id": avatar_id})
        raise HTTPException(status_code=500, detail="avatar preview encoding failed") from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.post("/nvidia-vsr/prepare")
async def prepare_nvidia_vsr(
    request: Request,
    input_h: int = Query(..., gt=0, le=720, multiple_of=2),
    input_w: int = Query(..., gt=0, le=1280, multiple_of=2),
    output_h: int = Query(..., gt=0, le=1080, multiple_of=2),
    output_w: int = Query(..., gt=0, le=1920, multiple_of=2),
    quality: NvidiaVSRQuality = "high_bitrate_high",
) -> dict[str, str | int]:
    """Load one VSR profile before a WebRTC session consumes its resources."""

    if output_h < input_h or output_w < input_w:
        raise HTTPException(
            status_code=422,
            detail="RTX output dimensions cannot be smaller than its input",
        )

    activity = _renderer_activity(request.app)
    activity.begin_live()
    try:
        pipeline, _registry = await _ensure_renderer_runtime_loaded(request.app)
        try:
            await run_in_thread(
                _run_live_preflight,
                activity,
                pipeline.prepare_nvidia_vsr,
                input_height=input_h,
                input_width=input_w,
                output_height=output_h,
                output_width=output_w,
                quality=quality,
            )
        except NvidiaVSRUnavailableError as exc:
            raise _nvidia_vsr_http_exception(exc) from exc
    finally:
        activity.end_live()

    return {
        "status": "ready",
        "input_height": input_h,
        "input_width": input_w,
        "output_height": output_h,
        "output_width": output_w,
        "quality": quality,
    }


@app.get("/avatars/{avatar_id}/idle-loop")
async def avatar_idle_loop(avatar_id: str, request: Request) -> FileResponse:
    """Return a cached transparent listening animation, generating if absent."""

    if avatar_id not in _available_avatar_ids(request.app):
        raise HTTPException(status_code=404, detail=f"unknown avatar: {avatar_id}")
    user_root: Path | None = getattr(request.app.state, "user_assets_root", None)
    if user_root is None:
        user_root = _user_assets_root()
    activity = _renderer_activity(request.app)
    cache_path = _avatar_idle_loop_path(user_root, avatar_id)
    if cache_path.is_file() and cache_path.stat().st_size > 12:
        return FileResponse(
            cache_path,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )
    if activity.has_active_live and not (
        cache_path.is_file() and cache_path.stat().st_size > 12
    ):
        raise _renderer_busy_http_exception()
    lock: asyncio.Lock | None = getattr(request.app.state, "user_asset_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.user_asset_lock = lock
    async with lock:
        pipeline, registry = await _ensure_renderer_runtime_loaded(request.app)
        try:
            path = await run_in_thread(
                _ensure_avatar_idle_loop,
                avatar_id=avatar_id,
                pipeline=pipeline,
                registry=registry,
                user_root=user_root,
                inference_guard=activity.idle_inference,
            )
        except HTTPException:
            raise
        except _RendererBusyError as exc:
            raise _renderer_busy_http_exception() from exc
        except Exception as exc:
            LOG.exception("avatar idle-loop generation failed", extra={"avatar_id": avatar_id})
            raise HTTPException(status_code=503, detail="人物待机动画生成失败") from exc
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@app.post("/release")
async def release_renderer_models(request: Request) -> dict[str, object]:
    """Release renderer models while leaving cached UI assets and API online."""

    if not _is_loopback_request(request):
        raise HTTPException(status_code=403, detail="model release is localhost-only")
    user_lock: asyncio.Lock | None = getattr(request.app.state, "user_asset_lock", None)
    if user_lock is None:
        user_lock = asyncio.Lock()
        request.app.state.user_asset_lock = user_lock
    runtime_lock: asyncio.Lock | None = getattr(request.app.state, "runtime_lock", None)
    if runtime_lock is None:
        runtime_lock = asyncio.Lock()
        request.app.state.runtime_lock = runtime_lock
    try:
        # Match uploads/idle generation lock ordering: user assets first, then
        # runtime administration. This prevents release racing GPU preparation.
        async with user_lock, runtime_lock:
            result = await run_in_thread(_release_renderer_runtime, request.app)
    except _RendererBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"service": "renderer", **result}


@app.get("/health")
async def health(request: Request):
    pipeline: Pipeline | None = getattr(request.app.state, "pipeline", None)
    runtime_status = getattr(request.app.state, "runtime_status", None)
    if pipeline is None and runtime_status == "released":
        return {"status": "released", "loaded": False}
    checker: CudaHealthChecker | None = getattr(request.app.state, "health", None)
    if checker is None:
        raise HTTPException(status_code=503, detail="starting")
    try:
        await checker.check()
    except Exception as exc:
        LOG.exception("cuda healthcheck failed")
        raise HTTPException(status_code=503, detail=f"cuda unhealthy: {exc}") from exc
    return {"status": "ok", "loaded": True}


class _DropSuccessfulAccess(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        status = record.args[4] if isinstance(record.args, tuple) and len(record.args) >= 5 else None
        return not (isinstance(status, int) and 200 <= status < 400)


# Applied at import time so it sticks whether this module is launched via
# `python -m avtr1_renderer.api.app` or via `python -m uvicorn avtr1_renderer.api.app:app`
# (the orchestrator uses the latter to pass --port). Uvicorn configures its own
# logging before importing the app, so our filter attaches after its handlers exist.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(_DropSuccessfulAccess())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "avtr1_renderer.api.app:app",
        host=os.environ.get("RENDERER_HOST", "127.0.0.1"),
        port=8000,
    )
