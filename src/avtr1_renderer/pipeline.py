# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Streaming face animation orchestrator.

Thin coordinator wiring three stages: a ``MotionGenerator`` produces
per-frame motion from audio, ``render_chunk_streaming`` yields per-frame
GPU ``(rgb, alpha)`` slices (warp runs once on the full chunk, every
downstream step runs per frame), and ``pack_frames`` converts each
slice to the requested pixel format and copies to host.

    motions, next_state = motion_generator.generate_chunk(chunk, avatar, state)
    for rgb_1, alpha_1 in render_chunk_streaming(motions, avatar, bg, ...):
        yield pack_frames(rgb_1, alpha_1, pixel_format=...)[0]

``Pipeline.process_chunk`` returns ``(state, frame_iterator)``; with
``stream_frames=True`` (default) the iterator is truly per-frame —
each ``next()`` runs decode + putback + matting + pack + H2D for
exactly one frame and yields it before the next frame starts.

Runtime backends
----------------
Native Windows defaults to the portable CUDA path: the released AVTR-1
TorchScript checkpoint runs in PyTorch and the remaining models run through
ONNX Runtime CUDA. Other platforms prefer locally built TensorRT engines and
fall back to the same portable path when those engines are absent. TensorRT
engines are always local artifacts and must be built for the current platform,
GPU, CUDA, and TensorRT versions.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as torch_functional

from avtr1_renderer.avatar_loader import Avatar, AvatarLoader
from avtr1_renderer.avtr1_artifact_manager import (
    find_engine_or_onnx,
    get_artifact_manager,
    get_storage_root,
    get_trt_engine_path,
)
from avtr1_renderer.avtr1_motion_generator import (
    AVTR1MotionGenerator,
    Normalizer,
)
from avtr1_renderer.backgrounds import load_background
from avtr1_renderer.frame_sink import pack_frames
from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeInput,
    Avtr1DecodeOutput,
    Avtr1EncodeInput,
    Avtr1EncodeOutput,
)
from avtr1_renderer.models.decoder import DecoderEngine, DecoderInput, DecoderOutput
from avtr1_renderer.models.hubert import HubertInput, HubertOutput
from avtr1_renderer.models.matting import MODNetEngine, MODNetInput, MODNetOutput
from avtr1_renderer.models.stitch import StitchEngine, StitchInput, StitchOutput
from avtr1_renderer.models.warp import WarpEngine, WarpInput, WarpOutput
from avtr1_renderer.motion_generator import MotionGenerator
from avtr1_renderer.nvidia_vsr import NvidiaVSR
from avtr1_renderer.renderer import render_chunk, render_chunk_streaming
from avtr1_renderer.runtime import load_engine
from avtr1_renderer.runtime.portable_onnx import (
    prepare_portable_hubert_onnx,
    prepare_portable_modnet_onnx,
    prepare_portable_stitch_onnx,
    prepare_portable_warp_onnx,
)
from avtr1_renderer.runtime.torchscript_avtr1 import TorchScriptAVTR1Backend
from avtr1_renderer.types import Chunk, FrameIterator, RenderOptions

# Reserved sentinel: composite onto zero so internal callers can recover the
# straight foreground from the returned premultiplied colour + matte.
TRANSPARENT_BG_ID = "transparent"

AVTR1Backend = Literal["auto", "tensorrt", "torchscript"]
ResolvedAVTR1Backend = Literal["tensorrt", "torchscript"]


def _center_crop_output(
    rgb: torch.Tensor,
    alpha: torch.Tensor | None,
    *,
    output_height: int,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Centre-crop rendered RGB/alpha tensors before pixel-format packing."""

    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output height and width must be positive")
    if output_height % 2 or output_width % 2:
        raise ValueError("I420 output height and width must be even")

    source_height, source_width = rgb.shape[-2:]
    if output_height > source_height or output_width > source_width:
        raise ValueError(
            f"Output {output_width}x{output_height} exceeds native "
            f"{source_width}x{source_height}"
        )

    top = (source_height - output_height) // 2
    left = (source_width - output_width) // 2
    rows = slice(top, top + output_height)
    columns = slice(left, left + output_width)
    cropped_rgb = rgb[..., rows, columns]
    cropped_alpha = alpha[..., rows, columns] if alpha is not None else None
    return cropped_rgb, cropped_alpha


def _straight_rgba_uint8(
    premultiplied_rgb: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Convert premultiplied RGB + alpha to straight NHWC RGBA bytes.

    This deliberately runs before I420 conversion. Unpremultiplying after 4:2:0
    chroma subsampling amplifies quantisation error around translucent hair.
    """

    if premultiplied_rgb.ndim != 4 or premultiplied_rgb.shape[1] != 3:
        raise ValueError(
            "premultiplied_rgb must be NCHW with three channels, got "
            f"{tuple(premultiplied_rgb.shape)}"
        )
    expected_alpha_shape = (
        premultiplied_rgb.shape[0],
        1,
        premultiplied_rgb.shape[2],
        premultiplied_rgb.shape[3],
    )
    if alpha.shape != expected_alpha_shape:
        raise ValueError(
            "alpha must be N1HW matching premultiplied_rgb, got "
            f"{tuple(alpha.shape)}"
        )
    alpha = alpha.clamp(0.0, 1.0)
    safe_alpha = alpha.clamp_min(1.0 / 255.0)
    straight_rgb = torch.where(
        alpha > (0.5 / 255.0),
        premultiplied_rgb.clamp(0.0, 1.0) / safe_alpha,
        torch.zeros_like(premultiplied_rgb),
    ).clamp_(0.0, 1.0)
    rgba = torch.cat([straight_rgb, alpha], dim=1).permute(0, 2, 3, 1)
    return rgba.mul(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8).contiguous()


def resolve_avtr1_backend(
    requested: AVTR1Backend | str = "auto",
    *,
    platform: str = sys.platform,
    encode_path: Path,
    decode_path: Path,
    engines_exist: bool | None = None,
) -> ResolvedAVTR1Backend:
    """Resolve the speech-to-motion backend without loading either runtime."""
    if requested not in {"auto", "tensorrt", "torchscript"}:
        raise ValueError(
            f"Unknown AVTR1 backend {requested!r}; expected auto, tensorrt, or torchscript"
        )
    if engines_exist is None:
        engines_exist = encode_path.is_file() and decode_path.is_file()
    if requested == "torchscript":
        return "torchscript"
    if requested == "tensorrt":
        if not engines_exist:
            raise RuntimeError(
                "AVTR1 TensorRT engines not found.\n"
                f"Expected files:\n  {encode_path}\n  {decode_path}"
            )
        return "tensorrt"
    return "tensorrt" if engines_exist else "torchscript"


def _load_avtr1_engines(
    backend: ResolvedAVTR1Backend,
    *,
    manager: Any,
    encode_path: Path,
    decode_path: Path,
) -> tuple[Any, Any, Normalizer]:
    if backend == "torchscript":
        checkpoint = manager.get_artifact_path("avtr1_scripted")
        portable = TorchScriptAVTR1Backend.from_file(checkpoint)
        normalizer = Normalizer.from_scripted(portable.scripted)
        return portable.encode, portable.decode, normalizer

    encode = load_engine(encode_path, Avtr1EncodeInput, Avtr1EncodeOutput)
    decode = load_engine(decode_path, Avtr1DecodeInput, Avtr1DecodeOutput)
    normalizer = Normalizer.from_safetensors(
        get_storage_root() / "avtr1_normalizer.safetensors"
    )
    return encode, decode, normalizer


def _resolve_component_path(
    backend: ResolvedAVTR1Backend,
    *,
    manager: Any,
    engine_name: str,
    onnx_artifact: str,
) -> Path:
    if backend == "torchscript":
        return manager.get_artifact_path(onnx_artifact)
    return find_engine_or_onnx(engine_name, onnx_artifact)


def _resolve_warp_path(
    backend: ResolvedAVTR1Backend,
    *,
    manager: Any,
) -> Path:
    if backend == "tensorrt":
        engine_path = get_trt_engine_path("warp_network")
        if engine_path.is_file():
            return engine_path
    source = manager.get_artifact_path("warp_network_ori_onnx")
    return prepare_portable_warp_onnx(source)


def _resolve_warp_plugin(
    *,
    platform: str = sys.platform,
    manager: Any,
) -> Path:
    override = os.environ.get("AVTR1_WARP_PLUGIN")
    if override:
        plugin = Path(override).expanduser()
    elif platform == "win32":
        plugin = get_trt_engine_path("warp_network").with_name(
            "grid_sample_3d_plugin.dll"
        )
    else:
        plugin = manager.storage_path("warp_plugin")
    if not plugin.is_file():
        platform_hint = (
            "Compile the Windows GridSample3D plugin DLL and set "
            "AVTR1_WARP_PLUGIN to its full path."
            if platform == "win32"
            else "Download the released warp plugin artifact."
        )
        raise RuntimeError(f"Warp TensorRT plugin not found: {plugin}. {platform_hint}")
    return plugin


def _resolve_dynamic_component_path(
    backend: ResolvedAVTR1Backend,
    *,
    manager: Any,
    engine_name: str,
    onnx_artifact: str,
    prepare: Callable[[Path], Path],
) -> Path:
    path = _resolve_component_path(
        backend,
        manager=manager,
        engine_name=engine_name,
        onnx_artifact=onnx_artifact,
    )
    return prepare(path) if path.suffix.lower() == ".onnx" else path


def _resolve_hubert_path(
    backend: ResolvedAVTR1Backend,
    *,
    manager: Any,
) -> Path:
    path = _resolve_component_path(
        backend,
        manager=manager,
        engine_name="hubert_lbs",
        onnx_artifact="hubert_onnx",
    )
    if path.suffix.lower() == ".onnx":
        return prepare_portable_hubert_onnx(path)
    return path


class Pipeline[StateT]:
    """Holds the motion generator + renderer engines + bg registry."""

    def __init__(
        self,
        *,
        motion_generator: MotionGenerator[StateT],
        stitch: StitchEngine,
        warp: WarpEngine,
        decoder: DecoderEngine,
        matting: MODNetEngine,
        backgrounds: dict[str, torch.Tensor],
        avatar_loader: AvatarLoader | None = None,
        nvidia_vsr: NvidiaVSR | None = None,
        out_size: tuple[int, int] = (720, 1280),
    ) -> None:
        self._motion_generator = motion_generator
        self._stitch = stitch
        self._warp = warp
        self._decoder = decoder
        self._matting = matting
        if not backgrounds:
            raise ValueError("backgrounds registry is empty")
        self._backgrounds = backgrounds
        self._avatar_loader = avatar_loader
        self._nvidia_vsr = nvidia_vsr or NvidiaVSR()
        self._out_size = out_size

    def close(self) -> None:
        """Release optional renderer resources owned by this pipeline."""

        self._nvidia_vsr.close()

    @staticmethod
    def _validate_nvidia_vsr_dimensions(
        *,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
    ) -> None:
        dimensions = (input_height, input_width, output_height, output_width)
        if any(value <= 0 or value % 2 for value in dimensions):
            raise ValueError("RTX super resolution dimensions must be positive and even")
        if output_height < input_height or output_width < input_width:
            raise ValueError("RTX output dimensions cannot be smaller than its input")

    def prepare_nvidia_vsr(
        self,
        *,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        quality: str,
    ) -> None:
        """Synchronously load the requested VSR configuration before streaming."""

        self._validate_nvidia_vsr_dimensions(
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
        )
        self._nvidia_vsr.prepare(
            output_height=output_height,
            output_width=output_width,
            quality=quality,
        )

    def _apply_nvidia_vsr(
        self,
        rgb: torch.Tensor,
        alpha: torch.Tensor | None,
        options: RenderOptions,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not options.rtx_super_resolution:
            return rgb, alpha

        dimensions = (
            options.rtx_input_height,
            options.rtx_input_width,
            options.rtx_output_height,
            options.rtx_output_width,
        )
        if any(value is None for value in dimensions):
            raise ValueError(
                "RTX super resolution requires input and output height/width"
            )
        input_height, input_width, output_height, output_width = (
            int(value) for value in dimensions
        )
        self._validate_nvidia_vsr_dimensions(
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
        )

        rgb = self._nvidia_vsr.enhance(
            rgb,
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
            quality=options.rtx_quality,
        )
        if alpha is not None and tuple(alpha.shape[-2:]) != (output_height, output_width):
            alpha = torch_functional.interpolate(
                alpha,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return rgb, alpha

    def prepare_avatar(self, path: str | Path, *, avatar_id: str) -> Avatar:
        """Preprocess one portrait into an immutable GPU-resident avatar.

        Preparation is deliberately separate from registry mutation so callers
        can publish an avatar only after every detector / landmark / appearance
        model has completed successfully.
        """
        if self._avatar_loader is None:
            raise RuntimeError("This pipeline was created without an AvatarLoader")
        return self._avatar_loader.load(path, avatar_id=avatar_id)

    def prepare_background(self, path: str | Path) -> torch.Tensor:
        """Decode and upload one background at the pipeline output size."""
        out_h, out_w = self._out_size
        return load_background(path, out_h, out_w)

    def register_background(self, background_id: str, background: torch.Tensor) -> None:
        """Publish a prepared background for subsequent render requests."""
        if background_id == TRANSPARENT_BG_ID:
            raise ValueError(f"{TRANSPARENT_BG_ID!r} is a reserved background id")
        self._backgrounds[background_id] = background

    @classmethod
    def from_artifacts(
        cls,
        *,
        avatar_ids: list[str] | None = None,
        portraits_dir: Path | str | None = None,
        background_paths: dict[str, Path | str] | None = None,
        out_size: tuple[int, int] = (720, 1280),
        download_workers: int = 4,
        avtr1_backend: AVTR1Backend | None = None,
    ) -> tuple[Pipeline, dict[str, Avatar]]:
        """Build the Pipeline + avatar registry from local artifacts.

        Required model files are resolved lazily from HuggingFace when absent,
        then the full speech-to-motion + renderer stack is wired.  Explicit
        portrait or background paths are never replaced by a catalog-wide
        avatar download.

        ``auto`` prefers local TensorRT engines on every platform, including
        native Windows, and falls back to TorchScript/ONNX when the AVTR-1
        encode/decode engines are absent. Explicit ``tensorrt`` requires those
        engines to have been built locally first.

        Args:
            avatar_ids:       Portrait IDs to pre-load (stem of ``{id}.png`` in
                              ``portraits_dir``). Pass ``None`` to skip.
            portraits_dir:    Directory of ``{avatar_id}.png`` files. Defaults to
                              the ``reference_frames`` artifact from HuggingFace.
            background_paths: ``{bg_id: png_path}`` mapping. When ``None`` all
                              PNGs from the ``backgrounds`` artifact are used.
            out_size:         ``(H, W)`` the pipeline operates at.
            download_workers: Retained for API compatibility. Missing model
                              artifacts are fetched on demand by their loaders.
            avtr1_backend:    ``auto`` uses TensorRT when local engines exist,
                              otherwise TorchScript/ONNX.
        """
        from avtr1_renderer.avatar_loader import AvatarLoader

        out_h, out_w = out_size

        # --- Resolve only the artifacts this configuration actually needs ---
        mgr = get_artifact_manager()

        # --- Backgrounds -----------------------------------------------------
        if background_paths is None:
            bg_dir = mgr.get_artifact_path("backgrounds")
            background_paths = {p.stem: p for p in sorted(bg_dir.glob("*.png"))}
        backgrounds: dict[str, torch.Tensor] = {
            bg_id: load_background(path, out_h, out_w)
            for bg_id, path in background_paths.items()
        }
        backgrounds[TRANSPARENT_BG_ID] = torch.zeros(
            (1, 3, out_h, out_w), dtype=torch.float32, device="cuda"
        )

        # --- AVTR1 speech-to-motion backend ---------------------------------
        encode_path = get_trt_engine_path("avtr1_encode")
        decode_path = get_trt_engine_path("avtr1_decode")
        requested_backend = avtr1_backend or os.environ.get("AVTR1_BACKEND", "auto")
        resolved_backend = resolve_avtr1_backend(
            requested_backend,
            encode_path=encode_path,
            decode_path=decode_path,
        )
        encode, decode, normalizer = _load_avtr1_engines(
            resolved_backend,
            manager=mgr,
            encode_path=encode_path,
            decode_path=decode_path,
        )

        # --- Hubert: TRT if built, else ONNX --------------------------------
        hubert = load_engine(
            _resolve_hubert_path(resolved_backend, manager=mgr),
            HubertInput,
            HubertOutput,
        )

        # --- Renderer engines: TRT if built, else ONNX ----------------------
        decoder = load_engine(
            _resolve_component_path(
                resolved_backend,
                manager=mgr,
                engine_name="decoder",
                onnx_artifact="decoder_onnx",
            ),
            DecoderInput,
            DecoderOutput,
        )

        # The portable path converts the standard 5-D GridSample graph to
        # opset 20. TensorRT keeps using the plugin graph and platform plugin.
        warp_path = _resolve_warp_path(resolved_backend, manager=mgr)
        if warp_path.suffix.lower() in {".engine", ".trt", ".plan"}:
            plugin_path = _resolve_warp_plugin(manager=mgr)
            warp = load_engine(
                warp_path, WarpInput, WarpOutput,
                plugin_files=[str(plugin_path)],
            )
        else:
            warp = load_engine(warp_path, WarpInput, WarpOutput)

        stitch = load_engine(
            _resolve_dynamic_component_path(
                resolved_backend,
                manager=mgr,
                engine_name="stitch_network",
                onnx_artifact="stitch_network_onnx",
                prepare=prepare_portable_stitch_onnx,
            ),
            StitchInput,
            StitchOutput,
        )
        matting = load_engine(
            _resolve_dynamic_component_path(
                resolved_backend,
                manager=mgr,
                engine_name="modnet",
                onnx_artifact="modnet_onnx",
                prepare=prepare_portable_modnet_onnx,
            ),
            MODNetInput,
            MODNetOutput,
        )

        # --- AvatarLoader (ONNX, GPU-independent) ----------------------------
        mask_path = (
            mgr.get_artifact_path("pasteback_mask")
            if "pasteback_mask" in mgr._artifacts
            else None
        )
        loader = AvatarLoader(
            engine_files={
                "insightface_det": mgr.get_artifact_path("insightface_det"),
                "landmark106": mgr.get_artifact_path("landmark106"),
                "landmark203": mgr.get_artifact_path("landmark203"),
                "appearance_extractor": mgr.get_artifact_path("appearance_extractor"),
                "motion_extractor": mgr.get_artifact_path("motion_extractor"),
            },
            mask_template_path=mask_path,
            out_h=out_h,
            out_w=out_w,
            max_dim=max(out_h, out_w),
        )

        # --- Avatar registry -------------------------------------------------
        if portraits_dir is None:
            portraits_dir = mgr.get_artifact_path("reference_frames")
        portraits_dir = Path(portraits_dir)

        registry: dict[str, Avatar] = {}
        for avatar_id in (avatar_ids or []):
            portrait = portraits_dir / f"{avatar_id}.png"
            if not portrait.is_file():
                raise FileNotFoundError(f"No portrait at {portrait}")
            registry[avatar_id] = loader.load(portrait, avatar_id=avatar_id)

        # --- Motion generator ------------------------------------------------
        motion_generator = AVTR1MotionGenerator(
            hubert=hubert,
            encode_engine=encode,
            decode_engine=decode,
            normalizer=normalizer,
        )
        return (
            cls(
                motion_generator=motion_generator,
                stitch=stitch,
                warp=warp,
                decoder=decoder,
                matting=matting,
                backgrounds=backgrounds,
                avatar_loader=loader,
                out_size=out_size,
            ),
            registry,
        )

    def initial_state(self, avatar: Avatar) -> StateT:
        return self._motion_generator.initial_state(avatar)

    def _generate_motions(
        self,
        avatar: Avatar,
        chunk: Chunk,
        state: StateT | None,
        options: RenderOptions,
    ):
        if state is None:
            state = self._motion_generator.initial_state(avatar)
        mg = self._motion_generator
        expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        got_len = len(chunk.audio_speech)
        if got_len != expected_len:
            raise ValueError(
                f"Chunk audio length mismatch: expected {expected_len} samples "
                f"({expected_len / 16000 * 1000:.0f} ms at 16 kHz), got {got_len}. "
                f"Use Pipeline helpers (_chunk_window / _chunk_step) or the "
                f"slice_chunks utility in scripts/generate_offline.py."
            )
        return self._motion_generator.generate_chunk(chunk, avatar, state, options)

    def process_transparent_rgba_chunk(
        self,
        avatar: Avatar,
        chunk: Chunk,
        state: StateT | None,
        options: RenderOptions | None = None,
    ) -> tuple[StateT, Iterator[np.ndarray]]:
        """Render one chunk as straight RGBA without passing through YUV.

        This is an internal asset-generation path, not a WebRTC pixel format.
        Keeping it here lets idle-loop generation reuse the exact live motion,
        warp, decoder, pasteback and matting stack without damaging fine alpha
        edges through an avoidable RGB -> I420 -> RGB round trip.
        """

        if options is None:
            options = RenderOptions()
        motions, next_state = self._generate_motions(avatar, chunk, state, options)
        transparent = self._backgrounds[TRANSPARENT_BG_ID]

        def rgba_frames() -> Iterator[np.ndarray]:
            stream = render_chunk_streaming(
                motions,
                avatar,
                transparent,
                stitch=self._stitch,
                warp=self._warp,
                decoder=self._decoder,
                matting=self._matting,
            )
            for premultiplied_rgb, alpha in stream:
                rgba = _straight_rgba_uint8(premultiplied_rgb, alpha)
                yield rgba[0].cpu().numpy()

        return next_state, rgba_frames()

    def process_chunk(
        self,
        avatar: Avatar,
        chunk: Chunk,
        state: StateT | None,
        options: RenderOptions | None = None,
    ) -> tuple[StateT, FrameIterator]:
        """Run one streaming chunk end-to-end.

        Pass ``state=None`` for the cold-start call — the pipeline builds
        the initial state for ``avatar`` itself.
        """
        if options is None:
            options = RenderOptions()
        if options.bg_id is None:
            raise ValueError("RenderOptions.bg_id is required (no implicit default)")
        try:
            bg = self._backgrounds[options.bg_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown bg_id {options.bg_id!r}; registered: {sorted(self._backgrounds)}"
            ) from exc
        output_height = (
            int(bg.shape[-2]) if options.output_height is None else options.output_height
        )
        output_width = int(bg.shape[-1]) if options.output_width is None else options.output_width
        motions, next_state = self._generate_motions(avatar, chunk, state, options)

        def frames_streaming() -> FrameIterator:
            stream = render_chunk_streaming(
                motions, avatar, bg,
                stitch=self._stitch,
                warp=self._warp,
                decoder=self._decoder,
                matting=self._matting,
            )
            for rgb, alpha in stream:
                rgb, alpha = _center_crop_output(
                    rgb,
                    alpha,
                    output_height=output_height,
                    output_width=output_width,
                )
                rgb, alpha = self._apply_nvidia_vsr(rgb, alpha, options)
                packed = pack_frames(rgb, alpha, pixel_format=options.pixel_format)
                yield packed[0]

        def frames_batched() -> FrameIterator:
            rgb, alpha = render_chunk(
                motions, avatar, bg,
                stitch=self._stitch,
                warp=self._warp,
                decoder=self._decoder,
                matting=self._matting,
            )
            rgb, alpha = _center_crop_output(
                rgb,
                alpha,
                output_height=output_height,
                output_width=output_width,
            )
            rgb, alpha = self._apply_nvidia_vsr(rgb, alpha, options)
            yield from pack_frames(rgb, alpha, pixel_format=options.pixel_format)

        frames = frames_streaming if options.stream_frames else frames_batched
        return next_state, frames()


__all__ = ["TRANSPARENT_BG_ID", "Pipeline"]
