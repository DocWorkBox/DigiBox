# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""``Avatar`` plus the loader that produces it from a portrait PNG.

Owns its own TRT engines and runs the full registration cascade:

    PNG --(load + alpha-composite over WHITE)-->
        face detect (insightface SCRFD)         --> bbox
        landmark106 (from bbox)                 --> 106 lmk
        crop to 224 (scale=1.5, vy_ratio=-0.1)  --> img_crop_224, M_c2o_224
        landmark203 on the 224 crop             --> 203 lmk in original frame
        crop to 512 (scale=2.6, vy_ratio=-0.1, rot)
                                                --> img_crop_512, M_c2o
        appearance extractor on the 256 crop    --> f_s
        motion extractor on the 256 crop        --> KPInfo
        mask warp via M_c2o                     --> per-avatar mask

The portrait is always composited over white -- per-frame MODNet matting
in the renderer recovers an alpha channel so the caller can composite
the rendered face over any background. This mirrors the reference's
``with_alpha=True`` path (its ``img_rgb_green_lst`` / ``f_s_green_lst``).

The crop scale/vy values come from the reference's inference config
(where ``crop_scale=2.6`` and ``crop_vy_ratio=-0.1`` are set for the
source crop).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from avtr1_renderer.components.face_detection import detect_faces
from avtr1_renderer.components.face_landmarks import landmark106, landmark203
from avtr1_renderer.components.liveportrait.motion_extractor import extract_motion
from avtr1_renderer.components.source_crop import crop_image, get_default_mask
from avtr1_renderer.models.appearance_extractor import (
    AEInput,
    AEOutput,
    AppearanceExtractorEngine,
)
from avtr1_renderer.models.face_detection import FaceDetEngine, FaceDetInput, FaceDetOutput
from avtr1_renderer.models.landmark106 import Lm106Engine, Lm106Input, Lm106Output
from avtr1_renderer.models.landmark203 import Lm203Engine, Lm203Input, Lm203Output
from avtr1_renderer.models.motion_extractor import (
    MotionExtractorEngine,
    MotionInput,
    MotionOutput,
)
from avtr1_renderer.runtime import load_engine
from avtr1_renderer.types import KPInfo

# Constants previously held on the wrapper classes; the cascade below
# uses them directly for the 224 crop (lm203's input size).
_LM203_DSIZE = 224


@dataclass(slots=True, frozen=True)
class Avatar:
    """Per-portrait, immutable. Produced by ``AvatarLoader.load``.

    Lives entirely on CUDA. The ``id`` is the caller-facing handle used to
    key the per-session ``State`` (the AVTR1 autoregressive memory is
    avatar-specific, so a state blob is meaningful only with its avatar).

    Backgrounds are *not* on the avatar -- they live in a separate
    registry on the ``Pipeline`` so callers can pick one per request.
    ``Pipeline.process_chunk`` looks the bg up by id and passes it
    through to the renderer.
    """

    id: str
    kp_info: KPInfo
    f_s: torch.Tensor  # (1, 32, 16, 64, 64) appearance feature volume
    M_grid: torch.Tensor  # (2, 3)            precomputed F.affine_grid input
    #                                          (inverse of the normalised
    #                                          crop->frame affine). Caches the
    #                                          ~40-kernel LU factorisation
    #                                          kornia.warp_affine would do at
    #                                          every putback; rebuilt only on
    #                                          avatar load.
    mask: torch.Tensor  # (1, H, W)           pasteback mask (pre-warped)
    source: torch.Tensor  # (3, H, W)         straight source RGB in [0, 1]
    no_matting: bool = False  # skip MODNet; source carries the real background
    source_alpha: torch.Tensor | None = None  # (1, H, W) trusted source-frame matte
    matting_source: torch.Tensor | None = None  # (3, H, W) neutral-bg MODNet fixture
    alpha_motion_mask: torch.Tensor | None = None  # (1, H, W) contour update band


def encode_avatar_preview_png(avatar: Avatar) -> bytes:
    """Encode an avatar's selected source layers as a browser-ready PNG.

    This intentionally uses the immutable source portrait, not a generated
    frame, so the settings panel previews exactly the person currently
    selected.  Trusted upload alpha remains straight alpha in the PNG.
    """

    source = avatar.source.detach().float().cpu().clamp(0.0, 1.0)
    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError(f"avatar source must have shape (3, H, W), got {tuple(source.shape)}")
    rgb = np.rint(source.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    if avatar.source_alpha is not None:
        alpha = avatar.source_alpha.detach().float().cpu().clamp(0.0, 1.0)
        if alpha.ndim != 3 or alpha.shape[0] != 1:
            raise ValueError(
                f"avatar source alpha must have shape (1, H, W), got {tuple(alpha.shape)}"
            )
        alpha_hwc = np.rint(alpha.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if alpha_hwc.shape[:2] != rgb.shape[:2]:
            raise ValueError("avatar source RGB and alpha dimensions do not match")
        pixels = cv2.cvtColor(
            np.concatenate((rgb, alpha_hwc), axis=2),
            cv2.COLOR_RGBA2BGRA,
        )
    else:
        pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    encoded, png = cv2.imencode(".png", pixels)
    if not encoded:
        raise RuntimeError("failed to encode avatar preview PNG")
    return png.tobytes()


@dataclass(slots=True, frozen=True)
class CropConfig:
    """Crop knobs taken from the reference config."""

    scale: float = 2.6
    vx_ratio: float = 0.0
    vy_ratio: float = -0.1
    flag_do_rot: bool = True


def _load_image_rgb(path: Path) -> tuple[np.ndarray, int, int]:
    """Load an image as (H, W, C) RGB(A) uint8.

    Mirrors the reference's ``load_image``: reads with cv2, converts to RGB
    (or RGBA when the source has 4 channels), and returns the original
    (h, w) so downstream code can decide whether to resize to a max_dim.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    assert img.ndim == 3
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    return img, h, w


def _prepare_portrait_layers(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """Split registration pixels from the layers used at render time.

    RGBA portraits still use the reference grey-200 composite for face
    registration. Pasteback, however, must retain straight RGB plus the
    uploaded alpha so semi-transparent edges are not composited over grey
    twice and MODNet does not need to infer the silhouette again.
    """

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"portrait must be RGB or RGBA, got {img.shape}")

    source_rgb = np.ascontiguousarray(img[:, :, :3])
    if img.shape[2] == 3:
        return source_rgb.astype(np.float64), source_rgb, None, True

    alpha_u8 = img[:, :, 3]
    source_alpha = np.ascontiguousarray(alpha_u8.astype(np.float32) / 255.0)
    registration = (
        source_rgb.astype(np.float64) * source_alpha[:, :, None]
        + 200.0 * (1.0 - source_alpha[:, :, None])
    )
    nonzero = alpha_u8 > 0
    opaque_rectangle = False
    if np.any(nonzero) and np.all((alpha_u8 == 0) | (alpha_u8 == 255)):
        ys, xs = np.nonzero(nonzero)
        bbox_area = (int(ys.max()) - int(ys.min()) + 1) * (
            int(xs.max()) - int(xs.min()) + 1
        )
        opaque_rectangle = int(nonzero.sum()) == bbox_area
    if opaque_rectangle:
        # Legacy FeyNoBg fallback assets are an opaque source rectangle with
        # transparent contain-padding. They need the old MODNet path instead
        # of treating that rectangle as an authoritative person silhouette.
        return registration, source_rgb, None, False
    return registration, source_rgb, source_alpha, False


def _clean_source_alpha(
    alpha: np.ndarray,
    *,
    min_component_area: int = 64,
) -> np.ndarray:
    """Remove detached mask specks without changing the main silhouette.

    Upload-time feathering already follows the real FeyNoBg contour.  A
    previous version also faded the rectangular support bounding box here;
    that removed valid shoulders / flowers whenever the person touched a
    crop edge and created a fixed rectangular transparency ramp.
    """

    cleaned = np.clip(alpha.astype(np.float32, copy=True), 0.0, 1.0)
    support = cleaned > (1.0 / 255.0)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros_like(support)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
            keep |= labels == label
    cleaned[~keep] = 0.0

    return np.ascontiguousarray(cleaned)


def _build_alpha_motion_mask(
    alpha: np.ndarray,
    *,
    radius: int = 32,
) -> np.ndarray:
    """Return the contour band in which a generated pose may change alpha.

    Deep foreground remains the trusted source matte, distant background can
    never become opaque, and only a local band on both sides of the real
    silhouette is delegated to per-frame matting.  This makes the head outline
    dynamic without allowing a bad MODNet frame to erase the whole person.
    """

    if radius < 1:
        raise ValueError("alpha motion radius must be positive")
    support = (np.clip(alpha, 0.0, 1.0) > (1.0 / 255.0)).astype(np.uint8)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    outer = cv2.dilate(support, kernel, iterations=1)
    inner = cv2.erode(support, kernel, iterations=1)
    return np.ascontiguousarray((outer.astype(np.float32) - inner).clip(0.0, 1.0))


def _check_resize(h: int, w: int, max_dim: int = 1920, division: int = 2) -> tuple[int, int, bool]:
    rsz = False
    if max_dim > 0 and max(h, w) > max_dim:
        rsz = True
        if h > w:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        else:
            new_w = max_dim
            new_h = int(h * max_dim / w)
    else:
        new_h, new_w = h, w
    if new_h % division != 0:
        new_h -= new_h % division
        rsz = True
    if new_w % division != 0:
        new_w -= new_w % division
        rsz = True
    return new_h, new_w, rsz


def _img_crop_to_bchw256(img_crop: np.ndarray) -> np.ndarray:
    """Resize a 512 (or arbitrary) crop to a (1, 3, 256, 256) float32 [0, 1]
    blob. Uses cv2.INTER_AREA -- same as the reference -- because we're
    downsampling."""
    rgb = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_AREA)
    rgb_bchw = (rgb.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(rgb_bchw)


def _kornia_warp_affine(target: torch.Tensor, M_c2o: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Warp ``target`` by the (1, 2, 3) affine ``M_c2o[:2, :]``.

    Uses ``torch.nn.functional.grid_sample`` with bilinear sampling and
    ``align_corners=False`` to match Kornia's ``warp_affine``. We avoid
    importing kornia here because we only need the one operation.

    ``target`` shape: (1, C, H_in, W_in). Returns (C, h, w).
    """
    if M_c2o.ndim == 2:
        M = M_c2o[:2, :].unsqueeze(0)  # (1, 2, 3)
    else:
        M = M_c2o
        if M.shape[-2:] == (3, 3):
            M = M[..., :2, :]
        if M.ndim == 2:
            M = M.unsqueeze(0)
    M = M.to(dtype=target.dtype)

    # Convert M (output->input pixel) into a normalized inverse for
    # grid_sample. grid_sample expects the inverse map (output coords ->
    # source coords) in normalised [-1, 1] coordinates.
    # Following kornia's warp_affine with align_corners=False, the
    # transform between pixel and normalized-coordinates frames matters,
    # so we just defer to kornia for parity.
    import kornia.geometry.transform as KGT

    out = KGT.warp_affine(
        target,
        M,
        dsize=(h, w),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return out.squeeze(0)


class AvatarLoader:
    """Build ``Avatar`` instances from portrait PNGs.

    Owns five engines (face det / landmark106 / landmark203 / appearance /
    motion). The default ``engines_dir`` points at the portable ``.onnx``
    files; registration runs once per portrait so the per-call latency
    isn't critical and the ONNX path skips the TRT build step. Pass an
    ``.engine`` directory if you have one and want the speedup.

    Construct once per process and reuse across portraits.
    """

    def __init__(
        self,
        *,
        engines_dir: str | Path | None = None,
        engine_files: dict[str, Path] | None = None,
        mask_template_path: str | Path | None = None,
        out_h: int = 720,
        out_w: int = 1280,
        max_dim: int = 1280,
        crop_cfg: CropConfig | None = None,
        engine_suffix: str = ".onnx",
    ) -> None:
        if engines_dir is None and engine_files is None:
            raise ValueError("provide either engines_dir or engine_files")
        if engines_dir is not None and engine_files is not None:
            raise ValueError("provide either engines_dir or engine_files, not both")

        if crop_cfg is None:
            crop_cfg = CropConfig()

        if engine_files is not None:
            # Explicit per-artifact paths — used when loading from ArtifactManager.
            self.det: FaceDetEngine = load_engine(
                engine_files["insightface_det"], FaceDetInput, FaceDetOutput
            )
            self.lmk106: Lm106Engine = load_engine(
                engine_files["landmark106"], Lm106Input, Lm106Output
            )
            self.lmk203: Lm203Engine = load_engine(
                engine_files["landmark203"], Lm203Input, Lm203Output
            )
            self.appearance: AppearanceExtractorEngine = load_engine(
                engine_files["appearance_extractor"], AEInput, AEOutput
            )
            self.motion: MotionExtractorEngine = load_engine(
                engine_files["motion_extractor"], MotionInput, MotionOutput
            )
        else:
            engines_dir = Path(engines_dir)  # type: ignore[arg-type]
            s = engine_suffix
            # The ``insightface_det`` / ``landmark*`` ONNX files don't carry the
            # ``_fp16`` suffix; the TRT engines do. The ``_fpXX`` part is folded
            # into the suffix string (e.g. ``"_fp16.engine"``) so callers can
            # pick a different precision tag if needed.
            if s == ".onnx":
                det_name, lm106_name, lm203_name = (
                    "insightface_det",
                    "landmark106",
                    "landmark203",
                )
                appearance_name = "appearance_extractor"
                motion_name = "motion_extractor"
            else:
                det_name, lm106_name, lm203_name = (
                    "insightface_det_fp16",
                    "landmark106_fp16",
                    "landmark203_fp16",
                )
                appearance_name = "appearance_extractor_fp16"
                motion_name = "motion_extractor_fp32"
            self.det = load_engine(
                engines_dir / f"{det_name}{s}", FaceDetInput, FaceDetOutput
            )
            self.lmk106 = load_engine(
                engines_dir / f"{lm106_name}{s}", Lm106Input, Lm106Output
            )
            self.lmk203 = load_engine(
                engines_dir / f"{lm203_name}{s}", Lm203Input, Lm203Output
            )
            self.appearance = load_engine(
                engines_dir / f"{appearance_name}{s}", AEInput, AEOutput
            )
            self.motion = load_engine(
                engines_dir / f"{motion_name}{s}", MotionInput, MotionOutput
            )

        self.out_h = out_h
        self.out_w = out_w
        self.max_dim = max_dim
        self.crop_cfg = crop_cfg

        # Default mask template (used when no per-avatar .pbmask.png is found).
        if mask_template_path is None:
            mask = get_default_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, axis=2)
        else:
            mask_img = cv2.imread(str(mask_template_path), cv2.IMREAD_COLOR)
            if mask_img is None:
                raise FileNotFoundError(f"Mask not found: {mask_template_path}")
            mask = mask_img.astype(np.float32) / 255.0
        # (H, W) first channel only.
        self._default_mask_hw = np.ascontiguousarray(mask[:, :, 0])

    # ---- crop pipeline ----

    def _crop_512(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run det -> lm106 -> 224-crop -> lm203 -> 512-crop. Returns
        ``(img_crop_512, M_c2o, lmk203_in_orig)``."""
        det, _ = detect_faces(img_rgb, det=self.det)
        if len(det) == 0:
            raise RuntimeError("No face detected in source image.")
        # Pick the largest box.
        order = np.argsort(-(det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))
        bbox = det[order[0]]

        lmk106_pts = landmark106(img_rgb, bbox, lm106=self.lmk106)

        crop_224 = crop_image(
            img_rgb,
            lmk106_pts,
            dsize=_LM203_DSIZE,
            scale=1.5,
            vx_ratio=0.0,
            vy_ratio=-0.1,
            flag_do_rot=True,
        )
        lmk203_pts = landmark203(crop_224.img_crop, crop_224.M_c2o, lm203=self.lmk203)

        crop_512 = crop_image(
            img_rgb,
            lmk203_pts,
            dsize=512,
            scale=self.crop_cfg.scale,
            vx_ratio=self.crop_cfg.vx_ratio,
            vy_ratio=self.crop_cfg.vy_ratio,
            flag_do_rot=self.crop_cfg.flag_do_rot,
        )
        return crop_512.img_crop, crop_512.M_c2o, lmk203_pts

    # ---- public ----

    def load(self, portrait_path: str | Path, *, avatar_id: str) -> Avatar:
        portrait_path = Path(portrait_path)

        img, h, w = _load_image_rgb(portrait_path)
        new_h, new_w, rsz = _check_resize(h, w, self.max_dim)
        if rsz:
            img = cv2.resize(img, (new_w, new_h))
        h, w = img.shape[:2]
        if h != self.out_h or w != self.out_w:
            img = cv2.resize(img, (self.out_w, self.out_h))
            h, w = self.out_h, self.out_w

        # Auto-detect matting regime from channels:
        # 4-ch (RGBA) → transparent portrait → MODNet needed.
        # 3-ch (RGB)  → background baked in → skip MODNet, use source bg directly.
        img_bg, source_rgb, source_alpha, no_matting = _prepare_portrait_layers(img)

        # Run the crop cascade on the float64 composite -- same as reference.
        img_crop_512, M_c2o, _ = self._crop_512(img_bg)

        # 256 crop -> appearance + motion extractors.
        rgb_256 = _img_crop_to_bchw256(img_crop_512)
        rgb_256_t = torch.from_numpy(rgb_256).cuda()
        f_s = self.appearance(AEInput(image=rgb_256_t.contiguous())).pred
        kp_info = extract_motion(rgb_256_t, motion=self.motion)

        # Source frame: straight HWC RGB uint8 -> CHW float32 [0, 1].
        source_t = (
            (torch.from_numpy(source_rgb).cuda().float() / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        source_alpha_t = None
        matting_source_t = None
        source_alpha_clean: np.ndarray | None = None
        if source_alpha is not None:
            source_alpha_clean = _clean_source_alpha(source_alpha)
            source_alpha_t = (
                torch.from_numpy(source_alpha_clean).cuda().unsqueeze(0).contiguous()
            )
            matting_source = (
                source_rgb.astype(np.float32) * source_alpha_clean[:, :, None]
                + 200.0 * (1.0 - source_alpha_clean[:, :, None])
            )
            matting_source_t = (
                torch.from_numpy(np.ascontiguousarray(matting_source))
                .cuda()
                .div_(255.0)
                .permute(2, 0, 1)
                .contiguous()
            )

        # Per-avatar mask: look for {portrait_stem}.pbmask.png next to the portrait.
        pbmask_path = portrait_path.parent / f"{portrait_path.stem}.pbmask.png"
        if pbmask_path.is_file():
            pbmask_img = cv2.imread(str(pbmask_path), cv2.IMREAD_GRAYSCALE)
            if pbmask_img is None:
                raise FileNotFoundError(f"Mask file found but unreadable: {pbmask_path}")
            mask_hw = np.ascontiguousarray(pbmask_img.astype(np.float32) / 255.0)
        else:
            mask_hw = self._default_mask_hw

        # Pre-warp the mask template to the original frame using M_c2o.
        M_c2o_t = torch.from_numpy(M_c2o).cuda().to(torch.float32)
        mask_template_cuda = (
            torch.from_numpy(mask_hw).float().cuda().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, H_mask, W_mask)
        mask_warped = _kornia_warp_affine(
            mask_template_cuda, M_c2o_t, self.out_h, self.out_w
        ).clamp(0.0, 1.0)
        # mask_warped: (1, H, W)
        mask = mask_warped.contiguous()
        alpha_motion_mask_t = None
        if source_alpha_clean is not None:
            alpha_motion_mask = _build_alpha_motion_mask(source_alpha_clean)
            alpha_motion_mask_t = (
                torch.from_numpy(alpha_motion_mask)
                .cuda()
                .unsqueeze(0)
                .mul_(mask)
                .clamp_(0.0, 1.0)
                .contiguous()
            )

        # Precompute the matrix ``F.affine_grid`` will consume at putback
        # time. kornia.warp_affine builds this on every call -- the path
        # is ``M -> M_3x3 -> normalize_homography -> torch.inverse``,
        # where the inverse triggers a 40-kernel LU factorisation with
        # two D2H sync points (the cuBLAS info-checks for singularity).
        # M_c2o is constant per avatar, so we run that chain once here
        # and cache the resulting (2, 3) tensor.
        from kornia.geometry.conversions import convert_affinematrix_to_homography
        from kornia.geometry.transform.imgwarp import normalize_homography

        M_c2o_2x3 = M_c2o_t[:2, :].unsqueeze(0)  # (1, 2, 3)
        M_3x3 = convert_affinematrix_to_homography(M_c2o_2x3)  # (1, 3, 3)
        # The ``warp_affine`` source dim is the face crop's (H, W) = (512, 512);
        # the dest dim is the original frame's (H, W).
        dst_norm_trans_src_norm = normalize_homography(
            M_3x3, (512, 512), (h, w)
        )  # (1, 3, 3)
        src_norm_trans_dst_norm = torch.linalg.inv(dst_norm_trans_src_norm)
        M_grid = src_norm_trans_dst_norm[:, :2, :].squeeze(0).contiguous()  # (2, 3)

        return Avatar(
            id=avatar_id,
            kp_info=kp_info,
            f_s=f_s,
            M_grid=M_grid,
            mask=mask,
            source=source_t,
            source_alpha=source_alpha_t,
            matting_source=matting_source_t,
            alpha_motion_mask=alpha_motion_mask_t,
            no_matting=no_matting,
        )


__all__ = ["Avatar", "AvatarLoader", "CropConfig"]
