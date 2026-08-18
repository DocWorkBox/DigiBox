from __future__ import annotations

import numpy as np
import torch

from avtr1_renderer import avatar_loader as subject


def _avatar_for_preview(
    source: torch.Tensor,
    source_alpha: torch.Tensor | None,
) -> subject.Avatar:
    return subject.Avatar(
        id="preview",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.empty(0),
        mask=torch.empty(0),
        source=source,
        source_alpha=source_alpha,
    )


def test_avatar_preview_png_preserves_straight_rgb_and_alpha() -> None:
    encode = getattr(subject, "encode_avatar_preview_png", None)
    assert callable(encode), "avatar preview PNG encoder is not implemented"
    avatar = _avatar_for_preview(
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.5]]]),
        torch.tensor([[[0.25, 1.0]]]),
    )

    decoded = subject.cv2.imdecode(
        np.frombuffer(encode(avatar), np.uint8),
        subject.cv2.IMREAD_UNCHANGED,
    )

    assert decoded.shape == (1, 2, 4)
    assert decoded[0, 0].tolist() == [0, 0, 255, 64]
    assert decoded[0, 1].tolist() == [128, 255, 0, 255]


def test_avatar_preview_png_without_alpha_is_rgb() -> None:
    encode = getattr(subject, "encode_avatar_preview_png", None)
    assert callable(encode), "avatar preview PNG encoder is not implemented"
    avatar = _avatar_for_preview(
        torch.tensor([[[10 / 255]], [[20 / 255]], [[30 / 255]]]),
        None,
    )

    decoded = subject.cv2.imdecode(
        np.frombuffer(encode(avatar), np.uint8),
        subject.cv2.IMREAD_UNCHANGED,
    )

    assert decoded.shape == (1, 1, 3)
    assert decoded[0, 0].tolist() == [30, 20, 10]


def test_rgba_portrait_keeps_uncomposited_rgb_and_source_alpha() -> None:
    prepare = getattr(subject, "_prepare_portrait_layers", None)
    assert callable(prepare), "portrait layer preparation is not implemented"

    rgba = np.array(
        [
            [[10, 20, 30, 0], [40, 50, 60, 128]],
            [[70, 80, 90, 255], [100, 110, 120, 64]],
        ],
        dtype=np.uint8,
    )

    registration, source_rgb, source_alpha, no_matting = prepare(rgba)

    expected_alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    expected_registration = (
        rgba[:, :, :3].astype(np.float64) * expected_alpha[:, :, None]
        + 200.0 * (1.0 - expected_alpha[:, :, None])
    )
    assert np.array_equal(source_rgb, rgba[:, :, :3])
    assert np.allclose(source_alpha, expected_alpha)
    assert np.allclose(registration, expected_registration)
    assert no_matting is False


def test_opaque_rgba_keeps_modnet_fallback_instead_of_masking_as_a_rectangle() -> None:
    rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    rgba[:, :, :3] = [25, 50, 75]

    registration, source_rgb, source_alpha, no_matting = (
        subject._prepare_portrait_layers(rgba)
    )

    assert np.array_equal(registration, rgba[:, :, :3].astype(np.float64))
    assert np.array_equal(source_rgb, rgba[:, :, :3])
    assert source_alpha is None
    assert no_matting is False


def test_padded_opaque_rgba_keeps_modnet_fallback_for_legacy_assets() -> None:
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[1:5, 2:6, :3] = [25, 50, 75]
    rgba[1:5, 2:6, 3] = 255

    _registration, _source_rgb, source_alpha, no_matting = (
        subject._prepare_portrait_layers(rgba)
    )

    assert source_alpha is None
    assert no_matting is False


def test_clean_source_alpha_removes_specks_without_rectangular_bbox_fade() -> None:
    alpha = np.zeros((10, 12), dtype=np.float32)
    alpha[2:10, 2:9] = 1.0
    alpha[1, 10] = 1.0

    cleaned = subject._clean_source_alpha(
        alpha,
        min_component_area=3,
    )

    assert cleaned[1, 10] == 0.0
    assert np.array_equal(cleaned[2:10, 2:9], np.ones((8, 7), dtype=np.float32))


def test_build_alpha_motion_mask_only_opens_a_band_around_the_real_contour() -> None:
    build = getattr(subject, "_build_alpha_motion_mask", None)
    assert callable(build), "alpha motion-mask construction is not implemented"

    alpha = np.zeros((7, 7), dtype=np.float32)
    alpha[2:5, 2:5] = 1.0
    motion = build(alpha, radius=1)

    assert motion[3, 3] == 0.0  # deep foreground remains the trusted static prior
    assert motion[2, 3] > 0.0  # inner contour can disappear as the head moves
    assert motion[1, 3] > 0.0  # outer contour can appear as the head moves
    assert motion[0, 0] == 0.0  # distant background can never become foreground


def test_default_source_crop_scale_matches_reference_and_contains_full_hair() -> None:
    assert subject.CropConfig().scale == 2.6
