from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from avtr1_renderer.api import app as subject


def _encode(image: np.ndarray, suffix: str = ".png") -> bytes:
    ok, encoded = cv2.imencode(suffix, image)
    assert ok
    return encoded.tobytes()


def _fixture_cutout(image: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255
    rgba[:8, :, 3] = 0
    return rgba


def test_user_assets_use_writable_overlay_and_migrate_legacy_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "immutable-artifacts"
    writable = tmp_path / "writable" / "user-assets"
    monkeypatch.setenv("AVTR1_LOCAL_STORAGE", str(storage))
    monkeypatch.setenv("AVTR1_USER_ASSETS_ROOT", str(writable))

    from avtr1_renderer.avtr1_artifact_manager import get_storage_root

    legacy_avatar = get_storage_root() / "user_assets" / "reference_frames" / "user_legacy.png"
    legacy_avatar.parent.mkdir(parents=True)
    legacy_avatar.write_bytes(b"legacy-avatar")

    assert subject._user_assets_root() == writable
    migrated_avatar = writable / "reference_frames" / legacy_avatar.name
    assert migrated_avatar.read_bytes() == b"legacy-avatar"

    # A deletion in the writable store must survive restart.  The immutable
    # legacy copy is a one-time migration source, never a live fallback.
    migrated_avatar.unlink()
    assert subject._user_assets_root() == writable
    assert not migrated_avatar.exists()


def test_user_asset_migration_never_overwrites_existing_writable_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "immutable-artifacts"
    writable = tmp_path / "writable" / "user-assets"
    monkeypatch.setenv("AVTR1_LOCAL_STORAGE", str(storage))
    monkeypatch.setenv("AVTR1_USER_ASSETS_ROOT", str(writable))

    from avtr1_renderer.avtr1_artifact_manager import get_storage_root

    relative = Path("backgrounds") / "user_existing.png"
    legacy = get_storage_root() / "user_assets" / relative
    current = writable / relative
    legacy.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    current.write_bytes(b"current")

    assert subject._user_assets_root() == writable
    assert current.read_bytes() == b"current"


def test_bundled_background_paths_include_every_theme_asset() -> None:
    paths = subject._bundled_background_paths()
    expected_ids = {
        "tech_particles_dark",
        "theme_aurora",
        "theme_winter_hearth",
        "theme_romantic",
        "theme_cozy_cabin",
        "theme_pearl",
        "theme_cyberspace",
        "theme_rainforest",
    }

    assert set(paths) == expected_ids
    for background_id, path in paths.items():
        assert Path(path).is_file(), f"missing bundled background for {background_id}: {path}"


def test_pipeline_does_not_eagerly_download_unused_avatar_artifacts() -> None:
    """A packaged renderer supplies its own backgrounds and portraits.

    ``Pipeline.from_artifacts`` must let the concrete paths bypass the optional
    HuggingFace avatar artifacts.  Eagerly ensuring the complete catalog makes
    every freshly extracted portable Runtime fetch the default backgrounds
    before the renderer can become healthy.
    """
    from avtr1_renderer import pipeline

    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    factory = source.split("    def from_artifacts(", 1)[1].split(
        "    def __init__(", 1
    )[0]

    assert "mgr.ensure_all_artifacts" not in factory
    assert 'mgr.get_artifact_path("backgrounds")' in factory


def test_avatar_upload_is_safe_rgba_png_at_renderer_resolution() -> None:
    image = np.full((360, 640, 3), 127, dtype=np.uint8)

    asset = subject._normalise_user_image(
        _encode(image, ".jpg"),
        filename=r"..\..\我的 人物.jpg",
        kind="avatar",
        cutout=_fixture_cutout,
    )

    decoded = cv2.imdecode(np.frombuffer(asset.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (720, 1280, 4)
    assert asset.asset_id.startswith("user_avatar_")
    assert ".." not in asset.asset_id
    assert "/" not in asset.asset_id
    assert "\\" not in asset.asset_id


def test_background_upload_is_safe_rgb_png_at_renderer_resolution() -> None:
    image = np.zeros((300, 500, 4), dtype=np.uint8)

    asset = subject._normalise_user_image(
        _encode(image),
        filename="studio.webp",
        kind="background",
    )

    decoded = cv2.imdecode(np.frombuffer(asset.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (720, 1280, 3)
    assert asset.asset_id.startswith("user_studio_")


def test_rgba_contain_resize_preserves_straight_edge_colour() -> None:
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[:, 0, 2] = 255
    image[:, 0, 3] = 255

    resized = subject._contain_image(image, channels=4)
    alpha = resized[:, :, 3]
    soft_edge = (alpha > 0) & (alpha < 255)

    assert np.any(soft_edge)
    assert np.all(resized[:, :, 2][soft_edge] >= 254)
    assert np.all(resized[:, :, :2][soft_edge] == 0)


def test_cutout_feather_stays_inside_the_original_alpha_support() -> None:
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[8:24, 8:24, :3] = 200
    image[8:24, 8:24, 3] = 255

    feathered = subject._feather_cutout_alpha(image)

    original_alpha = image[:, :, 3]
    assert np.all(feathered[:, :, 3][original_alpha == 0] == 0)
    assert 0 < feathered[8, 16, 3] < 255
    assert feathered[16, 16, 3] == 255


def test_cutout_edge_refinement_removes_white_spill_and_contracts_matte() -> None:
    height = width = 64
    yy, xx = np.indices((height, width))
    signed_distance = 19.0 - np.sqrt((xx - 32.0) ** 2 + (yy - 32.0) ** 2)
    alpha = np.clip((signed_distance + 2.0) / 4.0, 0.0, 1.0)
    foreground = np.array([28.0, 38.0, 48.0], dtype=np.float32)  # BGR
    old_white_background = np.array([245.0, 245.0, 245.0], dtype=np.float32)
    observed = (
        foreground[None, None, :] * alpha[:, :, None]
        + old_white_background[None, None, :] * (1.0 - alpha[:, :, None])
    )
    cutout = np.concatenate(
        (
            np.rint(observed).astype(np.uint8),
            np.rint(alpha * 255.0).astype(np.uint8)[:, :, None],
        ),
        axis=2,
    )

    refined = subject._refine_cutout_edges(cutout)

    refined_alpha = refined[:, :, 3].astype(np.float32) / 255.0
    new_background = np.array([120.0, 30.0, 4.0], dtype=np.float32)
    before = (
        observed * refined_alpha[:, :, None]
        + new_background[None, None, :] * (1.0 - refined_alpha[:, :, None])
    )
    after = (
        refined[:, :, :3].astype(np.float32) * refined_alpha[:, :, None]
        + new_background[None, None, :] * (1.0 - refined_alpha[:, :, None])
    )
    expected = (
        foreground[None, None, :] * refined_alpha[:, :, None]
        + new_background[None, None, :] * (1.0 - refined_alpha[:, :, None])
    )
    visible_edge = (refined_alpha > 0.02) & (refined_alpha < 0.98)
    before_error = np.abs(before - expected)[visible_edge].mean()
    after_error = np.abs(after - expected)[visible_edge].mean()

    # Projection-guided propagation must remove at least half of the visible
    # white-spill error without the black clipping caused by direct unmatting.
    assert after_error < before_error * 0.50
    assert np.all(refined[:, :, 3][cutout[:, :, 3] == 0] == 0)
    assert np.count_nonzero(refined[:, :, 3]) < np.count_nonzero(cutout[:, :, 3])
    assert np.array_equal(refined[32, 32, :3], cutout[32, 32, :3])


def test_cutout_edge_refinement_does_not_turn_semantic_alpha_into_black_halo() -> None:
    height = width = 72
    yy, xx = np.indices((height, width))
    distance = np.sqrt((xx - 36.0) ** 2 + (yy - 36.0) ** 2)
    alpha = np.clip((25.0 - distance) / 8.0, 0.0, 1.0)
    foreground = np.array([30.0, 42.0, 55.0], dtype=np.float32)
    white = np.array([248.0, 248.0, 248.0], dtype=np.float32)
    # The segmentation matte can be uncertain around a real dark strand even
    # when that strand's RGB is already clean. Treating semantic alpha as exact
    # optical coverage would subtract white and clip this valid colour to black.
    observed = np.broadcast_to(foreground, (height, width, 3)).copy()
    observed[alpha == 0.0] = white
    cutout = np.concatenate(
        (
            np.rint(observed).astype(np.uint8),
            np.rint(alpha * 255.0).astype(np.uint8)[:, :, None],
        ),
        axis=2,
    )

    refined = subject._refine_cutout_edges(cutout)

    visible_edge = (refined[:, :, 3] > 8) & (cutout[:, :, 3] < 240)
    assert np.any(visible_edge)
    assert np.max(
        np.abs(
            refined[:, :, :3][visible_edge].astype(np.int16)
            - np.rint(foreground).astype(np.int16)
        )
    ) <= 1


@pytest.mark.parametrize("case", ["empty", "garbage", "oversized"])
def test_invalid_or_oversized_upload_is_rejected(case: str) -> None:
    payload = {
        "empty": b"",
        "garbage": b"not-an-image",
        "oversized": b"x" * (12 * 1024 * 1024 + 1),
    }[case]
    with pytest.raises(HTTPException) as raised:
        subject._normalise_user_image(payload, filename="bad.png", kind="avatar")

    assert raised.value.status_code == 400


def test_grayscale_avatar_is_rejected() -> None:
    image = np.full((256, 256), 127, dtype=np.uint8)

    with pytest.raises(HTTPException, match="彩色"):
        subject._normalise_user_image(
            _encode(image),
            filename="gray.png",
            kind="avatar",
        )


class _FakePipeline:
    def __init__(self, *, fail_avatar: bool = False) -> None:
        self.fail_avatar = fail_avatar
        self._backgrounds: dict[str, object] = {}

    def prepare_avatar(self, path: Path, *, avatar_id: str) -> object:
        assert path.is_file()
        if self.fail_avatar:
            raise RuntimeError("No face detected")
        return {"id": avatar_id}

    def prepare_background(self, path: Path) -> object:
        assert path.is_file()
        return {"path": path.name}

    def register_background(self, background_id: str, background: object) -> None:
        self._backgrounds[background_id] = background


def _asset(kind: str) -> object:
    channels = 4 if kind == "avatar" else 3
    image = np.zeros((720, 1280, channels), dtype=np.uint8)
    return subject._normalise_user_image(
        _encode(image),
        filename=f"sample-{kind}.png",
        kind=kind,
        cutout=_fixture_cutout if kind == "avatar" else None,
    )


def test_avatar_install_is_atomic_and_idempotent(tmp_path: Path) -> None:
    pipeline = _FakePipeline()
    registry: dict[str, object] = {}
    asset = _asset("avatar")

    first = subject._install_user_asset(
        asset,
        pipeline=pipeline,
        registry=registry,
        user_root=tmp_path,
    )
    second = subject._install_user_asset(
        asset,
        pipeline=pipeline,
        registry=registry,
        user_root=tmp_path,
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert asset.asset_id in registry
    assert (tmp_path / "reference_frames" / f"{asset.asset_id}.png").is_file()


def test_failed_avatar_install_leaves_no_file_or_registry_entry(tmp_path: Path) -> None:
    pipeline = _FakePipeline(fail_avatar=True)
    registry: dict[str, object] = {}
    asset = _asset("avatar")

    with pytest.raises(RuntimeError, match="No face"):
        subject._install_user_asset(
            asset,
            pipeline=pipeline,
            registry=registry,
            user_root=tmp_path,
        )

    assert registry == {}
    assert list(tmp_path.rglob("*")) in ([tmp_path / "reference_frames"], [])


def test_background_install_registers_and_persists(tmp_path: Path) -> None:
    pipeline = _FakePipeline()
    registry: dict[str, object] = {}
    asset = _asset("background")

    result = subject._install_user_asset(
        asset,
        pipeline=pipeline,
        registry=registry,
        user_root=tmp_path,
    )

    assert result["kind"] == "background"
    assert asset.asset_id in pipeline._backgrounds
    assert (tmp_path / "backgrounds" / f"{asset.asset_id}.png").is_file()


def test_renderer_startup_allows_zero_presets_and_then_restores_user_avatars() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    lifespan = source.split("async def lifespan", 1)[1].split("\n\n@app.", 1)[0]

    assert "if not avatar_ids:" not in lifespan
    assert lifespan.index("Pipeline.from_artifacts(") < lifespan.index("_restore_user_assets(")
