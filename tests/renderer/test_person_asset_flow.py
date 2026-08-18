from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from avtr1_renderer.api import app as subject

_PERSON_BACKGROUND_ID = "tech_particles_dark"


def _encode_rgb(width: int = 640, height: int = 480) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint16) % 256
    image[:, :, 1] = 83
    image[:, :, 2] = 191
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _encode_rgba(width: int = 640, height: int = 480) -> bytes:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint16) % 256
    image[:, :, 1] = 83
    image[:, :, 2] = 191
    image[:, :, 3] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _normalise_person(
    payload: bytes,
    *,
    preserve_background: bool,
    cutout: Callable[[np.ndarray], np.ndarray],
):
    signature = inspect.signature(subject._normalise_user_image)
    assert "preserve_background" in signature.parameters, (
        "person uploads do not expose preserve_background"
    )
    assert "cutout" in signature.parameters, "person cutout is not injectable"
    return subject._normalise_user_image(
        payload,
        filename="person.png",
        kind="avatar",
        preserve_background=preserve_background,
        cutout=cutout,
    )


class _FakePipeline:
    def __init__(self) -> None:
        self.prepared_channels: list[int] = []
        self._backgrounds: dict[str, object] = {}

    def prepare_avatar(self, path: Path, *, avatar_id: str) -> object:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        channels = image.shape[2]
        self.prepared_channels.append(channels)
        return {"id": avatar_id, "no_matting": channels == 3}


def _install(asset, tmp_path: Path) -> tuple[dict[str, str | bool], _FakePipeline]:
    pipeline = _FakePipeline()
    result = subject._install_user_asset(
        asset,
        pipeline=pipeline,
        registry={},
        user_root=tmp_path,
    )
    return result, pipeline


def test_preserving_person_background_produces_rgb_without_cutout(tmp_path: Path) -> None:
    cutout_calls = 0

    def forbidden_cutout(_image: np.ndarray) -> np.ndarray:
        nonlocal cutout_calls
        cutout_calls += 1
        raise AssertionError("cutout must not run when the original background is preserved")

    asset = _normalise_person(
        _encode_rgb(),
        preserve_background=True,
        cutout=forbidden_cutout,
    )
    decoded = cv2.imdecode(np.frombuffer(asset.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    result, pipeline = _install(asset, tmp_path)

    assert decoded.shape == (720, 1280, 3)
    assert cutout_calls == 0
    assert pipeline.prepared_channels == [3]
    assert result["background_id"] == _PERSON_BACKGROUND_ID


def test_fey_nobg_cutout_produces_rgba_and_keeps_fixed_background_id(
    tmp_path: Path,
) -> None:
    cutout_inputs: list[np.ndarray] = []

    def fake_fey_nobg(image: np.ndarray) -> np.ndarray:
        cutout_inputs.append(image.copy())
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 137
        return rgba

    asset = _normalise_person(
        _encode_rgb(),
        preserve_background=False,
        cutout=fake_fey_nobg,
    )
    decoded = cv2.imdecode(np.frombuffer(asset.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    result, pipeline = _install(asset, tmp_path)

    assert len(cutout_inputs) == 1
    assert cutout_inputs[0].shape == (480, 640, 3)
    assert decoded.shape == (720, 1280, 4)
    assert np.any(decoded[:, :, 3] < 255)
    assert pipeline.prepared_channels == [4]
    assert result["background_id"] == _PERSON_BACKGROUND_ID


def test_opaque_rgba_person_still_runs_fey_nobg_cutout(tmp_path: Path) -> None:
    """A PNG alpha channel does not prove that its background was removed."""

    cutout_inputs: list[np.ndarray] = []

    def fake_fey_nobg(image: np.ndarray) -> np.ndarray:
        cutout_inputs.append(image.copy())
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 173
        return rgba

    asset = _normalise_person(
        _encode_rgba(),
        preserve_background=False,
        cutout=fake_fey_nobg,
    )
    decoded = cv2.imdecode(np.frombuffer(asset.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    result, pipeline = _install(asset, tmp_path)

    assert len(cutout_inputs) == 1
    assert cutout_inputs[0].shape == (480, 640, 3)
    assert decoded.shape == (720, 1280, 4)
    assert np.any(decoded[:, :, 3] < 255)
    assert pipeline.prepared_channels == [4]
    assert result["background_id"] == _PERSON_BACKGROUND_ID


def test_opaque_cutout_fallback_is_rejected_instead_of_saving_a_rectangle() -> None:
    def opaque_fallback(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    with pytest.raises(subject.HTTPException, match="FeyNoBg"):
        _normalise_person(
            _encode_rgb(),
            preserve_background=False,
            cutout=opaque_fallback,
        )


def test_rgba_upload_keeps_straight_rgb_at_semitransparent_edges() -> None:
    source = np.zeros((720, 1280, 4), dtype=np.uint8)
    source[:, :, :3] = [10, 20, 200]
    source[:, :, 3] = 128
    ok, encoded = cv2.imencode(".png", source)
    assert ok

    def alpha_only_cutout(image: np.ndarray) -> np.ndarray:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = 255
        return result

    asset = _normalise_person(
        encoded.tobytes(),
        preserve_background=False,
        cutout=alpha_only_cutout,
    )
    decoded = cv2.imdecode(np.frombuffer(asset.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    assert np.array_equal(decoded[360, 640, :3], source[360, 640, :3])
    assert decoded[360, 640, 3] == 128


def test_avatar_upload_route_accepts_preserve_background() -> None:
    assert "preserve_background" in inspect.signature(subject.upload_avatar).parameters


def test_fixed_person_background_asset_exists_and_helper_locates_it() -> None:
    helper = getattr(subject, "_resolve_preset_background_path", None)
    assert callable(helper), "preset background path helper is not implemented"

    path = Path(helper(_PERSON_BACKGROUND_ID))

    assert path.name == f"{_PERSON_BACKGROUND_ID}.png"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_uploaded_person_can_be_moved_to_recoverable_trash(tmp_path: Path) -> None:
    asset = _normalise_person(
        _encode_rgb(),
        preserve_background=True,
        cutout=lambda image: image,
    )
    pipeline = _FakePipeline()
    registry: dict[str, object] = {}
    subject._install_user_asset(
        asset,
        pipeline=pipeline,
        registry=registry,
        user_root=tmp_path,
    )

    result = subject._trash_user_avatar(
        asset.asset_id,
        registry=registry,
        user_root=tmp_path,
    )

    assert result == {
        "id": asset.asset_id,
        "kind": "avatar",
        "deleted": True,
        "recoverable": True,
    }
    assert asset.asset_id not in registry
    assert not (tmp_path / "reference_frames" / f"{asset.asset_id}.png").exists()
    trashed = list((tmp_path / ".trash" / "reference_frames").glob(f"{asset.asset_id}.*.png"))
    assert len(trashed) == 1


def test_builtin_person_cannot_be_deleted_as_a_user_upload(tmp_path: Path) -> None:
    with pytest.raises(subject.HTTPException) as error:
        subject._trash_user_avatar(
            "portrait",
            registry={"portrait": object()},
            user_root=tmp_path,
        )

    assert error.value.status_code == 404
