from __future__ import annotations

import torch

from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.components import putback as subject


def test_animated_head_alpha_replaces_static_outline_without_exposing_fill(
    monkeypatch,
) -> None:
    height = width = 4
    pasteback_mask = torch.zeros((1, height, width), dtype=torch.float32)
    pasteback_mask[:, :2, :] = 1.0

    source_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    source_alpha[0, 0, 1] = 1.0  # original head position
    source_alpha[0, 3, 2] = 1.0  # static body outside the animated region
    source = torch.zeros((3, height, width), dtype=torch.float32)
    source[:, 3, 2] = torch.tensor([0.0, 1.0, 0.0])

    avatar = Avatar(
        id="moving-head",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        mask=pasteback_mask,
        source=source,
        source_alpha=source_alpha,
        no_matting=False,
    )

    # The neutral fill represents the decoder's registration background; the
    # red point is the generated head moving one pixel between frames.
    face_crops = torch.full((2, 3, height, width), 0.8, dtype=torch.float32)
    face_crops[0, :, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    face_crops[1, :, 0, 2] = torch.tensor([1.0, 0.0, 0.0])

    def red_foreground_matte(chunk: torch.Tensor, *, modnet: object) -> torch.Tensor:
        del modnet
        return (
            (chunk[:, 0:1] > 0.9)
            & (chunk[:, 1:2] < 0.1)
            & (chunk[:, 2:3] < 0.1)
        ).to(chunk.dtype)

    monkeypatch.setattr(subject, "matte_chunk", red_foreground_matte)

    background = torch.zeros((1, 3, height, width), dtype=torch.float32)
    rgb, alpha = subject.putback_chunk(
        face_crops,
        avatar,
        background,
        matting=object(),  # type: ignore[arg-type]
    )

    assert torch.equal(
        torch.stack((alpha[:, 0, 0, 1], alpha[:, 0, 0, 2]), dim=1),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.allclose(rgb[1, :, 0, 1], torch.zeros(3))
    assert torch.allclose(rgb[1, :, 0, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(alpha[:, 0, 3, 2], torch.ones(2))
    assert torch.allclose(rgb[1, :, 3, 2], torch.tensor([0.0, 1.0, 0.0]))


def test_collapsed_dynamic_matte_falls_back_to_static_portrait(monkeypatch) -> None:
    height = width = 4
    source_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    source_alpha[0, 0, 1] = 1.0
    source = torch.zeros((3, height, width), dtype=torch.float32)
    source[:, 0, 1] = torch.tensor([0.2, 0.4, 0.6])
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[:, :2, :] = 1.0

    avatar = Avatar(
        id="collapsed-matte",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        mask=mask,
        source=source,
        source_alpha=source_alpha,
        no_matting=False,
    )

    monkeypatch.setattr(
        subject,
        "matte_chunk",
        lambda chunk, *, modnet: torch.zeros(
            (chunk.shape[0], 1, height, width), dtype=chunk.dtype
        ),
    )

    faces = torch.full((2, 3, height, width), 0.8, dtype=torch.float32)
    rgb, alpha = subject.putback_chunk(
        faces,
        avatar,
        torch.zeros((1, 3, height, width), dtype=torch.float32),
        matting=object(),  # type: ignore[arg-type]
    )

    assert torch.equal(alpha, source_alpha.unsqueeze(0).expand(2, -1, -1, -1))
    assert torch.allclose(
        rgb[:, :, 0, 1],
        torch.tensor([[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]]),
    )


def test_collapsed_matte_keeps_generated_rgb_outside_the_motion_band(
    monkeypatch,
) -> None:
    height = width = 5
    source_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    source_alpha[:, 1:4, 1:4] = 1.0
    source = torch.zeros((3, height, width), dtype=torch.float32)
    source[:, 1:4, 1:4] = torch.tensor([0.0, 0.0, 1.0])[:, None, None]
    pasteback_mask = torch.zeros((1, height, width), dtype=torch.float32)
    pasteback_mask[:, 1:4, 1:4] = 1.0
    motion_mask = pasteback_mask.clone()
    motion_mask[:, 2, 2] = 0.0  # Deep head pixels are not part of the outline.
    avatar = Avatar(
        id="collapsed-outline-only",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        mask=pasteback_mask,
        source=source,
        source_alpha=source_alpha,
        alpha_motion_mask=motion_mask,
        no_matting=False,
    )

    dynamic = torch.stack(
        (source_alpha, torch.zeros_like(source_alpha)),
        dim=0,
    )
    monkeypatch.setattr(subject, "matte_chunk", lambda chunk, *, modnet: dynamic)
    faces = torch.zeros((2, 3, height, width), dtype=torch.float32)
    faces[:, 0, 1:4, 1:4] = 1.0

    rgb, alpha = subject.putback_chunk(
        faces,
        avatar,
        torch.zeros((1, 3, height, width), dtype=torch.float32),
        matting=object(),  # type: ignore[arg-type]
    )

    assert torch.equal(alpha, source_alpha.unsqueeze(0).expand(2, -1, -1, -1))
    assert torch.allclose(rgb[:, :, 2, 2], torch.tensor([[1.0, 0.0, 0.0]] * 2))
    assert torch.allclose(rgb[1, :, 1, 1], torch.tensor([0.0, 0.0, 1.0]))


def test_dynamic_matte_decontaminates_neutral_fill_at_hair_edge(monkeypatch) -> None:
    height = width = 3
    source_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    source_alpha[0, 1, 1] = 1.0
    source = torch.zeros((3, height, width), dtype=torch.float32)
    pasteback_mask = torch.ones((1, height, width), dtype=torch.float32)
    motion_mask = torch.zeros((1, height, width), dtype=torch.float32)
    motion_mask[0, 0, 1] = 1.0
    motion_mask[0, 1, 1] = 1.0
    avatar = Avatar(
        id="decontaminated-hair",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        mask=pasteback_mask,
        source=source,
        source_alpha=source_alpha,
        alpha_motion_mask=motion_mask,
        no_matting=False,
    )

    dynamic_alpha = torch.zeros((1, 1, height, width), dtype=torch.float32)
    dynamic_alpha[0, 0, 0, 1] = 0.25
    dynamic_alpha[0, 0, 1, 1] = 0.75
    monkeypatch.setattr(
        subject,
        "matte_chunk",
        lambda chunk, *, modnet: dynamic_alpha,
    )
    desired_hair = torch.tensor([0.20, 0.10, 0.40])
    neutral = torch.full((3,), 200.0 / 255.0)
    contaminated = desired_hair * 0.25 + neutral * 0.75
    face = torch.full((1, 3, height, width), 200.0 / 255.0)
    face[0, :, 0, 1] = contaminated

    rgb, alpha = subject.putback_chunk(
        face,
        avatar,
        torch.zeros((1, 3, height, width), dtype=torch.float32),
        matting=object(),  # type: ignore[arg-type]
    )

    assert torch.allclose(alpha[0, 0, 0, 1], torch.tensor(0.25))
    assert torch.allclose(rgb[0, :, 0, 1], desired_hair * 0.25, atol=1e-5)


def test_dynamic_matte_rejects_neutral_only_expansion_outside_source_hair(
    monkeypatch,
) -> None:
    height = width = 3
    source_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    source_alpha[0, 1, 1] = 1.0
    motion_mask = torch.zeros_like(source_alpha)
    motion_mask[0, 0, 1] = 1.0
    motion_mask[0, 1, 1] = 1.0
    avatar = Avatar(
        id="neutral-only-outline",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        mask=torch.ones_like(source_alpha),
        source=torch.zeros((3, height, width), dtype=torch.float32),
        source_alpha=source_alpha,
        alpha_motion_mask=motion_mask,
        no_matting=False,
    )
    predicted = torch.zeros((1, 1, height, width), dtype=torch.float32)
    predicted[0, 0, 0, 1] = 0.25
    predicted[0, 0, 1, 1] = 0.75
    monkeypatch.setattr(subject, "matte_chunk", lambda chunk, *, modnet: predicted)
    neutral_face = torch.full(
        (1, 3, height, width),
        200.0 / 255.0,
        dtype=torch.float32,
    )

    rgb, alpha = subject.putback_chunk(
        neutral_face,
        avatar,
        torch.zeros((1, 3, height, width), dtype=torch.float32),
        matting=object(),  # type: ignore[arg-type]
    )

    assert alpha[0, 0, 0, 1] == 0.0
    assert torch.equal(rgb[0, :, 0, 1], torch.zeros(3))


def test_dynamic_matte_uses_neutral_fixture_not_hidden_source_background(
    monkeypatch,
) -> None:
    height = width = 3
    source_alpha = torch.zeros((1, height, width), dtype=torch.float32)
    source_alpha[0, 1, 1] = 1.0
    source = torch.zeros((3, height, width), dtype=torch.float32)
    source[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0])  # hidden old background RGB
    source[:, 1, 1] = torch.tensor([0.1, 0.2, 0.3])
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[:, 1, 1] = 1.0
    avatar = Avatar(
        id="neutral-fixture",
        kp_info=None,  # type: ignore[arg-type]
        f_s=torch.empty(0),
        M_grid=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        mask=mask,
        source=source,
        source_alpha=source_alpha,
        no_matting=False,
    )
    captured: dict[str, torch.Tensor] = {}

    def capture_fixture(chunk: torch.Tensor, *, modnet: object) -> torch.Tensor:
        del modnet
        captured["chunk"] = chunk.clone()
        return source_alpha.unsqueeze(0).expand(chunk.shape[0], -1, -1, -1)

    monkeypatch.setattr(subject, "matte_chunk", capture_fixture)
    subject.putback_chunk(
        torch.zeros((1, 3, height, width), dtype=torch.float32),
        avatar,
        torch.zeros((1, 3, height, width), dtype=torch.float32),
        matting=object(),  # type: ignore[arg-type]
    )

    assert torch.allclose(
        captured["chunk"][0, :, 0, 0],
        torch.full((3,), 200.0 / 255.0),
    )
