from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from avtr1_renderer.avtr1_motion_generator import Normalizer


def _scripted_buffers() -> SimpleNamespace:
    return SimpleNamespace(
        motion_offset=torch.arange(42, dtype=torch.float32),
        motion_scale=torch.arange(42, dtype=torch.float32) + 100,
        offset_so3=torch.arange(3, dtype=torch.float32),
        scale_so3=torch.arange(3, dtype=torch.float32) + 10,
        offset_kp=torch.arange(63, dtype=torch.float32).reshape(21, 3),
        scale_kp=torch.arange(63, dtype=torch.float32).reshape(21, 3) + 10,
        offset_exp=torch.arange(63, dtype=torch.float32).reshape(21, 3) + 20,
        scale_exp=torch.arange(63, dtype=torch.float32).reshape(21, 3) + 30,
        lipsync_coords=torch.arange(39, dtype=torch.int64),
    )


def test_from_scripted_uses_the_released_normalizer_buffers() -> None:
    scripted = _scripted_buffers()

    normalizer = Normalizer.from_scripted(scripted, device="cpu")

    assert normalizer.offset_so3.device.type == "cpu"
    assert normalizer.offset_kp.shape == (21, 3)
    assert normalizer.offset_exp.shape == (21, 3)
    assert torch.equal(normalizer.exp_lipsync_offset, scripted.motion_offset[3:])
    assert torch.equal(normalizer.exp_lipsync_scale, scripted.motion_scale[3:])
    assert normalizer.exp_lipsync_offset.is_contiguous()


def test_from_scripted_reports_a_missing_buffer() -> None:
    scripted = _scripted_buffers()
    del scripted.motion_scale

    with pytest.raises(RuntimeError, match="motion_scale"):
        Normalizer.from_scripted(scripted, device="cpu")
