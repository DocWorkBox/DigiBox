from __future__ import annotations

import torch

from avtr1_renderer.models.warp import WarpInput
from avtr1_renderer.renderer import _run_warp


class _RequiresPreallocatedOutput:
    def __init__(self) -> None:
        self.received_shape: tuple[int, ...] | None = None

    def __call__(self, inputs, out=None):
        assert out is not None
        self.received_shape = tuple(out.out.shape)
        out.out.fill_(7)
        return out


def test_run_warp_preallocates_the_dynamic_batch_output() -> None:
    engine = _RequiresPreallocatedOutput()
    inputs = WarpInput(
        feature_3d=torch.zeros(5, 32, 16, 64, 64),
        kp_source=torch.zeros(5, 21, 3),
        kp_driving=torch.zeros(5, 21, 3),
    )

    output = _run_warp(engine, inputs)

    assert engine.received_shape == (5, 256, 64, 64)
    assert output.shape == (5, 256, 64, 64)
    assert torch.all(output == 7)
