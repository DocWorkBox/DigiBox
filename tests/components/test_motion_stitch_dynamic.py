from __future__ import annotations

import torch

from avtr1_renderer.components.liveportrait.motion_stitch import (
    MotionFrame,
    motion_stitch,
)
from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.models.stitch import StitchOutput
from avtr1_renderer.types import KPInfo


class _DynamicStitch:
    def __call__(self, inputs: object, out: StitchOutput | None = None) -> StitchOutput:
        assert out is not None
        assert tuple(out.out.shape) == (5, 21, 3)
        out.out.copy_(inputs.kp_driving)
        return out


def test_motion_stitch_preallocates_dynamic_output() -> None:
    identity = torch.eye(3).unsqueeze(0)
    kp_info = KPInfo(
        kp=torch.zeros((1, 21, 3)),
        exp=torch.zeros((1, 21, 3)),
        scale=torch.ones((1, 1)),
        t=torch.zeros((1, 3)),
        pitch=torch.zeros((1, 1)),
        yaw=torch.zeros((1, 1)),
        roll=torch.zeros((1, 1)),
        R=identity,
    )
    motions = MotionFrame(
        R=identity.expand(5, -1, -1).contiguous(),
        exp=torch.zeros((5, len(LIPSYNC_COORDS))),
    )

    source, driving = motion_stitch(kp_info, motions, stitch=_DynamicStitch())

    assert tuple(source.shape) == (1, 21, 3)
    assert tuple(driving.shape) == (5, 21, 3)
