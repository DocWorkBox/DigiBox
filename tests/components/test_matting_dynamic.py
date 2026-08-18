from __future__ import annotations

import torch

from avtr1_renderer.components.matting import matte_chunk
from avtr1_renderer.models.matting import MODNetOutput


class _DynamicMODNet:
    def __call__(self, inputs: object, out: MODNetOutput | None = None) -> MODNetOutput:
        assert out is not None
        assert tuple(out.output.shape) == (5, 1, 288, 512)
        out.output.fill_(0.75)
        return out


def test_matte_chunk_preallocates_dynamic_output() -> None:
    frames = torch.zeros((5, 3, 72, 128), dtype=torch.float32)

    alpha = matte_chunk(frames, modnet=_DynamicMODNet())

    assert tuple(alpha.shape) == (5, 1, 72, 128)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.75))
