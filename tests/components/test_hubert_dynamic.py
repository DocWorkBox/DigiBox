from __future__ import annotations

import torch

from avtr1_renderer.components.hubert import _hubert_frame_count, run_hubert
from avtr1_renderer.models.hubert import HubertOutput


class _DynamicHubert:
    def __call__(self, inputs: object, out: HubertOutput | None = None) -> HubertOutput:
        assert out is not None
        assert tuple(out.last_hidden_state.shape) == (2, 26, 1024)
        out.last_hidden_state.fill_(1.0)
        return out


def test_run_hubert_preallocates_dynamic_output_shape() -> None:
    audio = torch.zeros((2, 8400), dtype=torch.float32)

    features = run_hubert(audio, n_motion_frames=10, hubert=_DynamicHubert())

    assert tuple(features.shape) == (2, 10, 1024)
    assert torch.all(features == 1.0)


def test_hubert_frame_count_uses_the_convolution_receptive_field() -> None:
    assert _hubert_frame_count(8400) == 26
    assert _hubert_frame_count(3240) == 9
