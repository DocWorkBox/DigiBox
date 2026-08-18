from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeInput,
    Avtr1DecodeOutput,
    Avtr1EncodeInput,
)
from avtr1_renderer.runtime.torchscript_avtr1 import TorchScriptAVTR1Backend


class _FakeAVTR1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            chunk_size=2,
            past_size=4,
            future_size=2,
            nfeats=3,
            cond_feature_dim=4,
            latent_dim=5,
        )
        self.decode_arguments: dict[str, torch.Tensor] | None = None

    def encode_conditions(
        self,
        *,
        past_cond: torch.Tensor,
        audio_cond: torch.Tensor,
        kp_cond: torch.Tensor,
        past_times: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = past_cond.shape[0]
        device = past_cond.device
        latent = self.model.latent_dim
        audio_len = self.model.past_size + self.model.chunk_size + self.model.future_size
        return (
            torch.full((batch, 1, latent), 1.0, device=device),
            torch.full((batch, 2, latent), 2.0, device=device),
            torch.full((batch, 2, latent), 3.0, device=device),
            torch.full((batch, audio_len, latent), 4.0, device=device),
            torch.full((batch, audio_len, latent), 5.0, device=device),
        )

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        self.decode_arguments = kwargs
        return (
            kwargs["x"]
            + kwargs["t"].reshape(-1, 1, 1)
            + kwargs["self_weights"][0]
            + kwargs["other_weights"][0]
            + kwargs["kp_weights"][0]
        )


def _encode_input() -> Avtr1EncodeInput:
    return Avtr1EncodeInput(
        past_cond=torch.zeros(1, 4, 3),
        audio_cond=torch.zeros(1, 8, 8),
        kp_cond=torch.zeros(1, 1, 129),
        past_times=torch.zeros(1, 2, 1),
    )


def _decode_input() -> Avtr1DecodeInput:
    return Avtr1DecodeInput(
        x=torch.ones(1, 2, 3),
        kp_tokens=torch.zeros(1, 1, 5),
        past_context=torch.zeros(1, 2, 5),
        past_last=torch.zeros(1, 2, 5),
        audio_self=torch.zeros(1, 8, 5),
        audio_other=torch.zeros(1, 8, 5),
        w_self=torch.full((5,), 2.0),
        w_other=torch.full((5,), 3.0),
        w_kp=torch.full((5,), 4.0),
        t=torch.full((1, 1), 0.5),
    )


def test_encode_matches_the_inference_engine_contract_and_reuses_outputs() -> None:
    backend = TorchScriptAVTR1Backend(_FakeAVTR1(), device="cpu")
    output = backend.encode.allocate_outputs()

    returned = backend.encode(_encode_input(), out=output)

    assert returned is output
    assert output.kp_tokens.shape == (1, 1, 5)
    assert output.past_context.shape == (1, 2, 5)
    assert output.past_last.shape == (1, 2, 5)
    assert output.audio_self.shape == (1, 8, 5)
    assert output.audio_other.shape == (1, 8, 5)
    assert torch.all(output.kp_tokens == 1)
    assert torch.all(output.audio_other == 5)


def test_decode_maps_runtime_names_to_the_released_scripted_api() -> None:
    scripted = _FakeAVTR1()
    backend = TorchScriptAVTR1Backend(scripted, device="cpu")
    output = backend.decode.allocate_outputs()

    returned = backend.decode(_decode_input(), out=output)

    assert returned is output
    assert output.output.shape == (1, 2, 3)
    assert torch.allclose(output.output, torch.full((1, 2, 3), 10.5))
    assert scripted.decode_arguments is not None
    assert scripted.decode_arguments["kp"] is not None
    assert scripted.decode_arguments["self_weights"][0].item() == 2.0
    assert scripted.decode_arguments["other_weights"][0].item() == 3.0
    assert scripted.decode_arguments["kp_weights"][0].item() == 4.0


def test_decode_rejects_output_buffer_with_wrong_dtype() -> None:
    backend = TorchScriptAVTR1Backend(_FakeAVTR1(), device="cpu")
    output = Avtr1DecodeOutput(output=torch.empty((1, 2, 3), dtype=torch.float16))

    with pytest.raises(ValueError, match="dtype"):
        backend.decode(_decode_input(), out=output)
