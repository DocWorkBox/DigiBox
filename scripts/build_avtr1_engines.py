# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Convert an AVTR1 checkpoint to two TensorRT engines + normalizer stats.

Loads the scripted AVTR1 checkpoint and exports two ONNX graphs in
memory -- one for ``encode_conditions`` and one for the guided CFG decode
pass -- then parses each into a TensorRT engine. The motion-vector
normalization stats (``motion_offset`` / ``motion_scale``) are pulled
straight off the scripted module and written as a safetensors sidecar.

The scripted checkpoint has a flat, keyword-only API:

    encode_conditions(*, past_cond, audio_cond, kp_cond, past_times)
        -> (kp, past_context, past_last, audio_self, audio_other)
    forward(*, x, kp, past_context, past_last, audio_self, audio_other,
            t, self_weights, other_weights, kp_weights) -> Tensor

so the wrappers below are essentially passthroughs. CFG weight tensors
are engine *inputs* (shape ``(latent_dim,)``) so the caller can retune
guidance strength per request without rebuilding; the runtime broadcasts
a single scalar across all coords.

Usage:
    pixi run python scripts/build_avtr1_engines.py               # downloads checkpoint from HF
    pixi run python scripts/build_avtr1_engines.py --ckpt /path/to/train-39.scripted.pt

Outputs (where ``suffix`` = ``fp16`` or ``fp32``):
    <prefix>_encode_<suffix>.engine
    <prefix>_decode_<suffix>.engine
    <prefix>_normalizer.safetensors   -- full set of normalization buffers
                                         lifted off the scripted wrapper:
                                         motion_offset/scale [42] (combined
                                         [so3 | exp_lipsync]), offset_so3/
                                         scale_so3 [3], offset_kp/scale_kp
                                         [21,3], offset_exp/scale_exp [21,3]
                                         (used by _build_kp_cond), and
                                         lipsync_coords [39].

Decode engine I/O (B = batch, C = chunk_size, P = past_size, F = future_size,
L = latent_dim, N = nfeats):

    in   x              FLOAT (B, C, N)
    in   kp_tokens      FLOAT (B, 1, L)
    in   past_context   FLOAT (B, P - C, L)
    in   past_last      FLOAT (B, C, L)
    in   audio_self     FLOAT (B, P + C + F, L)
    in   audio_other    FLOAT (B, P + C + F, L)
    in   t              FLOAT (B, 1)
    in   w_self         FLOAT (L,)   per-coord CFG weight (self audio)
    in   w_other        FLOAT (L,)   per-coord CFG weight (other audio)
    in   w_kp           FLOAT (L,)   per-coord CFG weight (kp / median)
    out  output         FLOAT (B, C, N)
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from avtr1_renderer.avtr1_artifact_manager import get_storage_root


@dataclass(frozen=True)
class ModelShapes:
    chunk_size: int
    past_size: int
    future_size: int
    nfeats: int
    cond_feature_dim: int
    latent_dim: int

    @property
    def audio_dim(self) -> int:
        return self.cond_feature_dim * 2

    @property
    def n_past_chunks(self) -> int:
        return self.past_size // self.chunk_size

    @property
    def audio_seq_len(self) -> int:
        return self.past_size + self.chunk_size + self.future_size

    @property
    def kp_dim(self) -> int:
        return 3 + 63 * 2  # so3 (3) + exp (63) + kp (63)


def _read_shapes(scripted: torch.jit.ScriptModule) -> ModelShapes:
    inner = scripted.model
    return ModelShapes(
        chunk_size=int(inner.chunk_size),
        past_size=int(inner.past_size),
        future_size=int(inner.future_size),
        nfeats=int(inner.nfeats),
        cond_feature_dim=int(inner.cond_feature_dim),
        latent_dim=int(inner.latent_dim),
    )


class EncodeWrapper(nn.Module):
    """Threads kwargs through to ``scripted.encode_conditions``.

    The scripted method already returns a flat 5-tuple of tensors, so
    nothing else to do here.
    """

    def __init__(self, scripted: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.scripted = scripted

    def forward(
        self,
        past_cond: torch.Tensor,
        audio_cond: torch.Tensor,
        kp_cond: torch.Tensor,
        past_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.scripted.encode_conditions(
            past_cond=past_cond,
            audio_cond=audio_cond,
            kp_cond=kp_cond,
            past_times=past_times,
        )


class DecodeWrapper(nn.Module):
    """Threads kwargs through to ``scripted.forward``.

    CFG weight tensors are forwarded straight through as engine inputs,
    so the runtime can vary guidance strength per request.
    """

    def __init__(self, scripted: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.scripted = scripted

    def forward(
        self,
        x: torch.Tensor,
        kp_tokens: torch.Tensor,
        past_context: torch.Tensor,
        past_last: torch.Tensor,
        audio_self: torch.Tensor,
        audio_other: torch.Tensor,
        t: torch.Tensor,
        w_self: torch.Tensor,
        w_other: torch.Tensor,
        w_kp: torch.Tensor,
    ) -> torch.Tensor:
        return self.scripted(
            x=x,
            kp=kp_tokens,
            past_context=past_context,
            past_last=past_last,
            audio_self=audio_self,
            audio_other=audio_other,
            t=t,
            self_weights=w_self,
            other_weights=w_other,
            kp_weights=w_kp,
        )


def _export_to_onnx_bytes(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> bytes:
    """Script the wrapper, then export to ONNX bytes.

    Both wrappers call into a ``ScriptModule`` (either
    ``encode_conditions`` directly, or the scripted ``forward`` with
    kwargs), and the classic tracer doesn't follow those calls reliably.
    """
    target = torch.jit.script(module)
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            target,
            args,
            buf,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
            verbose=False,
        )
    return buf.getvalue()


def _onnx_bytes_to_trt(
    onnx_bytes: bytes,
    engine_path: Path,
    *,
    fp16: bool,
    use_ampere_plus_cc: bool,
    pin_norm_layers_fp32: bool,
) -> None:
    """Parse ONNX bytes and serialise a TensorRT engine.

    Mirrors ``avtr1_onnx_to_trt_no_poly`` in the reference repo: pins
    LayerNorm-flavoured ops (``Norm`` in name, Pow->ReduceMean pattern,
    last 10 layers) to FP32 to keep numerics stable in fp16 mode.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"ONNX parsing failed for {engine_path}")

    config = builder.create_builder_config()

    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8 and use_ampere_plus_cc:
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    if pin_norm_layers_fp32:
        # Pin LayerNorm-flavoured ops to fp32 for numerical stability under
        # fp16. Only touch layers whose output is already float; some helper
        # layers (shape tensors, indices) carry int weights and reject a
        # float precision override.
        for i in range(1, network.num_layers - 1):
            layer = network.get_layer(i)
            layer_next = network.get_layer(i + 1)
            out_type = layer.get_output_type(0)
            if out_type not in (trt.DataType.FLOAT, trt.DataType.HALF):
                continue
            if "Norm" in str(layer.name):
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(0, trt.DataType.FLOAT)
            if (
                "ELEMENTWISE" in str(layer.type)
                and "REDUCE" in str(layer_next.type)
                and "Pow" in layer.name
            ):
                layer.precision = trt.DataType.FLOAT
                layer_next.precision = trt.DataType.FLOAT
                layer.set_output_type(0, trt.DataType.FLOAT)
                layer_next.set_output_type(0, trt.DataType.FLOAT)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build engine at {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine_bytes))
    print(f"  wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def _build_encode(
    scripted: torch.jit.ScriptModule,
    shapes: ModelShapes,
    out_path: Path,
    *,
    batch: int,
    fp16: bool,
    use_ampere_plus_cc: bool,
    opset: int,
) -> dict[str, tuple[int, ...]]:
    """Export encode_conditions to ONNX in memory and build a TRT engine.

    Returns the encode output shapes so the decode export can size its
    matching inputs.
    """
    print("[encode] exporting to ONNX...")
    wrapper = EncodeWrapper(scripted).eval().cuda()
    device = torch.device("cuda")

    past_cond = torch.randn(batch, shapes.past_size, shapes.nfeats, device=device)
    audio_cond = torch.randn(batch, shapes.audio_seq_len, shapes.audio_dim, device=device)
    kp_cond = torch.randn(batch, 1, shapes.kp_dim, device=device)
    past_times = torch.rand(batch, shapes.n_past_chunks, 1, device=device)

    with torch.no_grad():
        kp_t, past_ctx, past_last, audio_self, audio_other = wrapper(
            past_cond, audio_cond, kp_cond, past_times
        )
    out_shapes = {
        "kp_tokens": tuple(kp_t.shape),
        "past_context": tuple(past_ctx.shape),
        "past_last": tuple(past_last.shape),
        "audio_self": tuple(audio_self.shape),
        "audio_other": tuple(audio_other.shape),
    }
    print(f"[encode] output shapes: {out_shapes}")

    onnx_bytes = _export_to_onnx_bytes(
        wrapper,
        (past_cond, audio_cond, kp_cond, past_times),
        input_names=["past_cond", "audio_cond", "kp_cond", "past_times"],
        output_names=[
            "kp_tokens",
            "past_context",
            "past_last",
            "audio_self",
            "audio_other",
        ],
        opset=opset,
    )
    print(f"[encode] onnx size: {len(onnx_bytes) / 1e6:.1f} MB")

    print(f"[encode] building TRT engine -> {out_path}")
    _onnx_bytes_to_trt(
        onnx_bytes,
        out_path,
        fp16=fp16,
        use_ampere_plus_cc=use_ampere_plus_cc,
        pin_norm_layers_fp32=True,
    )
    return out_shapes


def _build_decode(
    scripted: torch.jit.ScriptModule,
    shapes: ModelShapes,
    encode_out_shapes: dict[str, tuple[int, ...]],
    out_path: Path,
    *,
    batch: int,
    fp16: bool,
    use_ampere_plus_cc: bool,
    opset: int,
) -> None:
    print("[decode] exporting to ONNX...")
    wrapper = DecodeWrapper(scripted).eval().cuda()
    device = torch.device("cuda")

    def _zeros(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=torch.float32)

    x = _zeros((batch, shapes.chunk_size, shapes.nfeats))
    kp_tokens = _zeros(encode_out_shapes["kp_tokens"])
    past_context = _zeros(encode_out_shapes["past_context"])
    past_last = _zeros(encode_out_shapes["past_last"])
    audio_self = _zeros(encode_out_shapes["audio_self"])
    audio_other = _zeros(encode_out_shapes["audio_other"])
    t = torch.rand(batch, 1, device=device)
    w_self = _zeros((shapes.latent_dim,))
    w_other = _zeros((shapes.latent_dim,))
    w_kp = _zeros((shapes.latent_dim,))

    with torch.no_grad():
        out = wrapper(
            x, kp_tokens, past_context, past_last, audio_self, audio_other, t,
            w_self, w_other, w_kp,
        )
    print(f"[decode] output shape: {tuple(out.shape)}")

    onnx_bytes = _export_to_onnx_bytes(
        wrapper,
        (
            x, kp_tokens, past_context, past_last, audio_self, audio_other, t,
            w_self, w_other, w_kp,
        ),
        input_names=[
            "x",
            "kp_tokens",
            "past_context",
            "past_last",
            "audio_self",
            "audio_other",
            "t",
            "w_self",
            "w_other",
            "w_kp",
        ],
        output_names=["output"],
        opset=opset,
    )
    print(f"[decode] onnx size: {len(onnx_bytes) / 1e6:.1f} MB")

    print(f"[decode] building TRT engine -> {out_path}")
    _onnx_bytes_to_trt(
        onnx_bytes,
        out_path,
        fp16=fp16,
        use_ampere_plus_cc=use_ampere_plus_cc,
        pin_norm_layers_fp32=True,
    )


_NORMALIZER_BUFFER_NAMES = (
    "motion_offset",     # (42,) -- combined [so3 (3) | exp_lipsync (39)]
    "motion_scale",      # (42,)
    "offset_so3",        # (3,)
    "scale_so3",         # (3,)
    "offset_kp",         # (21, 3)
    "scale_kp",          # (21, 3)
    "offset_exp",        # (21, 3) -- full expression, needed for kp_cond
    "scale_exp",         # (21, 3)
    "lipsync_coords",    # (39,) int64 -- indices into flattened exp
)


def _save_normalizer_stats(
    scripted: torch.jit.ScriptModule,
    out_path: Path,
) -> None:
    """Pull the normalizer buffers off the scripted wrapper and save
    them as a safetensors sidecar next to the engines.

    These cover every normalization the runtime needs:
    - ``motion_offset`` / ``motion_scale`` denormalize the 42-dim ODE
      output (``[so3 | exp_lipsync]``).
    - ``offset_so3`` / ``offset_kp`` / ``offset_exp`` (and scales) feed
      ``_build_kp_cond``'s 129-dim ``[so3 | kp | exp]`` input.
    - ``lipsync_coords`` are the flat indices into ``exp.flatten()``
      used to slice out the lipsync subset.
    """
    tensors = {}
    for name in _NORMALIZER_BUFFER_NAMES:
        if not hasattr(scripted, name):
            raise RuntimeError(
                f"scripted module missing required buffer '{name}'; the "
                f"checkpoint is likely an older revision (expected "
                f"train-40_scripted_v2.pt or newer)."
            )
        tensors[name] = getattr(scripted, name).detach().cpu().contiguous()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    shapes = {k: tuple(v.shape) for k, v in tensors.items()}
    print(f"  wrote {out_path} ({len(tensors)} tensors: {shapes})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to train-39.scripted.pt (default: downloaded from HuggingFace)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (flat layout); "
             "default writes each file to its storage subdir under AVTR1_LOCAL_STORAGE",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="avtr1",
        help=(
            "Engine filename prefix; produces <prefix>_encode_fp{16,32}.engine, "
            "<prefix>_decode_fp{16,32}.engine, and <prefix>_normalizer.safetensors"
        ),
    )
    parser.add_argument("--batch", type=int, default=1, help="Static batch size baked into the engine")
    parser.add_argument("--no-fp16", action="store_true", help="Build fp32 engines instead of fp16")
    parser.add_argument(
        "--ampere-plus",
        action="store_true",
        help="Set TRT hardware_compatibility_level=AMPERE_PLUS (sm80+)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args(argv)

    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager, get_trt_engine_path

    mgr = get_artifact_manager()
    ckpt: Path = args.ckpt if args.ckpt is not None else mgr.get_artifact_path("avtr1_scripted")

    if not ckpt.is_file():
        parser.error(f"Checkpoint not found: {ckpt}")

    fp16 = not args.no_fp16
    suffix = "fp16" if fp16 else "fp32"

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        encode_path = args.out_dir / f"{args.prefix}_encode_{suffix}.engine"
        decode_path = args.out_dir / f"{args.prefix}_decode_{suffix}.engine"
        normalizer_path = args.out_dir / f"{args.prefix}_normalizer.safetensors"
    else:
        encode_path = get_trt_engine_path("avtr1_encode")
        decode_path = get_trt_engine_path("avtr1_decode")
        normalizer_path = get_storage_root() / "avtr1_normalizer.safetensors"
        encode_path.parent.mkdir(parents=True, exist_ok=True)
        decode_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading scripted module from {ckpt}")
    scripted = torch.jit.load(str(ckpt), map_location="cuda").eval()
    shapes = _read_shapes(scripted)
    print(f"Model shapes: {shapes}")

    encode_out_shapes = _build_encode(
        scripted,
        shapes,
        encode_path,
        batch=args.batch,
        fp16=fp16,
        use_ampere_plus_cc=args.ampere_plus,
        opset=args.opset,
    )
    _build_decode(
        scripted,
        shapes,
        encode_out_shapes,
        decode_path,
        batch=args.batch,
        fp16=fp16,
        use_ampere_plus_cc=args.ampere_plus,
        opset=args.opset,
    )
    print("Saving normalizer stats...")
    _save_normalizer_stats(scripted, normalizer_path)
    print("Done.")


if __name__ == "__main__":
    main()
