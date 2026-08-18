# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Build a TRT engine for HuBERT with a profile wide enough to process
two tracks (speech + listen) in parallel.

The engine needs batch >= 2 because the generator runs HuBERT once per
chunk on a ``(2, 8400)`` window (speech + listen stacked along the batch
dim) and folds the resulting ``5`` "current" features into a 75-frame
feature buffer carried in ``State``.

Usage:
    pixi run python scripts/build_hubert_engine.py            # downloads ONNX from HF
    pixi run python scripts/build_hubert_engine.py --onnx /path/to/hubert-lbs-avtr1.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def build(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    min_batch: int,
    max_batch: int,
    opt_batch: int,
    min_len: int,
    opt_len: int,
    max_len: int,
) -> None:
    import onnx
    import tensorrt as trt

    # Make batch dimension dynamic so TRT accepts min_batch..max_batch profiles.
    model = onnx.load(str(onnx_path))
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"ONNX parse failed for {onnx_path}")

    config = builder.create_builder_config()

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    profile.set_shape(
        input_tensor.name,
        [min_batch, min_len],
        [opt_batch, opt_len],
        [max_batch, max_len],
    )
    config.add_optimization_profile(profile)

    # LayerNorm / Pow→ReduceMean stability, same trick as the reference.
    for i in range(1, network.num_layers - 1):
        layer = network.get_layer(i)
        layer_next = network.get_layer(i + 1)
        if (
            "ELEMENTWISE" in str(layer.type)
            and "REDUCE" in str(layer_next.type)
            and "Pow" in layer.name
        ):
            layer.precision = trt.DataType.FLOAT
            layer_next.precision = trt.DataType.FLOAT
            layer.set_output_type(0, trt.DataType.FLOAT)
            layer_next.set_output_type(0, trt.DataType.FLOAT)

    print(f"Building TRT engine -> {engine_path}")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TRT build failed for {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine_bytes))
    print(f"  wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to hubert-lbs-avtr1.onnx (default: downloaded from HuggingFace)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output engine path (default: <storage_root>/speech2motion_runtime_artifacts_cc/hubert_lbs_fp16.engine; storage root is <project_root>/artifacts/<HF_REVISION>/ or $AVTR1_LOCAL_STORAGE/<HF_REVISION>/)",
    )
    parser.add_argument("--no-fp16", action="store_true", help="Build fp32 engine")
    # Default profile covers a single chunk (3+5+5 frames @ 640 + 80 shift =
    # 8400) at batch up to 2 (speech + listen in parallel).
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--min-len", type=int, default=3240)
    parser.add_argument("--opt-len", type=int, default=8400)
    parser.add_argument("--max-len", type=int, default=12960)
    args = parser.parse_args(argv)

    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager, get_trt_engine_path

    suffix = "fp32" if args.no_fp16 else "fp16"
    onnx_path: Path = args.onnx if args.onnx is not None else get_artifact_manager().get_artifact_path("hubert_onnx")
    out_path: Path = args.out if args.out is not None else get_trt_engine_path("hubert_lbs")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        parser.error(f"ONNX not found: {onnx_path}")

    print(f"Building HuBERT engine from {onnx_path}")
    print(
        f"  profile: batch [{args.min_batch}..{args.max_batch}] (opt {args.opt_batch}), "
        f"len [{args.min_len}..{args.max_len}] (opt {args.opt_len})"
    )
    print(f"  fp16: {not args.no_fp16}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"  GPU compute capability: sm{cap[0]}{cap[1]}")

    build(
        onnx_path,
        out_path,
        fp16=not args.no_fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        min_len=args.min_len,
        opt_len=args.opt_len,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
