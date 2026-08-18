# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Validate the native Windows GridSample3D TensorRT plugin in isolation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as functional
from build_renderer_engines import _warp_plugin_smoke_batches

from avtr1_renderer.runtime.trt import _load_plugin_library, _trt_to_torch_dtype

INPUT_SHAPE = (2, 3, 4, 5)
GRID_SHAPE = (3, 4, 5, 3)


def _create_plugin(plugin_path: Path) -> trt.IPluginV2:
    _load_plugin_library(plugin_path)
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("GridSample3D", "1", "")
    if creator is None:
        raise RuntimeError("GridSample3D v1 was not registered")

    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "interpolation_mode",
                np.asarray([0], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "padding_mode",
                np.asarray([0], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "align_corners",
                np.asarray([0], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
        ]
    )
    plugin = creator.create_plugin("grid_sample_3d_smoke", fields)
    if plugin is None:
        raise RuntimeError("GridSample3D creator returned no plugin")
    return plugin


def _build_engine(
    plugin_path: Path,
    *,
    max_batch: int,
    fp16: bool,
) -> tuple[trt.ICudaEngine, bytes, float]:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    dtype = trt.float16 if fp16 else trt.float32

    input_tensor = network.add_input("input", dtype, (-1, *INPUT_SHAPE))
    grid_tensor = network.add_input("grid", dtype, (-1, *GRID_SHAPE))
    assert input_tensor is not None and grid_tensor is not None

    plugin = _create_plugin(plugin_path)
    layer = network.add_plugin_v2([input_tensor, grid_tensor], plugin)
    if layer is None:
        raise RuntimeError("TensorRT rejected the GridSample3D plugin")
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        (1, *INPUT_SHAPE),
        (max_batch, *INPUT_SHAPE),
        (max_batch, *INPUT_SHAPE),
    )
    profile.set_shape(
        "grid",
        (1, *GRID_SHAPE),
        (max_batch, *GRID_SHAPE),
        (max_batch, *GRID_SHAPE),
    )
    config.add_optimization_profile(profile)

    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the GridSample3D smoke engine")

    plan_bytes = bytes(plan)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the smoke engine")
    return engine, plan_bytes, build_seconds


def _run_case(
    context: trt.IExecutionContext,
    *,
    batch: int,
    fp16: bool,
    seed: int,
) -> tuple[float, float, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_host = torch.randn(
        (batch, *INPUT_SHAPE),
        generator=generator,
        dtype=torch.float32,
    )
    grid_host = (
        torch.rand(
            (batch, *GRID_SHAPE),
            generator=generator,
            dtype=torch.float32,
        )
        * 2.4
        - 1.2
    )
    reference = functional.grid_sample(
        input_host,
        grid_host,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    dtype = torch.float16 if fp16 else torch.float32
    input_cuda = input_host.to(device="cuda", dtype=dtype)
    grid_cuda = grid_host.to(device="cuda", dtype=dtype)

    context.set_input_shape("input", tuple(input_cuda.shape))
    context.set_input_shape("grid", tuple(grid_cuda.shape))
    actual_shape = tuple(context.get_tensor_shape("output"))
    if any(dimension <= 0 for dimension in actual_shape):
        raise RuntimeError(f"unresolved output shape for batch {batch}: {actual_shape}")
    output_cuda = torch.empty(
        actual_shape,
        device="cuda",
        dtype=_trt_to_torch_dtype(context.engine.get_tensor_dtype("output")),
    )
    if actual_shape != tuple(reference.shape):
        raise RuntimeError(
            f"unexpected output shape for batch {batch}: "
            f"engine={actual_shape}, reference={tuple(reference.shape)}"
        )

    context.set_tensor_address("input", input_cuda.data_ptr())
    context.set_tensor_address("grid", grid_cuda.data_ptr())
    context.set_tensor_address("output", output_cuda.data_ptr())
    stream = torch.cuda.current_stream()

    started = time.perf_counter()
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError(f"TensorRT execution failed for batch {batch}")
    stream.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000

    error = (output_cuda.float().cpu() - reference).abs()
    return float(error.max()), float(error.mean()), latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/build/warp-plugin-smoke"),
    )
    args = parser.parse_args()

    if not args.plugin.is_file():
        parser.error(f"plugin not found: {args.plugin}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; reboot first if nvidia-smi reports GPU lost")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Plugin: {args.plugin.resolve()}")

    for fp16 in (False, True):
        precision = "fp16" if fp16 else "fp32"
        engine, plan, build_seconds = _build_engine(
            args.plugin,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        engine_path = args.out_dir / f"grid_sample_3d_dynamic_{precision}.engine"
        engine_path.write_bytes(plan)
        print(
            f"{precision}: built in {build_seconds:.2f}s, {engine_path.stat().st_size / 1e6:.2f} MB"
        )
        context = engine.create_execution_context()

        max_limit = 2e-2 if fp16 else 2e-4
        mean_limit = 3e-3 if fp16 else 2e-5
        for index, batch in enumerate(_warp_plugin_smoke_batches(args.max_batch)):
            max_error, mean_error, latency_ms = _run_case(
                context,
                batch=batch,
                fp16=fp16,
                seed=20260803 + index,
            )
            print(
                f"  batch={batch}: {latency_ms:.3f} ms, "
                f"max_abs={max_error:.6g}, mean_abs={mean_error:.6g}"
            )
            if max_error > max_limit or mean_error > mean_limit:
                raise AssertionError(
                    f"{precision} batch={batch} exceeds tolerance: "
                    f"max={max_error}, mean={mean_error}"
                )

    print("GridSample3D dynamic batch 1 -> max -> 1 smoke passed.")


if __name__ == "__main__":
    main()
