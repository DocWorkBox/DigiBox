# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""ONNX compatibility transforms used by the portable Windows backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import onnx
import onnx_graphsurgeon as gs

_CACHE_VERSION = 1


def _temporary_model_path(destination: Path) -> Path:
    """Return a same-directory unique path suitable for atomic replacement."""
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )


def _i64(name: str, values: list[int]) -> gs.Constant:
    return gs.Constant(name=name, values=np.asarray(values, dtype=np.int64))


def _batch_scalar(graph: gs.Graph, source: gs.Variable) -> gs.Variable:
    prefix = "warp_dyn_batch"
    shape = gs.Variable(name=f"{prefix}/shape", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(op="Shape", name=f"{prefix}/Shape", inputs=[source], outputs=[shape])
    )
    batch = gs.Variable(name=f"{prefix}/batch1d", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Slice",
            name=f"{prefix}/Slice",
            inputs=[
                shape,
                _i64(f"{prefix}/start", [0]),
                _i64(f"{prefix}/end", [1]),
                _i64(f"{prefix}/axes", [0]),
                _i64(f"{prefix}/steps", [1]),
            ],
            outputs=[batch],
        )
    )
    return batch


def _concat_shape(
    graph: gs.Graph,
    parts: list[gs.Variable | gs.Constant],
    *,
    prefix: str,
) -> gs.Variable:
    output = gs.Variable(name=f"{prefix}/shape_concat", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Concat",
            name=f"{prefix}/ShapeConcat",
            attrs={"axis": 0},
            inputs=parts,
            outputs=[output],
        )
    )
    return output


def _multiply_batch(
    graph: gs.Graph,
    batch: gs.Variable,
    factor: int,
    *,
    prefix: str,
) -> gs.Variable:
    output = gs.Variable(name=f"{prefix}/batch_mul", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Mul",
            name=f"{prefix}/BatchMul",
            inputs=[batch, _i64(f"{prefix}/factor", [factor])],
            outputs=[output],
        )
    )
    return output


def _expand_constant(
    graph: gs.Graph,
    source: gs.Constant,
    target_shape: gs.Variable,
    *,
    prefix: str,
) -> gs.Variable:
    output = gs.Variable(
        name=f"{prefix}/expanded", dtype=source.values.dtype, shape=None
    )
    graph.nodes.append(
        gs.Node(
            op="Expand",
            name=f"{prefix}/Expand",
            inputs=[source, target_shape],
            outputs=[output],
        )
    )
    return output


def make_warp_batch_dynamic(graph: gs.Graph) -> None:
    """Replace the released warp graph's fixed batch constants with Shape ops."""
    feature = next(tensor for tensor in graph.inputs if tensor.name == "feature_3d")
    batch = _batch_scalar(graph, feature)

    def runtime_shape(rest: list[int], *, prefix: str) -> gs.Variable:
        return _concat_shape(
            graph,
            [batch, _i64(f"{prefix}/rest", rest)],
            prefix=prefix,
        )

    nodes = {node.name: node for node in graph.nodes}
    for name in (
        "/dense_motion_network/Reshape_1",
        "/dense_motion_network/Reshape_2",
    ):
        nodes[name].inputs[1] = runtime_shape(
            [21, 1, 1, 1, 3], prefix=f"warp_{name.replace('/', '_')}"
        )

    for name, rest in (
        ("/dense_motion_network/Reshape_3", [-1, 16, 64, 64]),
        ("/dense_motion_network/Reshape_4", [16, 64, 64, -1]),
    ):
        prefix = f"warp_{name.replace('/', '_')}"
        batch_times_keypoints = _multiply_batch(graph, batch, 22, prefix=prefix)
        nodes[name].inputs[1] = _concat_shape(
            graph,
            [batch_times_keypoints, _i64(f"{prefix}/rest", rest)],
            prefix=f"{prefix}/reshape",
        )

    nodes["/dense_motion_network/Reshape_5"].inputs[1] = runtime_shape(
        [22, -1, 16, 64, 64], prefix="warp_reshape_5"
    )
    nodes["/dense_motion_network/Reshape_10"].inputs[1] = runtime_shape(
        [-1, 16, 64, 64], prefix="warp_reshape_10"
    )
    nodes["/dense_motion_network/Reshape_11"].inputs[1] = runtime_shape(
        [-1, 64, 64], prefix="warp_reshape_11"
    )
    nodes["/Reshape"].inputs[1] = runtime_shape(
        [512, 64, 64], prefix="warp_final_reshape"
    )

    for name, expected, rest, prefix in (
        (
            "/dense_motion_network/Concat_1",
            (1, 1, 16, 64, 64, 3),
            [1, 16, 64, 64, 3],
            "warp_concat_1",
        ),
        (
            "/dense_motion_network/Concat_4",
            (1, 1, 16, 64, 64),
            [1, 16, 64, 64],
            "warp_concat_4",
        ),
    ):
        constant = nodes[name].inputs[0]
        if not isinstance(constant, gs.Constant) or tuple(constant.values.shape) != expected:
            raise ValueError(f"Unexpected fixed-batch constant at {name}")
        nodes[name].inputs[0] = _expand_constant(
            graph,
            constant,
            runtime_shape(rest, prefix=f"{prefix}/target"),
            prefix=prefix,
        )


def make_modnet_batch_dynamic(graph: gs.Graph) -> None:
    """Make the released MODNet SE-block Reshapes batch-dynamic."""
    input_tensor = next(tensor for tensor in graph.inputs if tensor.name == "input")
    batch = _batch_scalar(graph, input_tensor)
    nodes = {node.name: node for node in graph.nodes}
    for name, rest in (
        ("/lr_branch/se_block/Reshape", [1280]),
        ("/lr_branch/se_block/Reshape_1", [1280, 1, 1]),
    ):
        prefix = f"modnet_{name.replace('/', '_')}"
        nodes[name].inputs[1] = _concat_shape(
            graph,
            [batch, _i64(f"{prefix}/rest", rest)],
            prefix=prefix,
        )


def make_stitch_batch_dynamic(graph: gs.Graph) -> tuple[int, int]:
    """Make stitch Reshapes and fixed-batch ScatterND writes batch-safe."""
    source = next(tensor for tensor in graph.inputs if tensor.name == "kp_source")
    batch = _batch_scalar(graph, source)
    nodes = {node.name: node for node in graph.nodes}

    reshape_count = 0
    for name, rest in (
        ("/Reshape", [-1]),
        ("/Reshape_1", [-1]),
        ("/Reshape_2", [21, 3]),
        ("/Reshape_3", [1, 2]),
    ):
        node = nodes.get(name)
        if node is None:
            continue
        prefix = f"stitch_r_{name.replace('/', '_')}"
        node.inputs[1] = _concat_shape(
            graph,
            [batch, _i64(f"{prefix}/rest", rest)],
            prefix=prefix,
        )
        reshape_count += 1

    first = nodes.get("/ScatterND")
    second = nodes.get("/ScatterND_1")
    scatter_count = 0
    if first is not None and second is not None:
        original = first.inputs[0]
        updates = first.inputs[2]
        keep = gs.Variable(name="stitch_sc/keep_z", dtype=np.float32)
        graph.nodes.append(
            gs.Node(
                op="Slice",
                name="stitch_sc/Slice_keep",
                inputs=[
                    original,
                    _i64("stitch_sc/start", [2]),
                    _i64("stitch_sc/end", [3]),
                    _i64("stitch_sc/axes", [2]),
                    _i64("stitch_sc/steps", [1]),
                ],
                outputs=[keep],
            )
        )
        output = second.outputs[0]
        first.outputs.clear()
        second.outputs.clear()
        graph.nodes.append(
            gs.Node(
                op="Concat",
                name="stitch_sc/Concat_xy_z",
                attrs={"axis": 2},
                inputs=[updates, keep],
                outputs=[output],
            )
        )
        scatter_count = 1
    return reshape_count, scatter_count


def _symbolize_batch(model: onnx.ModelProto) -> None:
    del model.graph.value_info[:]
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"


def upgrade_grid_sample_to_opset20(model: onnx.ModelProto) -> onnx.ModelProto:
    """Upgrade volumetric GridSample nodes from opset 17 to opset 20.

    ONNX renamed the interpolation mode from ``bilinear`` to ``linear`` when
    GridSample gained rank-N support.  For a 5-D tensor this remains trilinear
    interpolation and matches ``torch.nn.functional.grid_sample`` semantics.
    """
    default_domain = next(
        (entry for entry in model.opset_import if entry.domain in {"", "ai.onnx"}),
        None,
    )
    if default_domain is None:
        raise ValueError("ONNX model has no default-domain opset import")
    default_domain.version = max(default_domain.version, 20)

    for node in model.graph.node:
        if node.domain not in {"", "ai.onnx"} or node.op_type != "GridSample":
            continue
        for attribute in node.attribute:
            if attribute.name == "mode" and attribute.s == b"bilinear":
                attribute.s = b"linear"
    return model


def prepare_portable_warp_onnx(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Create a dynamic-batch, standard-ONNX warp model for ORT CUDA."""
    source_path = Path(source)
    if destination is None:
        destination_path = source_path.with_name(
            f"{source_path.stem}_ort_v{_CACHE_VERSION}_opset20_dynamic.onnx"
        )
    else:
        destination_path = Path(destination)
    if (
        destination_path.is_file()
        and destination_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    ):
        return destination_path

    model = onnx.load(str(source_path))
    graph = gs.import_onnx(model)
    make_warp_batch_dynamic(graph)
    for tensor in list(graph.inputs) + list(graph.outputs):
        shape = list(tensor.shape)
        shape[0] = "batch"
        tensor.shape = shape
    graph.cleanup().toposort()
    converted = gs.export_onnx(graph)
    _symbolize_batch(converted)
    upgrade_grid_sample_to_opset20(converted)
    onnx.checker.check_model(converted)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_model_path(destination_path)
    try:
        onnx.save(converted, str(temporary))
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def prepare_portable_hubert_onnx(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Cache a HuBERT graph whose public batch dimension is symbolic."""
    source_path = Path(source)
    if destination is None:
        destination_path = source_path.with_name(
            f"{source_path.stem}_ort_v{_CACHE_VERSION}_dynamic.onnx"
        )
    else:
        destination_path = Path(destination)
    if (
        destination_path.is_file()
        and destination_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    ):
        return destination_path

    model = onnx.load(str(source_path))
    _symbolize_batch(model)
    onnx.checker.check_model(model)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_model_path(destination_path)
    try:
        onnx.save(model, str(temporary))
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def _prepare_dynamic_graph(
    source: str | Path,
    destination: str | Path | None,
    *,
    transform: Callable[[gs.Graph], Any],
) -> Path:
    source_path = Path(source)
    if destination is None:
        destination_path = source_path.with_name(
            f"{source_path.stem}_ort_v{_CACHE_VERSION}_dynamic.onnx"
        )
    else:
        destination_path = Path(destination)
    if (
        destination_path.is_file()
        and destination_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    ):
        return destination_path

    graph = gs.import_onnx(onnx.load(str(source_path)))
    transform(graph)
    for tensor in [*graph.inputs, *graph.outputs]:
        if not tensor.shape:
            raise ValueError(f"Tensor {tensor.name!r} has no declared shape")
        shape = list(tensor.shape)
        shape[0] = "batch"
        tensor.shape = shape
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    _symbolize_batch(model)
    onnx.checker.check_model(model)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_model_path(destination_path)
    try:
        onnx.save(model, str(temporary))
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def prepare_portable_modnet_onnx(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Cache a batch-dynamic MODNet graph for portable batched rendering."""
    return _prepare_dynamic_graph(
        source,
        destination,
        transform=make_modnet_batch_dynamic,
    )


def prepare_portable_stitch_onnx(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Cache a batch-dynamic stitch graph for portable rendering."""

    def transform(graph: gs.Graph) -> None:
        reshape_count, scatter_count = make_stitch_batch_dynamic(graph)
        if (reshape_count, scatter_count) != (4, 1):
            raise ValueError(
                "Unexpected stitch graph: expected 4 fixed Reshapes and "
                f"1 ScatterND pair, found {reshape_count} and {scatter_count}"
            )

    return _prepare_dynamic_graph(source, destination, transform=transform)


__all__ = [
    "make_modnet_batch_dynamic",
    "make_stitch_batch_dynamic",
    "make_warp_batch_dynamic",
    "prepare_portable_hubert_onnx",
    "prepare_portable_modnet_onnx",
    "prepare_portable_stitch_onnx",
    "prepare_portable_warp_onnx",
    "upgrade_grid_sample_to_opset20",
]
