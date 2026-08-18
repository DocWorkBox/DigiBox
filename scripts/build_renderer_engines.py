# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community
# Modified for DigiBox: native Windows TensorRT engine generation and packaging support.

"""Build the batched (b=5 dynamic) renderer TRT engines from ONNX.

Four engines are produced from their portable ONNX checkpoints:

- ``decoder_b5_fp16.engine``        (Resize ``sizes`` surgery on the
                                     SPADE conditioning path; the SPADE
                                     decoder otherwise needs no surgery)
- ``warp_network_b5_fp16.engine``   (Tile/Reshape/Concat surgery on the
                                     dense-motion stack)
- ``modnet_b5_fp16.engine``         (SE-block Reshape surgery)
- ``stitch_network_b5_fp16.engine`` (4x Reshape + redundant 2x ScatterND
                                     replaced with one batch-correct Concat)

Each builder loads the source ONNX, applies its surgery in memory, parses
straight into a TensorRT builder with a dynamic [1..max_batch] profile,
and writes the engine. No on-disk surged ONNX is kept.

Usage:
    pixi run python scripts/build_renderer_engines.py             # build all four
    pixi run python scripts/build_renderer_engines.py decoder     # build one
    pixi run python scripts/build_renderer_engines.py warp modnet
    pixi run python scripts/build_renderer_engines.py --max-batch 8

The exploratory originals are kept under ``scratchbook/surgery/`` for
reference; the runtime path consumes the engines this script writes.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnx.shape_inference
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from onnx.tools.update_model_dims import update_inputs_outputs_dims

# ONNX sources are downloaded from HuggingFace via the build artifact manager.
# Override with --*-onnx flags to use local files instead.

ENGINE_NAMES = ("decoder", "warp", "modnet", "stitch")
_RECOVERABLE_TACTIC_ERROR = re.compile(r"CUDA error 720(?!\d)")


def _warp_plugin_smoke_batches(max_batch: int) -> tuple[int, ...]:
    """Exercise both profile endpoints and the max-to-min shape transition."""
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    if max_batch == 1:
        return (1,)
    return (1, max_batch, 1)


def _persist_windows_warp_plugin(
    plugin: Path,
    engine: Path,
    *,
    platform: str = sys.platform,
) -> Path:
    """Install the Windows warp plugin beside its engine for later launches."""
    if platform != "win32":
        return plugin

    target = engine.with_name("grid_sample_3d_plugin.dll")
    if plugin.resolve() == target.resolve():
        return target
    if target.exists():
        if not filecmp.cmp(plugin, target, shallow=False):
            raise FileExistsError(f"Refusing to replace a different warp plugin: {target}")
        return target

    shutil.copy2(plugin, target)
    return target


class _FilteredLogger(trt.ILogger):
    """TensorRT logger that condenses one recoverable tactic-search failure."""

    def __init__(self, min_severity: trt.ILogger.Severity = trt.Logger.WARNING) -> None:
        super().__init__()
        self.min_severity = min_severity
        self.recoverable_tactic_skip_count = 0

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        if severity > self.min_severity:
            return
        if (
            severity == trt.Logger.ERROR
            and "Skipping tactic" in msg
            and _RECOVERABLE_TACTIC_ERROR.search(msg) is not None
        ):
            self.recoverable_tactic_skip_count += 1
            if self.recoverable_tactic_skip_count == 1:
                print(
                    "[TRT] [INFO] TensorRT skipped a tactic candidate after "
                    "CUDA error 720; the engine build is continuing to search "
                    "for another tactic. Repeated messages of this exact kind "
                    "are suppressed.",
                    file=sys.stderr,
                )
            return
        print(f"[TRT] [{severity.name}] {msg}", file=sys.stderr)

    def report_recoverable_tactic_skips(
        self,
        engine_name: str,
        *,
        success: bool,
    ) -> None:
        if not self.recoverable_tactic_skip_count:
            return
        outcome = (
            "the serialized engine plan was built successfully."
            if success
            else "the engine build ultimately failed or raised an exception."
        )
        print(
            f"[{engine_name}] [INFO] TensorRT skipped "
            f"{self.recoverable_tactic_skip_count} tactic candidate(s) after "
            f"CUDA error 720; {outcome}",
            file=sys.stderr,
        )


def _build_serialized_engine_plan(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
    logger: _FilteredLogger,
    *,
    engine_name: str,
) -> trt.IHostMemory:
    try:
        plan = builder.build_serialized_network(network, config)
    except Exception:
        logger.report_recoverable_tactic_skips(engine_name, success=False)
        raise
    if plan is None:
        logger.report_recoverable_tactic_skips(engine_name, success=False)
        raise RuntimeError(f"[{engine_name}] TRT build failed")
    logger.report_recoverable_tactic_skips(engine_name, success=True)
    return plan


# ---------------------------------------------------------------------------
# shared surger primitives
# ---------------------------------------------------------------------------


def _i64(name: str, values: list[int]) -> gs.Constant:
    return gs.Constant(name=name, values=np.asarray(values, dtype=np.int64))


def _make_batch_scalar(graph: gs.Graph, src: gs.Variable, prefix: str) -> gs.Variable:
    """Subgraph: ``Shape(src) -> Slice([0:1])`` → 1-D int64 of length 1."""
    shape_out = gs.Variable(name=f"{prefix}/shape", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(op="Shape", name=f"{prefix}/Shape", inputs=[src], outputs=[shape_out])
    )
    sliced = gs.Variable(name=f"{prefix}/batch1d", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Slice",
            name=f"{prefix}/Slice",
            inputs=[
                shape_out,
                _i64(f"{prefix}/start", [0]),
                _i64(f"{prefix}/end", [1]),
                _i64(f"{prefix}/axes", [0]),
                _i64(f"{prefix}/steps", [1]),
            ],
            outputs=[sliced],
        )
    )
    return sliced


def _concat_shape(
    graph: gs.Graph, parts: list[gs.Variable | gs.Constant], prefix: str
) -> gs.Variable:
    out = gs.Variable(name=f"{prefix}/shape_concat", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Concat",
            name=f"{prefix}/ShapeConcat",
            attrs={"axis": 0},
            inputs=parts,
            outputs=[out],
        )
    )
    return out


def _mul_batch_const(
    graph: gs.Graph, batch1d: gs.Variable, factor: int, prefix: str
) -> gs.Variable:
    out = gs.Variable(name=f"{prefix}/batch_mul", dtype=np.int64, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Mul",
            name=f"{prefix}/BatchMul",
            inputs=[batch1d, _i64(f"{prefix}/factor", [factor])],
            outputs=[out],
        )
    )
    return out


def _expand(
    graph: gs.Graph, src: gs.Constant, target_shape: gs.Variable, prefix: str
) -> gs.Variable:
    out = gs.Variable(name=f"{prefix}/expanded", dtype=src.values.dtype, shape=None)
    graph.nodes.append(
        gs.Node(
            op="Expand",
            name=f"{prefix}/Expand",
            inputs=[src, target_shape],
            outputs=[out],
        )
    )
    return out


def _symbolize_batch(model: onnx.ModelProto) -> None:
    """Symbolize dim 0 on each graph-level input/output and clear value_info."""
    del model.graph.value_info[:]
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim0 = tensor.type.tensor_type.shape.dim[0]
        dim0.ClearField("dim_value")
        dim0.dim_param = "batch"


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------


def _surge_decoder_resize(graph: gs.Graph) -> int:
    """Make the SPADE-stack ``Resize`` nodes batch-dynamic.

    The exported decoder has 4 ``Resize`` ops in its SPADE conditioning
    path with ``sizes`` constants that bake in batch=1 (e.g. ``[1, 256,
    64, 64]``, ``[1, 256, 128, 128]``). At runtime with input batch B>1
    these collapse the tensor to batch=1, which then broadcasts back to
    B downstream -- so frame 0's gamma/beta gets applied to every batch
    element, producing wrong outputs at indices 1..N-1.

    The fix: rewrite each ``sizes`` operand as
    ``Concat([Slice(Shape(feature), 0:1), Constant(C, H, W)])`` so the
    output's batch dim follows the input's. Same shape as the warp /
    modnet surgery.

    Returns the number of Resize nodes patched.
    """
    feature = next(t for t in graph.inputs if t.name == "feature")
    batch1d = _make_batch_scalar(graph, feature, prefix="dec_dyn_batch")
    patched = 0
    for node in graph.nodes:
        if node.op != "Resize":
            continue
        if len(node.inputs) < 4:
            continue
        sizes_inp = node.inputs[3]
        if not isinstance(sizes_inp, gs.Constant):
            continue
        v = sizes_inp.values.tolist()
        if len(v) != 4 or v[0] != 1:
            continue
        prefix = f"dec_resize_{node.name.replace('/', '_')}"
        new_sizes = _concat_shape(
            graph,
            [batch1d, _i64(f"{prefix}/rest", v[1:])],
            prefix=prefix,
        )
        node.inputs[3] = new_sizes
        patched += 1
    return patched


def _build_decoder(
    onnx_path: Path,
    out_path: Path,
    *,
    max_batch: int,
    fp16: bool,
) -> None:
    """Decoder needs no graph surgery — just relabel its batch dim to dynamic.

    Critically, we must also wipe the model's ``value_info`` entries.
    ``update_inputs_outputs_dims`` only relabels the I/O bindings; the
    194 internal ``value_info`` shape entries the original ONNX carries
    still pin batch=1 on every intermediate tensor. TRT honours those
    as kernel-selection hints and emits a graph whose b>1 outputs are
    silently wrong: batch index 0 is correct but indices 1..N-1
    receive frame 0's intermediate state instead of their own. We
    confirmed this by feeding ``[A, B, B, B, B]`` and observing the
    output at index 1..4 didn't depend on B at all.
    """
    print(f"[decoder] loading {onnx_path}")
    model = onnx.load(str(onnx_path))
    model = update_inputs_outputs_dims(
        model,
        input_dims={"feature": ["batch", 256, 64, 64]},
        output_dims={"output": ["batch", 3, 512, 512]},
    )

    # Patch the 4 SPADE Resize ``sizes`` constants from [1, C, H, W] to
    # runtime [B, C, H, W]. Without this the SPADE gamma/beta path is
    # collapsed to batch=1 and broadcast back to all elements,
    # producing identical (and wrong) modulation for every frame in a
    # b>1 chunk. See _surge_decoder_resize for the diagnosis.
    graph = gs.import_onnx(model)
    n_resize = _surge_decoder_resize(graph)
    print(f"[decoder] patched {n_resize} Resize nodes (sizes batch->dynamic)")
    for tensor in list(graph.inputs) + list(graph.outputs):
        new_shape = list(tensor.shape)
        new_shape[0] = "batch"
        tensor.shape = new_shape
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    _symbolize_batch(model)

    logger = _FilteredLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError("[decoder] ONNX parse failed")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "feature",
        (1, 256, 64, 64),
        (max_batch, 256, 64, 64),
        (max_batch, 256, 64, 64),
    )
    config.add_optimization_profile(profile)

    print(f"[decoder] building TRT engine -> {out_path}")
    t0 = time.time()
    plan = _build_serialized_engine_plan(
        builder,
        network,
        config,
        logger,
        engine_name="decoder",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(plan))
    print(
        f"[decoder] built in {time.time() - t0:.1f}s, size {out_path.stat().st_size / 1e6:.1f} MB"
    )


# ---------------------------------------------------------------------------
# warp
# ---------------------------------------------------------------------------


def _surge_warp(graph: gs.Graph) -> None:
    """Rewrite Reshape/Concat constants in warp_network so the batch dim is
    derived at runtime from ``Shape(feature_3d)``.

    The graph has multiple constants that bake batch=1: Reshape shape inputs
    of shape ``[1, ...]``, Reshape ops that collapse batch into the keypoint
    dim (``[22, -1, 16, 64, 64]``), and Concat ops whose other input is a
    constant of shape ``[1, ..., ...]`` (Concat doesn't broadcast).
    """
    feature_3d = next(t for t in graph.inputs if t.name == "feature_3d")
    batch1d = _make_batch_scalar(graph, feature_3d, prefix="dyn_batch")

    def runtime_shape(rest: list[int], prefix: str) -> gs.Variable:
        return _concat_shape(
            graph,
            [batch1d, _i64(f"{prefix}/rest", rest)],
            prefix=prefix,
        )

    name_to_node = {n.name: n for n in graph.nodes}

    # Reshape_1 / Reshape_2: shape const [1, 21, 1, 1, 1, 3] -> [B, 21, 1, 1, 1, 3]
    for node_name in (
        "/dense_motion_network/Reshape_1",
        "/dense_motion_network/Reshape_2",
    ):
        node = name_to_node[node_name]
        node.inputs[1] = runtime_shape([21, 1, 1, 1, 3], prefix=f"r1_{node_name.replace('/', '_')}")

    # Reshape_3 / Reshape_4: collapse batch into K dim
    # Old: [22, -1, 16, 64, 64] / [22, 16, 64, 64, -1]
    # New: [B*22, -1, 16, 64, 64] / [B*22, 16, 64, 64, -1]
    for node_name, rest in (
        ("/dense_motion_network/Reshape_3", [-1, 16, 64, 64]),
        ("/dense_motion_network/Reshape_4", [16, 64, 64, -1]),
    ):
        node = name_to_node[node_name]
        prefix = f"r3_{node_name.replace('/', '_')}"
        mul = _mul_batch_const(graph, batch1d, 22, prefix=prefix)
        node.inputs[1] = _concat_shape(
            graph,
            [mul, _i64(f"{prefix}/rest", rest)],
            prefix=f"r3_concat_{node_name.replace('/', '_')}",
        )

    # Reshape_5: un-collapse batch -> [B, 22, -1, 16, 64, 64]
    name_to_node["/dense_motion_network/Reshape_5"].inputs[1] = runtime_shape(
        [22, -1, 16, 64, 64], prefix="r5"
    )

    # Reshape_10 / Reshape_11 / final Reshape
    name_to_node["/dense_motion_network/Reshape_10"].inputs[1] = runtime_shape(
        [-1, 16, 64, 64], prefix="r10"
    )
    name_to_node["/dense_motion_network/Reshape_11"].inputs[1] = runtime_shape(
        [-1, 64, 64], prefix="r11"
    )
    name_to_node["/Reshape"].inputs[1] = runtime_shape([512, 64, 64], prefix="rfinal")

    # Concat_1: const [1, 1, 16, 64, 64, 3] concat with [B, 21, 16, 64, 64, 3]
    concat1 = name_to_node["/dense_motion_network/Concat_1"]
    const_inp = concat1.inputs[0]
    assert isinstance(const_inp, gs.Constant), f"unexpected: {const_inp}"
    assert tuple(const_inp.values.shape) == (1, 1, 16, 64, 64, 3), (
        f"unexpected Concat_1 const shape: {const_inp.values.shape}"
    )
    concat1.inputs[0] = _expand(
        graph, const_inp, runtime_shape([1, 16, 64, 64, 3], prefix="c1_expand"), prefix="c1"
    )

    # Concat_4: const [1, 1, 16, 64, 64] concat with [B, 21, 16, 64, 64]
    concat4 = name_to_node["/dense_motion_network/Concat_4"]
    const_inp = concat4.inputs[0]
    assert isinstance(const_inp, gs.Constant), f"unexpected: {const_inp}"
    assert tuple(const_inp.values.shape) == (1, 1, 16, 64, 64), (
        f"unexpected Concat_4 const shape: {const_inp.values.shape}"
    )
    concat4.inputs[0] = _expand(
        graph, const_inp, runtime_shape([1, 16, 64, 64], prefix="c4_expand"), prefix="c4"
    )


def _build_warp(
    onnx_path: Path,
    out_path: Path,
    plugin_path: Path,
    *,
    max_batch: int,
    fp16: bool,
) -> None:
    print(f"[warp] loading {onnx_path}")
    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)
    _surge_warp(graph)
    for tensor in list(graph.inputs) + list(graph.outputs):
        new_shape = list(tensor.shape)
        new_shape[0] = "batch"
        tensor.shape = new_shape
    graph.cleanup().toposort()
    new_model = gs.export_onnx(graph)
    _symbolize_batch(new_model)

    from avtr1_renderer.runtime.trt import _load_plugin_library

    _load_plugin_library(plugin_path)

    logger = _FilteredLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(new_model.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError("[warp] ONNX parse failed")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "feature_3d",
        (1, 32, 16, 64, 64),
        (max_batch, 32, 16, 64, 64),
        (max_batch, 32, 16, 64, 64),
    )
    profile.set_shape("kp_source", (1, 21, 3), (max_batch, 21, 3), (max_batch, 21, 3))
    profile.set_shape("kp_driving", (1, 21, 3), (max_batch, 21, 3), (max_batch, 21, 3))
    config.add_optimization_profile(profile)

    print(f"[warp] building TRT engine -> {out_path}")
    t0 = time.time()
    plan = _build_serialized_engine_plan(
        builder,
        network,
        config,
        logger,
        engine_name="warp",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(plan))
    print(f"[warp] built in {time.time() - t0:.1f}s, size {out_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# modnet
# ---------------------------------------------------------------------------


def _surge_modnet(graph: gs.Graph) -> None:
    """Rewrite the SE-block Reshape constants so batch is runtime-derived.

    Two Reshapes have hardcoded shape constants ``[1, 1280]`` and
    ``[1, 1280, 1, 1]``. Other apparent batch=1 constants in the
    /hr_branch/Resize ops are actually computed at runtime from
    ``Shape(input) -> Slice([0:2])`` already, so they need no fix.
    """
    input_tensor = next(t for t in graph.inputs if t.name == "input")
    batch1d = _make_batch_scalar(graph, input_tensor, prefix="dyn_batch")
    name_to_node = {n.name: n for n in graph.nodes}

    targets = {
        "/lr_branch/se_block/Reshape": [1280],
        "/lr_branch/se_block/Reshape_1": [1280, 1, 1],
    }
    for node_name, rest in targets.items():
        node = name_to_node[node_name]
        node.inputs[1] = _concat_shape(
            graph,
            [batch1d, _i64(f"{node_name.replace('/', '_')}_rest", rest)],
            prefix=f"se_{node_name.replace('/', '_')}",
        )


def _build_modnet(
    onnx_path: Path,
    out_path: Path,
    *,
    max_batch: int,
    fp16: bool,
) -> None:
    print(f"[modnet] loading {onnx_path}")
    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)
    _surge_modnet(graph)
    for tensor in list(graph.inputs) + list(graph.outputs):
        new_shape = list(tensor.shape)
        new_shape[0] = "batch"
        tensor.shape = new_shape
    graph.cleanup().toposort()
    new_model = gs.export_onnx(graph)
    _symbolize_batch(new_model)

    logger = _FilteredLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(new_model.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError("[modnet] ONNX parse failed")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        (1, 3, 288, 512),
        (max_batch, 3, 288, 512),
        (max_batch, 3, 288, 512),
    )
    config.add_optimization_profile(profile)

    print(f"[modnet] building TRT engine -> {out_path}")
    t0 = time.time()
    plan = _build_serialized_engine_plan(
        builder,
        network,
        config,
        logger,
        engine_name="modnet",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(plan))
    print(f"[modnet] built in {time.time() - t0:.1f}s, size {out_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# stitch
# ---------------------------------------------------------------------------


def _surge_stitch(graph: gs.Graph) -> tuple[int, int]:
    """Make ``stitch_network.onnx`` batch-dynamic.

    Two issues at b>1:
    1. Four ``Reshape`` nodes have shape constants ``[1, -1]``,
       ``[1, 21, 3]``, ``[1, 1, 2]`` -- collapse the batch dim to 1.
       Standard runtime-shape fix.
    2. Two ``ScatterND`` nodes have indices baked with batch=0:
       ``[(1, 21, 2, 3) of ([0, k, c] for k, c in product(range(21), range(2)))]``.
       Both nodes scatter ``input[:, :, :2] = updates`` -- an
       overcomplicated way to express ``Concat([updates,
       input[:, :, 2:3]], axis=2)``. The second ScatterND repeats
       the first's write with the same updates, so it's redundant.
       We replace the *output* of the second scatter (consumed
       downstream) with a single Concat that reads from the first
       scatter's *input* (``Add_output_0``) and the same ``updates``
       (``Add_1_output_0``). Both ScatterND nodes drop out under
       ``cleanup()``.

    Returns ``(reshapes_patched, scatter_pairs_replaced)``.
    """
    feature = next(t for t in graph.inputs if t.name == "kp_source")
    batch1d = _make_batch_scalar(graph, feature, prefix="stitch_dyn_batch")

    name_to_node = {n.name: n for n in graph.nodes}

    # Reshapes: [1, -1] x2 -> [B, -1]; [1, 21, 3] -> [B, 21, 3]; [1, 1, 2] -> [B, 1, 2].
    reshape_targets = {
        "/Reshape": [-1],
        "/Reshape_1": [-1],
        "/Reshape_2": [21, 3],
        "/Reshape_3": [1, 2],
    }
    n_reshape = 0
    for node_name, rest in reshape_targets.items():
        if node_name not in name_to_node:
            continue
        node = name_to_node[node_name]
        prefix = f"stitch_r_{node_name.replace('/', '_')}"
        new_shape = _concat_shape(
            graph,
            [batch1d, _i64(f"{prefix}/rest", rest)],
            prefix=prefix,
        )
        node.inputs[1] = new_shape
        n_reshape += 1

    # ScatterND pair -> single Concat.
    sn = name_to_node.get("/ScatterND")
    sn1 = name_to_node.get("/ScatterND_1")
    n_scatter = 0
    if sn is not None and sn1 is not None:
        # The first ScatterND writes input[:, :, :2] = updates over Add_output_0.
        # The second writes the same on top of that. Their combined effect:
        # Concat([updates (B, 21, 2), Add_output_0[:, :, 2:3]], axis=2).
        original_input = sn.inputs[0]  # Add_output_0
        updates = sn.inputs[2]  # Add_1_output_0
        prefix = "stitch_sc"
        # Slice the third channel of original_input: input[:, :, 2:3]
        keep = gs.Variable(name=f"{prefix}/keep_xy", dtype=np.float32)
        graph.nodes.append(
            gs.Node(
                op="Slice",
                name=f"{prefix}/Slice_keep",
                inputs=[
                    original_input,
                    _i64(f"{prefix}/start", [2]),
                    _i64(f"{prefix}/end", [3]),
                    _i64(f"{prefix}/axes", [2]),
                    _i64(f"{prefix}/steps", [1]),
                ],
                outputs=[keep],
            )
        )
        # Concat updates (B, 21, 2) || keep (B, 21, 1) along axis=2 -> (B, 21, 3).
        # Reuse the second ScatterND's output tensor name so consumers don't change.
        out_var = sn1.outputs[0]
        sn.outputs.clear()
        sn1.outputs.clear()
        graph.nodes.append(
            gs.Node(
                op="Concat",
                name=f"{prefix}/Concat_xy_z",
                attrs={"axis": 2},
                inputs=[updates, keep],
                outputs=[out_var],
            )
        )
        n_scatter = 1
    return n_reshape, n_scatter


def _build_stitch(
    onnx_path: Path,
    out_path: Path,
    *,
    max_batch: int,
    fp16: bool,
) -> None:
    print(f"[stitch] loading {onnx_path}")
    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)
    n_reshape, n_scatter = _surge_stitch(graph)
    print(f"[stitch] patched {n_reshape} Reshape + {n_scatter} ScatterND-pair -> Concat")
    for tensor in list(graph.inputs) + list(graph.outputs):
        new_shape = list(tensor.shape)
        new_shape[0] = "batch"
        tensor.shape = new_shape
    graph.cleanup().toposort()
    new_model = gs.export_onnx(graph)
    _symbolize_batch(new_model)

    logger = _FilteredLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(new_model.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError("[stitch] ONNX parse failed")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    profile.set_shape("kp_source", (1, 21, 3), (max_batch, 21, 3), (max_batch, 21, 3))
    profile.set_shape("kp_driving", (1, 21, 3), (max_batch, 21, 3), (max_batch, 21, 3))
    config.add_optimization_profile(profile)

    print(f"[stitch] building TRT engine -> {out_path}")
    t0 = time.time()
    plan = _build_serialized_engine_plan(
        builder,
        network,
        config,
        logger,
        engine_name="stitch",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(plan))
    print(f"[stitch] built in {time.time() - t0:.1f}s, size {out_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _inspect(engine_path: Path) -> None:
    logger = _FilteredLogger()
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    assert engine is not None, f"failed to deserialize {engine_path}"
    print(f"  {engine_path.name}:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name).name
        print(f"    [{mode:>6s}] {name}: {shape}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "engines",
        nargs="*",
        choices=ENGINE_NAMES,
        default=None,
        help=f"Subset of engines to build ({', '.join(ENGINE_NAMES)}). Default builds all four.",
    )
    parser.add_argument(
        "--decoder-onnx",
        type=Path,
        default=None,
        help="Path to decoder.onnx (default: downloaded from HuggingFace)",
    )
    parser.add_argument(
        "--warp-onnx",
        type=Path,
        default=None,
        help="Path to warp_network.onnx (default: downloaded from HuggingFace)",
    )
    parser.add_argument(
        "--warp-plugin",
        type=Path,
        default=None,
        help="Path to the GridSample3D plugin (.so on Linux, .dll on Windows)",
    )
    parser.add_argument(
        "--modnet-onnx",
        type=Path,
        default=None,
        help="Path to modnet.onnx (default: downloaded from HuggingFace)",
    )
    parser.add_argument(
        "--stitch-onnx",
        type=Path,
        default=None,
        help="Path to stitch_network.onnx (default: downloaded from HuggingFace)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory (flat layout); default writes each "
            "engine to its storage subdir under AVTR1_LOCAL_STORAGE"
        ),
    )
    parser.add_argument("--max-batch", type=int, default=5)
    parser.add_argument("--no-fp16", action="store_true", help="Build fp32 engines")
    args = parser.parse_args(argv)

    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager, get_trt_engine_path

    mgr = get_artifact_manager()

    def _onnx(arg: Path | None, artifact: str) -> Path:
        return arg if arg is not None else mgr.get_artifact_path(artifact)

    def _out(engine_name: str, filename: str) -> Path:
        if args.out_dir is not None:
            return args.out_dir / filename
        return get_trt_engine_path(engine_name)

    requested = tuple(args.engines) if args.engines else ENGINE_NAMES
    fp16 = not args.no_fp16
    suffix = "fp16" if fp16 else "fp32"

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"GPU compute capability: sm{cap[0]}{cap[1]}")
    print(f"Max batch: {args.max_batch}    fp16: {fp16}")
    print(f"Building: {', '.join(requested)}\n")

    written: list[Path] = []

    if "decoder" in requested:
        out = _out("decoder", f"decoder_b5_{suffix}.engine")
        out.parent.mkdir(parents=True, exist_ok=True)
        _build_decoder(
            _onnx(args.decoder_onnx, "decoder_onnx"),
            out,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        written.append(out)
    if "warp" in requested:
        out = _out("warp_network", f"warp_network_b5_{suffix}.engine")
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.warp_plugin is not None:
            plugin = args.warp_plugin
        elif os.environ.get("AVTR1_WARP_PLUGIN"):
            plugin = Path(os.environ["AVTR1_WARP_PLUGIN"])
        elif sys.platform == "win32":
            plugin = get_trt_engine_path("warp_network").with_name("grid_sample_3d_plugin.dll")
        else:
            plugin = mgr.storage_path("warp_plugin")
        if not plugin.is_file():
            parser.error(
                f"Warp plugin not found: {plugin}. Pass --warp-plugin or set AVTR1_WARP_PLUGIN."
            )
        installed_plugin = _persist_windows_warp_plugin(plugin, out)
        _build_warp(
            _onnx(args.warp_onnx, "warp_network_onnx"),
            out,
            installed_plugin,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        if installed_plugin != plugin:
            print(f"[warp] installed plugin -> {installed_plugin}")
        written.append(out)
    if "modnet" in requested:
        out = _out("modnet", f"modnet_b5_{suffix}.engine")
        out.parent.mkdir(parents=True, exist_ok=True)
        _build_modnet(
            _onnx(args.modnet_onnx, "modnet_onnx"),
            out,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        written.append(out)
    if "stitch" in requested:
        out = _out("stitch_network", f"stitch_network_b5_{suffix}.engine")
        out.parent.mkdir(parents=True, exist_ok=True)
        _build_stitch(
            _onnx(args.stitch_onnx, "stitch_network_onnx"),
            out,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        written.append(out)

    print("\nEngine I/O shapes:")
    for path in written:
        _inspect(path)
    print(f"\nDone. {len(written)} engine(s) written.")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
