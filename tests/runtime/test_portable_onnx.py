from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import torch
import torch.nn.functional as functional
from onnx import TensorProto, helper

from avtr1_renderer.runtime.portable_onnx import (
    _temporary_model_path,
    make_modnet_batch_dynamic,
    make_stitch_batch_dynamic,
    make_warp_batch_dynamic,
    prepare_portable_hubert_onnx,
    upgrade_grid_sample_to_opset20,
)


def _grid_sample_model() -> onnx.ModelProto:
    data = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT, [1, 1, 2, 2, 2]
    )
    grid = helper.make_tensor_value_info(
        "grid", TensorProto.FLOAT, [1, 1, 1, 1, 3]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 1, 1, 1]
    )
    node = helper.make_node(
        "GridSample",
        ["data", "grid"],
        ["output"],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=0,
    )
    graph = helper.make_graph([node], "grid-sample", [data, grid], [output])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )


def test_grid_sample_upgrade_preserves_pytorch_5d_semantics() -> None:
    model = upgrade_grid_sample_to_opset20(_grid_sample_model())

    assert model.opset_import[0].version == 20
    mode = next(
        attribute.s
        for attribute in model.graph.node[0].attribute
        if attribute.name == "mode"
    )
    assert mode == b"linear"

    data = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    grid = np.array([[[[[0.2, -0.1, 0.3]]]]], dtype=np.float32)
    expected = functional.grid_sample(
        torch.from_numpy(data),
        torch.from_numpy(grid),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).numpy()
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"data": data, "grid": grid})[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_warp_surgery_replaces_fixed_batch_shape_constants() -> None:
    feature = gs.Variable("feature_3d", dtype=np.float32, shape=[1, 32, 16, 64, 64])
    nodes: list[gs.Node] = []

    def reshape(name: str, shape: list[int]) -> gs.Node:
        node = gs.Node(
            op="Reshape",
            name=name,
            inputs=[
                gs.Variable(f"{name}/data", dtype=np.float32),
                gs.Constant(f"{name}/shape", np.asarray(shape, dtype=np.int64)),
            ],
            outputs=[gs.Variable(f"{name}/out", dtype=np.float32)],
        )
        nodes.append(node)
        return node

    for name, shape in (
        ("/dense_motion_network/Reshape_1", [1, 21, 1, 1, 1, 3]),
        ("/dense_motion_network/Reshape_2", [1, 21, 1, 1, 1, 3]),
        ("/dense_motion_network/Reshape_3", [22, -1, 16, 64, 64]),
        ("/dense_motion_network/Reshape_4", [22, 16, 64, 64, -1]),
        ("/dense_motion_network/Reshape_5", [1, 22, -1, 16, 64, 64]),
        ("/dense_motion_network/Reshape_10", [1, -1, 16, 64, 64]),
        ("/dense_motion_network/Reshape_11", [1, -1, 64, 64]),
        ("/Reshape", [1, 512, 64, 64]),
    ):
        reshape(name, shape)

    nodes.extend(
        [
            gs.Node(
                op="Concat",
                name="/dense_motion_network/Concat_1",
                inputs=[
                    gs.Constant(
                        "concat1/constant",
                        np.zeros((1, 1, 16, 64, 64, 3), dtype=np.float32),
                    ),
                    gs.Variable("concat1/other", dtype=np.float32),
                ],
                outputs=[gs.Variable("concat1/out", dtype=np.float32)],
            ),
            gs.Node(
                op="Concat",
                name="/dense_motion_network/Concat_4",
                inputs=[
                    gs.Constant(
                        "concat4/constant",
                        np.zeros((1, 1, 16, 64, 64), dtype=np.float32),
                    ),
                    gs.Variable("concat4/other", dtype=np.float32),
                ],
                outputs=[gs.Variable("concat4/out", dtype=np.float32)],
            ),
        ]
    )
    graph = gs.Graph(nodes=nodes, inputs=[feature], outputs=[])

    make_warp_batch_dynamic(graph)

    by_name = {node.name: node for node in graph.nodes}
    assert not isinstance(
        by_name["/dense_motion_network/Reshape_1"].inputs[1], gs.Constant
    )
    assert not isinstance(by_name["/Reshape"].inputs[1], gs.Constant)
    assert not isinstance(
        by_name["/dense_motion_network/Concat_1"].inputs[0], gs.Constant
    )
    assert "warp_dyn_batch/Shape" in by_name


def test_prepare_portable_hubert_symbolizes_batch_dimension(tmp_path: Path) -> None:
    source = tmp_path / "hubert.onnx"
    inputs = helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, 8400])
    outputs = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 26, 1024])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["input_values"], ["output"])],
            "hubert",
            [inputs],
            [outputs],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, source)

    converted_path = prepare_portable_hubert_onnx(source)
    converted = onnx.load(converted_path)

    assert converted_path != source
    for tensor in [*converted.graph.input, *converted.graph.output]:
        first = tensor.type.tensor_type.shape.dim[0]
        assert first.dim_param == "batch"
        assert not first.HasField("dim_value")


def test_modnet_surgery_replaces_fixed_batch_reshape_constants() -> None:
    input_tensor = gs.Variable("input", dtype=np.float32, shape=[1, 3, 288, 512])
    nodes = []
    for name, shape in (
        ("/lr_branch/se_block/Reshape", [1, 1280]),
        ("/lr_branch/se_block/Reshape_1", [1, 1280, 1, 1]),
    ):
        nodes.append(
            gs.Node(
                op="Reshape",
                name=name,
                inputs=[
                    gs.Variable(f"{name}/data", dtype=np.float32),
                    gs.Constant(f"{name}/shape", np.asarray(shape, dtype=np.int64)),
                ],
                outputs=[gs.Variable(f"{name}/out", dtype=np.float32)],
            )
        )
    graph = gs.Graph(nodes=nodes, inputs=[input_tensor], outputs=[])

    make_modnet_batch_dynamic(graph)

    by_name = {node.name: node for node in graph.nodes}
    assert not isinstance(by_name["/lr_branch/se_block/Reshape"].inputs[1], gs.Constant)
    assert not isinstance(
        by_name["/lr_branch/se_block/Reshape_1"].inputs[1], gs.Constant
    )


def test_stitch_surgery_replaces_fixed_shapes_and_scatter_pair() -> None:
    source = gs.Variable("kp_source", dtype=np.float32, shape=[1, 21, 3])
    nodes = []
    for name, shape in (
        ("/Reshape", [1, -1]),
        ("/Reshape_1", [1, -1]),
        ("/Reshape_2", [1, 21, 3]),
        ("/Reshape_3", [1, 1, 2]),
    ):
        nodes.append(
            gs.Node(
                op="Reshape",
                name=name,
                inputs=[
                    gs.Variable(f"{name}/data", dtype=np.float32),
                    gs.Constant(f"{name}/shape", np.asarray(shape, dtype=np.int64)),
                ],
                outputs=[gs.Variable(f"{name}/out", dtype=np.float32)],
            )
        )
    original = gs.Variable("original", dtype=np.float32)
    updates = gs.Variable("updates", dtype=np.float32)
    indices = gs.Constant("indices", np.zeros((1, 21, 2, 3), dtype=np.int64))
    scatter_out = gs.Variable("scatter_out", dtype=np.float32)
    final_out = gs.Variable("final_out", dtype=np.float32)
    nodes.extend(
        [
            gs.Node(
                op="ScatterND",
                name="/ScatterND",
                inputs=[original, indices, updates],
                outputs=[scatter_out],
            ),
            gs.Node(
                op="ScatterND",
                name="/ScatterND_1",
                inputs=[scatter_out, indices, updates],
                outputs=[final_out],
            ),
        ]
    )
    graph = gs.Graph(nodes=nodes, inputs=[source], outputs=[final_out])

    reshape_count, scatter_count = make_stitch_batch_dynamic(graph)

    by_name = {node.name: node for node in graph.nodes}
    assert reshape_count == 4
    assert scatter_count == 1
    assert not isinstance(by_name["/Reshape_2"].inputs[1], gs.Constant)
    assert by_name["/ScatterND"].outputs == []
    assert by_name["/ScatterND_1"].outputs == []
    assert by_name["stitch_sc/Concat_xy_z"].outputs == [final_out]


def test_portable_model_temporary_paths_are_unique(tmp_path: Path) -> None:
    destination = tmp_path / "converted.onnx"

    first = _temporary_model_path(destination)
    second = _temporary_model_path(destination)

    assert first != second
    assert first.parent == destination.parent
    assert second.parent == destination.parent
