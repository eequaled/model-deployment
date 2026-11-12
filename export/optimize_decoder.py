"""Decoder graph surgery: transpose cross-attention keys once in prefill.

Prefill emits present.{i}.encoder.key transposed with perm [0, 2, 3, 1] directly
(matching the fused [0, 2, 1, 3] + [0, 1, 3, 2] permutation), and step drops the
per-step Transpose_1 nodes that previously did perm [0, 1, 3, 2] on every token.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import onnx
from onnx import helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("optimize_decoder")


def optimize_prefill(model_path: Path, out_path: Path | None = None) -> None:
    if out_path is None:
        out_path = model_path
    m = onnx.load(str(model_path))
    modified_count = 0
    for node in m.graph.node:
        if node.op_type == "Transpose" and any("encoder.key" in o for o in node.output):
            # Original perm is [0, 2, 1, 3] from Reshape [batch, seq, heads, dim]
            # to [batch, heads, seq, dim].
            # Step then did Transpose_1 with [0, 1, 3, 2] to get [batch, heads, dim, seq].
            # Direct combined perm is [0, 2, 3, 1].
            for a in node.attribute:
                if a.name == "perm":
                    assert list(a.ints) == [0, 2, 1, 3], f"Unexpected perm {list(a.ints)}"
                    a.ints[:] = [0, 2, 3, 1]
                    modified_count += 1
                    break
    if modified_count != 6:
        raise RuntimeError(
            f"Expected 6 cross-key Transpose nodes in prefill, modified {modified_count}"
        )

    # Swap dimensions 2 and 3 in graph.output (and value_info) for present.{i}.encoder.key
    for out in list(m.graph.output) + list(m.graph.value_info):
        if "encoder.key" in out.name:
            dims = out.type.tensor_type.shape.dim
            if len(dims) == 4:
                # swap dims[2] and dims[3]
                d2_val, d2_param = dims[2].dim_value, dims[2].dim_param
                d3_val, d3_param = dims[3].dim_value, dims[3].dim_param
                if d2_val:
                    dims[3].dim_value = d2_val
                    dims[3].ClearField("dim_param")
                else:
                    dims[3].dim_param = d2_param
                    dims[3].ClearField("dim_value")
                if d3_val:
                    dims[2].dim_value = d3_val
                    dims[2].ClearField("dim_param")
                else:
                    dims[2].dim_param = d3_param
                    dims[2].ClearField("dim_value")

    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))
    logger.info(f"Optimized {modified_count} cross-key outputs in prefill -> {out_path}")


def optimize_step(model_path: Path, out_path: Path | None = None) -> None:
    if out_path is None:
        out_path = model_path
    m = onnx.load(str(model_path))

    # Find the 6 Transpose nodes: input past_key_values.{i}.encoder.key
    nodes_to_remove = set()
    replacement_map = {}

    for node in m.graph.node:
        if (
            node.op_type == "Transpose"
            and len(node.input) == 1
            and "encoder.key" in node.input[0]
            and node.input[0].startswith("past_key_values.")
        ):
            perm = None
            for a in node.attribute:
                if a.name == "perm":
                    perm = list(a.ints)
                    break
            if perm != [0, 1, 3, 2]:
                raise RuntimeError(
                    f"Unexpected perm {perm} on step cross-key node {node.name}"
                )
            in_name = node.input[0]
            out_name = node.output[0]
            nodes_to_remove.add(node.name)
            replacement_map[out_name] = in_name

    if len(nodes_to_remove) != 6:
        raise RuntimeError(
            f"Expected 6 step cross-key Transpose nodes to remove, found {len(nodes_to_remove)}"
        )

    # Rewrite downstream consumer inputs
    for node in m.graph.node:
        for idx, in_tensor in enumerate(node.input):
            if in_tensor in replacement_map:
                node.input[idx] = replacement_map[in_tensor]

    # Remove the 6 Transpose nodes from the graph
    new_nodes = [n for n in m.graph.node if n.name not in nodes_to_remove]
    m.graph.ClearField("node")
    m.graph.node.extend(new_nodes)

    # Swap dimensions 2 and 3 in graph.input (and value_info) for past_key_values.{i}.encoder.key
    for inp in list(m.graph.input) + list(m.graph.value_info):
        if "encoder.key" in inp.name and inp.name.startswith("past_key_values."):
            dims = inp.type.tensor_type.shape.dim
            if len(dims) == 4:
                # swap dims[2] and dims[3]
                d2_val, d2_param = dims[2].dim_value, dims[2].dim_param
                d3_val, d3_param = dims[3].dim_value, dims[3].dim_param
                if d2_val:
                    dims[3].dim_value = d2_val
                    dims[3].ClearField("dim_param")
                else:
                    dims[3].dim_param = d2_param
                    dims[3].ClearField("dim_value")
                if d3_val:
                    dims[2].dim_value = d3_val
                    dims[2].ClearField("dim_param")
                else:
                    dims[2].dim_param = d3_param
                    dims[2].ClearField("dim_value")

    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))
    logger.info(f"Removed {len(nodes_to_remove)} cross-key Transpose nodes in step -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="Directory containing models")
    args = parser.parse_args()

    models_dir: Path = args.models
    t0 = time.perf_counter()

    # 1. Optimize prefill
    prefill_path = models_dir / "decoder_prefill.onnx"
    if prefill_path.exists():
        optimize_prefill(prefill_path)

    # 2. Optimize step variants: fp32, int8, and int4
    for step_name in ("decoder_step.onnx", "decoder_step.int8.onnx", "decoder_step.int4.onnx"):
        p = models_dir / step_name
        if p.exists():
            optimize_step(p)

    logger.info(f"Decoder optimization complete in {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
