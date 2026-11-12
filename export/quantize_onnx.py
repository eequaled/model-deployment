"""Quantize ONNX models for Florence-2 CPU serving.

Builds weight-only 8-bit MatMulNBits for the decoder step graph (the proven accuracy winner),
plus mixed 4-bit transformer + 8-bit LM head step graph and QUInt8 dynamic MatMul text encoder.
decoder_prefill and embed_tokens remain fp32.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantize")

LM_HEAD_NAME = "/language_model/lm_head/MatMul"


def count_quantized_nodes(model_path: Path) -> dict[str, int]:
    m = onnx.load(str(model_path))
    counts: dict[str, int] = {}
    for n in m.graph.node:
        if n.op_type == "MatMulInteger":
            counts["MatMulInteger"] = counts.get("MatMulInteger", 0) + 1
        elif n.op_type == "MatMulNBits":
            bits = None
            for a in n.attribute:
                if a.name == "bits":
                    bits = a.i
                    break
            key = f"MatMulNBits_b{bits}" if bits else "MatMulNBits"
            counts[key] = counts.get(key, 0) + 1
    return counts


def quantize_decoder_step_8bit(models_dir: Path) -> dict:
    src = models_dir / "decoder_step.onnx"
    dst = models_dir / "decoder_step.int8.onnx"
    if not src.exists():
        raise FileNotFoundError(f"Source model missing: {src}")
    logger.info("Quantizing decoder_step to 8-bit MatMulNBits...")
    t0 = time.perf_counter()
    q = MatMulNBitsQuantizer(
        str(src),
        bits=8,
        block_size=32,
        is_symmetric=True,
        accuracy_level=4,
    )
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)
    elapsed = time.perf_counter() - t0
    counts = count_quantized_nodes(dst)
    if counts.get("MatMulNBits_b8", 0) == 0:
        raise RuntimeError("decoder_step.int8.onnx produced 0 8-bit MatMulNBits nodes")
    src_sz = src.stat().st_size / 1e6
    dst_sz = dst.stat().st_size / 1e6
    logger.info(
        f"Finished decoder_step 8-bit in {elapsed:.1f}s: {src_sz:.1f}MB -> {dst_sz:.1f}MB | {counts}"
    )
    return {
        "file": dst.name,
        "scheme": "matmulnbits-qint8-b32-a4",
        "size_mb": round(dst_sz, 1),
        "nodes": counts,
    }


def quantize_encoder_model_uint8(models_dir: Path) -> dict:
    src = models_dir / "encoder_model.onnx"
    dst = models_dir / "encoder_model.int8.onnx"
    if not src.exists():
        raise FileNotFoundError(f"Source model missing: {src}")
    logger.info("Quantizing encoder_model to QUInt8 MatMul dynamic quantization...")
    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        weight_type=QuantType.QUInt8,
    )
    elapsed = time.perf_counter() - t0
    counts = count_quantized_nodes(dst)
    src_sz = src.stat().st_size / 1e6
    dst_sz = dst.stat().st_size / 1e6
    logger.info(
        f"Finished encoder_model uint8 in {elapsed:.1f}s: {src_sz:.1f}MB -> {dst_sz:.1f}MB | {counts}"
    )
    return {
        "file": dst.name,
        "scheme": "dynamic-quint8-matmul-perchannel",
        "size_mb": round(dst_sz, 1),
        "nodes": counts,
    }


def quantize_decoder_step_mixed_4bit(models_dir: Path) -> dict:
    src = models_dir / "decoder_step.onnx"
    dst = models_dir / "decoder_step.int4.onnx"
    temp_pass1 = models_dir / "decoder_step.int4.pass1.tmp"
    if not src.exists():
        raise FileNotFoundError(f"Source model missing: {src}")
    logger.info("Quantizing decoder_step to mixed 4-bit + 8-bit LM head...")
    t0 = time.perf_counter()
    try:
        q4 = MatMulNBitsQuantizer(
            str(src),
            bits=4,
            block_size=32,
            is_symmetric=True,
            accuracy_level=4,
            nodes_to_exclude=[LM_HEAD_NAME],
        )
        q4.process()
        q4.model.save_model_to_file(str(temp_pass1), use_external_data_format=False)

        q8 = MatMulNBitsQuantizer(
            str(temp_pass1),
            bits=8,
            block_size=32,
            is_symmetric=True,
            accuracy_level=4,
        )
        q8.process()
        q8.model.save_model_to_file(str(dst), use_external_data_format=False)
    finally:
        if temp_pass1.exists():
            temp_pass1.unlink()

    elapsed = time.perf_counter() - t0
    counts = count_quantized_nodes(dst)
    b4 = counts.get("MatMulNBits_b4", 0)
    b8 = counts.get("MatMulNBits_b8", 0)
    if b4 != 48 or b8 != 1:
        raise RuntimeError(
            f"Expected 48 4-bit and 1 8-bit MatMulNBits in {dst}, got b4={b4}, b8={b8}"
        )
    src_sz = src.stat().st_size / 1e6
    dst_sz = dst.stat().st_size / 1e6
    logger.info(
        f"Finished decoder_step mixed in {elapsed:.1f}s: {src_sz:.1f}MB -> {dst_sz:.1f}MB | {counts}"
    )
    return {
        "file": dst.name,
        "scheme": "matmulnbits-mixed-4b8-b32-a4",
        "size_mb": round(dst_sz, 1),
        "nodes": counts,
    }


def write_quant_manifest(models_dir: Path) -> Path:
    manifest = {
        "onnxruntime": ort.__version__,
        "source": "microsoft/Florence-2-base",
        "variants": {
            "fp32": {
                "vision_encoder": "vision_encoder.onnx",
                "embed_tokens": "embed_tokens.onnx",
                "encoder_model": "encoder_model.onnx",
                "decoder_prefill": "decoder_prefill.onnx",
                "decoder_step": "decoder_step.onnx",
            },
            "int8": {
                "vision_encoder": "vision_encoder.onnx",
                "embed_tokens": "embed_tokens.onnx",
                "encoder_model": "encoder_model.int8.onnx",
                "decoder_prefill": "decoder_prefill.onnx",
                "decoder_step": "decoder_step.int8.onnx",
            },
            "int4": {
                "vision_encoder": "vision_encoder.onnx",
                "embed_tokens": "embed_tokens.onnx",
                "encoder_model": "encoder_model.int8.onnx",
                "decoder_prefill": "decoder_prefill.onnx",
                "decoder_step": "decoder_step.int4.onnx",
            },
        },
        "schemes": {
            "vision_encoder": "fp32",
            "embed_tokens": "fp32",
            "encoder_model": "fp32",
            "encoder_model.int8": "dynamic-quint8-matmul-perchannel",
            "decoder_prefill": "fp32",
            "decoder_step.int8": "matmulnbits-qint8-b32-a4",
            "decoder_step.int4": "matmulnbits-mixed-4b8-b32-a4",
        },
    }
    manifest_path = models_dir / "quant.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote {manifest_path}")
    return manifest_path


def extract_embed_tokens_npy(models_dir: Path) -> Path | None:
    npy_path = models_dir / "embed_tokens.npy"
    if npy_path.exists():
        logger.info(f"{npy_path} already exists")
        return npy_path
    onnx_path = models_dir / "embed_tokens.onnx"
    if not onnx_path.exists():
        return None
    logger.info("Extracting embed_tokens.npy from embed_tokens.onnx...")
    m = onnx.load(str(onnx_path))
    from onnx import numpy_helper
    for init in m.graph.initializer:
        if "weight" in init.name or init.name == "embeddings.weight":
            arr = numpy_helper.to_array(init)
            np.save(str(npy_path), arr)
            logger.info(f"Saved {npy_path} shape={arr.shape} dtype={arr.dtype}")
            return npy_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="Directory with ONNX models")
    args = parser.parse_args()

    models_dir: Path = args.models
    if not (models_dir / "meta.json").exists():
        logger.error(f"meta.json missing in {models_dir}")
        return 1

    t0 = time.perf_counter()
    extract_embed_tokens_npy(models_dir)
    quantize_encoder_model_uint8(models_dir)
    quantize_decoder_step_8bit(models_dir)
    quantize_decoder_step_mixed_4bit(models_dir)
    write_quant_manifest(models_dir)
    logger.info(f"Quantization completed in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
