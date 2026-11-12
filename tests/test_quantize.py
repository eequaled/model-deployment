"""Tests for the quantization export script and generated files."""
import json
from pathlib import Path
import pytest

MODELS_DIR = Path("/home/eequaled/florence-lab/onnx-export-v3")


@pytest.fixture
def manifest():
    manifest_path = MODELS_DIR / "quant.json"
    if not manifest_path.exists():
        pytest.skip("quant.json not present in models directory")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_quant_manifest_structure(manifest):
    assert "onnxruntime" in manifest
    assert "variants" in manifest
    assert "schemes" in manifest
    for key in ("fp32", "int8", "int4"):
        assert key in manifest["variants"]
        variant = manifest["variants"][key]
        assert "vision_encoder" in variant
        assert "embed_tokens" in variant
        assert "encoder_model" in variant
        assert "decoder_prefill" in variant
        assert "decoder_step" in variant


def test_quant_manifest_files_exist(manifest):
    for var_name, variant in manifest["variants"].items():
        for role, fname in variant.items():
            fpath = MODELS_DIR / fname
            assert fpath.exists(), f"File {fname} for variant {var_name}:{role} missing"
            assert fpath.stat().st_size > 0


def test_int4_mixed_node_count():
    from export.quantize_onnx import count_quantized_nodes

    int4_path = MODELS_DIR / "decoder_step.int4.onnx"
    if not int4_path.exists():
        pytest.skip("decoder_step.int4.onnx missing")
    counts = count_quantized_nodes(int4_path)
    assert counts.get("MatMulNBits_b4") == 48
    assert counts.get("MatMulNBits_b8") == 1


def test_int8_step_node_count():
    from export.quantize_onnx import count_quantized_nodes

    int8_path = MODELS_DIR / "decoder_step.int8.onnx"
    if not int8_path.exists():
        pytest.skip("decoder_step.int8.onnx missing")
    counts = count_quantized_nodes(int8_path)
    assert counts.get("MatMulNBits_b8") == 49
    assert counts.get("MatMulNBits_b4") is None
