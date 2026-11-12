"""Tests for precision loading and fallback behavior."""
import json
from pathlib import Path
import pytest

from server.ocr_engine import load_precision_files, PRECISION_MODES


def test_precision_modes_defined():
    assert "auto" in PRECISION_MODES
    assert "fp32" in PRECISION_MODES
    assert "int8" in PRECISION_MODES
    assert "int4" in PRECISION_MODES


def test_unknown_precision_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown precision"):
        load_precision_files(tmp_path, requested="bogus")


def test_load_precision_with_manifest(tmp_path):
    manifest = {
        "onnxruntime": "1.31.1",
        "variants": {
            "fp32": {
                "vision_encoder": "vis.onnx",
                "embed_tokens": "emb.onnx",
                "encoder_model": "enc.onnx",
                "decoder_prefill": "pref.onnx",
                "decoder_step": "step.onnx",
            },
            "int8": {
                "vision_encoder": "vis.int8.onnx",
                "embed_tokens": "emb.onnx",
                "encoder_model": "enc.int8.onnx",
                "decoder_prefill": "pref.onnx",
                "decoder_step": "step.int8.onnx",
            },
        },
    }
    (tmp_path / "quant.json").write_text(json.dumps(manifest))
    # Create fake files
    for fname in ("vis.onnx", "emb.onnx", "enc.onnx", "pref.onnx", "step.onnx",
                  "vis.int8.onnx", "enc.int8.onnx", "step.int8.onnx"):
        (tmp_path / fname).touch()

    # Request int8
    files_int8 = load_precision_files(tmp_path, requested="int8")
    assert files_int8["vision_encoder"] == tmp_path / "vis.int8.onnx"
    assert files_int8["decoder_step"] == tmp_path / "step.int8.onnx"

    # Request fp32
    files_fp32 = load_precision_files(tmp_path, requested="fp32")
    assert files_fp32["vision_encoder"] == tmp_path / "vis.onnx"

    # Request auto (resolves to int8 when quant.json present)
    files_auto = load_precision_files(tmp_path, requested="auto")
    assert files_auto["vision_encoder"] == tmp_path / "vis.int8.onnx"


def test_load_precision_missing_file_raises(tmp_path):
    manifest = {
        "variants": {
            "int8": {
                "vision_encoder": "vis.int8.onnx",
                "embed_tokens": "emb.onnx",
                "encoder_model": "enc.int8.onnx",
                "decoder_prefill": "pref.onnx",
                "decoder_step": "step.int8.onnx",
            }
        }
    }
    (tmp_path / "quant.json").write_text(json.dumps(manifest))
    # Only create one file
    (tmp_path / "vis.int8.onnx").touch()

    with pytest.raises(FileNotFoundError, match="missing required model files"):
        load_precision_files(tmp_path, requested="int8", allow_fallback=False)


def test_load_precision_fallback_to_fp32(tmp_path):
    manifest = {
        "variants": {
            "fp32": {
                "vision_encoder": "vis.onnx",
                "embed_tokens": "emb.onnx",
                "encoder_model": "enc.onnx",
                "decoder_prefill": "pref.onnx",
                "decoder_step": "step.onnx",
            },
            "int8": {
                "vision_encoder": "missing.onnx",
                "embed_tokens": "emb.onnx",
                "encoder_model": "enc.onnx",
                "decoder_prefill": "pref.onnx",
                "decoder_step": "step.onnx",
            },
        }
    }
    (tmp_path / "quant.json").write_text(json.dumps(manifest))
    for fname in ("vis.onnx", "emb.onnx", "enc.onnx", "pref.onnx", "step.onnx"):
        (tmp_path / fname).touch()

    # int8 missing a file with allow_fallback=True returns fp32 files
    files = load_precision_files(tmp_path, requested="int8", allow_fallback=True)
    assert files["vision_encoder"] == tmp_path / "vis.onnx"
