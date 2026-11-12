"""Graph contracts and CER scoring.

The graph tests read the *real* exported model when it is present, and skip
otherwise, so the fast suite still runs on a laptop. Run them inside the image or
with MODEL_DIR pointing at a local export to check the shapes the server relies on:

    MODEL_DIR=/models pytest tests/test_graphs.py -v

The metrics tests are pure functions and always run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from bench.metrics import accuracy, character_error_rate, edit_distance

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))

needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "meta.json").exists(),
    reason=f"no exported model at {MODEL_DIR} (set MODEL_DIR)",
)


# -- CER: the challenge's scoring rule ---------------------------------------
def test_edit_distance_basics():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("kitten", "sitting") == 3


def test_cer_is_relative_to_reference_length():
    assert character_error_rate("abcd", "abcd") == 0.0
    assert character_error_rate("abce", "abcd") == pytest.approx(0.25)
    assert character_error_rate("", "abcd") == 1.0


def test_accuracy_clamps_at_zero():
    # max(0, 1 - CER): a much longer prediction cannot score below zero
    assert accuracy("abcd", "abcd") == 1.0
    assert accuracy("abcdefgh", "ab") == 0.0
    assert accuracy("", "abcd") == 0.0


def test_accuracy_penalises_whitespace_differences():
    # OCR output is whitespace-sensitive: "a b" vs "ab" is a real CER difference
    assert accuracy("a b", "ab") < 1.0
    assert accuracy("ab", "a b") < 1.0


def test_empty_reference_is_handled_consistently():
    # ambiguous case kept explicit so nobody "fixes" it silently later
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("x", "") == 1.0


# -- exported graphs ---------------------------------------------------------
@needs_model
def test_meta_describes_a_consistent_model():
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["image_tokens"] == 577, "DaViT on 768x768 should emit 577 tokens"
    assert len(meta["prompt_ids"]) == 10, "the OCR prompt tokenizes to 10 ids"
    expected_len = meta["image_tokens"] + len(meta["prompt_ids"])
    assert meta.get("encoder_length", expected_len) == expected_len
    assert meta["decoder_layers"] == 6
    assert meta["prompt_text"].startswith("What is the text")
    assert meta.get("prefill_mode", "start") in ("prompt", "last", "start", "prompt_start")


@needs_model
def test_every_graph_loads_and_exposes_expected_io():
    expected = {
        "vision_encoder": (
            [["pixel_values"]],
            ["image_features"],
        ),
        "embed_tokens": (
            [["input_ids"]],
            ["inputs_embeds"],
        ),
        "encoder_model": (
            [["inputs_embeds", "attention_mask"]],
            ["last_hidden_state"],
        ),
        "decoder_prefill": (
            [
                ["encoder_hidden_states", "encoder_attention_mask", "decoder_inputs_embeds"],
                ["encoder_hidden_states", "encoder_attention_mask", "decoder_input_embeds"],
            ],
            ["logits"],
        ),
        "decoder_step": (
            [
                ["encoder_attention_mask", "decoder_inputs_embeds"],
                ["encoder_attention_mask", "decoder_input_embeds"],
                ["encoder_attention_mask", "input_ids"],
            ],
            ["logits"],
        ),
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    for name, (valid_inputs, outputs) in expected.items():
        path = MODEL_DIR / f"{name}.onnx"
        assert path.exists(), f"{name}.onnx missing"
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        declared_in = [i.name for i in session.get_inputs() if not i.name.startswith("past_key_values.")]
        if name == "decoder_step" and declared_in[:1] == ["encoder_hidden_states"]:
            declared_in = declared_in[1:]
        assert declared_in in valid_inputs, f"{name}: got {declared_in}, expected one of {valid_inputs}"
        assert [o.name for o in session.get_outputs()][: len(outputs)] == outputs, name


@needs_model
def test_cache_tensor_counts_match_the_layers():
    prefill = ort.InferenceSession(str(MODEL_DIR / "decoder_prefill.onnx"),
                                   providers=["CPUExecutionProvider"])
    step = ort.InferenceSession(str(MODEL_DIR / "decoder_step.onnx"),
                                providers=["CPUExecutionProvider"])
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    layers = meta["decoder_layers"]

    present = [o.name for o in prefill.get_outputs() if o.name.startswith("present.")]
    past = [i.name for i in step.get_inputs() if i.name.startswith("past_key_values.")]
    returned = [o.name for o in step.get_outputs() if o.name.startswith("present.")]

    assert len(present) == layers * 4, "self + cross key/value per layer"
    assert len(past) == layers * 4, "step consumes the whole cache"
    assert len(returned) == layers * 2, "step updates only self-attention"
    # every cache tensor the step graph wants must be produced by prefill
    assert {p[len("past_key_values."):] for p in past} == {
        p[len("present."):] for p in present
    }


@needs_model
def test_onnx_graphs_are_structurally_valid():
    """onnx.checker on each file: catches a truncated or hand-edited graph."""
    for path in sorted(MODEL_DIR.glob("*.onnx")):
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)


@needs_model
def test_weights_are_fp32_not_quantized():
    """Guards against silently shipping the INT8 graphs that drifted 7/64 tokens."""
    for path in sorted(MODEL_DIR.glob("*.onnx")):
        if ".int8." in path.name or ".int4." in path.name:
            continue
        model = onnx.load(str(path), load_external_data=False)
        types = {t.data_type for t in model.graph.initializer}
        assert onnx.TensorProto.FLOAT in types, f"{path.name} has no fp32 weights"
        assert onnx.TensorProto.INT8 not in types, f"{path.name} contains INT8 weights"


@needs_model
def test_vision_encoder_is_fixed_at_768():
    session = ort.InferenceSession(str(MODEL_DIR / "vision_encoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert list(shape) == [1, 3, 768, 768], f"unexpected vision input {shape}"
