"""Numerical equivalence test for decoder graph surgery.

Tests that prefill + step producing token 1 logits match before and after surgery
within 1e-4 tolerance across fp32, int8, and int4 variants.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest

MODELS_DIR = Path("/home/eequaled/florence-lab/onnx-export-v3")
SURGERY_DIR = Path("/tmp/opencode/surgery-test")


def build_mock_feeds(prefill_sess: ort.InferenceSession, step_sess: ort.InferenceSession):
    # Dummy prompt embeddings [1, 10, 768] and enc states [1, 587, 768]
    prompt_embeds = np.random.randn(1, 10, 768).astype(np.float32)
    last_hidden = np.random.randn(1, 587, 768).astype(np.float32)
    mask = np.ones((1, 587), dtype=np.float32)

    prefill_feed = {
        "inputs_embeds": prompt_embeds,
        "encoder_hidden_states": last_hidden,
        "encoder_attention_mask": mask,
    }
    return prefill_feed


@pytest.mark.parametrize("variant", ["fp32", "int8", "int4"])
def test_surgery_numerical_equivalence(variant):
    if not SURGERY_DIR.exists() or not (SURGERY_DIR / "decoder_prefill.onnx").exists():
        pytest.skip("surgery-test directory not populated")

    # Step file names
    step_map = {
        "fp32": "decoder_step.onnx",
        "int8": "decoder_step.int8.onnx",
        "int4": "decoder_step.int4.onnx",
    }
    step_file = step_map[variant]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1

    # Original sessions
    orig_pref_sess = ort.InferenceSession(
        str(MODELS_DIR / "decoder_prefill.onnx"), so, providers=["CPUExecutionProvider"]
    )
    orig_step_sess = ort.InferenceSession(
        str(MODELS_DIR / step_file), so, providers=["CPUExecutionProvider"]
    )

    # Surgered sessions
    surg_pref_sess = ort.InferenceSession(
        str(SURGERY_DIR / "decoder_prefill.onnx"), so, providers=["CPUExecutionProvider"]
    )
    surg_step_sess = ort.InferenceSession(
        str(SURGERY_DIR / step_file), so, providers=["CPUExecutionProvider"]
    )

    np.random.seed(42)
    prompt_embeds = np.random.randn(1, 10, 768).astype(np.float32)
    last_hidden = np.random.randn(1, 587, 768).astype(np.float32)
    mask = np.ones((1, 587), dtype=np.float32)

    p_feed = {
        "decoder_inputs_embeds": prompt_embeds,
        "encoder_hidden_states": last_hidden,
        "encoder_attention_mask": mask,
    }

    # Run prefill
    orig_pref_out = orig_pref_sess.run(None, p_feed)
    surg_pref_out = surg_pref_sess.run(None, p_feed)

    orig_pref_map = {o.name: v for o, v in zip(orig_pref_sess.get_outputs(), orig_pref_out)}
    surg_pref_map = {o.name: v for o, v in zip(surg_pref_sess.get_outputs(), surg_pref_out)}

    # Prefill logits must be identical
    np.testing.assert_allclose(
        orig_pref_map["logits"], surg_pref_map["logits"], rtol=1e-5, atol=1e-5
    )

    # Next feed step with token 0 embedding
    tok0_embed = np.random.randn(1, 1, 768).astype(np.float32)

    def make_step_feed(pref_map):
        feed = {
            "decoder_inputs_embeds": tok0_embed,
            "encoder_attention_mask": mask,
        }
        for i in range(6):
            feed[f"past_key_values.{i}.decoder.key"] = pref_map[f"present.{i}.decoder.key"]
            feed[f"past_key_values.{i}.decoder.value"] = pref_map[f"present.{i}.decoder.value"]
            feed[f"past_key_values.{i}.encoder.key"] = pref_map[f"present.{i}.encoder.key"]
            feed[f"past_key_values.{i}.encoder.value"] = pref_map[f"present.{i}.encoder.value"]
        return feed

    orig_step_out = orig_step_sess.run(None, make_step_feed(orig_pref_map))
    surg_step_out = surg_step_sess.run(None, make_step_feed(surg_pref_map))

    orig_step_map = {o.name: v for o, v in zip(orig_step_sess.get_outputs(), orig_step_out)}
    surg_step_map = {o.name: v for o, v in zip(surg_step_sess.get_outputs(), surg_step_out)}

    # Verify logits are numerically equivalent within 1e-4
    diff = np.max(np.abs(orig_step_map["logits"] - surg_step_map["logits"]))
    assert diff < 1e-4, f"Variant {variant} max logits diff {diff} exceeded 1e-4"
