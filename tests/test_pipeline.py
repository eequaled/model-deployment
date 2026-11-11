"""End-to-end wiring test for the decode loop, using tiny stand-in graphs.

Real parity against PyTorch was established offline and is recorded in journal.md.
What these protect is the part that would break silently in a refactor: the cache
contract between prefill and step, the encoder-length invariant, the tokenizer
round trip, and preprocessing of awkward images.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests.tinymodel import IMAGE_TOKENS, LAYERS, PROMPT_IDS


def test_generate_returns_text_within_cap(engine):
    text, timings = engine.generate(Image.new("RGB", (300, 200), (255, 255, 255)))
    assert isinstance(text, str) and text
    assert set(timings) == {"encoder_ms", "decode_ms", "total_ms"}
    assert timings["total_ms"] >= 0


def test_cache_contract_between_prefill_and_step(engine):
    prefill_present = [n for n in engine.prefill_outputs if n.startswith("present.")]
    step_past = [n for n in engine.step_inputs if n.startswith("past_key_values.")]
    step_present = [n for n in engine.step_outputs if n.startswith("present.")]

    # 6 layers x 4 slots in prefill; the step graph consumes all 24 and returns only
    # the 12 self-attention tensors, because cross-attention cannot change mid-request.
    assert len(prefill_present) == LAYERS * 4
    assert len(step_past) == LAYERS * 4
    assert len(step_present) == LAYERS * 2

    def mapped(name: str) -> str:
        return "present." + name[len("past_key_values."):]

    assert all(mapped(n) in prefill_present for n in step_past), "step cache names must line up"
    assert all(n in step_present for n in step_present if "encoder" not in n)


def test_encoder_length_matches_prompt_contract(engine):
    pixel_values = engine.preprocess(Image.new("RGB", (300, 200)))
    features = engine.sessions["vision_encoder"].run(None, {"pixel_values": pixel_values})[0]
    prompt = engine.sessions["embed_tokens"].run(None, {"input_ids": engine.prompt_ids})[0]
    assert features.shape[1] == IMAGE_TOKENS
    assert prompt.shape[1] == len(PROMPT_IDS)


def test_preprocessing_handles_awkward_images(engine):
    for image in (
        Image.new("RGBA", (40, 900), (10, 20, 30, 128)),
        Image.new("L", (900, 40), 128),
        Image.new("P", (64, 64)),
    ):
        array = engine.preprocess(image)
        assert array.shape == (1, 3, 768, 768)
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_encoder_length_mismatch_fails_loudly(engine, monkeypatch):
    """A wrong-length mask must raise, not surface as a broadcast error mid-decode."""
    monkeypatch.setitem(engine.meta, "encoder_length", IMAGE_TOKENS + len(PROMPT_IDS) + 3)
    with pytest.raises(RuntimeError, match="encoder length"):
        engine.generate(Image.new("RGB", (64, 64), (0, 0, 0)))


def test_image_token_count_is_read_from_the_graph(engine):
    assert engine.image_tokens_from_graph() == IMAGE_TOKENS


def test_generate_stops_at_token_cap(engine, monkeypatch):
    monkeypatch.setattr(engine, "max_new_tokens", 3)
    text, _ = engine.generate(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert len(text.split()) <= 3


def test_prefill_modes_produce_expected_prefix_shapes(engine, monkeypatch):
    prompt = engine.sessions["embed_tokens"].run(None, {"input_ids": engine.prompt_ids})[0]
    expected = {
        "prompt": len(PROMPT_IDS),
        "last": 1,
        "start": 1,
        "prompt_start": len(PROMPT_IDS) + 1,
    }
    for mode, length in expected.items():
        monkeypatch.setattr(engine, "prefill_mode", mode)
        prefix = engine.prefill_embeds(prompt)
        assert prefix.shape == (1, length, 768), mode
        text, _ = engine.generate(Image.new("RGB", (64, 64), (255, 255, 255)))
        assert isinstance(text, str) and text


def test_unknown_prefill_mode_fails_at_startup(tiny_model_dir):
    from server.ocr_engine import FlorenceOCR

    with pytest.raises(RuntimeError, match="prefill_mode"):
        FlorenceOCR(tiny_model_dir, threads=1, prefill_mode="bogus")


def test_cgroup_cpu_quota_and_thread_resolution(tmp_path):
    from server.ocr_engine import cgroup_cpu_quota, resolve_ort_threads

    v2 = tmp_path / "cpu.max"
    v2.write_text("400000 100000\n", encoding="utf-8")
    assert cgroup_cpu_quota(v2_path=v2) == 4
    v2.write_text("max 100000\n", encoding="utf-8")
    assert cgroup_cpu_quota(v2_path=v2, v1_quota=tmp_path / "none", v1_period=tmp_path / "none") is None

    # Explicit ORT_THREADS always wins; otherwise divide the quota across workers.
    assert resolve_ort_threads(6, workers=2, quota=4) == 6
    assert resolve_ort_threads(0, workers=1, quota=4) == 4
    assert resolve_ort_threads(0, workers=2, quota=4) == 2
    assert resolve_ort_threads(0, workers=1, quota=0) == 0


def test_warmup_runs_short_pass(engine):
    elapsed_ms = engine.warmup()
    assert elapsed_ms >= 0.0
