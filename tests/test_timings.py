"""Tests for timings, tokens per second, and instrumentation."""
from pathlib import Path
from PIL import Image
import pytest

from server.ocr_engine import FlorenceOCR

MODELS_DIR = Path("/home/eequaled/florence-lab/onnx-export-v3")


@pytest.fixture(scope="module")
def real_engine():
    if not MODELS_DIR.exists() or not (MODELS_DIR / "meta.json").exists():
        pytest.skip("real models directory missing")
    eng = FlorenceOCR(MODELS_DIR, threads=4, precision="fp32")
    eng.warmup()
    return eng


def test_timing_keys_and_formulas(real_engine):
    img = Image.open("samples/example.png")
    text, timings = real_engine.generate(img)

    required_keys = [
        "encoder_ms",
        "decode_ms",
        "total_ms",
        "vision_ms",
        "text_encoder_ms",
        "prefill_ms",
        "embed_ms",
        "steps",
        "tokens",
        "decode_tokens_per_s",
        "total_tokens_per_s",
    ]
    for k in required_keys:
        assert k in timings, f"Missing timing key {k}"

    steps = timings["steps"]
    tokens = timings["tokens"]
    decode_ms = timings["decode_ms"]
    total_ms = timings["total_ms"]

    # tokens = steps + 1 (prefill token + step tokens)
    assert tokens == steps + 1, f"Expected tokens ({tokens}) == steps ({steps}) + 1"

    # Formula check
    if decode_ms > 0:
        expected_decode_rate = round(steps * 1000.0 / decode_ms, 2)
        assert abs(timings["decode_tokens_per_s"] - expected_decode_rate) < 0.05
    if total_ms > 0:
        expected_total_rate = round(tokens * 1000.0 / total_ms, 2)
        assert abs(timings["total_tokens_per_s"] - expected_total_rate) < 0.05
