"""Tests for bench.load summary and timing aggregation."""
from bench.load import summarize


def test_summarize_with_timings():
    latencies = [0.1, 0.2, 0.3]
    errors = 0
    elapsed = 0.6
    timings = [
        {"decode_tokens_per_s": 50.0, "total_tokens_per_s": 30.0, "vision_ms": 100.0},
        {"decode_tokens_per_s": 60.0, "total_tokens_per_s": 35.0, "vision_ms": 110.0},
        {"decode_tokens_per_s": 70.0, "total_tokens_per_s": 40.0, "vision_ms": 120.0},
    ]
    row = summarize("test", latencies, errors, elapsed, timings_list=timings)
    assert row["requests"] == 3
    assert row["ok"] == 3
    assert row["decode_tokens_per_s_mean"] == 60.0
    assert row["decode_tokens_per_s_p50"] == 60.0
    assert row["vision_ms_mean"] == 110.0


def test_summarize_with_missing_fields():
    latencies = [0.1, 0.2]
    errors = 0
    elapsed = 0.3
    # Second row missing decode_tokens_per_s
    timings = [
        {"decode_tokens_per_s": 50.0},
        {"vision_ms": 110.0},
    ]
    row = summarize("test_missing", latencies, errors, elapsed, timings_list=timings)
    assert row["decode_tokens_per_s_mean"] == 50.0
    assert row["vision_ms_mean"] == 110.0
