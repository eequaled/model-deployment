"""Load-test math: percentiles and the challenge's sustained-rate rule.

These are pure functions, so they run without a server or weights. What they pin is
the Step 2 definition: a step's rate only counts if at least 99% of its requests
succeeded, and the rate itself counts successes, not attempts.
"""

from __future__ import annotations

from bench.load import percentile, summarize


def test_percentile_handles_empty_and_short_samples():
    assert percentile([], 0.5) == 0.0
    assert percentile([1.0], 0.99) == 1.0
    assert percentile([1.0, 2.0, 3.0], 0.50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_rps_counts_successes_and_a_99_percent_step_is_sustained(capsys):
    row = summarize("step", [1.0] * 99, 1, 10.0)
    assert row["requests"] == 100
    assert row["success_rate"] == 0.99
    assert row["rps"] == 9.9, "failed requests must not inflate the rate"
    assert row["attempted_rps"] == 10.0
    assert row["sustained_rps"] == 9.9
    capsys.readouterr()  # summarize prints its row


def test_a_step_below_99_percent_scores_zero_throughput(capsys):
    row = summarize("step", [1.0] * 98, 2, 10.0)
    assert row["success_rate"] == 0.98
    assert row["sustained_rps"] == 0.0
    capsys.readouterr()
