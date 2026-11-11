"""Load-test math: percentiles and the challenge's sustained-rate rule.

These are pure functions, so they run without a server or weights. What they pin is
the Step 2 definition: a step's rate only counts if at least 99% of its requests
succeeded, and the rate itself counts successes, not attempts.
"""

from __future__ import annotations

from pathlib import Path

from bench.load import open_loop, percentile, summarize


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


def test_open_loop_dispatches_at_target_rate_and_records_target(capsys):
    seen = []

    def fake_sender(url: str, image: Path, timeout: float) -> None:
        seen.append(image.name)

    row = open_loop(
        "http://localhost:8000",
        [Path("a.png"), Path("b.png")],
        rate=20.0,
        seconds=0.25,
        timeout=1.0,
        sender=fake_sender,
    )
    assert row["target_rps"] == 20.0
    assert row["ok"] == len(seen) >= 3
    assert row["errors"] == 0
    assert row["sustained_rps"] > 0.0
    capsys.readouterr()


def test_png_header_only_bomb_triggers_413_on_app(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from bench.accuracy import load_refs
    from bench.stress import png_header_only_bomb
    from server import app as app_module

    refs_file = tmp_path / "refs.json"
    refs_file.write_text('{"images": [{"filename": "a.png", "text": "hello"}]}', encoding="utf-8")
    assert load_refs(refs_file) == {"a.png": "hello"}

    monkeypatch.setattr(app_module, "WARMUP", False)
    monkeypatch.setattr(app_module, "build_engine", lambda: object())
    with TestClient(app_module.app) as client:
        resp = client.post(
            "/ocr",
            files={"image": ("bomb.png", png_header_only_bomb(8000, 8000), "image/png")},
        )
        assert resp.status_code == 413
        assert "pixel" in resp.json()["error"]


def test_overload_spike_counts_timeouts_as_failures(tmp_path, monkeypatch):
    """A client that times out must count against the step, not vanish from it."""
    from bench import stress

    image = tmp_path / "a.png"
    image.write_bytes(b"not really a png, never sent")

    calls = []

    def fake_request(url, path, method="GET", body=None, headers=None, timeout=30.0):
        calls.append(path)
        if len(calls) % 2 == 0:
            raise TimeoutError("no answer within the budget")
        return 200, {"text": "ok"}, 1.0

    monkeypatch.setattr(stress, "raw_request", fake_request)
    row = stress.run_overload_spike("http://localhost:8000", [image], clients=4, timeout=1.0)

    assert len(row["unexpected_statuses"]) + row["status_200"] + row["status_503"] == 4
    assert 0 in row["unexpected_statuses"], "timeouts must be recorded as status 0"
    assert row["pass"] is False
