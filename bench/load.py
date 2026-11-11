"""Sequential latency, percentile tails, and sustained load against a running server.

Four separate measurements, because they answer different questions:

  latency     one request at a time -> p50/p95/p99 for Step 1
  throughput  fixed concurrent clients (closed-loop) -> saturation RPS and 503s
  open        fixed arrival rates in req/s (open-loop) -> Step 2 sustained RPS
  soak        steady load for a while -> does memory or latency drift

  python -m bench.load --url http://localhost:8000 --images samples --mode latency
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .accuracy import IMAGE_SUFFIXES, post_image


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(
    name: str,
    latencies: list[float],
    errors: int,
    elapsed: float,
    target_rps: float | None = None,
) -> dict:
    """One row per step, in the shape the challenge scores.

    `rps` counts successful requests only. `sustained_rps` applies the Step 2 rule:
    the rate only counts for a step if at least 99% of that step's requests
    succeeded, so a step that is mostly 503s or timeouts scores zero rather than a
    flattering attempt rate.
    """
    ok = len(latencies)
    requests = ok + errors
    success_rate = ok / requests if requests else 0.0
    rps = ok / elapsed if elapsed else 0.0
    row = {
        "mode": name,
        "requests": requests,
        "ok": ok,
        "errors": errors,
        "success_rate": round(success_rate, 4),
        "wall_s": round(elapsed, 2),
        "rps": round(rps, 3),
        "attempted_rps": round(requests / elapsed, 3) if elapsed else 0.0,
        "sustained_rps": round(rps, 3) if success_rate >= 0.99 else 0.0,
        "p50_ms": round(1000 * percentile(latencies, 0.50), 1),
        "p95_ms": round(1000 * percentile(latencies, 0.95), 1),
        "p99_ms": round(1000 * percentile(latencies, 0.99), 1),
        "mean_ms": round(1000 * statistics.fmean(latencies), 1) if latencies else 0.0,
    }
    if target_rps is not None:
        row["target_rps"] = round(target_rps, 3)
    print(json.dumps(row), flush=True)
    return row


def sequential(url: str, images: list[Path], repeats: int, timeout: float) -> dict:
    for image in images[:1]:
        post_image(url, image, timeout)  # warm up, discarded
    latencies, errors = [], 0
    started = time.perf_counter()
    for _ in range(repeats):
        for image in images:
            t0 = time.perf_counter()
            try:
                post_image(url, image, timeout)
                latencies.append(time.perf_counter() - t0)
            except Exception as exc:  # a 503, a timeout, or a malformed body
                errors += 1
                print("error:", exc)
    return summarize("latency", latencies, errors, time.perf_counter() - started)


def concurrent(url: str, images: list[Path], clients: int, seconds: float, timeout: float) -> dict:
    latencies: list[float] = []
    errors = 0
    lock = threading.Lock()
    stop = time.perf_counter() + seconds

    def worker(offset: int) -> None:
        nonlocal errors
        index = offset
        while time.perf_counter() < stop:
            image = images[index % len(images)]
            index += 1
            t0 = time.perf_counter()
            try:
                post_image(url, image, timeout)
                with lock:
                    latencies.append(time.perf_counter() - t0)
            except Exception:  # never let one bad response kill a load worker
                with lock:
                    errors += 1

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return summarize(f"concurrent_{clients}", latencies, errors, time.perf_counter() - started)


def open_loop(
    url: str,
    images: list[Path],
    rate: float,
    seconds: float,
    timeout: float,
    sender=post_image,
) -> dict:
    """Dispatch requests at a fixed arrival rate `rate` req/s for `seconds` (Step 2).

    Unlike `concurrent`, which waits for one response before sending the next from
    that worker, an open-loop step keeps sending on schedule so queueing delay and
    fast 503s show up at the exact arrival rate that triggered them.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    latencies: list[float] = []
    errors = 0
    lock = threading.Lock()
    interval = 1.0 / rate
    total = max(1, int(round(rate * seconds)))
    max_workers = max(4, min(64, int(rate * timeout) + 4))

    def one(image: Path) -> None:
        nonlocal errors
        t0 = time.perf_counter()
        try:
            sender(url, image, timeout)
            with lock:
                latencies.append(time.perf_counter() - t0)
        except Exception:
            with lock:
                errors += 1

    started = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for index in range(total):
            due = started + index * interval
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            if time.perf_counter() >= started + seconds and index > 0:
                break
            futures.append(pool.submit(one, images[index % len(images)]))
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - started
    return summarize(f"open_{rate:g}rps", latencies, errors, elapsed, target_rps=rate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("latency", "throughput", "open", "soak", "all"), default="all"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--clients", default="1,2,4,8")
    parser.add_argument(
        "--rates",
        default="",
        help="comma-separated open-loop arrival rates in req/s (e.g. 0.1,0.2,0.4,0.8)",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    # 30s matches the challenge's per-request timeout rule in Steps 2 and 3.
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images in {args.images}")

    rows = []
    if args.mode in ("latency", "all"):
        rows.append(sequential(args.url, images, args.repeats, args.timeout))
    if args.mode in ("throughput", "all"):
        for client_count in (int(c) for c in args.clients.split(",") if c.strip()):
            rows.append(concurrent(args.url, images, client_count, args.seconds, args.timeout))
    if args.mode == "open" or (args.mode == "all" and args.rates.strip()):
        rates_str = args.rates.strip() or "0.1,0.2,0.4,0.8"
        for rate in (float(r) for r in rates_str.split(",") if r.strip()):
            rows.append(open_loop(args.url, images, rate, args.seconds, args.timeout))
    if args.mode == "soak":
        clients = int(args.clients.split(",")[-1])
        rows.append(concurrent(args.url, images, clients, args.seconds, args.timeout))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
