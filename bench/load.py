"""Sequential latency, percentile tails, and sustained load against a running server.

Three separate measurements, because they answer different questions:

  latency   one request at a time -> p50/p95/p99 for the scoring step
  throughput concurrent clients -> requests per second, and how many get a 503
  soak      steady load for a while -> does memory or latency drift

  python -m bench.load --url http://localhost:8000 --images samples --mode latency
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
from pathlib import Path

from .accuracy import IMAGE_SUFFIXES, post_image


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(name: str, latencies: list[float], errors: int, elapsed: float) -> dict:
    row = {
        "mode": name,
        "requests": len(latencies) + errors,
        "ok": len(latencies),
        "errors": errors,
        "wall_s": round(elapsed, 2),
        "rps": round((len(latencies) + errors) / elapsed, 3) if elapsed else 0.0,
        "p50_ms": round(1000 * percentile(latencies, 0.50), 1),
        "p95_ms": round(1000 * percentile(latencies, 0.95), 1),
        "p99_ms": round(1000 * percentile(latencies, 0.99), 1),
        "mean_ms": round(1000 * statistics.fmean(latencies), 1) if latencies else 0.0,
    }
    print(json.dumps(row))
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
            except (urllib.error.URLError, TimeoutError) as exc:
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
            except (urllib.error.URLError, TimeoutError):
                with lock:
                    errors += 1

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return summarize(f"concurrent_{clients}", latencies, errors, time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--mode", choices=("latency", "throughput", "soak", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--clients", default="1,2,4,8")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images in {args.images}")

    rows = []
    if args.mode in ("latency", "all"):
        rows.append(sequential(args.url, images, args.repeats, args.timeout))
    if args.mode in ("throughput", "all"):
        for client_count in (int(c) for c in args.clients.split(",")):
            rows.append(concurrent(args.url, images, client_count, args.seconds, args.timeout))
    if args.mode == "soak":
        clients = int(args.clients.split(",")[-1])
        rows.append(concurrent(args.url, images, clients, args.seconds, args.timeout))

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
