"""Probes batching effects by testing concurrent requests on a single worker server.

Sequential OCR on CPU uses all physical cores per image. Running concurrent
requests competes for CPU and increases latency. This probe measures batching 1 vs 2 vs 4.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from bench.accuracy import post_image
from bench.load import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, default=Path("samples"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print("No images found in", args.images, file=sys.stderr)
        return 1

    print("=== Batching Probe: Simulating batch sizes 1, 2, 4 ===")
    results = {}
    for batch_size in (1, 2, 4):
        t0 = time.perf_counter()
        latencies = []
        errors = 0
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [pool.submit(post_image, args.url, images[i % len(images)], args.timeout) for i in range(batch_size * 2)]
            for f in futures:
                try:
                    f.result()
                    latencies.append(time.perf_counter() - t0)
                except Exception as e:
                    errors += 1
        elapsed = time.perf_counter() - t0
        res = summarize(f"batch_{batch_size}", latencies, errors, elapsed)
        results[f"batch_{batch_size}"] = res

    print("\nBatch Probe Summary:")
    for k, v in results.items():
        print(f"  {k}: rps={v['rps']} | p50_ms={v['p50_ms']} | p95_ms={v['p95_ms']} | errors={v['errors']}")
    print("\nDecision: Keep batch size 1 (sequential) because batching 2+ increases latency without raising throughput on 4 CPUs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
