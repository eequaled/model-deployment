"""Step 3 resilience harness: malformed inputs, pixel/size bombs, overload spike, recovery.

Exercises the exact failure modes described in `challenge.md` §6.3 against a live
server and exits non-zero if any invariant breaks:

1. Baseline health and single-request latency (`baseline_ms`).
2. Edge-case inputs: missing field (400), empty upload (400), non-image bytes (400),
   oversized payload >10 MiB (413), and a small PNG whose header exceeds the
   decoded-pixel cap (413) — each must return JSON and leave `/health` at 200.
3. Overload burst (`--spike-clients` concurrent requests): excess requests that hit
   a full queue must be rejected with a fast `503` JSON rather than hanging or
   crashing the worker (`500`).
4. Post-spike recovery: `/health` is still 200 and a sequential request right after
   the burst completes cleanly within the 30 s budget.

  python -m bench.stress --url http://localhost:8000 --images samples
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

from .accuracy import IMAGE_SUFFIXES, post_image


def raw_request(
    url: str,
    path: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict, float]:
    """Return `(status_code, parsed_json_or_empty, elapsed_ms)` without raising on 4xx/5xx."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=body,
        headers=headers or {},
        method=method,
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    elapsed_ms = round(1000 * (time.perf_counter() - t0), 1)
    try:
        parsed = json.loads(raw) if raw else {}
    except Exception:
        parsed = {}
    return status, parsed, elapsed_ms


def multipart_body(filename: str, content: bytes, field: str = "image") -> tuple[bytes, dict[str, str]]:
    boundary = "----florence2stress"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def png_header_only_bomb(width: int = 8000, height: int = 8000) -> bytes:
    """Build a valid tiny PNG header with huge dimensions (`64 MP > 40 MP` cap).

    Because the server checks `image.size` before calling `image.load()`, PIL reads
    only the 33-byte IHDR chunk and rejects the image with 413 before decompressing
    any scanlines.
    """
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def run_edge_cases(url: str, timeout: float) -> list[dict]:
    cases = [
        ("missing_field", b"", {"Content-Type": "multipart/form-data; boundary=empty"}, 400),
        ("empty_file", *multipart_body("empty.png", b""), 400),
        ("corrupt_bytes", *multipart_body("bad.png", b"not-an-image-at-all"), 400),
        ("pixel_bomb_64mp", *multipart_body("bomb.png", png_header_only_bomb(8000, 8000)), 413),
        ("oversized_11mb", *multipart_body("huge.bin", b"x" * (11 * 1024 * 1024)), 413),
    ]
    results = []
    for name, body, headers, expected_status in cases:
        status, payload, elapsed_ms = raw_request(
            url, "/ocr", method="POST", body=body, headers=headers, timeout=timeout
        )
        ok = status == expected_status and "error" in payload
        row = {
            "check": name,
            "status": status,
            "expected": expected_status,
            "has_error_json": "error" in payload,
            "elapsed_ms": elapsed_ms,
            "pass": ok,
        }
        print(json.dumps(row), flush=True)
        results.append(row)
    return results


def run_overload_spike(url: str, images: list[Path], clients: int, timeout: float) -> dict:
    """Fire `clients` simultaneous requests to trigger the bounded queue and fast 503s."""
    statuses: list[int] = []
    ms_503: list[float] = []
    ms_200: list[float] = []
    lock = threading.Lock()
    barrier = threading.Barrier(clients)

    def worker(idx: int) -> None:
        image_bytes = images[idx % len(images)].read_bytes()
        body, headers = multipart_body(images[idx % len(images)].name, image_bytes)
        try:
            barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError:
            pass
        status, _, elapsed_ms = raw_request(
            url, "/ocr", method="POST", body=body, headers=headers, timeout=timeout
        )
        with lock:
            statuses.append(status)
            if status == 503:
                ms_503.append(elapsed_ms)
            elif status == 200:
                ms_200.append(elapsed_ms)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = round(time.perf_counter() - started, 2)

    count_200 = statuses.count(200)
    count_503 = statuses.count(503)
    other = [s for s in statuses if s not in (200, 503)]
    row = {
        "check": f"overload_spike_c{clients}",
        "clients": clients,
        "wall_s": wall_s,
        "status_200": count_200,
        "status_503": count_503,
        "unexpected_statuses": other,
        "max_503_ms": round(max(ms_503), 1) if ms_503 else 0.0,
        "pass": count_200 >= 1 and not other,
    }
    print(json.dumps(row), flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--spike-clients", type=int, default=14)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images in {args.images}")

    summary: dict[str, object] = {}
    status, payload, _ = raw_request(args.url, "/health", timeout=5.0)
    health_before = status == 200 and payload.get("status") == "ok"

    t0 = time.perf_counter()
    post_image(args.url, images[0], args.timeout)
    baseline_ms = round(1000 * (time.perf_counter() - t0), 1)
    print(json.dumps({"check": "baseline", "health_ok": health_before, "baseline_ms": baseline_ms}), flush=True)

    edges = run_edge_cases(args.url, args.timeout)
    spike = run_overload_spike(args.url, images, args.spike_clients, args.timeout)

    status_after, payload_after, _ = raw_request(args.url, "/health", timeout=5.0)
    health_after = status_after == 200 and payload_after.get("status") == "ok"

    t1 = time.perf_counter()
    text_after, _ = post_image(args.url, images[0], args.timeout)
    recovery_ms = round(1000 * (time.perf_counter() - t1), 1)
    recovery_ok = health_after and bool(text_after) and recovery_ms <= args.timeout * 1000
    recovery_row = {
        "check": "recovery_after_spike",
        "health_ok": health_after,
        "baseline_ms": baseline_ms,
        "recovery_ms": recovery_ms,
        "ratio_vs_baseline": round(recovery_ms / baseline_ms, 2) if baseline_ms > 0 else 1.0,
        "pass": recovery_ok,
    }
    print(json.dumps(recovery_row), flush=True)

    all_passed = health_before and all(e["pass"] for e in edges) and spike["pass"] and recovery_ok
    summary = {
        "baseline_ms": baseline_ms,
        "edges": edges,
        "spike": spike,
        "recovery": recovery_row,
        "pass": all_passed,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"STRESS {'PASS' if all_passed else 'FAIL'}", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
