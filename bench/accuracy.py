"""Score a running server against reference text using the challenge's CER rule.

References are a JSON list (or a directory of .txt files) mapping image filename to
the expected text. Nothing here knows how the references were produced; point it at
whatever the organizer ships, or at locally generated ones for a sanity check.

  python -m bench.accuracy --url http://localhost:8000 --images samples --refs refs.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .metrics import accuracy, character_error_rate

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def post_image(url: str, path: Path, timeout: float) -> tuple[str, dict]:
    boundary = "----florence2bench"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/ocr",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload.get("text", ""), payload.get("timings", {})


def load_refs(refs: Path) -> dict[str, str]:
    if refs.is_dir():
        return {p.stem: p.read_text(encoding="utf-8").strip() for p in refs.glob("*.txt")}
    data = json.loads(refs.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("images"), list):
            return {
                str(item["filename"]): str(item["text"])
                for item in data["images"]
                if isinstance(item, dict) and "filename" in item and "text" in item
            }
        return {str(k): str(v) for k, v in data.items()}
    raise SystemExit("references JSON must be an object of filename -> text")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    # The challenge counts a response slower than 30 s as a failure, so score with
    # the same budget instead of waiting five minutes for a request that lost.
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    refs = load_refs(args.refs)
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images in {args.images}")

    rows, missing = [], 0
    for path in files:
        reference = refs.get(path.name) or refs.get(path.stem)
        if reference is None:
            missing += 1
            print(f"skip {path.name}: no reference", file=sys.stderr)
            continue
        try:
            text, timings = post_image(args.url, path, args.timeout)
        except urllib.error.HTTPError as exc:
            print(f"{path.name}: HTTP {exc.code} {exc.read()[:200]!r}")
            rows.append({"file": path.name, "error": exc.code})
            continue
        rows.append(
            {
                "file": path.name,
                "cer": round(character_error_rate(text, reference), 4),
                "accuracy": round(accuracy(text, reference), 4),
                "chars": len(text),
                "timings": timings,
            }
        )
        row = rows[-1]
        print(f"{path.name}: accuracy={row['accuracy']:.3f} cer={row['cer']:.3f} chars={len(text)}")

    scored = [r for r in rows if "accuracy" in r]
    if scored:
        mean = sum(r["accuracy"] for r in scored) / len(scored)
        print(f"\nimages scored: {len(scored)} (missing refs: {missing})")
        print(f"mean accuracy: {mean:.4f}")
        print(f"above 70%: {'yes' if mean >= 0.70 else 'NO'}")
    if args.json_out:
        args.json_out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
