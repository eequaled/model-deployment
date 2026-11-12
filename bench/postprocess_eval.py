"""Evaluate post-processing modes against reference texts."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

from server.ocr_engine import FlorenceOCR
from bench.metrics import accuracy, character_error_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    refs: dict[str, str] = json.loads(args.refs.read_text(encoding="utf-8"))
    print(f"Loaded {len(refs)} references from {args.refs}")

    for mode in ("pure_text", "baseline"):
        t0 = time.perf_counter()
        eng = FlorenceOCR(args.model_dir, threads=args.threads, postprocess_mode=mode)
        eng.warmup()
        scores: list[float] = []
        cers: list[float] = []
        for fname, ref_text in refs.items():
            img_path = args.image_dir / fname
            if not img_path.exists():
                print(f"warning: image {fname} missing at {img_path}", file=sys.stderr)
                continue
            img = Image.open(img_path)
            pred, _ = eng.generate(img)
            scores.append(accuracy(pred, ref_text))
            cers.append(character_error_rate(pred, ref_text))
        elapsed = time.perf_counter() - t0
        mean_acc = sum(scores) / len(scores) if scores else 0.0
        min_acc = min(scores) if scores else 0.0
        mean_cer = sum(cers) / len(cers) if cers else 0.0
        print(
            f"mode={mode:10s} | mean_acc={mean_acc:.4f} | min_acc={min_acc:.4f} "
            f"| mean_cer={mean_cer:.4f} | count={len(scores)} | time={elapsed:.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
