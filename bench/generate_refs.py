"""Generate fp32 PyTorch reference OCR texts offline for an image directory.

Used to expand the local accuracy check beyond the initial 5 FUNSD images (e.g.
across all ~50 test images in `~/florence-lab/funsd/dataset/testing_data/images`).
Writes a `filename -> text` JSON mapping directly usable by `python -m bench.accuracy`,
updating the file after each image so a partial run is still usable.

  ~/florence-lab/.venv312/bin/python -m bench.generate_refs \
      --images ~/florence-lab/funsd/dataset/testing_data/images \
      --out ~/florence-lab/references-funsd-fp32.json --limit 50

Use the environment that matches `baseline.py` (here transformers 4.40.2). A newer
transformers changes `clean_up_tokenization_spaces`, which would move the reference
text by a few whitespace characters and quietly change the score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .accuracy import IMAGE_SUFFIXES

REPO = "microsoft/Florence-2-base"
REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch.set_num_threads(args.threads)
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images in {args.images}")

    existing: dict[str, str] = {}
    if args.out.exists():
        try:
            loaded = json.loads(args.out.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("images"), list):
                # Resume from the richer reference format as well as from a flat
                # filename -> text map, so pointing --out at an existing reference
                # file never silently drops the images it has not reached yet.
                existing = {
                    str(item["filename"]): str(item["text"])
                    for item in loaded["images"]
                    if isinstance(item, dict) and "filename" in item and "text" in item
                }
            elif isinstance(loaded, dict):
                existing = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        revision=REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu").eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(files, start=1):
        if path.name in existing:
            print(f"[{idx}/{len(files)}] {path.name}: cached ({len(existing[path.name])} chars)", flush=True)
            continue
        t0 = time.perf_counter()
        image = Image.open(path).convert("RGB")
        inputs = processor(text="<OCR>", images=image, return_tensors="pt")
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=args.max_new,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(decoded, task="<OCR>", image_size=image.size)
        text = parsed.get("<OCR>", decoded.replace("<s>", "").replace("</s>", ""))
        existing[path.name] = text
        args.out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{idx}/{len(files)}] {path.name}: {len(text)} chars in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    print(f"wrote {len(existing)} references to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
