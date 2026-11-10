#!/usr/bin/env python3
"""Which decoder prefix does the fp32 reference use, and can prefill skip positions?

The accuracy gate is measured against `baseline.py`, which calls `model.generate()`.
That makes one thing decisive rather than stylistic: what the decoder sees *before*
the first generated token. A causal decoder's first token depends on its own prefix,
so these are genuinely different computations, not equivalent spellings of one:

  prompt        all 10 prompt positions            (the old, parity-checked build)
  last          the final prompt position only     (one position, what the server does now)
  start         decoder_start_token_id only        (what a plain HF encoder-decoder does)
  prompt_start  prompt plus one start token        (11 positions)

This script runs the reference generate() and all four hand-rolled loops on the same
image, then says which prefix reproduces the reference and whether the one-position
prefill is safe. Runtime is a few minutes at --max-new 256, seconds at 64.

  cd ~/florence-lab
  .venv312/bin/python <repo>/bench/prefill_probe.py --image funsd/dataset/testing_data/images/82837252.png

Use .venv312 for the reference verdict: that is the environment and library version
that produced references-greedy-fp32.json. Re-run in .venv-onnx only to check whether
the answer depends on the transformers version.

The result is not a code change. Whatever prefix wins becomes `prefill_mode` in
meta.json, and the prefill graph is dynamic in its decoder length, so switching is an
edit to one string (or PREFILL_MODE at runtime) rather than a re-export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

REPO = "microsoft/Florence-2-base"
REV = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
DEFAULT_IMAGE = "funsd/dataset/testing_data/images/82837252.png"
MODES = ("prompt", "last", "start", "prompt_start")


def load(image_path: Path):
    processor = AutoProcessor.from_pretrained(REPO, revision=REV, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REV, trust_remote_code=True,
        torch_dtype=torch.float32, use_safetensors=True,
    ).to("cpu").eval()
    pins = processor(text="<OCR>", images=Image.open(image_path).convert("RGB"), return_tensors="pt")
    return processor, model, pins


def encoder_states(model, pins):
    """image tokens + prompt embeddings through the language encoder, as the server does."""
    lm = model.language_model
    feats = model._encode_image(pins["pixel_values"])
    prompt_embeds = lm.get_input_embeddings()(pins["input_ids"])
    merged = torch.cat([feats, prompt_embeds], dim=1)
    mask = torch.ones(merged.shape[:2], dtype=torch.float32)
    enc = lm.model.encoder(inputs_embeds=merged, attention_mask=mask).last_hidden_state
    return enc, mask, prompt_embeds


def reference_tokens(model, pins, max_new: int) -> tuple[list[int], list[int]]:
    """model.generate() with the baseline settings: greedy, cached, capped.

    Returns (raw, aligned). `raw` is exactly what generate() emitted, so a surprise in
    the first few ids (a decoder prefix we did not expect) is visible rather than
    silently trimmed into a misaligned comparison.
    """
    attempts = (
        dict(input_ids=pins["input_ids"], pixel_values=pins["pixel_values"],
             max_new_tokens=max_new, num_beams=1, do_sample=False),
        dict(**pins, max_new_tokens=max_new, num_beams=1, do_sample=False),
    )
    last_error = None
    for kwargs in attempts:
        try:
            with torch.inference_mode():
                ids = model.generate(**kwargs)
            raw = [int(t) for t in ids[0].tolist()]
            # generate() emits the decoder prefix as its first element; our hand-rolled
            # loops start at the first *predicted* token, so drop one start token.
            aligned = raw[1:] if raw and raw[0] == int(model.config.text_config.decoder_start_token_id) else raw
            return raw, aligned
        except Exception as exc:  # signature drift between remote-code versions
            last_error = exc
    raise SystemExit(f"model.generate() failed: {type(last_error).__name__}: {last_error}")


def run_mode(model, enc, mask, prompt_embeds, mode: str, max_new: int, eos: int) -> dict:
    """One full greedy decode with a given decoder prefix, mirroring the server."""
    lm = model.language_model
    embed = lm.get_input_embeddings()
    start = torch.tensor([[int(model.config.text_config.decoder_start_token_id)]], dtype=torch.long)
    start_embeds = embed(start)

    if mode == "prompt":
        prefix = prompt_embeds
    elif mode == "prompt_start":
        prefix = torch.cat([prompt_embeds, start_embeds], dim=1)
    elif mode == "last":
        prefix = prompt_embeds[:, -1:, :]
    elif mode == "start":
        prefix = start_embeds
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    past = None

    with torch.inference_mode():
        out = lm(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc),
            attention_mask=mask,
            decoder_inputs_embeds=prefix,
            use_cache=True,
            return_dict=True,
        )
        first = int(out.logits[0, -1, :].argmax())
        tokens = [first]
        past = out.past_key_values
        while tokens[-1] != eos and len(tokens) < max_new:
            out = lm(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                attention_mask=mask,
                decoder_inputs_embeds=embed(torch.tensor([[tokens[-1]]], dtype=torch.long)),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            tokens.append(int(out.logits[0, -1, :].argmax()))
            past = out.past_key_values

    return {
        "mode": mode,
        "prefix_positions": int(prefix.shape[1]),
        "cache_positions": int(past[0][0].shape[2]),
        "len": len(tokens),
        "tokens": tokens,
    }


def agreement(a: list[int], b: list[int]) -> tuple[int, int | None]:
    """(agreeing prefix length, first mismatch index or None)."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index, index
    return min(len(a), len(b)), None


def cer(prediction: str, reference: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    previous = list(range(len(reference) + 1))
    for i, ca in enumerate(prediction, start=1):
        current = [i]
        for j, cb in enumerate(reference, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / len(reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=None,
                        help="image to probe; defaults to the FUNSD sample")
    parser.add_argument("--max-new", type=int, default=64,
                        help="tokens per variant; 64 identifies a prefix, 256 proves a full decode")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image or (Path(os.getenv("FLORENCE_LAB", str(Path.home() / "florence-lab")))
                                / DEFAULT_IMAGE)
    if not image_path.exists():
        raise SystemExit(f"no such image: {image_path} (pass --image)")

    print(f"image: {image_path}")
    processor, model, pins = load(image_path)
    eos = int(processor.tokenizer.eos_token_id)
    enc, mask, prompt_embeds = encoder_states(model, pins)
    print(f"encoder states {tuple(enc.shape)}  prompt {tuple(prompt_embeds.shape)}  eos {eos}")

    print(f"\nreference: model.generate(max_new_tokens={args.max_new}, num_beams=1)")
    with torch.inference_mode():
        raw_ref, ref = reference_tokens(model, pins, args.max_new)
    print(f"  raw head {raw_ref[:12]}")
    print(f"  {len(ref)} tokens after alignment, head {ref[:12]}")

    results = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        row = run_mode(model, enc, mask, prompt_embeds, mode, args.max_new, eos)
        agree, first_bad = agreement(row["tokens"], ref)
        row["agree_with_reference"] = agree
        row["first_mismatch"] = first_bad
        results[mode] = row
        exact = first_bad is None and row["len"] == len(ref)
        verdict = "EXACT match" if exact else f"{agree} of {len(ref)} tokens match, first mismatch at {first_bad}"
        print(f"\n{mode:<13} {row['prefix_positions']:>2} prefix positions, "
              f"{row['len']:>3} tokens -> {verdict}")

    # The two questions, answered separately and printed as a machine-readable line.
    best = max(results, key=lambda m: (results[m]["agree_with_reference"], -results[m]["len"]))
    full_vs_last_at = None
    if "prompt" in results and "last" in results:
        agree, full_vs_last_at = agreement(results["prompt"]["tokens"], results["last"]["tokens"])

    payload = {
        "image": str(image_path),
        "max_new": args.max_new,
        "eos": eos,
        "reference_len": len(ref),
        "reference_head": ref[:16],
        "variants": {
            mode: {
                "prefix_positions": row["prefix_positions"],
                "cache_positions": row["cache_positions"],
                "len": row["len"],
                "agree_with_reference": row["agree_with_reference"],
                "first_mismatch": row["first_mismatch"],
                "head": row["tokens"][:16],
            }
            for mode, row in results.items()
        },
        "best_prefix": best,
        "one_position_prefill_safe": full_vs_last_at is None
        and "prompt" in results and "last" in results
        and results["prompt"]["len"] == results["last"]["len"],
    }

    print("\n" + json.dumps(payload, indent=2))

    print("\n--- verdict ---")
    print(f"PREFIX {best}: {results[best]['agree_with_reference']} tokens match generate(), "
          f"first mismatch at {results[best]['first_mismatch']}")
    if payload["one_position_prefill_safe"]:
        print("ONE_POSITION SAFE: dropping the earlier prompt positions does not change the output")
    else:
        print("ONE_POSITION UNSAFE: prefill must run the prefix that matches the reference; "
              "set prefill_mode accordingly instead of slicing to the last position")
    if best != "prompt":
        print(f"ACTION: serve with prefill_mode={best!r} (meta.json or PREFILL_MODE), not 'prompt'")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    # read the text too: a token difference that does not move CER is not worth a re-export
    decode = processor.tokenizer.decode
    ref_text = decode(ref, skip_special_tokens=True)
    print(f"\nreference text ({len(ref_text)} chars): {ref_text[:120]!r}")
    for mode, row in results.items():
        text = decode(row["tokens"], skip_special_tokens=True)
        print(f"  {mode:<13} cer vs reference = {cer(text, ref_text):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
