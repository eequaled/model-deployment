"""Compare two exported model directories token by token, with and without PyTorch.

Why this exists: an optimization that changes the graphs must be proved not to
change the text. The cheapest proof does not need PyTorch at all -- run the same
greedy loop twice, once on the old graphs and once on the new, and compare the
generated token ids. If they differ, the change altered behaviour and should not be
trusted until the difference is explained.

PyTorch is used only when it happens to be importable, as a second opinion on the
*old* graphs (which were verified against it earlier). It is never required.

  python bench/parity.py --old ~/florence-lab/onnx-export --new ~/florence-lab/onnx-export-v2 \
      --images ~/florence-lab/funsd/dataset/testing_data/images --max-new 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer


class Pipeline:
    """The serving loop, over an arbitrary export directory."""

    def __init__(self, model_dir: Path, threads: int = 12) -> None:
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.prompt_ids = np.asarray(self.meta["prompt_ids"], dtype=np.int64)[None, :]
        self.start_ids = np.asarray([[int(self.meta["decoder_start_token_id"])]], dtype=np.int64)
        self.eos = int(self.meta["eos_token_id"])
        self.prefill_mode = str(self.meta.get("prefill_mode", "start"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        names = ("vision_encoder", "embed_tokens", "encoder_model", "decoder_prefill", "decoder_step")
        missing = [n for n in names if not (self.dir / f"{n}.onnx").exists()]
        if missing:
            raise SystemExit(f"{self.dir} is missing {missing}")
        self.sessions = {
            n: ort.InferenceSession(
                str(self.dir / f"{n}.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for n in names
        }
        self.prefill_in = [i.name for i in self.sessions["decoder_prefill"].get_inputs()]
        self.prefill_out = [o.name for o in self.sessions["decoder_prefill"].get_outputs()]
        self.step_in = [i.name for i in self.sessions["decoder_step"].get_inputs()]
        self.step_out = [o.name for o in self.sessions["decoder_step"].get_outputs()]
        self.prefill_embeds_input = next(
            (n for n in ("decoder_inputs_embeds", "decoder_input_embeds") if n in self.prefill_in),
            "decoder_inputs_embeds",
        )
        self.step_embeds_input = next(
            (n for n in ("decoder_inputs_embeds", "decoder_input_embeds") if n in self.step_in),
            "decoder_inputs_embeds",
        )
        self.layout = "ids" if "input_ids" in self.step_in else "embeds"

    def preprocess(self, image: Image.Image) -> np.ndarray:
        size = int(self.meta["image_size"])
        image = image.convert("RGB").resize((size, size), resample=Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) * float(self.meta["rescale_factor"])
        array = (array - np.asarray(self.meta["image_mean"], dtype=np.float32)) / np.asarray(
            self.meta["image_std"], dtype=np.float32
        )
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32)

    def prefill_embeds(self, prompt_embeds: np.ndarray) -> np.ndarray:
        if self.prefill_mode == "prompt":
            return prompt_embeds
        if self.prefill_mode == "last":
            return prompt_embeds[:, -1:, :]
        start = self.sessions["embed_tokens"].run(None, {"input_ids": self.start_ids})[0]
        if self.prefill_mode == "start":
            return start
        return np.concatenate([prompt_embeds, start], axis=1).astype(np.float32)

    def run(self, image: Image.Image, max_new: int) -> list[int]:
        s = self.sessions
        pixels = self.preprocess(image)
        features = s["vision_encoder"].run(None, {"pixel_values": pixels})[0]
        prompt_embeds = s["embed_tokens"].run(None, {"input_ids": self.prompt_ids})[0]
        merged = np.concatenate([features, prompt_embeds], axis=1).astype(np.float32)
        mask = np.ones((1, merged.shape[1]), dtype=np.float32)
        enc = s["encoder_model"].run(None, {"inputs_embeds": merged, "attention_mask": mask})[0]

        feed = {
            "encoder_hidden_states": enc,
            "encoder_attention_mask": mask,
            self.prefill_embeds_input: self.prefill_embeds(prompt_embeds),
        }
        out = s["decoder_prefill"].run(None, feed)
        cache = dict(zip(self.prefill_out[1:], out[1:]))
        tokens = [int(out[0][0, -1].argmax())]

        while tokens[-1] != self.eos and len(tokens) < max_new:
            step_feed = {"encoder_attention_mask": mask}
            if "encoder_hidden_states" in self.step_in:
                step_feed["encoder_hidden_states"] = enc
            if self.layout == "ids":
                step_feed["input_ids"] = np.array([[tokens[-1]]], dtype=np.int64)
            else:
                step_feed[self.step_embeds_input] = s["embed_tokens"].run(
                    None, {"input_ids": np.array([[tokens[-1]]], dtype=np.int64)}
                )[0]
            for name in self.step_in:
                if name.startswith("past_key_values."):
                    step_feed[name] = cache["present." + name[len("past_key_values."):]]
            out = s["decoder_step"].run(None, step_feed)
            tokens.append(int(out[0][0, -1].argmax()))
            cache.update(zip(self.step_out[1:], out[1:]))
        return tokens


def compare(label: str, a: list[int], b: list[int]) -> bool:
    length = min(len(a), len(b))
    first_bad = next((i for i in range(length) if a[i] != b[i]), None)
    same = first_bad is None and len(a) == len(b)
    print(
        json.dumps(
            {
                "check": label,
                "identical": same,
                "len_a": len(a),
                "len_b": len(b),
                "agree_tokens": length if first_bad is None else first_bad,
                "first_mismatch": first_bad,
            }
        ),
        flush=True,
    )
    if first_bad is not None:
        lo = max(0, first_bad - 4)
        print(f"  a[{lo}:{first_bad + 5}] = {a[lo:first_bad + 5]}", flush=True)
        print(f"  b[{lo}:{first_bad + 5}] = {b[lo:first_bad + 5]}", flush=True)
    return same


def pytorch_tokens(model_dir: Path, images: list[Path], max_new: int,
                   prefill_mode: str = "prompt") -> dict[str, list[int]]:
    """Greedy loop in PyTorch over the *original* model, for a second opinion."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.modeling_outputs import BaseModelOutput

    meta = json.loads((Path(model_dir) / "meta.json").read_text(encoding="utf-8"))
    repo, revision = meta["repo"], meta["revision"]
    processor = AutoProcessor.from_pretrained(repo, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, trust_remote_code=True,
        torch_dtype=torch.float32, use_safetensors=True,
    ).to("cpu").eval()
    lm = model.language_model
    results: dict[str, list[int]] = {}
    for path in images:
        pins = processor(text="<OCR>", images=Image.open(path).convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            feats = model._encode_image(pins["pixel_values"])
            prompt_embeds = lm.get_input_embeddings()(pins["input_ids"])
            merged = torch.cat([feats, prompt_embeds], dim=1)
            mask = torch.ones(merged.shape[:2], dtype=torch.float32)
            enc = lm.model.encoder(inputs_embeds=merged, attention_mask=mask).last_hidden_state
            start_ids = torch.tensor([[int(meta["decoder_start_token_id"])]], dtype=torch.long)
            if prefill_mode == "prompt":
                prefix = prompt_embeds
            elif prefill_mode == "last":
                prefix = prompt_embeds[:, -1:, :]
            elif prefill_mode == "start":
                prefix = lm.get_input_embeddings()(start_ids)
            else:
                prefix = torch.cat([prompt_embeds, lm.get_input_embeddings()(start_ids)], dim=1)
            out = lm(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                attention_mask=mask,
                decoder_inputs_embeds=prefix,
                use_cache=True,
                return_dict=True,
            )
            tokens = [int(out.logits[0, -1, :].argmax())]
            while tokens[-1] != int(meta["eos_token_id"]) and len(tokens) < max_new:
                past = out.past_key_values
                nxt = torch.tensor([[tokens[-1]]])
                out = lm(
                    encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                    attention_mask=mask,
                    decoder_inputs_embeds=lm.get_input_embeddings()(nxt),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                tokens.append(int(out.logits[0, -1, :].argmax()))
        results[path.name] = tokens
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--no-torch", action="store_true")
    args = parser.parse_args()

    names = ["82837252.png", "85201976.png", "86263525.png", "82251504.png", "93106788.png"]
    images = [args.images / n for n in names if (args.images / n).exists()]
    if not images:
        images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg"})[:5]
    print(f"comparing on {[p.name for p in images]}", flush=True)

    old = Pipeline(args.old, args.threads)
    new = Pipeline(args.new, args.threads)
    print(f"old ({old.prefill_mode}) step inputs: {old.step_in[:4]}...", flush=True)
    print(f"new ({new.prefill_mode}) step inputs: {new.step_in[:4]}...", flush=True)

    torch_tokens: dict[str, list[int]] = {}
    if not args.no_torch:
        try:
            print(f"running the PyTorch reference loop (prefill_mode={new.prefill_mode!r})...", flush=True)
            torch_tokens = pytorch_tokens(args.new, images, args.max_new, prefill_mode=new.prefill_mode)
        except Exception as exc:  # torch absence is the normal case in the runtime image
            print(f"PyTorch reference skipped: {type(exc).__name__}: {exc}", flush=True)

    all_same = True
    for path in images:
        old_tokens = old.run(Image.open(path), args.max_new)
        new_tokens = new.run(Image.open(path), args.max_new)
        print(f"\n{path.name}: old {len(old_tokens)} tokens, new {len(new_tokens)} tokens", flush=True)
        all_same &= compare("ORT_OLD_vs_ORT_NEW", old_tokens, new_tokens)
        if path.name in torch_tokens:
            all_same &= compare("PT_vs_ORT_OLD", torch_tokens[path.name], old_tokens)
            all_same &= compare("PT_vs_ORT_NEW", torch_tokens[path.name], new_tokens)

    print(f"\nPARITY {'PASS' if all_same else 'FAIL'} (max_new={args.max_new})")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
