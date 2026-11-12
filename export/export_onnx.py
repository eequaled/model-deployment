#!/usr/bin/env python3
"""Export microsoft/Florence-2-base to the ONNX graphs the server runs.

This runs in the PyTorch build stage of the Dockerfile. The final image does not
contain PyTorch, so every graph used at serving time is produced here, from the
original weights, and nothing is downloaded pre-converted.

The model is split into five graphs:

  vision_encoder.onnx  image -> 577 image tokens          (DaViT, fixed 768x768)
  embed_tokens.onnx    token ids -> embeddings            (shared by prompt and steps)
  encoder_model.onnx   merged embeddings -> encoder states
  decoder_prefill.onnx encoder states + decoder prefix -> logits + 24-tensor cache
  decoder_step.onnx    24-tensor cache -> logits + 12 updated self-attention tensors

Why two decoder graphs instead of one: the cross-attention key/value tensors are
computed once in prefill and reused unchanged for the whole request, so the step
graph never has to recompute them. Tracing also removes the encoder-feature input
from the step graph because those values are unused once the cache exists -- only
their length survives, through the attention mask.

Dynamic axes are declared only where we actually exercised them against PyTorch:
the encoder sequence length (tested at 587 and 590) and the decoder past length.
The vision encoder is deliberately fixed at 768x768 because that is the only size
the processor produces.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

REPO = "microsoft/Florence-2-base"
REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
OPSET = 17
IMAGE_SIZE = 768
SLOTS = ("decoder.key", "decoder.value", "encoder.key", "encoder.value")

# What the decoder sees before it predicts its first token. These are different
# computations, not spellings of one: a causal decoder's first token depends on its
# own prefix. bench/prefill_probe.py says which one reproduces model.generate().
#   prompt        all prompt positions (the build the parity runs used)
#   last          final prompt position only
#   start         decoder_start_token_id only
#   prompt_start  prompt followed by one start token
PREFILL_MODES = ("prompt", "last", "start", "prompt_start")


def prefill_positions(mode: str, prompt_len: int) -> int:
    """How many decoder positions the traced prefill graph stands for.

    Trace at the length the server will actually use. A traced graph keeps the shapes
    it saw wherever a shape only influences mask construction, so exporting prefill at
    one position while serving ten is exactly the kind of silent error this file is
    trying to avoid.
    """
    if mode == "prompt":
        return prompt_len
    if mode == "prompt_start":
        return prompt_len + 1
    return 1


def export(module, args, path, input_names, output_names, dynamic_axes):
    """torch.onnx.export with the TorchScript tracer, tolerating API drift.

    The legacy tracer is what the parity runs used. Newer PyTorch defaults to the
    dynamo exporter, which produces different graphs, so it is requested
    explicitly when the installed version understands the argument.
    """
    kwargs = dict(
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    started = time.perf_counter()
    with torch.no_grad():
        try:
            torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(module, args, str(path), **kwargs)
    print(f"  {path.name}: {path.stat().st_size / 2**20:.1f} MiB in {time.perf_counter() - started:.1f}s")


class VisionEncoder(nn.Module):
    """The model's own image path, not a reimplementation of it.

    Wrapping model._encode_image keeps the DaViT tower, the projection, the
    positional embedding, and the feature-source concatenation in the exact order
    the original forward pass uses them. Hand-copying that chain is where public
    export scripts have gone wrong.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model._encode_image(pixel_values)


class EmbedTokens(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids):
        return self.embeddings(input_ids)


class Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state


class Prefill(nn.Module):
    """First decoder pass: consumes the decoder prefix, returns logits and the full cache.

    In Florence-2 the 10-token task prompt goes into the encoder alongside the 577
    image tokens; the decoder starts from a single decoder_start_token_id (</s> = 2),
    which predicts <s> (0) and seeds the 1-position self-attention cache.
    """

    def __init__(self, language_model, layers):
        super().__init__()
        self.language_model = language_model
        self.layers = layers

    def forward(self, encoder_hidden_states, encoder_attention_mask, decoder_inputs_embeds):
        out = self.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        return (out.logits, *[out.past_key_values[i][j] for i in range(self.layers) for j in range(4)])


class Step(nn.Module):
    """One cached decoding step: takes embedded token, returns logits and 12 self-attention tensors.

    Keeping embed_tokens in its own 150.3 MiB session instead of folding it into the
    step graph avoids duplicating the 51289x768 embedding matrix on disk and in RAM
    (342.9 MiB vs 493.2 MiB under the 4 GB container limit).
    """

    def __init__(self, language_model, layers):
        super().__init__()
        self.language_model = language_model
        self.layers = layers

    def forward(self, encoder_hidden_states, encoder_attention_mask, decoder_inputs_embeds, *past):
        cache = tuple(tuple(past[i * 4 + j] for j in range(4)) for i in range(self.layers))
        out = self.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return (out.logits, *[out.past_key_values[i][j] for i in range(self.layers) for j in (0, 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/models"), help="where the graphs are written")
    parser.add_argument("--prefill-mode", choices=PREFILL_MODES, default="start",
                        help="decoder prefix the prefill graph stands for (see PREFILL_MODES)")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {REPO}@{REVISION}")
    processor = AutoProcessor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        revision=REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cpu").eval()

    config = model.config
    text_config = config.text_config
    language_model = model.language_model
    layers = text_config.decoder_layers
    heads = text_config.decoder_attention_heads
    d_model = text_config.d_model
    head_dim = d_model // heads

    # The OCR prompt. Florence-2 does not tokenize "<OCR>" itself: the processor
    # substitutes a natural-language instruction and tokenizes that. Getting this
    # wrong changes the token ids silently, so it is read from the processor and
    # written to meta.json for the server to reuse verbatim.
    prompts = getattr(processor, "task_prompts_without_inputs", {})
    prompt_text = prompts["<OCR>"] if "<OCR>" in prompts else "What is the text in the image?"
    prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = prompt_ids.ids if hasattr(prompt_ids, "ids") else prompt_ids
    prompt_ids = list(prompt_ids)
    print(f"prompt: {prompt_text!r} -> {prompt_ids}")

    device_ids = torch.tensor([prompt_ids], dtype=torch.long)
    start_id = config.text_config.decoder_start_token_id
    # Seeded random inputs rather than zeros: tracing follows the same operator
    # path either way, but non-degenerate activations keep any data-dependent
    # branch on the path the model actually takes.
    generator = torch.Generator().manual_seed(0)
    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    with torch.no_grad():
        image_tokens = model._encode_image(dummy_image).shape[1]
    encoder_len = image_tokens + len(prompt_ids)
    print(f"encoder sequence length for OCR: {image_tokens} image tokens + {len(prompt_ids)} prompt = {encoder_len}")

    embeddings = language_model.get_input_embeddings()
    embed_weights = embeddings.weight.detach().cpu().numpy()
    np.save(out / "embed_tokens.npy", embed_weights)
    print(f"  embed_tokens.npy: shape {embed_weights.shape}, dtype {embed_weights.dtype}")

    print("exporting graphs")
    export(
        VisionEncoder(model).eval(),
        (dummy_image,),
        out / "vision_encoder.onnx",
        ["pixel_values"],
        ["image_features"],
        {},
    )

    export(
        EmbedTokens(embeddings).eval(),
        (device_ids,),
        out / "embed_tokens.onnx",
        ["input_ids"],
        ["inputs_embeds"],
        {"input_ids": {1: "sequence_length"}, "inputs_embeds": {1: "sequence_length"}},
    )

    encoder_inputs = torch.randn(1, encoder_len, d_model, generator=generator)
    encoder_mask = torch.ones(1, encoder_len, dtype=torch.float32)
    export(
        Encoder(language_model.model.encoder).eval(),
        (encoder_inputs, encoder_mask),
        out / "encoder_model.onnx",
        ["inputs_embeds", "attention_mask"],
        ["last_hidden_state"],
        {
            "inputs_embeds": {1: "encoder_sequence_length"},
            "attention_mask": {1: "encoder_sequence_length"},
            "last_hidden_state": {1: "encoder_sequence_length"},
        },
    )

    # Prefill stands for the decoder prefix in --prefill-mode, traced at its real length.
    positions = prefill_positions(args.prefill_mode, len(prompt_ids))
    decoder_embeds = torch.randn(1, positions, d_model, generator=generator)
    print(f"prefill mode {args.prefill_mode!r}: tracing with {positions} decoder position(s)")
    prefill = Prefill(language_model, layers).eval()
    past_len = 6
    past = tuple(
        torch.zeros(1, heads, past_len if slot.startswith("decoder") else encoder_len, head_dim)
        for _ in range(layers)
        for slot in SLOTS
    )
    prefill_names = ["logits"] + [f"present.{i}.{name}" for i in range(layers) for name in SLOTS]
    export(
        prefill,
        (encoder_inputs, encoder_mask, decoder_embeds),
        out / "decoder_prefill.onnx",
        ["encoder_hidden_states", "encoder_attention_mask", "decoder_inputs_embeds"],
        prefill_names,
        {
            "encoder_hidden_states": {1: "encoder_sequence_length"},
            "encoder_attention_mask": {1: "encoder_sequence_length"},
            "decoder_inputs_embeds": {1: "decoder_sequence_length"},
            "logits": {1: "decoder_sequence_length"},
            **{
                name: {2: "decoder_sequence_length" if "decoder" in name else "encoder_sequence_length"}
                for name in prefill_names[1:]
            },
        },
    )

    step = Step(language_model, layers).eval()
    step_embeds = torch.randn(1, 1, d_model, generator=generator)
    step_inputs = ["encoder_hidden_states", "encoder_attention_mask", "decoder_inputs_embeds"] + [
        f"past_key_values.{i}.{name}" for i in range(layers) for name in SLOTS
    ]
    step_outputs = ["logits"] + [
        f"present.{i}.{name}" for i in range(layers) for name in ("decoder.key", "decoder.value")
    ]
    export(
        step,
        (encoder_inputs, encoder_mask, step_embeds, *past),
        out / "decoder_step.onnx",
        step_inputs,
        step_outputs,
        {
            "encoder_attention_mask": {1: "encoder_sequence_length"},
            "decoder_inputs_embeds": {1: "decoder_sequence_length"},
            "logits": {1: "decoder_sequence_length"},
            **{
                name: {
                    2: "past_sequence_length" if "decoder" in name else "encoder_sequence_length"
                }
                for name in step_inputs[3:]
            },
            **{name: {2: "present_sequence_length"} for name in step_outputs[1:]},
        },
    )

    tokenizer_json = hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
    shutil.copyfile(tokenizer_json, out / "tokenizer.json")

    image_processor = processor.image_processor
    size = getattr(image_processor, "size", {}) or {}
    meta = {
        "repo": REPO,
        "revision": REVISION,
        "opset": OPSET,
        "image_size": int(size.get("height", IMAGE_SIZE)) if isinstance(size, dict) else IMAGE_SIZE,
        "image_mean": list(getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])),
        "image_std": list(getattr(image_processor, "image_std", [0.229, 0.224, 0.225])),
        "rescale_factor": float(getattr(image_processor, "rescale_factor", 1 / 255.0)),
        "resample": "bicubic",
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "image_tokens": int(image_tokens),
        # 577 image tokens + 10 prompt tokens. Recorded so the server can reject a
        # mask/graph mismatch with one clear message instead of a broadcast error
        # partway through decoding.
        "encoder_length": int(encoder_len),
        "decoder_start_token_id": int(start_id),
        "eos_token_id": int(processor.tokenizer.eos_token_id),
        "pad_token_id": int(processor.tokenizer.pad_token_id),
        "decoder_layers": int(layers),
        # The server reads this to decide what to feed prefill. Changing it is an edit to
        # one string, not a re-export: the prefill graph is dynamic in its decoder length.
        "prefill_mode": args.prefill_mode,
        "prefill_positions": positions,
        "max_new_tokens": 256,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total = sum(p.stat().st_size for p in out.glob("*.onnx")) / 2**20
    print(f"wrote {len(list(out.glob('*.onnx')))} graphs, {total:.1f} MiB total, into {out}")


if __name__ == "__main__":
    main()
