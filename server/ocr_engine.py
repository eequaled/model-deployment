"""Florence-2 OCR over ONNX Runtime, with no PyTorch anywhere.

Layout proven by the offline parity experiments: the vision tower turns the image
into 577 tokens, the prompt contributes 10 more, one encoder pass fuses them, a
prefill pass produces the first token plus the full attention cache, and each later
step takes that cache and returns one token.

Two details matter more than they look:

* Prompt text. Florence-2 does not tokenize the literal string "<OCR>". The
  processor swaps in a sentence and tokenizes that. The ids are baked into
  meta.json by the exporter; use them and nothing else.
* Encoder length. The step graph broadcasts its attention mask against the cached
  cross-attention tensors, so mask length must equal the prefill encoder length
  exactly. A mismatch is a hard runtime error, not a silent drift.
* The prefill prefix. A causal decoder's first token depends on what came before it, so
  "the prompt", "just the last prompt token" and "just the start token" are different
  computations. bench/prefill_probe.py compares them against model.generate(), the
  exporter records the winner as `prefill_mode` in meta.json, and PREFILL_MODE can
  override it at runtime for an A/B over HTTP. Only the last row of prefill logits is
  ever used; the extra positions exist to give that row its context.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

PREFILL_MODES = ("prompt", "last", "start", "prompt_start")


class FlorenceOCR:
    def __init__(self, model_dir: Path | str, threads: int = 0, max_new_tokens: int = 0,
                 prefill_mode: str = "") -> None:
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        self.image_size = int(self.meta["image_size"])
        self.mean = np.asarray(self.meta["image_mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["image_std"], dtype=np.float32)
        self.rescale = float(self.meta["rescale_factor"])
        self.prompt_ids = np.asarray(self.meta["prompt_ids"], dtype=np.int64)[None, :]
        self.start_id = int(self.meta["decoder_start_token_id"])
        self.eos_id = int(self.meta["eos_token_id"])
        self.layers = int(self.meta["decoder_layers"])
        self.max_new_tokens = max_new_tokens or int(self.meta["max_new_tokens"])
        self.prefill_mode = prefill_mode or str(self.meta.get("prefill_mode", "prompt"))
        if self.prefill_mode not in PREFILL_MODES:
            raise RuntimeError(
                f"unknown prefill_mode {self.prefill_mode!r}; expected one of {PREFILL_MODES}"
            )
        self.start_ids = np.asarray([[self.start_id]], dtype=np.int64)

        self.threads = threads
        self.sessions = self._load_sessions()
        self.prefill_outputs = [o.name for o in self.sessions["decoder_prefill"].get_outputs()]
        self.prefill_inputs = [i.name for i in self.sessions["decoder_prefill"].get_inputs()]
        self.step_inputs = [i.name for i in self.sessions["decoder_step"].get_inputs()]
        self.step_outputs = [o.name for o in self.sessions["decoder_step"].get_outputs()]
        # The graph may name its embedding input either way depending on the export; take
        # whichever the file declares instead of guessing and failing at request time.
        self.prefill_embeds_input = next(
            (name for name in ("decoder_input_embeds", "decoder_inputs_embeds")
             if name in self.prefill_inputs),
            "decoder_input_embeds",
        )
        self._check_cache_contract()

        expected = self.meta.get("image_tokens")
        if expected is not None:
            declared = self.image_tokens_from_graph()
            if declared != int(expected):
                raise RuntimeError(
                    f"vision encoder emits {declared} image tokens but meta.json says {expected}; "
                    "the graphs and meta.json came from different runs"
                )
        if self.meta.get("cache_layout", "self_and_cross") != "self_and_cross":
            raise RuntimeError(f"unsupported cache layout {self.meta['cache_layout']!r}")

    def _load_sessions(self) -> dict[str, ort.InferenceSession]:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        # Fixed encoder length and a single image per request: let ORT keep memory
        # low instead of scaling for the worst case.
        options.enable_cpu_mem_arena = True
        sessions = {}
        started = time.perf_counter()
        for name in (
            "vision_encoder",
            "embed_tokens",
            "encoder_model",
            "decoder_prefill",
            "decoder_step",
        ):
            path = self.model_dir / f"{name}.onnx"
            sessions[name] = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        logger.info("loaded 5 sessions in %.1fs", time.perf_counter() - started)
        return sessions

    def _check_cache_contract(self) -> None:
        """Fail at startup if the graphs and meta.json disagree about the cache.

        A mismatch here is only visible a hundred tokens into decoding, as a broadcast
        error or a silently truncated cache, so it is worth one eager check.
        """
        present = [name for name in self.prefill_outputs if name.startswith("present.")]
        past = [name for name in self.step_inputs if name.startswith("past_key_values.")]
        returned = [name for name in self.step_outputs if name.startswith("present.")]
        known = {"input_ids", "encoder_attention_mask", "encoder_hidden_states"}
        unknown = [n for n in self.step_inputs if not n.startswith("past_key_values.") and n not in known]
        problems = [
            f"prefill emits {len(present)} present tensors, expected {4 * self.layers}",
            f"step takes {len(past)} past tensors, expected {4 * self.layers}",
            f"step returns {len(returned)} present tensors, expected {2 * self.layers}",
            f"step wants unrecognised inputs {unknown}",
        ]
        wrong = [p for p, ok in zip(problems,
                                    (len(present) == 4 * self.layers,
                                     len(past) == 4 * self.layers,
                                     len(returned) == 2 * self.layers,
                                     not unknown)) if not ok]
        if wrong:
            raise RuntimeError(
                f"cache contract mismatch for {self.layers} layers in {self.model_dir}: "
                + "; ".join(wrong)
            )

    # -- preprocessing -------------------------------------------------------
    def preprocess(self, image) -> np.ndarray:
        # Byte-exact with the reference preprocessing measured offline: RGB,
        # bicubic to 768x768, rescale, ImageNet normalize, CHW.
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) * self.rescale
        array = (array - self.mean) / self.std
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32)

    # -- inference -----------------------------------------------------------
    def _run(self, name: str, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.sessions[name].run(None, feed)

    def image_tokens_from_graph(self) -> int:
        """Read the vision output length from the graph instead of assuming 577."""
        shape = self.sessions["vision_encoder"].get_outputs()[0].shape
        if len(shape) == 3 and isinstance(shape[1], int):
            return shape[1]
        return -1

    def prefill_embeds(self, prompt_embeds: np.ndarray) -> np.ndarray:
        """The decoder prefix this build serves, per meta.json / PREFILL_MODE."""
        if self.prefill_mode == "prompt":
            return prompt_embeds
        if self.prefill_mode == "last":
            return prompt_embeds[:, -1:, :]
        start = self._run("embed_tokens", {"input_ids": self.start_ids})[0]
        if self.prefill_mode == "start":
            return start
        return np.concatenate([prompt_embeds, start], axis=1).astype(np.float32)

    def step_feed(self, mask: np.ndarray, token: int, cache: dict[str, np.ndarray],
                  encoder_states: np.ndarray) -> dict[str, np.ndarray]:
        """Everything the step graph declares, keyed by the names it declared."""
        feed = {
            "encoder_attention_mask": mask,
            "input_ids": np.array([[token]], dtype=np.int64),
        }
        if "encoder_hidden_states" in self.step_inputs:
            # Tracing normally drops this input, because the cached cross-attention makes
            # the values unused. When it survives, the values still do not matter -- only
            # the mask length does -- so feeding the real states is the cheapest way to
            # satisfy it.
            feed["encoder_hidden_states"] = encoder_states
        for name in self.step_inputs:
            if name.startswith("past_key_values."):
                feed[name] = cache["present." + name[len("past_key_values."):]]
        missing = [name for name in self.step_inputs if name not in feed]
        if missing:
            raise RuntimeError(f"step graph wants inputs this server does not provide: {missing}")
        return feed

    def generate(self, image) -> tuple[str, dict[str, float]]:
        started = time.perf_counter()
        pixel_values = self.preprocess(image)

        image_features = self._run("vision_encoder", {"pixel_values": pixel_values})[0]
        prompt_embeds = self._run("embed_tokens", {"input_ids": self.prompt_ids})[0]
        merged = np.concatenate([image_features, prompt_embeds], axis=1).astype(np.float32)
        encoder_length = merged.shape[1]
        # The step graph broadcasts this mask against the cached cross-attention
        # tensors. Check here so a mismatch is one clear error rather than a
        # broadcast failure a hundred tokens into decoding.
        expected = int(self.meta.get("encoder_length") or 0)
        if expected and encoder_length != expected:
            raise RuntimeError(
                f"encoder length {encoder_length} does not match meta.json ({expected}); "
                "check the prompt ids and the vision graph"
            )
        mask = np.ones((1, encoder_length), dtype=np.float32)
        encoder_states = self._run("encoder_model", {"inputs_embeds": merged, "attention_mask": mask})[0]
        prefill_at = time.perf_counter()

        # The decoder prefix: see the module docstring. Only the last row of logits is
        # kept; the positions exist so that row has the context the reference gave it.
        outputs = self._run(
            "decoder_prefill",
            {
                "encoder_hidden_states": encoder_states,
                "encoder_attention_mask": mask,
                self.prefill_embeds_input: self.prefill_embeds(prompt_embeds),
            },
        )
        cache = dict(zip(self.prefill_outputs[1:], outputs[1:]))
        tokens = [int(outputs[0][0, -1].argmax())]

        while tokens[-1] != self.eos_id and len(tokens) < self.max_new_tokens:
            # The embedding lookup happens inside the step graph now, so this is a
            # token id rather than a 768-float tensor.
            outputs = self._run("decoder_step", self.step_feed(mask, tokens[-1], cache, encoder_states))
            tokens.append(int(outputs[0][0, -1].argmax()))
            cache.update(zip(self.step_outputs[1:], outputs[1:]))

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        finished = time.perf_counter()
        timings = {
            "encoder_ms": round(1000 * (prefill_at - started), 1),
            "decode_ms": round(1000 * (finished - prefill_at), 1),
            "total_ms": round(1000 * (finished - started), 1),
        }
        return text, timings
