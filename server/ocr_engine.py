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


class FlorenceOCR:
    def __init__(self, model_dir: Path | str, threads: int = 0, max_new_tokens: int = 0) -> None:
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

        # One encoder length per request is enough because the OCR prompt is fixed,
        # but it is read from the graphs rather than hardcoded.
        self.threads = threads
        self.sessions = self._load_sessions()
        self.prefill_outputs = [o.name for o in self.sessions["decoder_prefill"].get_outputs()]
        self.step_inputs = [i.name for i in self.sessions["decoder_step"].get_inputs()]
        self.step_outputs = [o.name for o in self.sessions["decoder_step"].get_outputs()]

        expected = self.meta.get("image_tokens")
        if expected is not None:
            declared = self.image_tokens_from_graph()
            if declared != int(expected):
                raise RuntimeError(
                    f"vision encoder emits {declared} image tokens but meta.json says {expected}; "
                    "the graphs and meta.json came from different runs"
                )

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
    # -- inference -----------------------------------------------------------
    def _run(self, name: str, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.sessions[name].run(None, feed)

    def image_tokens_from_graph(self) -> int:
        """Read the vision output length from the graph instead of assuming 577."""
        shape = self.sessions["vision_encoder"].get_outputs()[0].shape
        if len(shape) == 3 and isinstance(shape[1], int):
            return shape[1]
        return -1

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

        start_ids = np.array([[self.start_id]], dtype=np.int64)
        start_embeds = self._run("embed_tokens", {"input_ids": start_ids})[0]
        outputs = self._run(
            "decoder_prefill",
            {
                "encoder_hidden_states": encoder_states,
                "encoder_attention_mask": mask,
                "decoder_inputs_embeds": start_embeds,
            },
        )
        cache = dict(zip(self.prefill_outputs[1:], outputs[1:]))
        tokens = [int(outputs[0][0, -1].argmax())]

        while tokens[-1] != self.eos_id and len(tokens) < self.max_new_tokens:
            embeds = self._run(
                "embed_tokens", {"input_ids": np.array([[tokens[-1]]], dtype=np.int64)}
            )[0]
            feed = {"encoder_attention_mask": mask, "decoder_inputs_embeds": embeds}
            for name in self.step_inputs:
                if name.startswith("past_key_values."):
                    slot = name[len("past_key_values."):]
                    feed[name] = cache[f"present.{slot}"]
            outputs = self._run("decoder_step", feed)
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
