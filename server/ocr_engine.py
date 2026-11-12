"""Florence-2 OCR over ONNX Runtime, with no PyTorch anywhere.

Layout proven by the offline parity experiments: the vision tower turns the image
into 577 tokens, the prompt contributes 10 more, one encoder pass fuses them, a
prefill pass produces the first token plus the full attention cache, and each later
step takes that cache and returns one token.

A few details matter more than they look:

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
* Decoding. `<OCR>` is post-processed as `pure_text`: the ids are decoded with special
  tokens left in, then only the literal `<s>` and `</s>` strings are removed. Nothing
  is stripped, so the whitespace the reference keeps survives, and any other special
  token (a `<pad>`, say) stays in the text exactly as the reference leaves it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

PREFILL_MODES = ("prompt", "last", "start", "prompt_start")


def cgroup_cpu_quota(
    v2_path: Path = Path("/sys/fs/cgroup/cpu.max"),
    v1_quota: Path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
    v1_period: Path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
) -> int | None:
    """Return the container CPU limit as an integer core count, or None if unlimited.

    Docker's `--cpus=4` sets a cgroup quota rather than a CPU affinity mask, so
    `os.cpu_count()` inside the container still reports all host threads. Reading the
    quota lets us size the intra-op pool to the CPUs the container is actually allowed
    to burn instead of oversubscribing the host count.
    """
    try:
        if v2_path.exists():
            parts = v2_path.read_text(encoding="utf-8").strip().split()
            if len(parts) == 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, quota // period)
        if v1_quota.exists() and v1_period.exists():
            quota = int(v1_quota.read_text(encoding="utf-8").strip())
            period = int(v1_period.read_text(encoding="utf-8").strip())
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        return None
    return None


def resolve_ort_threads(configured: int, workers: int = 1, quota: int | None = None) -> int:
    """Pick the per-session intra-op thread count.

    An explicit `ORT_THREADS > 0` always wins. When `ORT_THREADS == 0`, keep ORT's
    own default (`0`, which uses physical cores and enables thread affinity) on an
    unconstrained single-worker host, but divide the active cgroup CPU quota (or host
    core count when multiple workers share the process) across workers so concurrent
    sessions do not oversubscribe the quota.
    """
    if configured > 0:
        return configured
    effective_quota = cgroup_cpu_quota() if quota is None else quota
    if effective_quota is not None and effective_quota > 0:
        return max(1, effective_quota // max(1, workers))
    if workers > 1:
        return max(1, (os.cpu_count() or 1) // workers)
    return 0


def clean_up_tokenization_spaces(text: str) -> str:
    """Clean English tokenization artifacts to match transformers' baseline decode."""
    return (
        text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )


def decode_pure_text(tokenizer: Tokenizer, tokens: list[int]) -> str:
    """Decode exactly like Florence-2's ``pure_text`` post-processing.

    The reference decodes with ``skip_special_tokens=False`` and then removes the
    literal ``<s>`` and ``</s>`` strings. It does not strip whitespace and does not
    remove any other special token, so neither do we.
    """
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    return text.replace("<s>", "").replace("</s>", "")


POSTPROCESS_MODES = ("pure_text", "baseline")
PRECISION_MODES = ("auto", "fp32", "int8", "int4")


def has_avx_vnni() -> bool:
    """Detect if the CPU hardware supports Vector Neural Network Instructions (VNNI).
    Works in Linux/Docker (/proc/cpuinfo) and bare-metal Windows."""
    # 1. Linux & Docker on Windows (WSL2 / Docker Desktop)
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("flags"):
                    flags = set(line.split()[2:])
                    return bool({"avx_vnni", "avx512_vnni"} & flags)
    except Exception:
        pass

    # 2. Native Windows fallback
    if platform.system() == "Windows":
        try:
            proc = platform.processor().lower()
            if any(term in proc for term in ["12th gen", "13th gen", "14th gen", "ultra", "xeon", "genoa"]):
                return True
        except Exception:
            pass

    return False


def load_precision_files(models_dir: Path, requested: str = "auto",
                         allow_fallback: bool = False) -> dict[str, Path]:
    """Resolve the five ONNX model paths from quant.json or directory defaults."""
    if requested not in PRECISION_MODES:
        raise ValueError(
            f"unknown precision {requested!r}; expected one of {PRECISION_MODES!r}"
        )
    manifest_path = models_dir / "quant.json"
    target_var = requested
    if requested == "auto":
        target_var = "int4" if manifest_path.exists() else "fp32"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variants = manifest.get("variants", {})
        if requested == "auto":
            if "int4" in variants:
                target_var = "int4"
            elif "int8" in variants:
                target_var = "int8"
            else:
                target_var = "fp32"
        if target_var in variants:
            files = {}
            missing = []
            # Hardware adaptation is an `auto` policy, not an override: an explicit
            # PRECISION=fp32|int8|int4 must load exactly the variant it names.
            vnni_upgrade = requested == "auto" and has_avx_vnni()
            for role, fname in variants[target_var].items():
                p = models_dir / fname
                if vnni_upgrade and role == "encoder_model":
                    int8_enc = models_dir / "encoder_model.int8.onnx"
                    if int8_enc.exists():
                        p = int8_enc
                        logger.info(
                            "Host CPU supports AVX-VNNI: using encoder_model.int8.onnx."
                        )
                if not p.exists():
                    missing.append(p.name)
                files[role] = p
            if missing:
                if allow_fallback and target_var != "fp32":
                    logger.warning(
                        "precision %s missing files %s; falling back to fp32",
                        target_var, missing
                    )
                    return load_precision_files(models_dir, requested="fp32", allow_fallback=False)
                raise FileNotFoundError(
                    f"precision {target_var} missing required model files in {models_dir}: {missing}"
                )
            return files

    # Fallback to standard filenames
    files = {}
    missing = []
    for role in ("vision_encoder", "embed_tokens", "encoder_model", "decoder_prefill", "decoder_step"):
        p = models_dir / f"{role}.onnx"
        if not p.exists():
            missing.append(p.name)
        files[role] = p
    if missing:
        raise FileNotFoundError(f"Missing default model files in {models_dir}: {missing}")
    return files


def decode_text(tokenizer: Tokenizer, tokens: list[int], mode: str = "pure_text") -> str:
    """Decode tokens under the requested post-processing mode."""
    if mode not in POSTPROCESS_MODES:
        raise ValueError(
            f"unknown postprocess mode {mode!r}; expected one of {POSTPROCESS_MODES!r}"
        )
    text = decode_pure_text(tokenizer, tokens)
    if mode == "baseline":
        return clean_up_tokenization_spaces(text)
    return text


class FlorenceOCR:
    def __init__(self, model_dir: Path | str, threads: int = 0, max_new_tokens: int = 0,
                 prefill_mode: str = "", workers: int = 1,
                 postprocess_mode: str = "pure_text",
                 precision: str = "int8",
                 allow_fp32_fallback: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        if postprocess_mode not in POSTPROCESS_MODES:
            raise ValueError(
                f"unknown postprocess_mode {postprocess_mode!r}; "
                f"expected one of {POSTPROCESS_MODES!r}"
            )
        self.postprocess_mode = postprocess_mode
        self.precision = precision
        self.allow_fp32_fallback = allow_fp32_fallback
        self.model_paths = load_precision_files(
            self.model_dir, requested=precision, allow_fallback=allow_fp32_fallback
        )

        self.image_size = int(self.meta["image_size"])
        self.mean = np.asarray(self.meta["image_mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["image_std"], dtype=np.float32)
        self.rescale = float(self.meta["rescale_factor"])
        self.prompt_ids = np.asarray(self.meta["prompt_ids"], dtype=np.int64)[None, :]
        self.start_id = int(self.meta["decoder_start_token_id"])
        self.eos_id = int(self.meta["eos_token_id"])
        self.layers = int(self.meta["decoder_layers"])
        self.max_new_tokens = max_new_tokens or int(self.meta["max_new_tokens"])
        self.prefill_mode = prefill_mode or str(self.meta.get("prefill_mode", "start"))
        if self.prefill_mode not in PREFILL_MODES:
            raise RuntimeError(
                f"unknown prefill_mode {self.prefill_mode!r}; expected one of {PREFILL_MODES}"
            )
        self.start_ids = np.asarray([[self.start_id]], dtype=np.int64)

        self.threads = resolve_ort_threads(threads, workers=workers)
        self.sessions = self._load_sessions()
        self.prefill_outputs = [o.name for o in self.sessions["decoder_prefill"].get_outputs()]
        self.prefill_inputs = [i.name for i in self.sessions["decoder_prefill"].get_inputs()]
        self.step_inputs = [i.name for i in self.sessions["decoder_step"].get_inputs()]
        self.step_outputs = [o.name for o in self.sessions["decoder_step"].get_outputs()]
        # The graph may name its embedding input either way depending on the export; take
        # whichever the file declares instead of guessing and failing at request time.
        self.prefill_embeds_input = next(
            (name for name in ("decoder_inputs_embeds", "decoder_input_embeds")
             if name in self.prefill_inputs),
            "decoder_inputs_embeds",
        )
        self.step_embeds_input = next(
            (name for name in ("decoder_inputs_embeds", "decoder_input_embeds")
             if name in self.step_inputs),
            "",
        )
        self._check_cache_contract()

        # If embed_tokens.npy exists, load directly for O(1) in-memory table slicing
        # without invoking the ONNX session during decoding.
        embed_npy = self.model_dir / "embed_tokens.npy"
        self.embed_table = np.load(str(embed_npy)) if embed_npy.exists() else None

        # Precompute the static prompt and start-token embeddings (~33 KB total) and
        # the present->past slot mappings once at startup so neither is recomputed on
        # every request or every token.
        if self.embed_table is not None:
            self._prompt_embeds = self.embed_table[self.prompt_ids]
            self._start_embeds = self.embed_table[self.start_ids]
        else:
            self._prompt_embeds = self._run("embed_tokens", {"input_ids": self.prompt_ids})[0]
            self._start_embeds = self._run("embed_tokens", {"input_ids": self.start_ids})[0]
        step_input_set = set(self.step_inputs)
        self._prefill_cache_map = [
            (idx, "past_key_values." + name[len("present."):])
            for idx, name in enumerate(self.prefill_outputs)
            if name.startswith("present.")
            and ("past_key_values." + name[len("present."):]) in step_input_set
        ]
        self._step_cache_map = [
            (idx, "past_key_values." + name[len("present."):])
            for idx, name in enumerate(self.step_outputs)
            if name.startswith("present.")
            and ("past_key_values." + name[len("present."):]) in step_input_set
        ]

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
        options.enable_cpu_mem_arena = os.getenv("ORT_MEM_ARENA", "1") != "0"
        spinning = os.getenv("ORT_ALLOW_SPINNING", "").strip()
        if spinning in ("0", "1"):
            options.add_session_config_entry("session.intra_op.allow_spinning", spinning)
        sessions = {}
        started = time.perf_counter()
        for name, path in self.model_paths.items():
            sessions[name] = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        logger.info("loaded 5 sessions (%s) in %.1fs (intra_op_threads=%s)",
                    self.precision, time.perf_counter() - started, self.threads or "auto")
        return sessions

    def _check_cache_contract(self) -> None:
        """Fail at startup if the graphs and meta.json disagree about the cache.

        A mismatch here is only visible a hundred tokens into decoding, as a broadcast
        error or a silently truncated cache, so it is worth one eager check.
        """
        present = [name for name in self.prefill_outputs if name.startswith("present.")]
        past = [name for name in self.step_inputs if name.startswith("past_key_values.")]
        returned = [name for name in self.step_outputs if name.startswith("present.")]
        known = {
            "input_ids",
            "decoder_inputs_embeds",
            "decoder_input_embeds",
            "encoder_attention_mask",
            "encoder_hidden_states",
        }
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
        # Deliberately byte-exact with the reference CLIPImageProcessor: RGB, bicubic
        # to 768x768, rescale, ImageNet normalize, CHW. No EXIF rotation, because the
        # reference processor does not apply any either.
        image = image.convert("RGB")
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
        if self.prefill_mode == "start":
            return self._start_embeds
        return np.concatenate([prompt_embeds, self._start_embeds], axis=1).astype(np.float32)

    def step_feed(self, mask: np.ndarray, token: int, cache: dict[str, np.ndarray],
                  encoder_states: np.ndarray) -> dict[str, np.ndarray]:
        """Everything the step graph declares, keyed by the names it declared."""
        token_ids = np.array([[token]], dtype=np.int64)
        feed: dict[str, np.ndarray] = {"encoder_attention_mask": mask}
        if self.step_embeds_input:
            if self.embed_table is not None:
                feed[self.step_embeds_input] = self.embed_table[token_ids]
            else:
                feed[self.step_embeds_input] = self._run("embed_tokens", {"input_ids": token_ids})[0]
        if "input_ids" in self.step_inputs:
            feed["input_ids"] = token_ids
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

    def warmup(self) -> float:
        """Run one short pass through every session before /health reports ready.

        Two tokens is enough to touch vision_encoder, encoder_model, decoder_prefill,
        embed_tokens and decoder_step so ORT's memory arena and kernel plans are warm
        before the first timed request arrives.
        """
        started = time.perf_counter()
        self.generate(Image.new("RGB", (64, 64), (255, 255, 255)), max_tokens=2)
        elapsed_ms = round(1000 * (time.perf_counter() - started), 1)
        logger.info("warmed up all sessions in %.1fms", elapsed_ms)
        return elapsed_ms

    def generate(self, image, max_tokens: int = 0) -> tuple[str, dict[str, float]]:
        t_start = time.perf_counter()
        pixel_values = self.preprocess(image)

        t_vis_0 = time.perf_counter()
        image_features = self._run("vision_encoder", {"pixel_values": pixel_values})[0]
        vision_ms = (time.perf_counter() - t_vis_0) * 1000.0

        prompt_embeds = self._prompt_embeds
        merged = np.concatenate([image_features, prompt_embeds], axis=1).astype(np.float32)
        encoder_length = merged.shape[1]
        expected = int(self.meta.get("encoder_length") or 0)
        if expected and encoder_length != expected:
            raise RuntimeError(
                f"encoder length {encoder_length} does not match meta.json ({expected}); "
                "check the prompt ids and the vision graph"
            )
        mask = np.ones((1, encoder_length), dtype=np.float32)

        t_enc_0 = time.perf_counter()
        encoder_states = self._run("encoder_model", {"inputs_embeds": merged, "attention_mask": mask})[0]
        text_encoder_ms = (time.perf_counter() - t_enc_0) * 1000.0

        t_pref_0 = time.perf_counter()
        outputs = self._run(
            "decoder_prefill",
            {
                "encoder_hidden_states": encoder_states,
                "encoder_attention_mask": mask,
                self.prefill_embeds_input: self.prefill_embeds(prompt_embeds),
            },
        )
        first_token = int(outputs[0][0, -1].argmax())
        tokens = [first_token]
        prefill_ms = (time.perf_counter() - t_pref_0) * 1000.0

        t_dec_0 = time.perf_counter()
        embed_ms = 0.0
        step_calls = 0
        cap = max_tokens or self.max_new_tokens

        if tokens[-1] != self.eos_id and len(tokens) < cap:
            feed: dict[str, np.ndarray] = {"encoder_attention_mask": mask}
            if "encoder_hidden_states" in self.step_inputs:
                feed["encoder_hidden_states"] = encoder_states
            for out_idx, past_name in self._prefill_cache_map:
                feed[past_name] = outputs[out_idx]
            token_ids = np.empty((1, 1), dtype=np.int64)

            while tokens[-1] != self.eos_id and len(tokens) < cap:
                token_ids[0, 0] = tokens[-1]
                if self.step_embeds_input:
                    t_emb = time.perf_counter()
                    if self.embed_table is not None:
                        feed[self.step_embeds_input] = self.embed_table[token_ids]
                    else:
                        feed[self.step_embeds_input] = self._run("embed_tokens", {"input_ids": token_ids})[0]
                    embed_ms += (time.perf_counter() - t_emb) * 1000.0
                if "input_ids" in self.step_inputs:
                    feed["input_ids"] = token_ids
                outputs = self._run("decoder_step", feed)
                step_calls += 1
                tokens.append(int(outputs[0][0, -1].argmax()))
                for out_idx, past_name in self._step_cache_map:
                    feed[past_name] = outputs[out_idx]

        decode_ms = (time.perf_counter() - t_dec_0) * 1000.0
        text = decode_text(self.tokenizer, tokens, mode=self.postprocess_mode)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        num_tokens = len(tokens)
        decode_tok_s = round(step_calls * 1000.0 / decode_ms, 2) if decode_ms > 0 else 0.0
        total_tok_s = round(num_tokens * 1000.0 / total_ms, 2) if total_ms > 0 else 0.0

        timings = {
            "encoder_ms": round(vision_ms + text_encoder_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "total_ms": round(total_ms, 1),
            "vision_ms": round(vision_ms, 1),
            "text_encoder_ms": round(text_encoder_ms, 1),
            "prefill_ms": round(prefill_ms, 1),
            "embed_ms": round(embed_ms, 1),
            "steps": step_calls,
            "tokens": num_tokens,
            "decode_tokens_per_s": decode_tok_s,
            "total_tokens_per_s": total_tok_s,
        }
        return text, timings
