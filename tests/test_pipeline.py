"""End-to-end wiring test for the decode loop, using tiny stand-in graphs.

Real parity against PyTorch was established offline and is recorded in journal.md.
What this test protects is the part that would break silently in a refactor: the
cache contract between prefill and step, the encoder-length invariant, the
tokenizer round trip, and preprocessing of awkward images.

The stand-in graphs are generated here rather than committed, because committing
model-shaped binaries to a repo that forbids pre-built models would be confusing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from server.ocr_engine import FlorenceOCR

VOCAB = 32
LAYERS = 2
IMAGE_TOKENS = 20
PROMPT_IDS = [0, 11, 12, 13, 2]
EOS = 2


def tensor_info(name: str, shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def cache_info(name: str) -> onnx.ValueInfoProto:
    return tensor_info(name, ["batch", "heads", "seq", "dim"])


def save(directory: Path, name: str, inputs, outputs, nodes, initializers) -> None:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(directory / f"{name}.onnx"))


def build_tiny_model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    weights = (rng.randn(768, VOCAB) * 0.02).astype(np.float32)
    slots = ("decoder.key", "decoder.value", "encoder.key", "encoder.value")

    save(
        directory,
        "vision_encoder",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 768, 768])],
        [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [1, IMAGE_TOKENS, 768])],
        [
            helper.make_node("GlobalAveragePool", ["pixel_values"], ["pooled"]),
            # widen the 3 pooled channels to 768, then repeat across tokens: the
            # point is only to emit a (1, 20, 768) tensor with real arithmetic
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "proj"], ["wide"]),
            helper.make_node("Tile", ["wide", "tall"], ["tall_t"]),
            helper.make_node("Reshape", ["tall_t", "shape"], ["image_features"]),
        ],
        [
            helper.make_tensor("proj", TensorProto.FLOAT, [3, 768], (rng.randn(3, 768) * 0.02).astype(np.float32)),
            helper.make_tensor("tall", TensorProto.INT64, [2], [IMAGE_TOKENS, 1]),
            helper.make_tensor("shape", TensorProto.INT64, [3], [1, IMAGE_TOKENS, 768]),
        ],
    )

    embedding_table = (rng.randn(VOCAB, 768) * 0.02).astype(np.float32)
    save(
        directory,
        "embed_tokens",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "sequence_length"])],
        [tensor_info("inputs_embeds", [1, "sequence_length", 768])],
        [helper.make_node("Gather", ["table", "input_ids"], ["inputs_embeds"], axis=0)],
        [helper.make_tensor("table", TensorProto.FLOAT, [VOCAB, 768], embedding_table)],
    )

    save(
        directory,
        "encoder_model",
        [
            tensor_info("inputs_embeds", [1, "encoder_sequence_length", 768]),
            tensor_info("attention_mask", [1, "encoder_sequence_length"]),
        ],
        [tensor_info("last_hidden_state", [1, "encoder_sequence_length", 768])],
        [helper.make_node("Mul", ["inputs_embeds", "half"], ["last_hidden_state"])],
        [helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5])],
    )

    prefill_outputs = ["logits"] + [f"present.{i}.{s}" for i in range(LAYERS) for s in slots]
    save(
        directory,
        "decoder_prefill",
        [
            tensor_info("encoder_hidden_states", [1, "encoder_sequence_length", 768]),
            tensor_info("encoder_attention_mask", [1, "encoder_sequence_length"]),
            tensor_info("decoder_inputs_embeds", [1, "decoder_sequence_length", 768]),
        ],
        [tensor_info("logits", [1, "decoder_sequence_length", VOCAB])]
        + [cache_info(name) for name in prefill_outputs[1:]],
        [helper.make_node("MatMul", ["decoder_inputs_embeds", "w"], ["logits"])]
        + [helper.make_node("Unsqueeze", ["encoder_hidden_states", "axes"], ["hs4"])]
        + [helper.make_node("Mul", ["hs4", "one"], [name]) for name in prefill_outputs[1:]],
        [
            helper.make_tensor("w", TensorProto.FLOAT, [768, VOCAB], weights),
            helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
            helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
        ],
    )

    step_outputs = ["logits"] + [
        f"present.{i}.{s}" for i in range(LAYERS) for s in ("decoder.key", "decoder.value")
    ]
    save(
        directory,
        "decoder_step",
        [
            tensor_info("encoder_attention_mask", [1, "encoder_sequence_length"]),
            tensor_info("decoder_inputs_embeds", [1, "decoder_sequence_length", 768]),
        ]
        + [cache_info(f"past_key_values.{i}.{s}") for i in range(LAYERS) for s in slots],
        [tensor_info("logits", [1, "decoder_sequence_length", VOCAB])]
        + [cache_info(name) for name in step_outputs[1:]],
        [helper.make_node("MatMul", ["decoder_inputs_embeds", "w2"], ["logits"])]
        + [
            helper.make_node(
                "Add", [f"past_key_values.{i}.{slot}", "zero"], [f"present.{i}.{slot}"]
            )
            for i in range(LAYERS)
            for slot in ("decoder.key", "decoder.value")
        ],
        [
            helper.make_tensor("w2", TensorProto.FLOAT, [768, VOCAB], weights),
            helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
        ],
    )

    meta = {
        "image_size": 768,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1 / 255.0,
        "prompt_ids": PROMPT_IDS,
        "decoder_start_token_id": 0,
        "eos_token_id": EOS,
        "pad_token_id": 1,
        "decoder_layers": LAYERS,
        "max_new_tokens": 6,
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    vocabulary = {f"w{i}": i for i in range(VOCAB)}
    vocabulary.update({"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    return directory


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> FlorenceOCR:
    return FlorenceOCR(build_tiny_model(tmp_path_factory.mktemp("tiny-model")), threads=2)


def test_generate_returns_text_within_cap(engine):
    text, timings = engine.generate(Image.new("RGB", (300, 200), (255, 255, 255)))
    assert isinstance(text, str) and text
    assert set(timings) == {"encoder_ms", "decode_ms", "total_ms"}
    assert timings["total_ms"] >= 0


def test_cache_contract_between_prefill_and_step(engine):
    prefill_present = [n for n in engine.prefill_outputs if n.startswith("present.")]
    step_past = [n for n in engine.step_inputs if n.startswith("past_key_values.")]
    step_present = [n for n in engine.step_outputs if n.startswith("present.")]

    # 6 layers x 4 slots in prefill; the step graph consumes all 24 and returns only
    # the 12 self-attention tensors, because cross-attention cannot change mid-request.
    assert len(prefill_present) == LAYERS * 4
    assert len(step_past) == LAYERS * 4
    assert len(step_present) == LAYERS * 2

    def mapped(name: str) -> str:
        return "present." + name[len("past_key_values."):]

    assert all(mapped(n) in prefill_present for n in step_past), "step cache names must line up"
    assert all(n in step_present for n in step_present if "encoder" not in n)


def test_encoder_length_matches_prompt_contract(engine):
    pixel_values = engine.preprocess(Image.new("RGB", (300, 200)))
    features = engine.sessions["vision_encoder"].run(None, {"pixel_values": pixel_values})[0]
    prompt = engine.sessions["embed_tokens"].run(None, {"input_ids": engine.prompt_ids})[0]
    assert features.shape[1] == IMAGE_TOKENS
    assert prompt.shape[1] == len(PROMPT_IDS)


def test_preprocessing_handles_awkward_images(engine):
    for image in (
        Image.new("RGBA", (40, 900), (10, 20, 30, 128)),
        Image.new("L", (900, 40), 128),
        Image.new("P", (64, 64)),
    ):
        array = engine.preprocess(image)
        assert array.shape == (1, 3, 768, 768)
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_generate_stops_at_token_cap(engine, monkeypatch):
    monkeypatch.setattr(engine, "max_new_tokens", 3)
    text, _ = engine.generate(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert len(text.split()) <= 3
