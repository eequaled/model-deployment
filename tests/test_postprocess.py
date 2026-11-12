"""Unit tests for decode post-processing modes."""
import pytest
from tokenizers import Tokenizer

from server.ocr_engine import (
    clean_up_tokenization_spaces,
    decode_text,
    POSTPROCESS_MODES,
)


def test_clean_up_tokenization_spaces_punctuation():
    raw = "Hello , world ! Is it working ? Yes . Indeed ."
    expected = "Hello, world! Is it working? Yes. Indeed."
    assert clean_up_tokenization_spaces(raw) == expected


def test_clean_up_tokenization_spaces_contractions():
    raw = "it 's here and they 're coming and I 've seen it and I 'm ready and ca n't go"
    cleaned = clean_up_tokenization_spaces(raw)
    assert "it's" in cleaned
    assert "they're" in cleaned
    assert "I've" in cleaned
    assert "I'm" in cleaned
    assert "can't" in cleaned


def test_decode_text_rejects_unknown_mode():
    from pathlib import Path
    tok_path = Path("/home/eequaled/florence-lab/onnx-export-v3/tokenizer.json")
    if not tok_path.exists():
        pytest.skip("no tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tok_path))
    with pytest.raises(ValueError, match="unknown postprocess mode"):
        decode_text(tokenizer, [0], mode="bogus")


def test_postprocess_modes_list():
    assert "pure_text" in POSTPROCESS_MODES
    assert "baseline" in POSTPROCESS_MODES
