"""Post-processing parity: `pure_text` decoding and EXIF-free preprocessing.

Both behaviours come from the reference processor and are easy to "tidy up" by
accident: `.strip()` plus `skip_special_tokens=True` looks harmless, and honouring
EXIF orientation looks like a fix. Neither matches the reference, so these tests pin
the reference behaviour rather than the tidier one.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from tokenizers import Tokenizer, models, pre_tokenizers

from server.ocr_engine import decode_pure_text

# Ids 0-3 are the special tokens the WordLevel model is built around; w1/w2 stand in
# for ordinary words and " x " is a token whose text carries its own whitespace.
VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "w1": 4, "w2": 5, " x ": 6}
SOS, PAD, EOS = 0, 1, 2
W1, W2, PADDED_WORD = 4, 5, 6


def tiny_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_decode_keeps_middle_special_tokens_literal():
    """`[</s>, <s>, w1, <pad>, w2, </s>]` decodes to "</s> <s> w1 <pad> w2 </s>".

    Only the literal `<s>`/`</s>` strings are removed, so the expected text is
    exactly "  w1 <pad> w2 " -- the middle `<pad>` survives, and the two spaces
    left behind by the stripped wrappers are not collapsed.
    """
    text = decode_pure_text(tiny_tokenizer(), [EOS, SOS, W1, PAD, W2, EOS])
    assert text == "  w1 <pad> w2 "


def test_decode_preserves_whitespace_and_does_not_strip():
    tokenizer = tiny_tokenizer()
    # The token's text is " x ", so a `.strip()` would turn this into "x".
    assert decode_pure_text(tokenizer, [PADDED_WORD]) == " x "
    # Same token wrapped in the special tokens: the wrappers go, their spaces stay.
    assert decode_pure_text(tokenizer, [SOS, PADDED_WORD, EOS]) == "  x  "


def test_preprocess_ignores_exif_orientation(engine):
    pixels = np.arange(40 * 20 * 3, dtype=np.uint8).reshape(20, 40, 3)
    plain = Image.fromarray(pixels, "RGB")

    tagged = plain.copy()
    exif = tagged.getexif()
    exif[274] = 6  # Orientation: rotate 90 degrees clockwise
    tagged.info["exif"] = exif.tobytes()

    # Guard that the tag is actually attached and would rotate the image, so the
    # equality below is about ignoring EXIF rather than about a missing tag.
    assert tagged.getexif().get(274) == 6
    assert ImageOps.exif_transpose(tagged).size == (20, 40)

    assert np.array_equal(engine.preprocess(tagged), engine.preprocess(plain))
