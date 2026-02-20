"""Tests for encoder tokenization helpers."""

import numpy as np

from nanodistill.encoder.tokenization import tokenize_texts


class _DummyTokenizerNoTokenType:
    def __call__(self, texts, **kwargs):
        max_len = max(len(t.split()) for t in texts)
        return {
            "input_ids": np.ones((len(texts), max_len), dtype=np.int32),
            "attention_mask": np.ones((len(texts), max_len), dtype=np.int32),
        }


def test_tokenize_texts_infers_token_type_ids_when_missing():
    tokenizer = _DummyTokenizerNoTokenType()
    out = tokenize_texts(tokenizer, ["hello world", "test"], max_length=8)

    assert "token_type_ids" in out
    assert out["token_type_ids"].shape == out["input_ids"].shape
    assert np.all(out["token_type_ids"] == 0)
