"""Tokenization helpers for encoder sequence classification."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..utils.errors import ConfigError


def load_hf_tokenizer(backbone: str):
    """Load a Hugging Face tokenizer for the encoder backbone."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ConfigError(
            "transformers is required for encoder tokenization. "
            "Install with: pip install transformers"
        ) from e

    return AutoTokenizer.from_pretrained(backbone)


def tokenize_texts(
    tokenizer,
    texts: List[str],
    max_length: int,
) -> Dict[str, np.ndarray]:
    """Tokenize texts into numpy arrays suitable for MLX tensors."""
    if not texts:
        return {
            "input_ids": np.zeros((0, max_length), dtype=np.int32),
            "attention_mask": np.zeros((0, max_length), dtype=np.int32),
            "token_type_ids": np.zeros((0, max_length), dtype=np.int32),
        }

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="np",
    )

    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise ConfigError("Tokenizer output missing required fields: input_ids/attention_mask")

    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = np.zeros_like(encoded["input_ids"], dtype=np.int32)

    return {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int32),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int32),
        "token_type_ids": np.asarray(token_type_ids, dtype=np.int32),
    }
