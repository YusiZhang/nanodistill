"""Inference helpers for sequence-classification encoder models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..utils.errors import ConfigError
from .modeling import create_encoder_model
from .tokenization import load_hf_tokenizer, tokenize_texts


@dataclass
class Prediction:
    """Single sequence-classification prediction."""

    label: str
    confidence: float
    probabilities: Dict[str, float]


def predict_sequence_class(model_path: str, texts: Sequence[str]) -> List[Prediction]:
    """Run sequence-classification inference for one or more texts."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise ConfigError(f"Model path not found: {model_path}")

    config_file = model_dir / "encoder_config.json"
    label_file = model_dir / "label_map.json"
    weights_file = model_dir / "encoder_model.safetensors"
    npz_weights_file = model_dir / "encoder_model.npz"

    if not config_file.exists() or not label_file.exists():
        raise ConfigError(
            "Missing encoder artifacts. Expected encoder_config.json and label_map.json"
        )

    with open(config_file) as f:
        cfg = json.load(f)

    with open(label_file) as f:
        label_maps = json.load(f)

    id_to_label = {int(k): v for k, v in label_maps["id_to_label"].items()}
    num_labels = len(id_to_label)

    tokenizer = load_hf_tokenizer(cfg["encoder_backbone"])
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))

    model, _ = create_encoder_model(
        backbone=cfg["encoder_backbone"],
        vocab_size=vocab_size,
        num_labels=num_labels,
        max_length=int(cfg.get("encoder_max_length", 256)),
    )

    try:
        import mlx.core as mx
    except ImportError as e:
        raise ConfigError("mlx is required for encoder inference") from e

    if cfg.get("encoder_use_lora"):
        from .lora import apply_encoder_lora

        apply_encoder_lora(
            model.model,
            rank=int(cfg.get("encoder_lora_rank", 8)),
            targets=cfg.get("encoder_lora_targets", ["query", "value"]),
        )

    weights = _load_weights(weights_file, npz_weights_file)
    if weights:
        try:
            mx_weights = [(k, mx.array(v)) for k, v in weights.items()]
            model.load_weights(mx_weights)
        except Exception as e:
            raise ConfigError(f"Failed loading encoder weights: {e}") from e

    tokenized = tokenize_texts(
        tokenizer=tokenizer,
        texts=list(texts),
        max_length=int(cfg.get("encoder_max_length", 256)),
    )

    input_ids = mx.array(tokenized["input_ids"])
    attention_mask = mx.array(tokenized["attention_mask"])
    token_type_ids = mx.array(tokenized["token_type_ids"])

    model.model.eval()
    logits = model(input_ids, attention_mask, token_type_ids)
    probs = _softmax(np.asarray(logits))

    results: List[Prediction] = []
    for row in probs:
        best_idx = int(np.argmax(row))
        label = id_to_label.get(best_idx, str(best_idx))
        prob_map = {id_to_label.get(i, str(i)): float(p) for i, p in enumerate(row)}
        results.append(
            Prediction(
                label=label,
                confidence=float(row[best_idx]),
                probabilities=prob_map,
            )
        )

    return results


def _load_weights(weights_file: Path, npz_weights_file: Path) -> Dict[str, np.ndarray]:
    """Load model weights from safetensors or npz fallback."""
    if weights_file.exists():
        try:
            from safetensors.numpy import load_file

            return load_file(str(weights_file))
        except Exception:
            pass

    if npz_weights_file.exists():
        loaded = np.load(npz_weights_file)
        return {k: loaded[k] for k in loaded.files}

    return {}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
