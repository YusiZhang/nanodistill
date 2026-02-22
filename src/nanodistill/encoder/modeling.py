"""MLX-native BERT-style encoder model for sequence classification.

This module provides:
- A BERT-like encoder classifier implemented in MLX
- Hugging Face checkpoint loading from safetensors
- Parameter-name mapping from HF BERT to MLX module names
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.errors import ConfigError

if TYPE_CHECKING:
    import mlx.core as mlx_mx
    import mlx.nn as mlx_nn
else:
    try:
        import mlx.core as mlx_mx
        import mlx.nn as mlx_nn
    except ImportError:
        mlx_mx = None  # type: ignore[assignment]
        mlx_nn = None  # type: ignore[assignment]


@dataclass
class EncoderBackboneSpec:
    """Backbone hyperparameters for the sequence-classification encoder."""

    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    type_vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float


def resolve_backbone_spec(backbone: str) -> EncoderBackboneSpec:
    """Resolve backbone architecture from Hugging Face config when available."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(backbone)
        model_type = str(getattr(cfg, "model_type", "")).lower()
        if model_type and model_type != "bert":
            raise ConfigError(
                f"encoder_backbone '{backbone}' has model_type='{model_type}'. "
                "V1 only supports BERT checkpoints."
            )

        return EncoderBackboneSpec(
            hidden_size=int(getattr(cfg, "hidden_size")),
            intermediate_size=int(getattr(cfg, "intermediate_size")),
            max_position_embeddings=int(getattr(cfg, "max_position_embeddings", 512)),
            vocab_size=int(getattr(cfg, "vocab_size")),
            type_vocab_size=int(getattr(cfg, "type_vocab_size", 2)),
            num_hidden_layers=int(getattr(cfg, "num_hidden_layers", 12)),
            num_attention_heads=int(getattr(cfg, "num_attention_heads", 12)),
            hidden_dropout_prob=float(getattr(cfg, "hidden_dropout_prob", 0.1)),
            attention_probs_dropout_prob=float(
                getattr(cfg, "attention_probs_dropout_prob", 0.1)
            ),
            layer_norm_eps=float(getattr(cfg, "layer_norm_eps", 1e-12)),
        )
    except ConfigError:
        raise
    except Exception:
        pass

    # Fallback for bert-base-like defaults.
    return EncoderBackboneSpec(
        hidden_size=768,
        intermediate_size=3072,
        max_position_embeddings=512,
        vocab_size=30522,
        type_vocab_size=2,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    )


if TYPE_CHECKING:
    _MLXModuleBase = mlx_nn.Module
else:
    _MLXModuleBase = mlx_nn.Module if mlx_nn is not None else object


class BertEmbeddings(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """BERT embedding stack: token + position + token type + layer norm."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertEmbeddings")
        super().__init__()
        self.word_embeddings = mlx_nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.position_embeddings = mlx_nn.Embedding(spec.max_position_embeddings, spec.hidden_size)
        self.token_type_embeddings = mlx_nn.Embedding(spec.type_vocab_size, spec.hidden_size)
        self.layer_norm = mlx_nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.dropout = mlx_nn.Dropout(spec.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids):
        if mlx_mx is None:
            raise ConfigError("mlx is required for BertEmbeddings")

        batch_size, seq_len = input_ids.shape
        position_ids = mlx_mx.arange(seq_len)[None, :]
        position_ids = mlx_mx.broadcast_to(position_ids, (batch_size, seq_len))

        token_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        type_embeds = self.token_type_embeddings(token_type_ids)

        hidden = token_embeds + pos_embeds + type_embeds
        hidden = self.layer_norm(hidden)
        return self.dropout(hidden)


class BertSelfAttention(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """Multi-head self-attention block used inside a BERT encoder layer."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertSelfAttention")
        super().__init__()
        self.query = mlx_nn.Linear(spec.hidden_size, spec.hidden_size)
        self.key = mlx_nn.Linear(spec.hidden_size, spec.hidden_size)
        self.value = mlx_nn.Linear(spec.hidden_size, spec.hidden_size)
        self.dropout = mlx_nn.Dropout(spec.attention_probs_dropout_prob)
        self.num_heads = spec.num_attention_heads
        self.head_dim = spec.hidden_size // spec.num_attention_heads

    def __call__(self, hidden_states, attention_mask):
        if mlx_mx is None:
            raise ConfigError("mlx is required for BertSelfAttention")

        batch_size, seq_len, hidden_size = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = mlx_mx.softmax(scores, axis=-1)
        probs = self.dropout(probs)

        context = probs @ v
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        return context


class BertSelfOutput(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """Output projection + residual + layer norm for self-attention."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertSelfOutput")
        super().__init__()
        self.dense = mlx_nn.Linear(spec.hidden_size, spec.hidden_size)
        self.layer_norm = mlx_nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.dropout = mlx_nn.Dropout(spec.hidden_dropout_prob)

    def __call__(self, hidden_states, residual):
        hidden = self.dense(hidden_states)
        hidden = self.dropout(hidden)
        return self.layer_norm(hidden + residual)


class BertAttention(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """BERT attention block (self-attention + output projection)."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertAttention")
        super().__init__()
        self.self = BertSelfAttention(spec)
        self.output = BertSelfOutput(spec)

    def __call__(self, hidden_states, attention_mask):
        attended = self.self(hidden_states, attention_mask)
        return self.output(attended, hidden_states)


class BertIntermediate(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """BERT intermediate feed-forward projection."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertIntermediate")
        super().__init__()
        self.dense = mlx_nn.Linear(spec.hidden_size, spec.intermediate_size)

    def __call__(self, hidden_states):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertIntermediate")
        return mlx_nn.gelu(self.dense(hidden_states))


class BertOutput(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """BERT output feed-forward projection + residual + layer norm."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertOutput")
        super().__init__()
        self.dense = mlx_nn.Linear(spec.intermediate_size, spec.hidden_size)
        self.layer_norm = mlx_nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.dropout = mlx_nn.Dropout(spec.hidden_dropout_prob)

    def __call__(self, hidden_states, residual):
        hidden = self.dense(hidden_states)
        hidden = self.dropout(hidden)
        return self.layer_norm(hidden + residual)


class BertLayer(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """Single transformer encoder block in BERT."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertLayer")
        super().__init__()
        self.attention = BertAttention(spec)
        self.intermediate = BertIntermediate(spec)
        self.output = BertOutput(spec)

    def __call__(self, hidden_states, attention_mask):
        attn_out = self.attention(hidden_states, attention_mask)
        interm = self.intermediate(attn_out)
        return self.output(interm, attn_out)


class BertEncoder(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """Stack of BERT encoder layers."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertEncoder")
        super().__init__()
        self.layers = [BertLayer(spec) for _ in range(spec.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertPooler(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """BERT pooler over CLS token."""

    def __init__(self, spec: EncoderBackboneSpec):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertPooler")
        super().__init__()
        self.dense = mlx_nn.Linear(spec.hidden_size, spec.hidden_size)

    def __call__(self, hidden_states):
        if mlx_mx is None:
            raise ConfigError("mlx is required for BertPooler")
        cls = hidden_states[:, 0]
        return mlx_mx.tanh(self.dense(cls))


class BertForSequenceClassification(_MLXModuleBase):  # type: ignore[misc,valid-type]
    """BERT encoder with sequence-classification head."""

    def __init__(self, spec: EncoderBackboneSpec, num_labels: int):
        if mlx_nn is None:
            raise ConfigError("mlx is required for BertForSequenceClassification")
        super().__init__()
        self.embeddings = BertEmbeddings(spec)
        self.encoder = BertEncoder(spec)
        self.pooler = BertPooler(spec)
        self.dropout = mlx_nn.Dropout(spec.hidden_dropout_prob)
        self.classifier = mlx_nn.Linear(spec.hidden_size, num_labels)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        if mlx_mx is None:
            raise ConfigError("mlx is required for BertForSequenceClassification")

        ext_mask = (1.0 - attention_mask.astype(mlx_mx.float32)) * -1e4
        ext_mask = ext_mask[:, None, None, :]

        hidden = self.embeddings(input_ids, token_type_ids)
        hidden = self.encoder(hidden, ext_mask)
        pooled = self.pooler(hidden)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class MLXEncoderSequenceClassifier:
    """A BERT-style encoder classifier implemented with MLX modules."""

    def __init__(
        self,
        backbone: str,
        vocab_size: int,
        num_labels: int,
        max_length: int,
    ):
        if mlx_mx is None or mlx_nn is None:
            raise ConfigError(
                "mlx is required for encoder sequence classification. "
                "Install with: pip install mlx"
            )

        if num_labels < 2:
            raise ConfigError("num_labels must be >= 2 for sequence classification")

        spec = resolve_backbone_spec(backbone)
        if spec.hidden_size % spec.num_attention_heads != 0:
            raise ConfigError(
                "Invalid backbone config: hidden_size must be divisible by num_attention_heads"
            )

        # Keep tokenizer-derived vocab size authoritative for local tokenization.
        spec = EncoderBackboneSpec(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            max_position_embeddings=spec.max_position_embeddings,
            vocab_size=vocab_size,
            type_vocab_size=spec.type_vocab_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            hidden_dropout_prob=spec.hidden_dropout_prob,
            attention_probs_dropout_prob=spec.attention_probs_dropout_prob,
            layer_norm_eps=spec.layer_norm_eps,
        )

        self.spec = spec
        self.backbone = backbone
        self.max_length = max_length
        self.num_labels = num_labels
        self.model = BertForSequenceClassification(spec=spec, num_labels=num_labels)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask, token_type_ids)

    def parameters(self):
        return self.model.parameters()

    def load_weights(self, weights, strict: bool = True):
        return self.model.load_weights(weights, strict=strict)


def flatten_parameter_tree(tree: Any) -> Dict[str, Any]:
    """Flatten an MLX parameter tree into a flat name->tensor mapping."""
    try:
        from mlx.utils import tree_flatten
    except ImportError as e:
        raise ConfigError("mlx is required to flatten parameter trees") from e

    flattened = tree_flatten(tree)
    result: Dict[str, Any] = {}

    for key, value in flattened:
        if isinstance(key, tuple):
            name = ".".join(str(k) for k in key)
        else:
            name = str(key)
        result[name] = value

    return result


def map_mlx_param_to_hf(mlx_name: str) -> Optional[str]:
    """Map an MLX BERT parameter path to a Hugging Face BERT state key."""
    if mlx_name.startswith("embeddings.word_embeddings."):
        suffix = mlx_name.replace("embeddings.word_embeddings.", "", 1)
        return f"bert.embeddings.word_embeddings.{suffix}"

    if mlx_name.startswith("embeddings.position_embeddings."):
        suffix = mlx_name.replace("embeddings.position_embeddings.", "", 1)
        return f"bert.embeddings.position_embeddings.{suffix}"

    if mlx_name.startswith("embeddings.token_type_embeddings."):
        suffix = mlx_name.replace("embeddings.token_type_embeddings.", "", 1)
        return f"bert.embeddings.token_type_embeddings.{suffix}"

    if mlx_name.startswith("embeddings.layer_norm."):
        suffix = mlx_name.replace("embeddings.layer_norm.", "", 1)
        return f"bert.embeddings.LayerNorm.{suffix}"

    layer_match = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", mlx_name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)
        rest = rest.replace("attention.output.layer_norm.", "attention.output.LayerNorm.")
        rest = rest.replace("output.layer_norm.", "output.LayerNorm.")
        return f"bert.encoder.layer.{layer_idx}.{rest}"

    if mlx_name.startswith("pooler.dense."):
        suffix = mlx_name.replace("pooler.dense.", "", 1)
        return f"bert.pooler.dense.{suffix}"

    if mlx_name.startswith("classifier."):
        return mlx_name

    return None


def load_pretrained_bert_weights(
    model: MLXEncoderSequenceClassifier,
    backbone: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load pretrained HF BERT weights into MLX model modules.

    Returns a summary dict with counts and skipped keys.
    """
    hf_weights = _load_hf_checkpoint_arrays(backbone)
    mlx_params = flatten_parameter_tree(model.parameters())

    assignments: List[Tuple[str, np.ndarray]] = []
    missing: List[str] = []
    shape_mismatch: List[str] = []

    for mlx_name, mlx_value in mlx_params.items():
        hf_name = map_mlx_param_to_hf(mlx_name)
        if hf_name is None:
            continue

        hf_value = hf_weights.get(hf_name)
        # Older BERT checkpoints use gamma/beta instead of weight/bias
        # for LayerNorm parameters.
        if hf_value is None and hf_name.endswith(".weight"):
            hf_value = hf_weights.get(hf_name[:-6] + "gamma")
        if hf_value is None and hf_name.endswith(".bias"):
            hf_value = hf_weights.get(hf_name[:-4] + "beta")
        if hf_value is None:
            missing.append(mlx_name)
            continue

        if tuple(hf_value.shape) != tuple(mlx_value.shape):
            shape_mismatch.append(
                f"{mlx_name}: expected {tuple(mlx_value.shape)} got {tuple(hf_value.shape)}"
            )
            continue

        assignments.append((mlx_name, hf_value))

    if not assignments:
        raise ConfigError(
            f"No compatible pretrained weights found for backbone '{backbone}'. "
            "Ensure it is a BERT checkpoint with safetensors weights."
        )

    mx_assignments = [(name, mlx_mx.array(val)) for name, val in assignments]
    model.load_weights(mx_assignments, strict=strict)

    return {
        "loaded": len(assignments),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


def _load_hf_checkpoint_arrays(backbone: str) -> Dict[str, np.ndarray]:
    """Load HF checkpoint arrays from safetensors files (single or sharded)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ConfigError(
            "huggingface_hub is required to download pretrained encoder weights"
        ) from e

    try:
        from safetensors.numpy import load_file
    except ImportError as e:
        raise ConfigError(
            "safetensors is required to load pretrained encoder weights"
        ) from e

    files = set(list_repo_files(backbone))

    if "model.safetensors" in files:
        ckpt = hf_hub_download(backbone, "model.safetensors")
        return dict(load_file(ckpt))

    if "model.safetensors.index.json" in files:
        index_path = hf_hub_download(backbone, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ConfigError(f"Invalid safetensors index for backbone '{backbone}'")

        merged: Dict[str, np.ndarray] = {}
        shard_names = sorted(set(weight_map.values()))
        for shard_name in shard_names:
            shard_path = hf_hub_download(backbone, shard_name)
            merged.update(dict(load_file(shard_path)))

        return merged

    safetensor_files = sorted([f for f in files if f.endswith(".safetensors")])
    if safetensor_files:
        ckpt = hf_hub_download(backbone, safetensor_files[0])
        return dict(load_file(ckpt))

    raise ConfigError(
        f"No safetensors checkpoint found for backbone '{backbone}'. "
        "Only safetensors checkpoints are supported in V1."
    )


def create_encoder_model(
    backbone: str,
    vocab_size: int,
    num_labels: int,
    max_length: int,
) -> Tuple[MLXEncoderSequenceClassifier, Dict[str, Any]]:
    """Create sequence-classification model and return model metadata."""
    model = MLXEncoderSequenceClassifier(
        backbone=backbone,
        vocab_size=vocab_size,
        num_labels=num_labels,
        max_length=max_length,
    )

    metadata = {
        "backbone": backbone,
        "vocab_size": vocab_size,
        "num_labels": num_labels,
        "max_length": max_length,
        "hidden_size": model.spec.hidden_size,
        "intermediate_size": model.spec.intermediate_size,
        "num_hidden_layers": model.spec.num_hidden_layers,
        "num_attention_heads": model.spec.num_attention_heads,
    }

    return model, metadata
