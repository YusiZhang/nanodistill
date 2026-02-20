"""MLX trainer for encoder sequence classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..config import DistillationConfig
from ..utils.errors import TrainingError
from .data import LabeledExample
from .lora import apply_encoder_lora
from .modeling import (
    create_encoder_model,
    flatten_parameter_tree,
    load_pretrained_bert_weights,
)
from .tokenization import load_hf_tokenizer, tokenize_texts


@dataclass
class EncoderTrainingResult:
    """Result of an encoder sequence-classification training run."""

    model_path: str
    metrics: Dict[str, float]


class MLXEncoderTrainer:
    """Train a sequence-classification encoder with MLX."""

    def __init__(self, config: DistillationConfig):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
        except ImportError as e:
            raise TrainingError(
                "MLX is required for encoder training. Install with: pip install mlx"
            ) from e

        self.mx = mx
        self.nn = nn
        self.optim = optim
        self.config = config

        self.model = None
        self.model_meta: Dict[str, object] = {}
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.metrics: Dict[str, float] = {}

    def train(
        self,
        train_examples: Sequence[LabeledExample],
        val_examples: Sequence[LabeledExample],
        label_to_id: Dict[str, int],
        id_to_label: Dict[int, str],
    ) -> str:
        """Train model and save artifacts.

        Returns:
            Model directory path.
        """
        if not train_examples:
            raise TrainingError("Cannot train sequence-classification model on empty dataset")

        self.label_to_id = dict(label_to_id)
        self.id_to_label = dict(id_to_label)

        tokenizer = load_hf_tokenizer(self.config.encoder_backbone)
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))

        self.model, self.model_meta = create_encoder_model(
            backbone=self.config.encoder_backbone,
            vocab_size=vocab_size,
            num_labels=len(label_to_id),
            max_length=self.config.encoder_max_length,
        )

        if self.config.encoder_load_pretrained:
            try:
                preload_summary = load_pretrained_bert_weights(
                    self.model,
                    backbone=self.config.encoder_backbone,
                    strict=False,
                )
                self.model_meta["pretrained_weights"] = preload_summary
            except Exception as e:
                raise TrainingError(
                    f"Failed to load pretrained weights for '{self.config.encoder_backbone}': {e}"
                ) from e

        lora_modules: List[str] = []
        if self.config.encoder_use_lora:
            try:
                lora_modules = apply_encoder_lora(
                    self.model.model,
                    rank=self.config.encoder_lora_rank,
                    targets=self.config.encoder_lora_targets,
                )
            except Exception as e:
                raise TrainingError(f"Failed to apply encoder LoRA: {e}") from e

            if lora_modules:
                # Freeze the entire model, then selectively unfreeze only
                # LoRA adapter weights (lora_a, lora_b) and the classifier.
                self.model.model.freeze()
                for module_path in lora_modules:
                    parts = module_path.split(".")
                    obj = self.model.model
                    for p in parts:
                        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
                    obj.unfreeze(keys=["lora_a", "lora_b"])
                self.model.model.classifier.unfreeze()

        optimizer = self._build_optimizer(self.config.learning_rate)
        loss_and_grad_fn = self.nn.value_and_grad(self.model.model, self._loss_fn)

        self.model.model.train()
        epoch_losses: List[float] = []
        for _ in range(self.config.num_train_epochs):
            batch_losses: List[float] = []
            for batch in self._batch_iter(train_examples, self.config.batch_size):
                mx_batch = self._examples_to_batch(tokenizer, batch)
                loss, grads = loss_and_grad_fn(self.model.model, mx_batch)
                optimizer.update(self.model.model, grads)
                self.mx.eval(self.model.model.parameters(), optimizer.state, loss)
                batch_losses.append(float(loss.item()))

            if batch_losses:
                epoch_losses.append(float(sum(batch_losses) / len(batch_losses)))

        train_eval = self._evaluate(tokenizer, train_examples)
        val_eval = self._evaluate(tokenizer, val_examples) if val_examples else {}

        self.metrics = {
            "train_accuracy": float(train_eval.get("accuracy", 0.0)),
            "train_macro_f1": float(train_eval.get("macro_f1", 0.0)),
            "val_accuracy": float(val_eval.get("accuracy", 0.0)) if val_eval else 0.0,
            "val_macro_f1": float(val_eval.get("macro_f1", 0.0)) if val_eval else 0.0,
            "train_examples": float(len(train_examples)),
            "val_examples": float(len(val_examples)),
            "num_labels": float(len(label_to_id)),
            "loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        }

        model_dir = Path(self.config.output_dir) / self.config.name
        model_dir.mkdir(parents=True, exist_ok=True)

        self._save_artifacts(
            output_dir=model_dir,
            lora_modules=lora_modules,
            val_confusion_matrix=val_eval.get("confusion_matrix", []),
        )

        return str(model_dir)

    def _build_optimizer(self, learning_rate: float):
        """Build optimizer with MLX-version compatibility."""
        if hasattr(self.optim, "AdamW"):
            return self.optim.AdamW(learning_rate=learning_rate)
        return self.optim.Adam(learning_rate=learning_rate)

    def _loss_fn(self, model, batch):
        """Cross-entropy objective for sequence classification."""
        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        )
        labels = batch["labels"]
        loss = self.nn.losses.cross_entropy(logits, labels, reduction="mean")
        return loss

    def _examples_to_batch(self, tokenizer, batch_examples: Sequence[LabeledExample]):
        """Tokenize and convert a list of LabeledExample objects into MLX tensors."""
        texts = [ex.text for ex in batch_examples]
        labels = np.asarray([ex.label for ex in batch_examples], dtype=np.int32)

        tokenized = tokenize_texts(
            tokenizer=tokenizer,
            texts=texts,
            max_length=self.config.encoder_max_length,
        )

        return {
            "input_ids": self.mx.array(tokenized["input_ids"]),
            "attention_mask": self.mx.array(tokenized["attention_mask"]),
            "token_type_ids": self.mx.array(tokenized["token_type_ids"]),
            "labels": self.mx.array(labels),
        }

    def _evaluate(self, tokenizer, examples: Sequence[LabeledExample]) -> Dict[str, object]:
        """Evaluate current model and compute metrics."""
        if not examples:
            return {}

        self.model.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        for batch in self._batch_iter(examples, self.config.batch_size):
            mx_batch = self._examples_to_batch(tokenizer, batch)
            logits = self.model(
                mx_batch["input_ids"],
                mx_batch["attention_mask"],
                mx_batch["token_type_ids"],
            )
            probs = self._softmax(np.asarray(logits))
            preds = probs.argmax(axis=1).tolist()
            labels = np.asarray(mx_batch["labels"]).astype(int).tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

        accuracy = _accuracy(all_labels, all_preds)
        macro_f1 = _macro_f1(all_labels, all_preds, num_classes=len(self.id_to_label))
        confusion = _confusion_matrix(all_labels, all_preds, num_classes=len(self.id_to_label))

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "confusion_matrix": confusion,
        }

    def _save_artifacts(
        self,
        output_dir: Path,
        lora_modules: List[str],
        val_confusion_matrix: List[List[int]],
    ) -> None:
        """Save model, config, label map, and metrics."""
        model_path = output_dir / "encoder_model.safetensors"
        label_map_path = output_dir / "label_map.json"
        config_path = output_dir / "encoder_config.json"
        metrics_path = output_dir / "metrics.json"

        flat_params = flatten_parameter_tree(self.model.parameters())
        np_weights = {
            key: np.asarray(value)
            for key, value in flat_params.items()
        }

        weights_format = "safetensors"
        try:
            from safetensors.numpy import save_file

            save_file(np_weights, str(model_path))
        except Exception:
            npz_path = output_dir / "encoder_model.npz"
            np.savez(npz_path, **np_weights)
            weights_format = "npz"

        with open(label_map_path, "w") as f:
            json.dump(
                {
                    "label_to_id": self.label_to_id,
                    "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
                },
                f,
                indent=2,
            )

        with open(config_path, "w") as f:
            json.dump(
                {
                    "task_type": "sequence_classification",
                    "encoder_backbone": self.config.encoder_backbone,
                    "encoder_max_length": self.config.encoder_max_length,
                    "encoder_use_lora": self.config.encoder_use_lora,
                    "encoder_lora_rank": self.config.encoder_lora_rank,
                    "encoder_lora_targets": self.config.encoder_lora_targets,
                    "weights_format": weights_format,
                    "lora_modules": lora_modules,
                    "model": self.model_meta,
                },
                f,
                indent=2,
            )

        serialized_metrics = dict(self.metrics)
        serialized_metrics["val_confusion_matrix"] = val_confusion_matrix
        with open(metrics_path, "w") as f:
            json.dump(serialized_metrics, f, indent=2)

    @staticmethod
    def _batch_iter(examples: Sequence[LabeledExample], batch_size: int):
        """Yield mini-batches from a list of examples."""
        for i in range(0, len(examples), batch_size):
            yield examples[i : i + batch_size]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        if logits.ndim != 2:
            raise TrainingError(f"Expected 2D logits, got shape {logits.shape}")
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _accuracy(labels: Sequence[int], preds: Sequence[int]) -> float:
    """Compute accuracy metric."""
    if not labels:
        return 0.0
    correct = sum(1 for y, p in zip(labels, preds) if y == p)
    return correct / len(labels)


def _macro_f1(labels: Sequence[int], preds: Sequence[int], num_classes: int) -> float:
    """Compute unweighted macro F1 across classes."""
    if not labels or num_classes <= 0:
        return 0.0

    f1_scores: List[float] = []
    for cls in range(num_classes):
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))

    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


def _confusion_matrix(
    labels: Sequence[int],
    preds: Sequence[int],
    num_classes: int,
) -> List[List[int]]:
    """Build confusion matrix with shape [num_classes, num_classes]."""
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels, preds):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            matrix[y][p] += 1
    return matrix
