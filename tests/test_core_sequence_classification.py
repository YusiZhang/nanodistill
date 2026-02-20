"""Tests for sequence-classification routing in core.distill."""

from pathlib import Path

from nanodistill.core import distill


class _FakeEncoderTrainer:
    def __init__(self, config):
        self.config = config
        self.metrics = {
            "train_accuracy": 1.0,
            "train_macro_f1": 1.0,
            "val_accuracy": 1.0,
            "val_macro_f1": 1.0,
        }

    def train(self, train_examples, val_examples, label_to_id, id_to_label):
        output = Path(self.config.output_dir) / self.config.name
        output.mkdir(parents=True, exist_ok=True)
        return str(output)


def test_distill_sequence_classification_training_data(monkeypatch, tmp_path):
    """distill() should route encoder task_type to MLXEncoderTrainer."""

    monkeypatch.setattr("nanodistill.core.MLXEncoderTrainer", _FakeEncoderTrainer)

    result = distill(
        name="encoder-route-test",
        task_type="sequence_classification",
        training_data=[
            {"input": "great", "label": "positive"},
            {"input": "bad", "label": "negative"},
            {"input": "okay", "label": "neutral"},
        ],
        text_field="input",
        label_field="label",
        output_dir=str(tmp_path),
        val_split=0.33,
    )

    assert result.config.task_type == "sequence_classification"
    assert result.config.encoder_backbone == "bert-base-uncased"
    assert result.metrics["training_examples"] == 3
    assert result.model_path == tmp_path / "encoder-route-test"
