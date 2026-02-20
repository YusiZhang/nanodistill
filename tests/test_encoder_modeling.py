"""Tests for encoder modeling + HF weight mapping utilities."""

import numpy as np

from nanodistill.encoder import modeling


def _tiny_spec():
    return modeling.EncoderBackboneSpec(
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=32,
        vocab_size=64,
        type_vocab_size=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    )


def test_map_mlx_param_to_hf_basic_paths():
    assert (
        modeling.map_mlx_param_to_hf("embeddings.word_embeddings.weight")
        == "bert.embeddings.word_embeddings.weight"
    )
    assert (
        modeling.map_mlx_param_to_hf("embeddings.layer_norm.bias")
        == "bert.embeddings.LayerNorm.bias"
    )
    assert (
        modeling.map_mlx_param_to_hf("encoder.layers.0.attention.self.query.weight")
        == "bert.encoder.layer.0.attention.self.query.weight"
    )
    assert (
        modeling.map_mlx_param_to_hf("encoder.layers.1.output.layer_norm.weight")
        == "bert.encoder.layer.1.output.LayerNorm.weight"
    )
    assert modeling.map_mlx_param_to_hf("classifier.bias") == "classifier.bias"


def test_load_pretrained_bert_weights_from_mocked_hf(monkeypatch):
    monkeypatch.setattr(modeling, "resolve_backbone_spec", lambda _: _tiny_spec())

    model, _ = modeling.create_encoder_model(
        backbone="tiny-bert-test",
        vocab_size=64,
        num_labels=3,
        max_length=16,
    )

    mlx_params = modeling.flatten_parameter_tree(model.parameters())

    hf_weights = {}
    for mlx_name, value in mlx_params.items():
        hf_name = modeling.map_mlx_param_to_hf(mlx_name)
        if hf_name is not None:
            hf_weights[hf_name] = np.asarray(value)

    monkeypatch.setattr(modeling, "_load_hf_checkpoint_arrays", lambda _: hf_weights)

    summary = modeling.load_pretrained_bert_weights(
        model,
        backbone="tiny-bert-test",
        strict=False,
    )

    assert summary["loaded"] > 0
    assert isinstance(summary["missing"], list)
    assert isinstance(summary["shape_mismatch"], list)
