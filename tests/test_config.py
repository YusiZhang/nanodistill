"""Tests for config module."""

import pytest

from nanodistill.config import DistillationConfig


def test_config_valid(sample_seed_data, sample_instruction):
    """Test creating valid DistillationConfig."""
    config = DistillationConfig(
        name="test-run",
        seed=sample_seed_data,
        instruction=sample_instruction,
    )

    assert config.name == "test-run"
    assert len(config.seed) == len(sample_seed_data)
    assert config.augment_factor == 50  # Default
    assert config.teacher == "claude-sonnet-4-5"  # Default


def test_config_missing_seed():
    """Test that config fails with missing seed."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[],  # Empty seed
            instruction="Test instruction",
        )


def test_config_missing_input_field():
    """Test that config fails if seed is missing 'input' field."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[
                {"output": "answer"},  # Missing 'input'
            ],
            instruction="Test instruction",
        )


def test_config_missing_output_field():
    """Test that config fails if seed is missing 'output' field."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[
                {"input": "question"},  # Missing 'output'
            ],
            instruction="Test instruction",
        )


def test_config_empty_input():
    """Test that config fails if seed has empty input."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[
                {"input": "", "output": "answer"},  # Empty input
            ],
            instruction="Test instruction",
        )


def test_config_sequence_classification_defaults(sample_seed_data, sample_instruction):
    """Test encoder sequence-classification config values."""
    config = DistillationConfig(
        name="encoder-test",
        seed=sample_seed_data,
        instruction=sample_instruction,
        task_type="sequence_classification",
        encoder_backbone="bert-base-uncased",
    )

    assert config.task_type == "sequence_classification"
    assert config.encoder_backbone == "bert-base-uncased"
    assert config.text_field == "input"
    assert config.label_field == "label"
    assert config.encoder_lora_targets == ["query", "value"]
    assert config.encoder_load_pretrained is True


def test_config_invalid_encoder_target(sample_seed_data, sample_instruction):
    """Test that empty LoRA targets fail validation."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="encoder-test",
            seed=sample_seed_data,
            instruction=sample_instruction,
            task_type="sequence_classification",
            encoder_lora_targets=["query", ""],
        )
