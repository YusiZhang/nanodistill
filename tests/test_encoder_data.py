"""Tests for encoder sequence-classification data helpers."""

from nanodistill.encoder.data import build_label_maps, split_examples, to_labeled_examples


def test_build_label_maps_string_labels_deterministic():
    rows = [
        {"input": "b", "label": "negative"},
        {"input": "a", "label": "positive"},
        {"input": "c", "label": "negative"},
    ]

    label_to_id, id_to_label = build_label_maps(rows, label_field="label")

    assert label_to_id == {"negative": 0, "positive": 1}
    assert id_to_label == {0: "negative", 1: "positive"}


def test_to_labeled_examples_encodes_labels():
    rows = [
        {"input": "first", "label": "neutral"},
        {"input": "second", "label": "positive"},
    ]

    label_to_id = {"neutral": 0, "positive": 1}
    encoded = to_labeled_examples(rows, "input", "label", label_to_id)

    assert len(encoded) == 2
    assert encoded[0].text == "first"
    assert encoded[0].label == 0
    assert encoded[1].label == 1


def test_split_examples_has_train_and_val():
    rows = [
        {"input": "x1", "label": "a"},
        {"input": "x2", "label": "b"},
        {"input": "x3", "label": "a"},
        {"input": "x4", "label": "b"},
    ]
    label_to_id, _ = build_label_maps(rows, label_field="label")
    examples = to_labeled_examples(rows, "input", "label", label_to_id)

    train, val = split_examples(examples, val_split=0.25, seed=123)

    assert len(train) == 3
    assert len(val) == 1
