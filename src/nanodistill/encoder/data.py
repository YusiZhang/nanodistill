"""Data utilities for encoder sequence-classification training."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..teacher.schemas import ThinkingTrace
from ..utils.errors import ConfigError


@dataclass
class LabeledExample:
    """A single sequence-classification example."""

    text: str
    label: int


def traces_to_labeled_examples(traces: List[ThinkingTrace]) -> List[Dict[str, Union[str, int]]]:
    """Convert ThinkingTrace rows to sequence-classification records.

    The teacher pipeline stores class labels in `output`. We map that directly
    to a classification label and keep `input` as text.
    """
    rows: List[Dict[str, Union[str, int]]] = []
    for trace in traces:
        rows.append({"input": trace.input, "label": trace.output})
    return rows


def load_sequence_classification_data(
    dataset: Union[List[Dict[str, Union[str, int]]], str, Path],
    text_field: str,
    label_field: str,
) -> List[Dict[str, Union[str, int]]]:
    """Load sequence-classification records from list/JSON/JSONL/CSV.

    Required fields are configurable via `text_field` and `label_field`.
    """
    if isinstance(dataset, list):
        return _validate_rows(dataset, text_field=text_field, label_field=label_field)

    path = Path(dataset)
    if not path.exists():
        raise ConfigError(f"Classification dataset file not found: {path}")

    rows: List[Dict[str, Union[str, int]]] = []
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ConfigError("JSON dataset must contain a list of records")
        rows = data
    elif path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    elif path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
    else:
        raise ConfigError(
            f"Unsupported classification dataset format: {path.suffix}. "
            "Supported: .json, .jsonl, .csv"
        )

    return _validate_rows(rows, text_field=text_field, label_field=label_field)


def build_label_maps(
    rows: List[Dict[str, Union[str, int]]],
    label_field: str,
    num_labels: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build deterministic label maps.

    - Integer labels preserve numeric ordering.
    - String labels use lexicographic ordering for determinism.
    """
    raw_labels = [row[label_field] for row in rows]
    if not raw_labels:
        raise ConfigError("Classification dataset is empty")

    if all(_is_int_like(v) for v in raw_labels):
        unique_ints = sorted({int(v) for v in raw_labels})
        label_to_id = {str(i): i for i in unique_ints}
        id_to_label = {i: str(i) for i in unique_ints}
    else:
        unique_labels = sorted({str(v) for v in raw_labels})
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

    if num_labels is not None and len(label_to_id) != num_labels:
        raise ConfigError(
            f"num_labels={num_labels} but found {len(label_to_id)} unique labels in dataset"
        )

    return label_to_id, id_to_label


def to_labeled_examples(
    rows: List[Dict[str, Union[str, int]]],
    text_field: str,
    label_field: str,
    label_to_id: Dict[str, int],
) -> List[LabeledExample]:
    """Convert raw rows to typed `LabeledExample` records."""
    examples: List[LabeledExample] = []
    for row in rows:
        text = str(row[text_field]).strip()
        raw_label = row[label_field]
        int_key = str(int(raw_label)) if _is_int_like(raw_label) else None
        if int_key is not None and int_key in label_to_id:
            key = int_key
        else:
            key = str(raw_label)
        if key not in label_to_id:
            raise ConfigError(f"Unknown label '{raw_label}' encountered while encoding labels")
        examples.append(LabeledExample(text=text, label=label_to_id[key]))
    return examples


def split_examples(
    examples: List[LabeledExample],
    val_split: float,
    seed: int = 42,
) -> Tuple[List[LabeledExample], List[LabeledExample]]:
    """Deterministically split examples into train/validation sets."""
    if not examples:
        raise ConfigError("Cannot split empty sequence-classification dataset")

    if val_split <= 0.0:
        return examples, []

    shuffled = list(examples)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)

    val_count = int(len(shuffled) * val_split)
    if val_count <= 0 and len(shuffled) >= 2:
        val_count = 1
    if val_count >= len(shuffled):
        val_count = max(1, len(shuffled) - 1)

    val = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, val


def _validate_rows(
    rows: List[Dict[str, Union[str, int]]],
    text_field: str,
    label_field: str,
) -> List[Dict[str, Union[str, int]]]:
    """Validate rows and required fields."""
    if not rows:
        raise ConfigError("Classification dataset must contain at least 1 record")

    validated: List[Dict[str, Union[str, int]]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConfigError(f"classification_data[{i}] must be a dict, got {type(row)}")
        if text_field not in row:
            raise ConfigError(
                f"classification_data[{i}] missing required text field '{text_field}'"
            )
        if label_field not in row:
            raise ConfigError(
                f"classification_data[{i}] missing required label field '{label_field}'"
            )

        text_value = str(row[text_field]).strip()
        if not text_value:
            raise ConfigError(f"classification_data[{i}] has empty text field '{text_field}'")

        validated.append(row)

    return validated


def _is_int_like(value: Union[str, int]) -> bool:
    """Return True if value can be safely interpreted as an int."""
    try:
        int(str(value))
        return True
    except (TypeError, ValueError):
        return False
