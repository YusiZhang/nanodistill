# Encoder Fine-Tuning (Sequence Classification)

NanoDistill now supports an encoder-only path for sequence classification on Apple Silicon using MLX.

## What this mode does

- Uses `task_type="sequence_classification"` in `distill()`
- Trains an encoder classifier with MLX (BERT-base default backbone name)
- Supports LoRA injection into configurable encoder modules
- Writes encoder artifacts under `outputs/<run_name>/`
- Loads pretrained HF BERT safetensors by default (`encoder_load_pretrained=True`)

## API

```python
from nanodistill import distill

result = distill(
    name="sentiment-encoder",
    task_type="sequence_classification",
    training_data=[
        {"input": "Great product", "label": "positive"},
        {"input": "Very bad", "label": "negative"},
    ],
    text_field="input",
    label_field="label",
    encoder_backbone="bert-base-uncased",
    encoder_lora_rank=8,
    encoder_lora_targets=["query", "value"],
)
```

## Data format

For direct labeled training (`training_data`), each row must include:

- text field (`text_field`, default `input`)
- label field (`label_field`, default `label`)

Supported file formats:

- `.json` (array of objects)
- `.jsonl`
- `.csv`

## Artifacts

- `encoder_model.safetensors` (or `encoder_model.npz` fallback)
- `label_map.json`
- `encoder_config.json`
- `metrics.json`
- `summary.json`

## Inference

```python
from nanodistill import predict_sequence_class

predictions = predict_sequence_class(
    "./outputs/sentiment-encoder",
    ["I love this", "This is awful"],
)
for p in predictions:
    print(p.label, p.confidence)
```

## Notes

- `sequence_classification` does not use the decoder `mlx-lm` LoRA CLI path.
- If you do not pass `training_data`, NanoDistill can still bootstrap labels from seed `input/output` pairs and optional synthetic amplification.
- The encoder trainer now loads pretrained HF BERT safetensors weights before fine-tuning.
- V1 pretrained loading supports BERT checkpoints with safetensors (`model.safetensors` or sharded safetensors index).
