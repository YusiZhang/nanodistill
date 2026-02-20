"""Example: encoder-only sequence-classification fine-tuning with MLX."""

from nanodistill import distill, predict_sequence_class


if __name__ == "__main__":
    training_data = [
        {"input": "I love this product.", "label": "positive"},
        {"input": "The experience was terrible.", "label": "negative"},
        {"input": "The app works as expected.", "label": "neutral"},
        {"input": "Amazing support and fast response.", "label": "positive"},
        {"input": "It keeps crashing every day.", "label": "negative"},
    ]

    result = distill(
        name="encoder-sentiment-demo",
        task_type="sequence_classification",
        training_data=training_data,
        text_field="input",
        label_field="label",
        encoder_backbone="bert-base-uncased",
        num_train_epochs=1,
        batch_size=2,
        val_split=0.2,
        output_dir="./outputs",
    )

    print(f"Model saved to: {result.model_path}")

    preds = predict_sequence_class(
        str(result.model_path),
        [
            "This is fantastic.",
            "Worst update ever.",
        ],
    )

    for text, pred in zip(["This is fantastic.", "Worst update ever."], preds):
        print(f"Text: {text}")
        print(f"Predicted label: {pred.label} (confidence={pred.confidence:.3f})")
