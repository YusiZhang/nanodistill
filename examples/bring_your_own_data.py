"""Example: Bring Your Own Training Data.

Skip the CoT synthesis and amplification stages entirely by supplying
pre-formatted training data directly. No teacher API key is required.

Two approaches are shown below:
  1. Inline list of dicts
  2. JSONL file on disk
"""

from nanodistill import distill

# ---------------------------------------------------------------------------
# Option 1: Pass training data as a Python list
# ---------------------------------------------------------------------------
# Each record must have 'input', 'thinking', and 'output' fields.
# 'confidence' is optional (defaults to 0.9).

training_data = [
    # --- Greetings ---
    {
        "input": "Translate 'hello' to French.",
        "thinking": (
            "The English word 'hello' is a common greeting. "
            "In French the standard translation is 'bonjour'."
        ),
        "output": "bonjour",
    },
    {
        "input": "Translate 'goodbye' to French.",
        "thinking": "'Goodbye' is a parting greeting. In French this is 'au revoir'.",
        "output": "au revoir",
        "confidence": 0.95,
    },
    {
        "input": "Translate 'good morning' to French.",
        "thinking": "'Good morning' is a morning greeting. In French it is 'bonjour'.",
        "output": "bonjour",
    },
    {
        "input": "Translate 'good evening' to French.",
        "thinking": "'Good evening' is an evening greeting. In French it is 'bonsoir'.",
        "output": "bonsoir",
    },
    {
        "input": "Translate 'good night' to French.",
        "thinking": "'Good night' is said before sleeping. In French it is 'bonne nuit'.",
        "output": "bonne nuit",
    },
    # --- Polite expressions ---
    {
        "input": "Translate 'thank you' to French.",
        "thinking": (
            "'Thank you' is a polite expression of gratitude. " "The French equivalent is 'merci'."
        ),
        "output": "merci",
    },
    {
        "input": "Translate 'please' to French.",
        "thinking": (
            "'Please' is used to make polite requests. " "The French phrase is 's'il vous plaît'."
        ),
        "output": "s'il vous plaît",
    },
    {
        "input": "Translate 'excuse me' to French.",
        "thinking": (
            "'Excuse me' is used to get attention or apologize. " "In French it is 'excusez-moi'."
        ),
        "output": "excusez-moi",
    },
    {
        "input": "Translate 'sorry' to French.",
        "thinking": "'Sorry' expresses an apology. In French this is 'désolé'.",
        "output": "désolé",
    },
    {
        "input": "Translate 'you're welcome' to French.",
        "thinking": "'You're welcome' is a polite response to thanks. In French it is 'de rien'.",
        "output": "de rien",
    },
    # --- Affirmatives & common words ---
    {
        "input": "Translate 'yes' to French.",
        "thinking": "'Yes' is an affirmative response. In French the word is 'oui'.",
        "output": "oui",
    },
    {
        "input": "Translate 'no' to French.",
        "thinking": "'No' is a negative response. In French the word is 'non'.",
        "output": "non",
    },
    {
        "input": "Translate 'maybe' to French.",
        "thinking": "'Maybe' indicates uncertainty. In French the word is 'peut-être'.",
        "output": "peut-être",
    },
    # --- Numbers ---
    {
        "input": "Translate 'one' to French.",
        "thinking": "'One' is the first counting number. In French it is 'un'.",
        "output": "un",
    },
    {
        "input": "Translate 'two' to French.",
        "thinking": "'Two' is the second counting number. In French it is 'deux'.",
        "output": "deux",
    },
    {
        "input": "Translate 'three' to French.",
        "thinking": "'Three' is the third counting number. In French it is 'trois'.",
        "output": "trois",
    },
    # --- Colors ---
    {
        "input": "Translate 'red' to French.",
        "thinking": "'Red' is a primary color. In French the word is 'rouge'.",
        "output": "rouge",
    },
    {
        "input": "Translate 'blue' to French.",
        "thinking": "'Blue' is a primary color. In French the word is 'bleu'.",
        "output": "bleu",
    },
    {
        "input": "Translate 'green' to French.",
        "thinking": "'Green' is the color of grass. In French the word is 'vert'.",
        "output": "vert",
    },
    {
        "input": "Translate 'white' to French.",
        "thinking": "'White' is the lightest color. In French the word is 'blanc'.",
        "output": "blanc",
    },
    {
        "input": "Translate 'black' to French.",
        "thinking": "'Black' is the darkest color. In French the word is 'noir'.",
        "output": "noir",
    },
    # --- Common phrases ---
    {
        "input": "Translate 'How are you?' to French.",
        "thinking": (
            "'How are you?' asks about someone's well-being. "
            "In French it is 'comment allez-vous ?'."
        ),
        "output": "comment allez-vous ?",
    },
    {
        "input": "Translate 'My name is' to French.",
        "thinking": "'My name is' introduces oneself. In French the phrase is 'je m'appelle'.",
        "output": "je m'appelle",
    },
    # --- Food ---
    {
        "input": "Translate 'bread' to French.",
        "thinking": "'Bread' is a staple food. In French the word is 'pain'.",
        "output": "pain",
    },
    {
        "input": "Translate 'water' to French.",
        "thinking": "'Water' is a basic drink. In French the word is 'eau'.",
        "output": "eau",
    },
]

result = distill(
    name="translator-from-list",
    training_data=training_data,
    student="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    output_dir="./outputs",
    batch_size=1,
)

print(f"Model saved to: {result.model_path}")
print(f"Training examples: {result.metrics['training_examples']}")

# ---------------------------------------------------------------------------
# Option 2: Pass a path to a JSONL file
# ---------------------------------------------------------------------------
# Useful when you already have training data exported from another tool or
# generated by a separate script. One JSON object per line with the same
# required fields: input, thinking, output.

# Create a sample JSONL file for demonstration
# import json, tempfile
# from pathlib import Path
# sample_jsonl = Path(tempfile.mkdtemp()) / "my_traces.jsonl"
# with open(sample_jsonl, "w") as f:
#     for record in training_data:
#         f.write(json.dumps(record) + "\n")
#
# print(f"\nSample JSONL written to: {sample_jsonl}")
#
# result = distill(
#     name="translator-from-file",
#     training_data=str(sample_jsonl),  # also accepts a Path object
#     student="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
#     output_dir="./outputs",
# )
#
# print(f"Model saved to: {result.model_path}")
# print(f"Training examples: {result.metrics['training_examples']}")
