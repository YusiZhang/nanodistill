# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Install package in development mode with dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from nanodistill import distill; print('✓ Installation successful')"
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_config.py -v

# Run specific test function
pytest tests/test_config.py::test_config_name -v

# Run tests with coverage report
pytest --cov=src/nanodistill --cov-report=term-missing
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

### Running the Distillation Pipeline
```bash
# Minimal example (requires ANTHROPIC_API_KEY set)
python -c "
from nanodistill import distill
result = distill(
    name='test-run',
    seed=[
        {'input': 'What is 2+2?', 'output': '4'},
        {'input': 'What is 3+5?', 'output': '8'},
    ],
    instruction='You are a helpful math tutor.',
)
print(f'✓ Model saved to: {result.model_path}')
"

# Load seed data from JSON file
python -c "
from nanodistill import distill
result = distill(
    name='my-model',
    seed='seeds.json',  # List of {input, output} dicts
    instruction='Your task description here',
)
"
```

## Project Architecture

NanoDistill transforms seed examples through a 4-stage pipeline, with each stage handled by a specialized module:

### Entry Point: `src/nanodistill/core.py`
- **`distill()` function** - Main entry point that orchestrates the entire pipeline
- **`DistillationResult`** - Dataclass containing model_path, metrics, and config
- Handles validation, component initialization, and progress reporting
- Can skip API calls if amplified data already exists (`traces_amplified.jsonl`)

### Stage 1: Policy Extraction & CoT Synthesis
**Module:** `src/nanodistill/teacher/client.py`
- **`TeacherClient`** - Wraps LiteLLM for unified API access (Claude, GPT, Gemini, etc.)
- **`synthesize_cot()`** - Generates Chain-of-Thought reasoning traces for seed examples
- Uses `instructor` library for structured output with Pydantic schemas
- Supports retry logic with exponential backoff on API failures
- **Schemas in `teacher/schemas.py`**: `ThinkingTrace`, `TaskPolicy`, `TeacherResponse`

### Stage 2: Data Amplification
**Module:** `src/nanodistill/amplifier/pipeline.py`
- **`AmplificationPipeline`** - Two-phase data expansion strategy:
  1. Extract task policy from seed data and CoT traces (`_extract_policy`)
  2. Generate synthetic examples constrained by policy (`_generate_synthetic_examples`)
- Scales seed data by `augment_factor` (default: 50x → 10 seeds becomes 500 examples)
- Optional schema validation via `response_model` parameter for structured outputs

### Stage 3: Fine-tuning
**Module:** `src/nanodistill/distiller/trainer.py`
- **`MLXTrainer`** - Apple Silicon native training using MLX-LM with LoRA
- Loads student model from HuggingFace, fine-tunes with LoRA (not full parameters)
- Saves model to `{output_dir}/{name}/model/`
- Key config parameters: `batch_size`, `learning_rate`, `num_train_epochs`, `max_seq_length`

### Configuration & Data
- **`config.py`** - `DistillationConfig` Pydantic model with field validators for all parameters
- **`data/loader.py`** - Supports loading seed data from: Python list, JSON, JSONL, CSV
- **`data/formatter.py`** - Converts traces to HuggingFace Dataset format
- **`utils/errors.py`** - Custom exception hierarchy and validation functions

## Data Flow & File Outputs

```
Inputs (from user):
  - seed: List[Dict[input, output]] or file path
  - instruction: str (system prompt)
  - teacher: str (model name, default: "claude-sonnet-4-5")
  - student: str (model ID, default: "mlx-community/Llama-3-8B-Instruct-4bit")
  - augment_factor: int (default: 50)

Outputs (in {output_dir}/{name}/ directory):
  - traces_cot.jsonl: CoT traces generated from seed examples
  - traces_amplified.jsonl: Final training dataset (seed + synthetic examples)
  - model/: Fine-tuned MLX model directory
  - model.gguf: Quantized model (if exported)
```

## Key Design Patterns

### LiteLLM Abstraction
Teacher model is abstracted through LiteLLM, enabling easy swapping between Claude, GPT-4o, Gemini, or Ollama without code changes. Just pass `teacher="model-name"` to `distill()`.

### Instructor for Structured Output
Uses `instructor` library to enforce Pydantic schemas on LLM outputs. Automatically retries if output doesn't match schema and filters extra fields.

### Caching & Resumption
If `traces_amplified.jsonl` exists in output directory, skips API calls entirely and jumps to fine-tuning. Useful for iterating on training parameters without re-running synthesis.

### Optional Schema Validation
Pass `response_model` (a Pydantic BaseModel) to `distill()` to enforce structure on synthetic examples. Empty fields are automatically filtered.

## Common Modifications

**Changing the teacher model:**
```python
distill(..., teacher="gpt-4o")  # Uses OpenAI (requires OPENAI_API_KEY)
distill(..., teacher="gemini-pro")  # Uses Google (requires GOOGLE_API_KEY)
```

**Changing the student model:**
```python
distill(..., student="mlx-community/Mistral-7B-Instruct-4bit")
```

**Tuning training parameters:**
```python
distill(
    ...,
    batch_size=2,  # Lower for memory-constrained systems
    learning_rate=2e-5,  # Conservative default
    num_train_epochs=2,
    max_seq_length=512,
)
```

**Using structured output:**
```python
from pydantic import BaseModel

class TaskOutput(BaseModel):
    answer: str
    reasoning: str

distill(..., response_model=TaskOutput)
```

## Testing Strategy

- **`tests/conftest.py`** - Fixtures for common test data (seed examples, configs)
- **`tests/test_config.py`** - Validation of `DistillationConfig` Pydantic model
- **`tests/test_data.py`** - Data loading from various formats (JSON, JSONL, CSV)
- **`tests/test_teacher_schemas.py`** - Schema validation for structured outputs
- **`tests/test_errors.py`** - Error handling and validation logic

Tests avoid actual API calls by mocking LiteLLM. Focus is on data flow, configuration validation, and error handling.

## Environment Requirements

- **Python:** 3.9+
- **Hardware:** macOS with Apple Silicon (M1/M2/M3+) for MLX training
- **API Keys:**
  - `ANTHROPIC_API_KEY` for Claude models
  - `OPENAI_API_KEY` for GPT models (if used)
  - `GOOGLE_API_KEY` for Gemini (if used)

Set via `.env` file or environment variable:
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```
