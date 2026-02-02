# NanoDistill - Knowledge distillation for small language models

<p align="center">
  <strong>10 examples. One API key. Your own model.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" alt="build" />
  <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=for-the-badge" alt="tests" />
  <img src="https://img.shields.io/badge/python-3.9+-green?style=for-the-badge" alt="python" />
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-green?style=for-the-badge" alt="platform" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="license" />
</p>

**NanoDistill** is a _pipeline_ that turns a small set of seed examples and a task instruction into a custom small language model you run locally. You give it around 10 input/output pairs and an API key; it uses a teacher model to generate reasoning traces and hundreds of synthetic examples, then fine-tunes a student model (e.g. Llama 3 8B) with MLX on Apple Silicon. The result is a model that follows your task without ongoing API calls.

If you want a small, local model that does one thing well from a handful of examples, this is it.

[Quick Start](docs/QUICK_START.md) · [Workflow](docs/WORKFLOW.md) · [Model Setup](docs/MODEL_SETUP.md) · [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)

Preferred setup: `uv pip install -e .` (or `pip install -e .`). Runs on **macOS with Apple Silicon (M1/M2/M3+)**, 16GB+ RAM. Needs an API key for the teacher: [Anthropic](https://console.anthropic.com) for Claude; other teachers (OpenAI, Google, Ollama) work via LiteLLM. New install? Start here: [Getting started](docs/QUICK_START.md)

---

## Examples

### Install and run

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

```python
from nanodistill import distill

result = distill(
    name='stock-sentiment',
    seed='seeds.json',  # list of {"input": "...", "output": "..."}
    instruction='You analyze financial news and headlines. Output sentiment (bullish/bearish/neutral) and brief reasoning.',
    teacher='claude-sonnet-4-5',
)
print(f'Model saved to: {result.model_path}')
```

### Seed file (`seeds.json`)

```json
[
  {"input": "Tesla down 8% today. Elon says 'best quarter ever coming'.", "output": "{\"sentiment\": \"bearish\", \"reasoning\": \"Price drop and skeptical tone outweigh positive statement.\"}"},
  {"input": "AAPL beats earnings by 12% but warns of supply chain issues. Stock flat after-hours.", "output": "{\"sentiment\": \"neutral\", \"reasoning\": \"Strong beat offset by guidance; flat reaction suggests mixed view.\"}"},
  {"input": "NVDA 200% YoY growth. IV crush post-earnings, premium sellers loving it.", "output": "{\"sentiment\": \"bullish\", \"reasoning\": \"Strong fundamentals; IV crush reflects reduced uncertainty.\"}"}
]
```

### Run the model locally

```python
from mlx_lm import load, generate

model, tokenizer = load('./outputs/stock-sentiment/model')
response = generate(model, tokenizer, 'META announces layoffs affecting 15% of staff.', max_tokens=150)
print(response)
```

### Pipeline in one picture

```
Your 10 examples
    ↓
    ├→ Generate reasoning with Claude
    ├→ Extract task pattern
    ├→ Create hundreds of synthetic examples
    └→ Fine-tune Llama-3-8B (MLX)
    ↓
Locally runnable model
```

---

## How it works

1. **Policy extraction** - Infers the task pattern from your seed data and optional Chain-of-Thought traces.
2. **Synthetic generation** - Uses the teacher (e.g. Claude) to produce many new examples that match the pattern.
3. **Data amplification** - Turns seed + synthetic examples into training data (optionally with CoT).
4. **Fine-tuning** - Trains a student model on Apple Silicon with MLX-LM (LoRA).

You can swap the teacher (Claude, GPT-4o, Gemini, Ollama via LiteLLM) and the student (e.g. different MLX community models). Optional Pydantic `response_model` support lets you distill structured outputs.

---

## Configuration

**Environment**

- `ANTHROPIC_API_KEY` -Required for default Claude teacher
- `HF_HUB_TIMEOUT` -Optional (default 300s)

**Key parameters**

- `name` -Run identifier
- `seed` -Path to JSON/JSONL/CSV or list of `{"input", "output"}` (recommend 10+ examples)
- `instruction` -System/task description
- `teacher` -Teacher model (default: `claude-sonnet-4-5`)
- `student` -Student model (default: `mlx-community/Llama-3-8B-Instruct-4bit`)
- `augment_factor` -Data multiplier (default: 50)
- `output_dir` -Output directory (default: `./outputs`)

---

## Output

Per run, under `{output_dir}/{name}/`:

- `model/` -Fine-tuned model (MLX)
- `model.gguf` -Quantized model
- Training logs and metrics

---

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** -Export your key: `export ANTHROPIC_API_KEY='sk-ant-...'`

**Out of memory** -Lower `augment_factor` (e.g. 20–30), use a smaller student, or close other GPU-heavy apps.

**MLX** -Requires macOS 13+. See [MLX](https://github.com/ml-explore/mlx).

---

## Development

```bash
uv pip install -e ".[dev]"
pytest
black src/
mypy src/
```

**Layout**

```
src/nanodistill/
├── config.py       # Configuration
├── core.py         # Orchestrator
├── teacher/        # Teacher API (LiteLLM)
├── amplifier/      # Policy + synthetic data
├── distiller/      # MLX-LM training
├── data/           # Loaders and formatters
└── utils/          # Errors and helpers
```

---

## Roadmap

**Current:** MLX-LM on Apple Silicon, Claude Sonnet as default teacher, policy-based synthetic generation.

**Planned:** Cross-platform (e.g. Unsloth), more teachers (GPT-4o, Gemini, Ollama), richer amplification and evaluation, CLI.

---

## License

MIT

---

## Contributing

Contributions are welcome. See CONTRIBUTING.md (coming soon).

---

## Citation

```bibtex
@software{nanodistill2025,
  title = {NanoDistill: Knowledge Distillation for Small Language Models},
  author = {NanoDistill Contributors},
  year = {2025},
  url = {https://github.com/yourusername/nanodistill}
}
```
