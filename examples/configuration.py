"""
Configuration Examples for NanoDistill

This file demonstrates common configuration patterns for different scenarios.
Each example is self-contained and can be run independently.

To run any example:
    python examples/configuration.py

Requires: ANTHROPIC_API_KEY environment variable set
"""

# ============================================================================
# EXAMPLE 1: Minimal (No Configuration)
# ============================================================================
# Use this when you want to get started quickly with sensible defaults.
# Perfect for testing and prototyping.

def example_minimal():
    """Absolute minimum configuration - just the required parameters."""
    from nanodistill import distill

    result = distill(
        name="math-tutor-v1",
        seed=[
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+5?", "output": "8"},
            {"input": "What is 10-4?", "output": "6"},
            {"input": "What is 5×3?", "output": "15"},
            {"input": "What is 20÷4?", "output": "5"},
        ],
        instruction="You are a helpful math tutor. Show your reasoning.",
    )

    print(f"✅ Model saved to: {result.model_path}")
    print(f"   Training examples: {result.metrics['training_examples']}")


# ============================================================================
# EXAMPLE 2: Quick Prototyping (Fast Iteration)
# ============================================================================
# Use this when you're testing ideas and want fast feedback.
# Reduces dataset size and training time significantly.

def example_quick_prototype():
    """Fast prototyping configuration - small dataset, quick training."""
    from nanodistill import distill

    result = distill(
        name="quick-test",
        seed=[
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+5?", "output": "8"},
        ],
        instruction="You are a helpful math tutor.",
        # Fast training
        augment_factor=10,          # Small dataset (10 * 2 = 20 examples)
        num_train_epochs=1,         # Single epoch
        batch_size=1,
        # Simpler model
        lora_rank=4,
        lora_layers=2,
    )

    print(f"✅ Quick model trained: {result.model_path}")
    print(f"   Training time: <5 minutes (typical)")


# ============================================================================
# EXAMPLE 3: Memory-Constrained (M1/M2 8GB MacBook)
# ============================================================================
# Use this on systems with limited RAM.
# Conservative settings prevent out-of-memory errors.

def example_memory_constrained():
    """Configuration for memory-limited systems (M1/M2 8GB)."""
    from nanodistill import distill

    result = distill(
        name="m1-model",
        seed=[
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+5?", "output": "8"},
            {"input": "What is 10-4?", "output": "6"},
            {"input": "What is 5×3?", "output": "15"},
            {"input": "What is 20÷4?", "output": "5"},
            {"input": "What is 100+50?", "output": "150"},
            {"input": "What is 100-30?", "output": "70"},
            {"input": "What is 12×5?", "output": "60"},
            {"input": "What is 144÷12?", "output": "12"},
            {"input": "What is 7×8?", "output": "56"},
        ],
        instruction="You are a helpful math tutor. Show your reasoning.",

        # Conservative training settings
        batch_size=1,               # Minimal memory usage
        max_seq_length=256,         # Shorter sequences
        learning_rate=5e-6,         # Very conservative
        num_train_epochs=1,         # Single epoch

        # Smaller model
        lora_rank=4,
        lora_layers=2,

        # Conservative system limits
        max_memory_gb=4,            # Use only 4GB
        memory_hard_limit_gb=5,     # Hard stop at 5GB
    )

    print(f"✅ Memory-friendly model: {result.model_path}")
    print(f"   Configured for M1/M2 8GB systems")


# ============================================================================
# EXAMPLE 4: High-Performance (M3 Pro/Max 36GB+)
# ============================================================================
# Use this on powerful systems for higher quality models.
# Leverages available resources for better results.

def example_high_performance():
    """Configuration for high-performance systems (M3 Pro/Max with 36GB+)."""
    from nanodistill import distill

    # Create a larger seed dataset
    seed_data = [
        {"input": f"What is {a}+{b}?", "output": str(a + b)}
        for a, b in [(2, 2), (3, 5), (10, 4), (5, 3), (20, 4),
                     (100, 50), (15, 25), (7, 8), (12, 18), (99, 1)]
    ]

    result = distill(
        name="high-quality-model",
        seed=seed_data,
        instruction="You are a helpful math tutor. Show your reasoning step-by-step.",

        # Larger dataset for better quality
        augment_factor=100,         # 10 seeds × 100 = 1000 training examples

        # Better training settings
        batch_size=4,               # Larger batches
        max_seq_length=1024,        # Longer sequences
        learning_rate=5e-5,         # Standard rate
        num_train_epochs=3,         # Multiple epochs

        # More expressive model
        lora_rank=16,               # Larger adapter rank
        lora_layers=8,              # More layers affected

        # Higher temperatures for diversity
        temperature=0.8,

        # Use available system resources
        # (auto-detected, but can override)
        max_memory_gb=16,
        memory_hard_limit_gb=24,
        cpu_capacity_percent=0.9,
    )

    print(f"✅ High-quality model: {result.model_path}")
    print(f"   Metrics: {result.metrics}")


# ============================================================================
# EXAMPLE 5: Production Quality (Conservative, High-Quality)
# ============================================================================
# Use this for models you're shipping to production.
# Prioritizes consistency and robustness over speed.

def example_production_quality():
    """Configuration for production-grade models."""
    from nanodistill import distill

    # High-quality seed data (50+ examples)
    seed_data = [
        {"input": f"What is {a}+{b}?", "output": str(a + b)}
        for a, b in [(2, 2), (3, 5), (10, 4), (5, 3), (20, 4),
                     (100, 50), (15, 25), (7, 8), (12, 18), (99, 1),
                     (45, 55), (33, 67), (111, 89), (123, 456), (789, 123)]
    ]

    result = distill(
        name="production-math-tutor",
        seed=seed_data,
        instruction="You are a professional math tutor. "
                   "Always show step-by-step reasoning. "
                   "Explain concepts clearly and accurately.",

        # Larger dataset
        augment_factor=100,

        # Conservative training (prioritizes stability)
        batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=2,
        max_seq_length=512,

        # Good balance of expressiveness
        lora_rank=12,
        lora_layers=6,

        # Lower temperature for consistency
        temperature=0.3,

        # 80/20 train/val split (default, but explicit)
        val_split=0.2,
    )

    print(f"✅ Production model: {result.model_path}")
    print(f"   Training examples: {result.metrics['training_examples']}")
    print(f"   Epochs: {result.metrics['num_epochs']}")


# ============================================================================
# EXAMPLE 6: Structured Output (Schema Enforcement)
# ============================================================================
# Use this when you need consistent, structured output from the model.

def example_structured_output():
    """Configuration with schema enforcement for structured outputs."""
    from pydantic import BaseModel
    from nanodistill import distill

    class MathExplanation(BaseModel):
        """Schema for math tutor responses."""
        answer: str
        reasoning: str
        difficulty_level: str  # "easy", "medium", "hard"

    seed_data = [
        {"input": "What is 2+2?", "output": '{"answer": "4", "reasoning": "Adding 2 and 2 gives 4", "difficulty_level": "easy"}'},
        {"input": "What is 3+5?", "output": '{"answer": "8", "reasoning": "3 plus 5 equals 8", "difficulty_level": "easy"}'},
    ]

    result = distill(
        name="structured-math-tutor",
        seed=seed_data,
        instruction="You are a math tutor. Provide structured responses with answer, reasoning, and difficulty level.",

        # Enforce schema
        response_model=MathExplanation,

        # Lower temperature for consistency with schema
        temperature=0.3,
        num_train_epochs=2,
    )

    print(f"✅ Structured model: {result.model_path}")
    print(f"   Using schema: {MathExplanation.__name__}")


# ============================================================================
# EXAMPLE 7: Custom Teacher Model
# ============================================================================
# Use this to use a different teacher model (GPT-4, Gemini, etc.)
# Any LiteLLM-compatible model works.

def example_custom_teacher():
    """Configuration with alternative teacher model."""
    from nanodistill import distill

    result = distill(
        name="gpt4-student",
        seed=[
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+5?", "output": "8"},
        ],
        instruction="You are a helpful math tutor.",

        # Use GPT-4 instead of Claude (requires OPENAI_API_KEY)
        teacher="gpt-4o",

        # Or use Gemini (requires GOOGLE_API_KEY)
        # teacher="gemini-pro",

        # Or use local Ollama
        # teacher="ollama/llama2",

        num_train_epochs=1,
    )

    print(f"✅ Model trained with alternative teacher: {result.model_path}")


# ============================================================================
# EXAMPLE 8: Custom Student Model
# ============================================================================
# Use this to fine-tune a different student model.

def example_custom_student():
    """Configuration with alternative student model."""
    from nanodistill import distill

    result = distill(
        name="mistral-student",
        seed=[
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+5?", "output": "8"},
        ],
        instruction="You are a helpful math tutor.",

        # Use Mistral instead of Llama (any MLX-compatible model)
        student="mlx-community/Mistral-7B-Instruct-4bit",

        num_train_epochs=1,
    )

    print(f"✅ Model trained with alternative student: {result.model_path}")


# ============================================================================
# EXAMPLE 9: Aggressive Training (Maximum Resources)
# ============================================================================
# Use this on a dedicated M3 Max machine for maximum quality.
# Aggressively uses system resources.

def example_aggressive_training():
    """Maximum resource utilization on powerful dedicated systems."""
    from nanodistill import distill

    # Large seed dataset
    seed_data = [
        {"input": f"What is {a}+{b}?", "output": str(a + b)}
        for a, b in [
            (2, 2), (3, 5), (10, 4), (5, 3), (20, 4),
            (100, 50), (15, 25), (7, 8), (12, 18), (99, 1),
            (45, 55), (33, 67), (111, 89), (123, 456), (789, 123),
            (11, 22), (44, 55), (77, 88), (91, 9), (50, 50),
        ]
    ]

    result = distill(
        name="maximum-quality-model",
        seed=seed_data,
        instruction="You are an expert math tutor with deep knowledge.",

        # Maximum dataset size
        augment_factor=200,         # 20 seeds × 200 = 4000 examples

        # Aggressive training
        batch_size=8,               # Large batches
        max_seq_length=2048,        # Full context
        learning_rate=1e-4,         # Higher learning rate
        num_train_epochs=5,         # Many epochs

        # Maximum expressiveness
        lora_rank=32,
        lora_layers=16,

        # Aggressive system usage
        max_memory_gb=48,           # Use 48GB RAM
        memory_hard_limit_gb=64,    # Hard stop at 64GB
        cpu_capacity_percent=0.95,  # Use up to 95% CPU

        # Balanced temperature
        temperature=0.7,
    )

    print(f"✅ Maximum quality model: {result.model_path}")
    print(f"   Metrics: {result.metrics}")


# ============================================================================
# EXAMPLE 10: Balancing Speed vs. Quality
# ============================================================================
# Use this to find the sweet spot for your use case.

def example_balanced():
    """Balanced configuration for speed and quality."""
    from nanodistill import distill

    result = distill(
        name="balanced-model",
        seed=[
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+5?", "output": "8"},
            {"input": "What is 10-4?", "output": "6"},
            {"input": "What is 5×3?", "output": "15"},
            {"input": "What is 20÷4?", "output": "5"},
        ],
        instruction="You are a helpful math tutor. Show your reasoning.",

        # Medium dataset (good quality without excessive training)
        augment_factor=50,          # Default

        # Balanced training settings
        batch_size=2,               # Reasonable batch size
        learning_rate=1e-5,         # Conservative default
        num_train_epochs=2,         # One more epoch than minimum
        max_seq_length=512,         # 2x minimum

        # Balanced LoRA
        lora_rank=8,                # Default
        lora_layers=4,              # Default

        # Balanced temperature
        temperature=0.7,            # Default

        # Let system auto-detect (no explicit limits)
    )

    print(f"✅ Balanced model: {result.model_path}")
    print(f"   Good quality with reasonable training time")


# ============================================================================
# Main: Run all examples
# ============================================================================

if __name__ == "__main__":
    import os

    # Check for required API key
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo use these examples, set your API key:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("\nThen run this script again.")
        exit(1)

    examples = [
        ("Minimal (No Configuration)", example_minimal),
        ("Quick Prototyping", example_quick_prototype),
        ("Memory-Constrained", example_memory_constrained),
        ("High-Performance", example_high_performance),
        ("Production Quality", example_production_quality),
        ("Structured Output", example_structured_output),
        ("Custom Teacher Model", example_custom_teacher),
        ("Custom Student Model", example_custom_student),
        ("Aggressive Training", example_aggressive_training),
        ("Balanced", example_balanced),
    ]

    print("=" * 70)
    print("NanoDistill Configuration Examples")
    print("=" * 70)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("\nTo run a specific example, modify the __main__ section.")
    print("\nExample usage:")
    print("  # Run example 2 (Quick Prototyping)")
    print("  python examples/configuration.py")
    print("  # Then uncomment the example function call below\n")

    # Uncomment the example you want to run:
    # example_minimal()
    # example_quick_prototype()
    # example_memory_constrained()
    # example_high_performance()
    # example_production_quality()
    # example_structured_output()
    # example_custom_teacher()
    # example_custom_student()
    # example_aggressive_training()
    # example_balanced()

    print("✅ Configuration examples loaded successfully!")
    print("\nTo run an example, uncomment it in the __main__ section.")
