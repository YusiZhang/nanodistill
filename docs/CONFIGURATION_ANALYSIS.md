# NanoDistill Configuration Analysis & Enhancement Plan

**Date**: 2026-02-01
**Status**: For Review
**Purpose**: Identify generic vs. opinionated components and provide options for making the library more configurable for open source release.

---

## Executive Summary

NanoDistill is a well-designed knowledge distillation library with **sensible defaults** but **limited flexibility** in key areas.

### What This Document Covers

1. **Analysis**: Identifies what's configurable vs. hardcoded (including external library calls)
2. **Approach**: User-approved plan using generic kwargs dicts for maximum flexibility
3. **Implementation**: Specific code changes and new configuration fields
4. **Critical Fixes**: 3 bugs identified and fixed in Phase 1

### User-Approved Final Approach ✅

**Instead of**: 30+ individual config fields
**We use**: 7 core fields + 2 generic kwargs dicts

```python
# Core explicit parameters
temperature: float = 0.7
lora_rank: int = 8
lora_layers: int = 4
val_split: float = 0.2
max_memory_gb: int = (auto-detect)
memory_hard_limit_gb: int = (auto-detect)
cpu_capacity_percent: float = 0.8

# Generic passthrough - supports ANY library parameter
litellm_kwargs: Dict[str, Any] = {}    # Any LiteLLM param
mlx_lm_kwargs: Dict[str, Any] = {}     # Any MLX-LM param
```

**Benefits**:
- ✅ Fixes 3 critical bugs (hardcoded temps, max_seq_length, undocumented LoRA params)
- ✅ Supports ANY LiteLLM/MLX-LM parameter without code changes
- ✅ No function signature bloat (stays clean with `**kwargs`)
- ✅ ~100 lines of code changes total (minimal!)
- ✅ Backward compatible (all defaults unchanged)

### Key Finding: Clean API with kwargs Power

**Current state**:
- 8 explicit parameters in function signature (core requirements)
- 4 hidden kwargs for training (batch_size, learning_rate, num_train_epochs, max_seq_length)

**After Phase 1** (via kwargs - no signature changes):
- ✅ 7 new configurable parameters via kwargs
- ✅ **All passed through `**kwargs` - signature stays clean!**
- ✅ Full Pydantic validation for all parameters
- ✅ Auto-detection for system limits

**New configurable parameters (all via kwargs)**:
- Temperature (sampling control)
- Train/validation split ratio
- LoRA rank and layers
- System RAM limits (with auto-detection!)
- System CPU threshold

**Still hardcoded (for Phase 3+)**:
- Chain-of-Thought format (`<thinking>` tags)
- System prompts and prompt templates
- Response parsing strategies
- LoRA advanced settings (alpha, dropout, target modules)

---

## Part 1: Current Configuration Landscape

### ✅ Explicit Parameters (Core API)

These are in the `distill()` function signature:

```python
distill(
    # Required (no defaults)
    name: str,                                    # Run identifier
    seed: List[Dict[str, str]],                 # Training examples
    instruction: str,                            # Task description

    # Optional with sensible defaults
    teacher: str = "claude-sonnet-4-5",         # Any LiteLLM model
    student: str = "mlx-community/Llama-3-8B-Instruct-4bit",  # MLX model
    augment_factor: int = 50,                   # Range: 1-500
    output_dir: str = "./outputs",
    response_model: Optional[Type[BaseModel]] = None,  # Schema enforcement

    # Everything else via kwargs (see below)
    **kwargs,
)
```

### ✅ Configurable via kwargs (Flexible Parameters)

These are passed as keyword arguments via `**kwargs`. All are validated by `DistillationConfig`:

**Training Parameters**:
```python
batch_size: int = 1                    # Range: 1-32
learning_rate: float = 1e-5            # Range: >0 to <1
num_train_epochs: int = 1              # Range: 1-10
max_seq_length: int = 256              # Range: 32-2048
```

**LoRA Parameters (v0.2.0+)**:
```python
lora_rank: int = 8                     # Range: 1-64
lora_layers: int = 4                   # Range: 1-32
```

**Generation Parameters (v0.2.0+)**:
```python
temperature: float = 0.7               # Range: 0.0-2.0
```

**Data Split (v0.2.0+)**:
```python
val_split: float = 0.2                 # Range: 0.0-0.5
```

**System Configuration (v0.2.0+)**:
```python
max_memory_gb: int = 12                # Auto-detect, cap at 12GB
memory_hard_limit_gb: int = 12         # Auto-detect, cap at 12GB
cpu_capacity_percent: float = 0.8      # 80% default threshold
```

### 🔒 Completely Hardcoded (8 Major Areas)

#### 1️⃣ Chain-of-Thought Format

**Files**: `src/nanodistill/data/formatter.py:31, 103, 116`

**Current**: Fixed `<thinking>` XML tags

```python
# formatter.py:31
content = f"<thinking>\n{trace.thinking}\n</thinking>\n\n{trace.output}"

# formatter.py:103
return f"<thinking>\n{thinking}\n</thinking>\n\n{output}"

# formatter.py:116
if "<thinking>" in response and "</thinking>" in response:
    # parsing logic
```

**Impact**: Cannot use alternative formats (JSON, Markdown, plain text)

---

#### 2️⃣ System Prompts & Templates

**Files**: `src/nanodistill/teacher/prompts.py`

**Current**: 4 hardcoded system prompts

| Prompt | Lines | Purpose | Customizable |
|--------|-------|---------|--------------|
| `COT_SYSTEM_PROMPT` | 8-21 | Reasoning style instructions | ❌ No |
| `POLICY_EXTRACTION_SYSTEM_PROMPT` | 51-60 | Policy analysis framework | ❌ No |
| `SYNTHETIC_GENERATION_SYSTEM_PROMPT` | 109-118 | Generation guidelines | ❌ No |
| Policy dimensions (in schema) | TaskPolicy | 10 fixed fields | ❌ No |

**Fixed Policy Dimensions** (Cannot be changed):
1. task_description
2. input_format
3. output_format
4. reasoning_style
5. key_constraints
6. difficulty_level
7. reasoning_patterns
8. examples_summary
9. input_length_range
10. output_length_range

**Impact**: Cannot customize how the teacher analyzes tasks or generates examples

---

#### 3️⃣ Temperature & Sampling

**Files**: `src/nanodistill/teacher/client.py:262, 320`

**Current**: Fixed `temperature=0.7`

```python
# Line 262 (text parsing mode)
response = self.client.messages.create(
    ...,
    temperature=0.7,  # HARDCODED
)

# Line 320 (schema mode)
response = self.client.messages.create(
    ...,
    temperature=0.7,  # HARDCODED
)
```

**No Control Over**:
- top_p (nucleus sampling)
- top_k (top-k sampling)
- frequency_penalty
- presence_penalty

**Impact**: Cannot adjust creativity vs. consistency in synthetic examples

---

#### 4️⃣ Response Parsing Strategies

**Files**: `src/nanodistill/teacher/client.py:394-511`

**Current**: 3 rigid parsing strategies

```
Strategy 1: Markdown format
## Example N
**Input:** [text]
**Output:** [text]

Strategy 2: Colon-separated
Input: [text]
Output: [text]

Strategy 3: JSON array
[{"input": ..., "output": ...}]
```

**Impact**: Cannot add custom parsing strategies; fails if format doesn't match

---

#### 5️⃣ Training Data Format

**Files**: `src/nanodistill/distiller/trainer.py:144-148`

**Current**: Fixed template

```python
text = f"""Input: {example['input']}

Thinking: {example['thinking']}

Output: {example['output']}"""
```

**Also Fixed**:
- Conversation structure: Always `user → assistant` (no system messages)
- Chat template application: Delegates to tokenizer

**Impact**: Cannot customize how training examples are formatted

---

#### 6️⃣ Training Configuration Details

**Files**: `src/nanodistill/distiller/trainer.py`

| Setting | Value | Line | Customizable |
|---------|-------|------|--------------|
| Train/val split | 80/20 | 219 | ❌ No |
| Progress reports | ~20 per training | 241 | ❌ No |
| Memory hard cap | 12GB | 186 | ❌ No |
| CPU capacity threshold | 80% | 196 | ❌ No |
| CPU thread limiting | 80% of available | 267 | ❌ No |
| Optimizer | AdamW (via MLX-LM) | implicit | ❌ No |

**No Control Over**:
- Gradient clipping
- Gradient accumulation
- Warmup steps
- Learning rate scheduling
- Early stopping
- Checkpoint frequency

**Impact**: Cannot fine-tune advanced training strategies

---

#### 7️⃣ LoRA Advanced Settings

**Files**: `src/nanodistill/distiller/trainer.py:121-129`

**Exposed** (via getattr):
- lora_rank (default: 8)
- lora_layers (default: 4)

**Hidden/Not Exposed**:
- LoRA alpha (scaling factor)
- LoRA dropout
- Target modules (which attention layers)
- Quantization settings

**Impact**: Limited control over LoRA fine-tuning behavior

---

#### 8️⃣ API Behavior

**Files**: `src/nanodistill/teacher/client.py:40`

**Hardcoded**:
- `max_retries: 3` - API retry count
- `confidence: 0.9` - Default confidence (not computed)

**Impact**: Cannot adjust retry strategy or confidence scoring

---

## Part 2: Enhancement Options

### Option A: Conservative Approach

**Philosophy**: Minimal API surface increase, address high-ROI parameters

#### What Gets Added
1. **Temperature control**
2. **Thinking format alternatives**
3. **LoRA parameters (explicit)**
4. **Train/val split control**

#### Code Impact
```python
def distill(
    ...,
    temperature: float = 0.7,           # NEW
    thinking_format: str = "xml",       # NEW
    lora_rank: int = 8,                 # NEW (from getattr)
    lora_layers: int = 4,               # NEW (from getattr)
    val_split: float = 0.2,             # NEW
)
```

#### Pros
✅ Minimal API increase (5 new parameters)
✅ Addresses most common customization needs
✅ Easy to understand and document

#### Cons
❌ Still limited for power users
❌ Prompts remain hardcoded
❌ Advanced training settings inaccessible

---

### Option B: Moderate Approach (RECOMMENDED)

**Philosophy**: Tiered configuration - simple by default, powerful when needed

#### Tier 1: Main API Parameters (Recommended Approach)

Keep signature clean, move new params to kwargs:

```python
def distill(
    # Core parameters (explicit)
    name: str,
    seed: Union[List[Dict[str, str]], str, Path],
    instruction: str,
    teacher: str = "claude-sonnet-4-5",
    student: str = "mlx-community/Llama-3-8B-Instruct-4bit",
    augment_factor: int = 50,
    output_dir: str = "./outputs",
    response_model: Optional[Type[BaseModel]] = None,

    # All other params go into kwargs (cleaner API)
    **kwargs,  # temperature, lora_rank, lora_layers, val_split,
               # batch_size, learning_rate, num_train_epochs, max_seq_length,
               # max_memory_gb, memory_hard_limit_gb, cpu_capacity_percent
) -> DistillationResult:
```

**What goes in kwargs**:
```python
# Training parameters (already supported via kwargs)
batch_size=1
learning_rate=1e-5
num_train_epochs=1
max_seq_length=256

# NEW in Phase 1: Generation & LoRA
temperature=0.7
lora_rank=8
lora_layers=4
val_split=0.2

# NEW in Phase 1: System configuration
max_memory_gb=12
memory_hard_limit_gb=12
cpu_capacity_percent=0.8
```

#### Tier 2: AdvancedConfig Class (Optional)
```python
from nanodistill import AdvancedConfig

advanced = AdvancedConfig(
    # Sampling
    top_p: float = 1.0,
    top_k: int = -1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,

    # Training
    gradient_clip: float = 1.0,
    warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    # LoRA
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "v_proj"],

    # System
    memory_limit_gb: int = 12,
    cpu_limit_percent: float = 0.8,
)

result = distill(..., advanced_config=advanced)
```

#### Tier 3: Prompt Overrides (Escape Hatch)
```python
from nanodistill import PromptTemplates

prompts = PromptTemplates(
    cot_system="Your custom CoT instructions...",
    policy_extraction_system="Your custom policy prompt...",
    synthetic_generation_system="Your custom generation prompt...",
)

result = distill(..., prompt_templates=prompts)
```

#### Pros
✅ Progressive disclosure: simple → power users
✅ Escape hatches for all major decisions
✅ Backward compatible
✅ Extensible for future needs

#### Cons
⚠️ More API surface to document
⚠️ More validation needed

---

### Option C: Aggressive Approach

**Philosophy**: Plugin system for maximum extensibility

#### Plugin Types
```python
from nanodistill.plugins import (
    ThinkingFormat,
    TrainingStrategy,
    ResponseParser,
)

class JSONThinkingFormat(ThinkingFormat):
    def format(self, thinking: str, output: str) -> str:
        return json.dumps({"reasoning": thinking, "answer": output})

    def parse(self, response: str) -> tuple[str, str]:
        data = json.loads(response)
        return data["reasoning"], data["answer"]

class CustomResponseParser(ResponseParser):
    def parse_synthetic_examples(self, text: str) -> List[Dict[str, str]]:
        # Custom logic
        ...

result = distill(
    ...,
    thinking_format=JSONThinkingFormat(),
    response_parser=CustomResponseParser(),
)
```

#### Pros
✅ Maximum flexibility
✅ Extensible for any use case

#### Cons
❌ High complexity
❌ Steep learning curve
❌ Significant maintenance burden

---

## Part 3: Recommended Approach

### 🎯 Start with **Option B (Moderate)** - Phased Implementation

This balances ease-of-use with flexibility while avoiding premature over-engineering.

---

### Phase 1: Expose Hidden Parameters (CRITICAL - Pre-Release)

**Timeline**: 2-4 hours
**Impact**: High - Addresses most common user needs

#### 1.1 Update Config Schema

**File**: `src/nanodistill/config.py`

**Dependencies**: Add psutil to requirements (for system auto-detection)
```
psutil>=5.8.0
```

Add 3 new fields to `DistillationConfig` (+ 4 previously added fields from Phase 1):

```python
temperature: float = Field(
    default=0.7,
    description="Sampling temperature for synthetic generation (0.0-2.0)",
    ge=0.0,
    le=2.0,
)

lora_rank: int = Field(
    default=8,
    description="LoRA adapter rank (higher = more parameters)",
    ge=1,
    le=64,
)

lora_layers: int = Field(
    default=4,
    description="Number of layers to apply LoRA adapters",
    ge=1,
    le=32,
)

val_split: float = Field(
    default=0.2,
    description="Validation set split ratio (0.0-0.5)",
    ge=0.0,
    le=0.5,
)

# System Configuration - NEW for Phase 1
@staticmethod
def get_system_defaults() -> tuple[int, int, float]:
    """Auto-detect system capabilities and return safe defaults.

    Returns:
        (max_memory_gb, memory_hard_limit_gb, cpu_capacity_percent)
    """
    import psutil

    try:
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        # Conservative defaults: use up to 90% of RAM, hard stop at 95%
        max_memory = min(total_memory_gb * 0.9, 12)  # Cap at 12GB
        hard_limit = min(total_memory_gb * 0.95, 12)  # Cap at 12GB
        cpu_threshold = 0.8  # 80% capacity threshold

        return int(max_memory), int(hard_limit), cpu_threshold
    except Exception:
        # Fallback if psutil fails
        return 12, 12, 0.8

max_memory_gb: int = Field(
    default_factory=lambda: DistillationConfig.get_system_defaults()[0],
    description="Maximum RAM to use during training (GB). Auto-detects system, defaults to 12GB.",
    ge=2,
    le=128,
)

memory_hard_limit_gb: int = Field(
    default_factory=lambda: DistillationConfig.get_system_defaults()[1],
    description="Hard stop limit - training stops if memory exceeds this (GB). Should be >= max_memory_gb.",
    ge=2,
    le=128,
)

cpu_capacity_percent: float = Field(
    default=0.8,
    description="CPU/memory threshold - training pauses if system > this percent (0.1-1.0)",
    ge=0.1,
    le=1.0,
)

@field_validator("memory_hard_limit_gb")
@classmethod
def validate_memory_limits(cls, v: int, info) -> int:
    """Ensure hard limit >= max memory."""
    if info.data.get("max_memory_gb") and v < info.data["max_memory_gb"]:
        raise ValueError(
            f"memory_hard_limit_gb ({v}) must be >= max_memory_gb ({info.data['max_memory_gb']})"
        )
    return v
```

#### 1.2 Update Teacher Client

**File**: `src/nanodistill/teacher/client.py`

Replace hardcoded temperatures at lines 262 and 320:

```python
# Before (line 262):
temperature=0.7,

# After:
temperature=getattr(self.config, "temperature", 0.7),

# Same change at line 320
```

#### 1.3 Update Trainer

**File**: `src/nanodistill/distiller/trainer.py`

Update train/val split at line 219 and system monitoring throughout:

```python
# Update train/val split (line 219)
# Before:
split_idx = int(len(train_data) * 0.8)

# After:
val_ratio = getattr(self.config, "val_split", 0.2)
split_idx = int(len(train_data) * (1 - val_ratio))


# Update system monitoring (lines 177-206: check_system_capacity() function)
# Before:
def check_system_capacity() -> bool:
    """Check if system is below 80% capacity and 12GB memory hard limit."""
    mem_info = psutil.virtual_memory()
    memory_used_gb = mem_info.used / (1024 ** 3)

    if memory_used_gb > 12:  # HARDCODED
        print(f"\n🛑 MEMORY HARD CAP HIT: {memory_used_gb:.2f}GB > 12GB limit!")
        return False

    if mem_info.percent > 80:  # HARDCODED
        print(f"\n⚠️  System capacity at {mem_info.percent:.1f}%")
        return False

    return True


# After:
def check_system_capacity() -> bool:
    """Check if system is below configured capacity limits."""
    mem_info = psutil.virtual_memory()
    memory_used_gb = mem_info.used / (1024 ** 3)

    # Get configured limits (with defaults)
    hard_limit = getattr(self.config, "memory_hard_limit_gb", 12)
    capacity_threshold = getattr(self.config, "cpu_capacity_percent", 0.8)

    if memory_used_gb > hard_limit:
        print(f"\n🛑 MEMORY HARD LIMIT HIT: {memory_used_gb:.2f}GB > {hard_limit}GB!")
        print(f"   Stopping training to prevent system crash...")
        return False

    cpu_percent = psutil.cpu_percent(interval=0.1)
    current_capacity = max(mem_info.percent, cpu_percent)

    if current_capacity > (capacity_threshold * 100):
        print(f"\n⚠️  System capacity at {current_capacity:.1f}% (threshold: {capacity_threshold*100:.0f}%)")
        print(f"   Memory: {memory_used_gb:.2f}GB / {hard_limit}GB ({mem_info.percent:.1f}%)")
        print(f"   CPU: {cpu_percent:.1f}%")
        print(f"   Waiting for system to cool down...")
        return False

    return True
```

#### 1.4 Update Core Signature (No changes needed!)

**File**: `src/nanodistill/core.py`

**Good news**: The `distill()` function already accepts `**kwargs` and passes them to `DistillationConfig`!

Current implementation already does exactly what we need:

```python
def distill(
    name: str,
    seed: Union[List[Dict[str, str]], str, Path],
    instruction: str,
    teacher: str = "claude-sonnet-4-5",
    student: str = "mlx-community/Llama-3-8B-Instruct-4bit",
    augment_factor: int = 50,
    output_dir: str = "./outputs",
    response_model: Optional[Type[BaseModel]] = None,
    **kwargs,  # Already captures temperature, lora_rank, etc.
) -> DistillationResult:
    ...
    config = DistillationConfig(
        name=name,
        seed=seed,
        instruction=instruction,
        teacher=teacher,
        student=student,
        augment_factor=augment_factor,
        output_dir=output_dir,
        **kwargs,  # Already passes through!
    )
```

**No changes needed** - the architecture already supports this pattern!

---

### Phase 2: Documentation (CRITICAL - Pre-Release)

**Timeline**: 4-6 hours
**Impact**: High - Users understand what's configurable and why

**Key Message**:
> NanoDistill keeps the main API clean with 8 explicit parameters. Everything else is configured via kwargs - simple by default, powerful when needed.

#### 2.1 Create Configuration Reference

**File**: `docs/CONFIGURATION.md`

```markdown
# Configuration Reference

## Overview

NanoDistill provides sensible defaults - just call `distill(name, seed, instruction)` and it works!

For advanced customization, pass any of these parameters as keyword arguments. All are validated and optional.

## Quick Start (No Configuration Needed)
```python
from nanodistill import distill

result = distill(
    name="my-model",
    seed=[{"input": "...", "output": "..."}, ...],
    instruction="...",
)
```

## Customize via Keyword Arguments (Optional)
```python
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",

    # Training (optional)
    batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=2,
    max_seq_length=512,

    # LoRA (optional)
    lora_rank=16,
    lora_layers=8,

    # Generation (optional)
    temperature=0.7,

    # Data split (optional)
    val_split=0.2,

    # System (optional - auto-detected if not specified)
    max_memory_gb=12,
    memory_hard_limit_gb=16,
    cpu_capacity_percent=0.8,
)
```

## Core Parameters

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| name | str | Required | 1-100 chars | Run identifier |
| seed | List[Dict] | Required | min 1 | Training examples |
| instruction | str | Required | min 10 chars | Task description |
| teacher | str | "claude-sonnet-4-5" | Any LiteLLM model | Teacher model |
| student | str | "mlx-community/Llama-3-8B-Instruct-4bit" | MLX model ID | Student model |
| augment_factor | int | 50 | 1-500 | Data multiplication |
| output_dir | str | "./outputs" | Valid path | Output directory |

## All Optional Parameters (passed via kwargs)

**These can all be passed as keyword arguments to `distill()`**

### Training Parameters

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| batch_size | int | 1 | 1-32 | Training batch size |
| learning_rate | float | 1e-5 | >0, <1 | Learning rate |
| num_train_epochs | int | 1 | 1-10 | Training epochs |
| max_seq_length | int | 256 | 32-2048 | Token limit |

### LoRA Parameters (New in v0.2.0)

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| lora_rank | int | 8 | 1-64 | Adapter rank |
| lora_layers | int | 4 | 1-32 | Target layers |

### Generation Parameters (New in v0.2.0)

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| temperature | float | 0.7 | 0.0-2.0 | Sampling temperature |

### Data Split (New in v0.2.0)

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| val_split | float | 0.2 | 0.0-0.5 | Validation ratio |

## System Configuration Parameters (New in v0.2.0)

NanoDistill can auto-detect your system capabilities and apply sensible defaults, but you can override them:

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| max_memory_gb | int | Auto-detect, max 12 | 2-128 | Maximum RAM to use during training |
| cpu_capacity_percent | float | 0.8 | 0.1-1.0 | CPU threshold before slowdown (80% = pause if > 80%) |
| memory_hard_limit_gb | int | Auto-detect, max 12 | 2-128 | Hard stop if memory exceeds this (prevents system crash) |

### System Auto-Detection

If you don't specify system parameters, NanoDistill automatically detects your system and sets safe defaults:

```python
import psutil

# Auto-detection logic:
total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)

# Conservative defaults:
max_memory_gb = min(total_memory_gb * 0.9, 12)  # Use up to 90% of RAM, but cap at 12GB
memory_hard_limit_gb = min(total_memory_gb * 0.95, 12)  # Hard stop at 95% or 12GB
cpu_capacity_percent = 0.8  # Stop if system > 80% loaded
```

### Examples by System Type

#### M1 MacBook Air (8GB RAM)
```python
# Auto-detected defaults
distill(
    name="my-model",
    seed=[...],
    instruction="...",
    # System config auto-detected:
    # max_memory_gb = 7.2 (8 * 0.9)
    # memory_hard_limit_gb = 7.6 (8 * 0.95)
)

# Or explicitly override:
distill(
    ...,
    max_memory_gb=6,  # Conservative
    memory_hard_limit_gb=7,
)
```

#### M2 MacBook Pro (16GB RAM)
```python
distill(
    ...,
    # Auto-detected:
    # max_memory_gb = 12 (capped at 12)
    # memory_hard_limit_gb = 12
)
```

#### M3 Max (128GB RAM)
```python
distill(
    ...,
    # Auto-detected:
    # max_memory_gb = 12 (capped at 12)
    # memory_hard_limit_gb = 12

    # Or increase limit for powerful systems:
    max_memory_gb=48,
    memory_hard_limit_gb=64,
)
```

### System Configuration Details

#### max_memory_gb
- **What it does**: Training pauses if memory usage exceeds this threshold
- **Default**: 90% of system RAM, capped at 12GB
- **Why capped at 12GB**: Conservative default prevents swapping on most systems
- **When to increase**: If you have 64GB+ RAM and want to use more
- **When to decrease**: If you're getting OOM errors on smaller systems

#### memory_hard_limit_gb
- **What it does**: Training STOPS immediately if memory exceeds this (prevents system crash)
- **Default**: 95% of system RAM, capped at 12GB
- **Why capped at 12GB**: Extra safety margin
- **When to increase**: Only if you know your system can handle it
- **Critical**: Should always be >= max_memory_gb

#### cpu_capacity_percent
- **What it does**: If CPU or memory usage > this threshold, training pauses to cool down
- **Default**: 0.8 (80%)
- **Why 80%**: Allows other system processes (Spotlight, browser, etc.)
- **When to increase**: If system is dedicated to training (0.9+)
- **When to decrease**: If your system is slow with training active (0.6-0.7)

## Usage Examples

### Memory-Constrained System (M1 8GB)
```python
from nanodistill import distill

result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    batch_size=1,
    max_seq_length=256,
    lora_rank=4,
)
```

### High-Performance System (M3 Pro 36GB)
```python
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    batch_size=4,
    max_seq_length=1024,
    lora_rank=16,
    lora_layers=8,
    learning_rate=5e-5,
    # System config auto-detected for 36GB system
)
```

### Maximum Performance (M3 Max 128GB+)
```python
# System auto-detects 128GB, but caps at 12GB by default (conservative)
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    batch_size=8,
    max_seq_length=2048,
    lora_rank=32,
    lora_layers=16,
    learning_rate=1e-4,
    num_train_epochs=5,

    # Override conservative defaults for dedicated training machine
    max_memory_gb=48,           # Use up to 48GB
    memory_hard_limit_gb=64,    # Hard stop at 64GB (leaves 64GB for OS/other)
    cpu_capacity_percent=0.95,  # Aggressive: stop only if > 95% loaded
)
```

### Quick Prototyping
```python
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    augment_factor=10,  # Smaller dataset
    num_train_epochs=1,
)
```

### High-Quality Production
```python
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    augment_factor=100,  # Larger dataset
    num_train_epochs=3,
    batch_size=2,
    temperature=0.5,  # Less diversity (more consistent)
    lora_rank=12,      # Decent expressiveness
)
```

## What's Fixed by Design

NanoDistill makes opinionated choices in these areas:

### Chain-of-Thought Format
- Fixed: `<thinking>` XML-style tags
- Rationale: Universal across models, easy to parse
- Cannot be changed (would require extensive refactoring)

### Training Data Structure
- Fixed: User → Assistant conversation format
- Fixed: Merged thinking + output
- Rationale: Aligns with standard LLM instruction tuning

### System Monitoring (New in v0.2.0: Now Configurable!)
- **Before**: 80% capacity threshold, 12GB memory limit (fixed)
- **Now**: Auto-detects system RAM and allows configuration
- **Default Behavior**: Conservative limits (12GB cap) prevent OOM on most systems
- **Customizable**: Override with `max_memory_gb`, `memory_hard_limit_gb`, `cpu_capacity_percent`
- **Rationale**: Prevent system crashes while allowing power users to customize for their hardware

### Optimizer
- Fixed: AdamW (via MLX-LM)
- Rationale: Industry standard for LLM fine-tuning

### Retry Behavior
- Fixed: 3 retries for API failures
- Rationale: Balance between robustness and speed

## Future Enhancements

Coming in future versions:
- Custom prompt templates (Tier 2)
- Advanced training config (Tier 2)
- Custom thinking format plugins (Tier 3)
```

#### 2.2 Create Design Decisions Document

**File**: `docs/DESIGN_DECISIONS.md`

```markdown
# Design Decisions

This document explains why certain choices are hardcoded in NanoDistill.

## Chain-of-Thought Format

**Decision**: Use `<thinking>` XML-style tags
**Alternatives Considered**:
- JSON format: `{"reasoning": "...", "answer": "..."}`
- Markdown: `**Thinking:** ... **Output:** ...`
- Plain text: Just concatenate with separator

**Chosen**: XML tags with newline delimiters

**Rationale**:
- Works consistently across all models
- Easy to parse with simple string operations
- Prevents conflicts with content that includes JSON or markdown
- Widely recognized by Claude models specifically

**Impact**: If you need a different format, you would need to:
1. Modify `data/formatter.py` (format_for_training)
2. Modify `teacher/client.py` (CoT trace parsing)
3. Modify inference code to extract thinking

**Future**: May support configurable templates in v0.3.0

---

## 80/20 Train/Validation Split

**Decision**: Fixed 80% training, 20% validation
**Rationale**:
- Industry standard for supervised learning
- Prevents overfitting on small datasets
- Sufficient validation data for meaningful metrics

**Customizable As Of**: v0.2.0 via `val_split` parameter

---

## Temperature = 0.7

**Decision**: Fixed temperature for synthetic generation
**Rationale**:
- 0.7 balances creativity and consistency
- Lower (0.2-0.3): Very consistent, less diverse
- Higher (1.2+): Very creative, potentially inconsistent
- 0.7: Good default for most tasks

**Customizable As Of**: v0.2.0 via `temperature` parameter

---

## AdamW Optimizer

**Decision**: Use AdamW (via MLX-LM)
**Rationale**:
- Industry standard for transformer fine-tuning
- Better than vanilla Adam for language models
- Handles large parameter spaces well

**Cannot Be Changed**: AdamW is hardcoded in MLX-LM CLI. If you need a different optimizer:
- Consider using MLX-LM directly instead of NanoDistill
- Submit an issue to request optimizer selection

---

## System Monitoring Thresholds

**Decision**:
- Default: Auto-detect system RAM, cap at 12GB hard limit, 80% capacity threshold
- Hard stop if memory > 12GB (default) or configured limit
- Pause if capacity > 80% (default) or configured threshold
- Limit CPU to 80% of available cores

**Rationale**:
- Prevents system crashes during training
- Allows other system processes to run
- 12GB is conservative default for safety margin (M1/M2 typically have 8-16GB)
- Auto-detection adapts to different system sizes

**Customizable As Of**: v0.2.0 via these parameters:
- `max_memory_gb` - Memory threshold before pause (default: auto-detect, capped at 12GB)
- `memory_hard_limit_gb` - Hard stop limit to prevent crash (default: auto-detect, capped at 12GB)
- `cpu_capacity_percent` - Capacity threshold for pause (default: 0.8 or 80%)

**Examples**:
```python
# M1 MacBook Air (8GB) - conservative
distill(..., max_memory_gb=6, memory_hard_limit_gb=7)

# M3 Max (128GB) - aggressive
distill(..., max_memory_gb=48, memory_hard_limit_gb=64, cpu_capacity_percent=0.95)

# Custom for your system
distill(...,
    max_memory_gb=24,              # Pause at 24GB
    memory_hard_limit_gb=32,       # Stop at 32GB
    cpu_capacity_percent=0.85,     # Pause at 85% capacity
)
```

**How to Check Your System Specs**:
```python
import psutil

total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
cpu_count = psutil.cpu_count()

print(f"Total RAM: {total_memory_gb:.1f} GB")
print(f"Available RAM: {available_memory_gb:.1f} GB")
print(f"CPU Cores: {cpu_count}")

# NanoDistill auto-detects:
max_memory = min(total_memory_gb * 0.9, 12)
hard_limit = min(total_memory_gb * 0.95, 12)

print(f"\nNanoDistill defaults (if not overridden):")
print(f"  max_memory_gb: {int(max_memory)}")
print(f"  memory_hard_limit_gb: {int(hard_limit)}")
print(f"  cpu_capacity_percent: 0.8 (80%)")
```

#### 2.3 Create Configuration Examples

**File**: `examples/configuration.py`

```python
"""
Configuration examples for different use cases.
Run any of these examples to see how configuration affects distillation.
"""

from nanodistill import distill

# Example seed data
SEED = [
    {"input": "What is 2+2?", "output": "4"},
    {"input": "What is 3+5?", "output": "8"},
    {"input": "What is 10×5?", "output": "50"},
]

INSTRUCTION = """You are a math tutor. Provide clear, step-by-step explanations."""


# 1. MINIMAL CONFIG (Use all defaults)
def example_minimal():
    """Minimal configuration - uses all defaults."""
    result = distill(
        name="math-minimal",
        seed=SEED,
        instruction=INSTRUCTION,
    )
    return result


# 2. MEMORY-CONSTRAINED (M1/M2 8GB)
def example_low_memory():
    """Low-memory configuration for M1 MacBook Air."""
    result = distill(
        name="math-low-memory",
        seed=SEED,
        instruction=INSTRUCTION,

        # Reduce batch size
        batch_size=1,

        # Shorter sequences
        max_seq_length=256,

        # Smaller LoRA adapters
        lora_rank=4,
        lora_layers=2,

        # Fewer synthetic examples
        augment_factor=20,
    )
    return result


# 3. HIGH-PERFORMANCE (M3 Pro/Max 36GB+)
def example_high_performance():
    """High-performance configuration for well-resourced systems."""
    result = distill(
        name="math-high-performance",
        seed=SEED,
        instruction=INSTRUCTION,

        # Larger batch size
        batch_size=4,

        # Longer sequences
        max_seq_length=1024,

        # Larger LoRA adapters
        lora_rank=16,
        lora_layers=8,

        # More training data
        augment_factor=100,

        # More training
        num_train_epochs=3,

        # Slightly lower learning rate (more stable)
        learning_rate=5e-5,
    )
    return result


# 4. QUICK PROTOTYPING
def example_quick_prototype():
    """Quick prototyping - fast iteration."""
    result = distill(
        name="math-quick",
        seed=SEED,
        instruction=INSTRUCTION,

        # Small dataset for fast iteration
        augment_factor=10,

        # Single epoch
        num_train_epochs=1,

        # Conservative batch size
        batch_size=1,
    )
    return result


# 5. PRODUCTION QUALITY
def example_production_quality():
    """Production-quality configuration - high quality output."""
    result = distill(
        name="math-production",
        seed=SEED,
        instruction=INSTRUCTION,

        # Large dataset
        augment_factor=100,

        # Multiple epochs
        num_train_epochs=3,

        # Conservative learning rate
        learning_rate=1e-5,

        # Longer context
        max_seq_length=512,

        # Lower temperature = more consistent
        temperature=0.5,
    )
    return result


# 6. EXPLORATION MODE (High diversity)
def example_exploration():
    """High diversity mode - explore different example variations."""
    result = distill(
        name="math-exploration",
        seed=SEED,
        instruction=INSTRUCTION,

        # High temperature = more diverse
        temperature=1.5,

        # More examples
        augment_factor=150,
    )
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_name = sys.argv[1]

        examples = {
            "minimal": example_minimal,
            "low-memory": example_low_memory,
            "high-performance": example_high_performance,
            "quick": example_quick_prototype,
            "production": example_production_quality,
            "exploration": example_exploration,
        }

        if example_name in examples:
            print(f"Running {example_name} configuration...")
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("Usage: python examples/configuration.py <example>")
        print("Examples: minimal, low-memory, high-performance, quick, production, exploration")
```

#### 2.4 Update README

**File**: `README.md` - Add Configuration Section

```markdown
## Configuration

NanoDistill works great with default settings, but you can customize key parameters:

### Basic Usage (Defaults)
```python
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
)
```

### Customized Setup (with Optional Parameters)
```python
result = distill(
    # Core/explicit parameters
    name="my-model",
    seed=[...],
    instruction="...",
    teacher="gpt-4o",                    # Optional, custom teacher
    student="mlx-community/Mistral-7B-Instruct-4bit",  # Optional, custom student
    augment_factor=100,                  # Optional, data amplification

    # Everything else via kwargs (clean API, powerful customization)

    # Training (already supported)
    batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=2,
    max_seq_length=512,

    # NEW: Generation & LoRA (Phase 1)
    temperature=0.7,                     # 0.0-2.0 (creativity)
    lora_rank=16,                        # 1-64 (model size)
    lora_layers=8,                       # 1-32 (which layers)

    # NEW: Data split (Phase 1)
    val_split=0.2,                       # 0.0-0.5 (validation ratio)

    # NEW: System config (Phase 1) - auto-detects if not specified
    max_memory_gb=12,                    # Pause if exceeds this
    memory_hard_limit_gb=16,             # Stop if exceeds this
    cpu_capacity_percent=0.8,            # Pause if system > 80% loaded
)
```

### System-Specific Configuration

NanoDistill auto-detects your system capabilities, but you can override via kwargs:

```python
# Let system auto-detect (recommended for most users)
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
)

# Customize for specific hardware
# M1 MacBook Air (8GB) - conservative
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    max_memory_gb=6,
    memory_hard_limit_gb=7,
)

# M3 Max (128GB) - aggressive
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    max_memory_gb=48,
    memory_hard_limit_gb=64,
    cpu_capacity_percent=0.95,
)
```

See [Configuration Guide](docs/CONFIGURATION.md) for detailed parameter reference, system specs, and use case examples.
```

#### 2.5 Update CLAUDE.md

**File**: `CLAUDE.md` - Add to Development Commands section

```markdown
### Configuration

```bash
# View all configurable parameters
python -c "from nanodistill import distill; help(distill)"

# Check your system capabilities
python -c "
import psutil
total_ram = psutil.virtual_memory().total / (1024**3)
available_ram = psutil.virtual_memory().available / (1024**3)
cpus = psutil.cpu_count()
print(f'System: {total_ram:.1f}GB RAM, {cpus} CPU cores, {available_ram:.1f}GB available')
"

# Example: Low-memory setup
python examples/configuration.py low-memory

# Example: High-performance setup
python examples/configuration.py high-performance
```

See `docs/CONFIGURATION.md` for comprehensive configuration guide with examples for different systems (M1/M2/M3, 8GB-128GB+ RAM).
```

---

### Phase 3: Advanced Features (Post-Launch, Optional)

**Timeline**: 8-12 hours
**Impact**: Medium - For power users who need more control

#### 3.1 Thinking Format Templates

**File**: `src/nanodistill/config.py`

Add configuration for thinking format:

```python
class DistillationConfig(BaseModel):
    # ... existing fields ...

    thinking_format: str = Field(
        default="xml",
        description="Format for thinking tags: xml, markdown, json, plain",
        pattern="^(xml|markdown|json|plain)$",
    )
```

**File**: `src/nanodistill/data/formatter.py`

Add template-based formatting:

```python
def get_thinking_template(format_type: str) -> str:
    """Get template for thinking/output format."""
    templates = {
        "xml": "<thinking>\n{thinking}\n</thinking>\n\n{output}",
        "markdown": "**Thinking:**\n{thinking}\n\n**Output:**\n{output}",
        "json": '{{"reasoning": "{thinking}", "answer": "{output}"}}',
        "plain": "{thinking}\n\n{output}",
    }
    return templates.get(format_type, templates["xml"])

def format_for_training(trace: ThinkingTrace, tokenizer, config) -> Dict:
    """Format with configurable thinking template."""
    template = get_thinking_template(config.thinking_format)
    content = template.format(thinking=trace.thinking, output=trace.output)

    conversation = [
        {"role": "user", "content": trace.input},
        {"role": "assistant", "content": content},
    ]
    # ... rest of function
```

#### 3.2 Custom Prompt Overrides

**File**: `src/nanodistill/config.py`

Add optional prompt configuration:

```python
class DistillationConfig(BaseModel):
    # ... existing fields ...

    custom_cot_system_prompt: Optional[str] = Field(
        default=None,
        description="Override default CoT system prompt",
    )

    custom_policy_extraction_prompt: Optional[str] = Field(
        default=None,
        description="Override policy extraction system prompt",
    )

    custom_synthetic_generation_prompt: Optional[str] = Field(
        default=None,
        description="Override synthetic generation system prompt",
    )
```

**File**: `src/nanodistill/teacher/prompts.py`

Add prompt getters:

```python
def get_cot_system_prompt(config: DistillationConfig) -> str:
    """Get CoT system prompt (custom or default)."""
    return config.custom_cot_system_prompt or COT_SYSTEM_PROMPT

def get_policy_extraction_system_prompt(config: DistillationConfig) -> str:
    """Get policy extraction prompt (custom or default)."""
    return config.custom_policy_extraction_prompt or POLICY_EXTRACTION_SYSTEM_PROMPT

def get_synthetic_generation_system_prompt(config: DistillationConfig) -> str:
    """Get synthetic generation prompt (custom or default)."""
    return config.custom_synthetic_generation_prompt or SYNTHETIC_GENERATION_SYSTEM_PROMPT
```

---

## Part 4: Implementation Checklist

### ✅ Phase 1 (Pre-Release)

**Critical Fixes + New Fields** (3-4 hours):

*Adding 7 new config fields:*
- [ ] `synthesis_temperature: float = 0.7` (was hardcoded in teacher/client.py)
- [ ] `lora_rank: int = 8` (currently uses undocumented getattr fallback)
- [ ] `lora_layers: int = 4` (currently uses undocumented getattr fallback)
- [ ] `val_split: float = 0.2` (new feature)
- [ ] `max_memory_gb: int` (new feature, auto-detect)
- [ ] `memory_hard_limit_gb: int` (new feature, auto-detect)
- [ ] `cpu_capacity_percent: float = 0.8` (new feature)

*Add helper methods to DistillationConfig:*
- [ ] `get_system_defaults()` - static method using psutil for auto-detection
- [ ] `get_litellm_kwargs()` - returns dict of LiteLLM parameters
- [ ] `get_mlx_lm_kwargs()` - returns dict of MLX-LM parameters
- [ ] Add field validators for numeric ranges
- [ ] Add cross-field validator: `memory_hard_limit_gb >= max_memory_gb`

**Integration - Fix Critical Bugs** (2-3 hours):

*Teacher/Client - Fix hardcoded temperature:*
- [ ] `teacher/client.py:262` - Replace `temperature=0.7` with `temperature=getattr(self.config, "synthesis_temperature", 0.7)`
- [ ] `teacher/client.py:320` - Same fix

*Trainer - Fix max_seq_length bug (CRITICAL):*
- [ ] `distiller/trainer.py:278-290` - Add `"--max-seq-length", str(max_seq_length)` to MLX-LM CLI
- [ ] `distiller/trainer.py:151-152` - Remove manual truncation (MLX-LM now handles it)

*Trainer - Fix undocumented LoRA params:*
- [ ] `distiller/trainer.py:123-124` - Change from `getattr()` to `self.config.lora_rank` (now validated)
- [ ] `distiller/trainer.py:211-212` - Same change

*Trainer - Fix system monitoring config:*
- [ ] `distiller/trainer.py` system monitoring function - Use config values instead of hardcoded 12GB/80%

*Dependencies:*
- [ ] Add `psutil>=5.8.0` to requirements.txt

**Documentation** (1 hour):
- [ ] Update `core.py` docstring with kwargs examples
- [ ] Update type hints and docstrings in config.py

**NO changes needed to function signature** - `**kwargs` already handles everything!

### ✅ Phase 2 (Pre-Release)

- [ ] Create `docs/CONFIGURATION.md` with comprehensive reference
- [ ] Create `docs/DESIGN_DECISIONS.md` explaining hardcoded choices
- [ ] Create `examples/configuration.py` with 6 use case examples
- [ ] Update `README.md` with configuration section
- [ ] Update `CLAUDE.md` with new configuration examples
- [ ] Add configuration section to API documentation

### ✅ Phase 3 (Post-Launch, Optional)

- [ ] Add thinking_format parameter and template support
- [ ] Add custom prompt override fields
- [ ] Create prompt template documentation
- [ ] Add examples for custom thinking formats
- [ ] Add examples for custom prompts

### 🧪 Testing (All Phases)

- [ ] Test all new config parameters with valid values
- [ ] Test config parameter validation (min/max ranges)
- [ ] Test backward compatibility (old code still works)
- [ ] Test with different hardware (M1, M3 Pro, M3 Max)
- [ ] Test with different model combinations
- [ ] Run full test suite with coverage

---

## Part 5: Verification Checklist

After implementation is complete:

### Documentation
- [ ] All parameters documented with types, defaults, ranges
- [ ] Design decisions explain every hardcoded choice
- [ ] Examples show 5+ use cases (memory, performance, speed, quality, etc.)
- [ ] Configuration guide has troubleshooting section
- [ ] README updated with configuration section

### Code Quality
- [ ] All new parameters validated with Pydantic
- [ ] Config changes tested with unit tests
- [ ] Type hints complete and correct
- [ ] Docstrings updated
- [ ] No breaking changes to existing API

### User Experience
- [ ] Default values unchanged (backward compatible)
- [ ] Error messages helpful when invalid config provided
- [ ] Configuration examples run without errors
- [ ] Progressive disclosure (simple default, powerful when needed)

---

## Part 6: Summary Table

### What Users Can Configure Now vs. After

| **Category** | **Currently** | **After Phase 1** | **After Phase 3** |
|---|---|---|---|
| **Explicit Params** | 8 params | 8 params (unchanged) | 8 params |
| **Training** | ⚠️ batch_size, lr, epochs (kwargs) | ✅ All via kwargs | ✅ All via kwargs |
| **LoRA** | ⚠️ rank, layers (kwargs) | ✅ Configurable via kwargs | ✅ Configurable via kwargs |
| **Temperature** | ❌ Fixed 0.7 | ✅ Configurable via kwargs | ✅ Configurable via kwargs |
| **Val Split** | ❌ Fixed 80/20 | ✅ Configurable via kwargs | ✅ Configurable via kwargs |
| **System Memory** | ❌ Fixed 12GB | ✅ Auto-detect + kwargs | ✅ Auto-detect + kwargs |
| **System CPU** | ❌ Fixed 80% threshold | ✅ Configurable via kwargs | ✅ Configurable via kwargs |
| **CoT Format** | ❌ Fixed XML | ❌ Fixed XML | ✅ Template-based (Phase 3) |
| **Prompts** | ❌ Hardcoded | ❌ Hardcoded | ✅ Override option (Phase 3) |
| **Sampling** | ❌ Fixed | ✅ Temperature only | 🔮 More controls (Phase 3+) |
| **Optimizer** | ❌ AdamW only | ❌ AdamW only | ❌ MLX-LM limitation |
| **Parsing** | ❌ 3 fixed strategies | ❌ 3 fixed strategies | 🔮 Future enhancement |

**Legend**:
- ✅ Fully configurable (via explicit params or kwargs)
- ⚠️ Works but undocumented
- ❌ Hardcoded
- 🔮 Future enhancement

### Signature Remains Clean

```python
# Always this simple
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
)

# With optional overrides via kwargs
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    batch_size=2,
    learning_rate=5e-5,
    temperature=0.7,
    max_memory_gb=24,
    # ... etc
)
```

**No function signature bloat - all power users need is documented in kwargs!**

---

## Part 7: Recommendation

### 🎯 **Start with Option B (Moderate) - Phased Implementation**

**With Clean kwargs Architecture**

#### Why This Approach?

1. **Phase 1 (Pre-Release)** - 3-5 hours implementation
   - ✅ Addresses 95% of user needs
   - ✅ Minimal code changes (only add fields to DistillationConfig)
   - ✅ **No signature changes** - uses existing `**kwargs` pattern
   - ✅ Comprehensive documentation
   - ✅ Maintains "batteries included" philosophy

2. **Phase 2 (Post-Launch)** - If needed based on feedback
   - ✅ Monitor user feedback
   - ✅ Add thinking templates if requested
   - ✅ Add prompt overrides if power users demand

3. **Phase 3 (If Needed)** - Only if proven necessary
   - ✅ Plugin system only if community shows clear need
   - ✅ Don't over-engineer prematurely

#### API Design Benefits

**Before (No Configuration)**:
```python
result = distill(name="my-model", seed=[...], instruction="...")
```

**After Phase 1 (Powerful but Clean)**:
```python
# Simple by default
result = distill(name="my-model", seed=[...], instruction="...")

# Power users add kwargs as needed
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    batch_size=2,
    temperature=0.7,
    max_memory_gb=24,
    # ... any of 20+ optional params
)
```

**Function signature never changes!** 8 explicit params + flexible kwargs.

#### This approach balances:

- ✅ **Ease of use** - Simple function signature, clear defaults
- ✅ **Transparency** - All configurable params documented
- ✅ **Flexibility** - 20+ tunable parameters via kwargs
- ✅ **Maintainability** - No signature bloat, clean architecture
- ✅ **Extensibility** - Easy to add more params in future

---

## Part 7.5: Critical External Library Integration Issues

### 🚨 Issues Found During Library Call Analysis

#### 1. **Max Seq Length Bug (CRITICAL)**
**Location**: `distiller/trainer.py:141, 278-290`

**Issue**: `max_seq_length` is read from config but **NEVER passed to MLX-LM CLI**

```python
# Line 141: We read it
max_seq_length = getattr(self.config, "max_seq_length", 512)

# Line 151-152: We use it for manual truncation
if len(tokens) > max_seq_length:
    tokens = tokens[:max_seq_length]

# Lines 278-290: But we DON'T pass it to MLX-LM!
cmd = [
    "python", "-m", "mlx_lm", "lora",
    # ... missing: "--max-seq-length", str(max_seq_length)
]
```

**Fix**: Pass `--max-seq-length` to MLX-LM CLI instead of manual truncation

---

#### 2. **LoRA Parameters Not in Config (CRITICAL)**
**Location**: `distiller/trainer.py:123-124, 211-212`

**Issue**: `lora_rank` and `lora_layers` use `getattr()` fallback but aren't in `DistillationConfig`

```python
# Current: Using getattr with magic defaults
lora_rank = getattr(self.config, "lora_rank", 8)      # Undocumented!
lora_layers = getattr(self.config, "lora_layers", 4)  # Undocumented!
```

**Fix**: Add formal Pydantic fields to `DistillationConfig` with validation

---

#### 3. **Hardcoded Temperature in LiteLLM (HIGH PRIORITY)**
**Location**: `teacher/client.py:262, 320`

**Issue**: Synthetic generation uses fixed `temperature=0.7` hardcoded in two places

```python
# Line 262
response = completion(
    ...,
    temperature=0.7,  # HARDCODED - should be from config!
)

# Line 320 - same issue
response = self.client.chat.completions.create(
    ...,
    temperature=0.7,  # HARDCODED
)
```

**Fix**: Move to config as `synthesis_temperature` parameter

---

#### 4. **Library Parameters Not Exposed (MEDIUM PRIORITY)**

**LiteLLM**:
- `top_p`, `top_k` for sampling diversity
- `max_tokens` for response length control
- `timeout` for API calls
- `seed` for reproducibility

**Instructor**:
- `mode` in `model_dump()` - hardcoded to `'json'` (line 334)
- Validation retry logic

**HuggingFace Tokenizer**:
- `max_length` - currently doing manual truncation instead
- `padding` strategy
- `return_attention_mask` for future batching

**MLX-LM CLI**:
- `--lora-dropout` (not exposed)
- `--gradient-checkpointing` (not exposed)
- `--seed` for reproducibility
- `--warmup-steps` (not exposed)
- `--weight-decay` (not exposed)

---

### Recommended Pattern: Pass kwargs to External Libraries

Instead of hardcoding values, follow this pattern:

1. **Config fields with defaults** - Define in Pydantic with sensible defaults
2. **Pass through to libraries** - Use `getattr()` or direct access
3. **Support kwargs passthrough** - Allow users to pass library-specific params

#### Pattern Comparison

**BAD - hardcoded, not configurable**:
```python
response = completion(
    model=self.model,
    messages=[...],
    temperature=0.7,  # HARDCODED - can't change!
)
```

**GOOD - from config with defaults**:
```python
response = completion(
    model=self.model,
    messages=[...],
    temperature=getattr(self.config, "synthesis_temperature", 0.7),
)
```

**BETTER - with optional kwargs passthrough**:
```python
# Users can pass extra params they discover
# Example: distill(..., litellm_kwargs={"top_p": 0.9, "timeout": 30})

litellm_extra = self.config.get("litellm_kwargs", {})
response = completion(
    model=self.model,
    messages=[...],
    temperature=self.config.synthesis_temperature,
    **litellm_extra,  # Pass any library params user wants!
)
```

---

## Part 8: Implementation Overview

### Generic Kwargs Passthrough Pattern (RECOMMENDED)

Instead of explicitly defining every possible parameter, allow users to pass ANY parameter via generic kwargs dicts:

```python
class DistillationConfig(BaseModel):
    # ... existing fields ...

    # === Core parameters we explicitly define ===
    temperature: float = Field(default=0.7, description="Synthetic generation temperature")
    lora_rank: int = Field(default=8, description="LoRA adapter rank")
    lora_layers: int = Field(default=4, description="LoRA target layers")
    val_split: float = Field(default=0.2, description="Validation split ratio")

    # === System parameters ===
    max_memory_gb: int = Field(default_factory=lambda: get_system_defaults()[0])
    memory_hard_limit_gb: int = Field(default_factory=lambda: get_system_defaults()[1])
    cpu_capacity_percent: float = Field(default=0.8)

    # === GENERIC KWARGS - Allow ANY library parameter ===
    litellm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any LiteLLM parameter (temperature, top_p, timeout, seed, etc.)"
    )
    mlx_lm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any MLX-LM parameter (lora_dropout, warmup_steps, seed, etc.)"
    )

    def get_litellm_synthesis_kwargs(self) -> Dict[str, Any]:
        """Get kwargs dict for LiteLLM synthetic generation calls.

        Merges explicit config with user-provided kwargs (user overrides explicit).
        """
        base = {"temperature": self.temperature}
        # User-provided kwargs override everything
        base.update(self.litellm_kwargs)
        return base

    def get_mlx_lm_cli_args(self) -> List[str]:
        """Get CLI arguments for MLX-LM training.

        Converts mlx_lm_kwargs dict to CLI arguments.
        Example: {"warmup_steps": 100} -> ["--warmup-steps", "100"]
        """
        args = []
        for key, value in self.mlx_lm_kwargs.items():
            # Convert underscore to hyphen (Python style to CLI style)
            cli_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(cli_key)
            else:
                args.extend([cli_key, str(value)])
        return args
```

#### How to Use in Code

**In teacher/client.py**:
```python
# Line 262, 320: Replace hardcoded temperature
response = completion(
    model=self.model,
    messages=[...],
    **self.config.get_litellm_synthesis_kwargs(),  # Unpack all LiteLLM params!
)
```

**In distiller/trainer.py**:
```python
# Line 278-290: Add MLX-LM CLI args from config
cmd = [
    "python", "-m", "mlx_lm", "lora",
    "--model", self.student_model,
    "--train",
    "--data", str(data_dir),
    "--iters", str(iters),
    "--batch-size", str(batch_size),
    "--learning-rate", str(learning_rate),
    "--num-layers", str(self.config.lora_layers),
    "--adapter-path", str(adapter_path),
    "--steps-per-report", str(steps_per_report),
    "--save-every", str(iters),
    "--max-seq-length", str(max_seq_length),  # FIX: Was missing!
]
# Add any user-provided MLX-LM parameters
cmd.extend(self.config.get_mlx_lm_cli_args())

result = subprocess.run(cmd, check=True, ...)
```

#### User Examples

```python
# Basic usage - all defaults
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
)

# Custom synthesis parameters via litellm_kwargs
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    temperature=0.5,  # Explicit config
    litellm_kwargs={
        "top_p": 0.9,
        "timeout": 60,
        "seed": 42,  # For reproducibility
        "api_base": "https://custom-endpoint.com",  # Override default
    },
)

# Custom training parameters via mlx_lm_kwargs
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    lora_rank=16,  # Explicit config
    mlx_lm_kwargs={
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "seed": 42,
        "gradient_checkpointing": True,
    },
)

# Everything customized
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",

    # Explicit parameters
    temperature=0.7,
    lora_rank=16,
    lora_layers=8,
    val_split=0.2,

    # Any LiteLLM param user wants
    litellm_kwargs={
        "top_p": 0.95,
        "timeout": 120,
    },

    # Any MLX-LM param user wants
    mlx_lm_kwargs={
        "lora_dropout": 0.1,
        "warmup_steps": 500,
    },
)
```

---

### What Changes?

#### Code Changes (Minimal + Clean)
- ✅ Add 7 new fields to `DistillationConfig`:
  - `temperature: float = 0.7` (rename from hardcoded)
  - `lora_rank: int = 8` (move from getattr)
  - `lora_layers: int = 4` (move from getattr)
  - `val_split: float = 0.2` (NEW)
  - `max_memory_gb: int` (NEW, auto-detect)
  - `memory_hard_limit_gb: int` (NEW, auto-detect)
  - `cpu_capacity_percent: float = 0.8` (NEW)
  - `litellm_kwargs: Dict[str, Any] = {}` (NEW - generic passthrough)
  - `mlx_lm_kwargs: Dict[str, Any] = {}` (NEW - generic passthrough)

- ✅ Add 2 helper methods (~50 lines):
  - `get_litellm_synthesis_kwargs()` - merge explicit + kwargs
  - `get_mlx_lm_cli_args()` - convert dict to CLI arguments

- ✅ Fix critical bugs:
  - Replace `temperature=0.7` hardcoded with config value (2 places)
  - Add `--max-seq-length` to MLX-LM CLI (CRITICAL FIX)
  - Replace undocumented `getattr()` with explicit config fields

- ✅ Update system monitoring to use config values

- ✅ Add `psutil>=5.8.0` to requirements.txt

**Total code changes**: ~100 lines (minimal!)
**Key benefit**: Users can pass ANY parameter without us having to explicitly define it!

#### What Doesn't Change
- ✅ Function signature - already has `**kwargs`!
- ✅ User-facing API - works same as before
- ✅ Defaults - all maintained for backward compatibility
- ✅ Existing code - still works without any changes

#### Documentation Changes (Important)
- ✅ Configuration reference with all parameters
- ✅ Design decisions explaining hardcoded choices
- ✅ Examples for different systems
- ✅ System auto-detection explanation
- ✅ README section on configuration

---

---

## Implementation Approved ✅

Based on your feedback:

1. ✅ **Fix 3 critical bugs** - Max seq length, hardcoded temperatures, undocumented LoRA params
2. ✅ **Generic kwargs passthrough** - Allow ANY library parameter via `litellm_kwargs` and `mlx_lm_kwargs`
3. ✅ **Include in Phase 1** - Generic dicts give maximum flexibility without maintaining long parameter lists
4. ✅ **Keep defaults as-is** - No changes to current default values

**Approach Summary**:
- 9 new config fields (core parameters + system settings + generic dicts)
- 3 helper methods for merging/converting kwargs
- Fix all 3 critical bugs
- Support ANY parameter users discover without code changes needed

---

## Part 9: Implementation Checklist - All Changes

### Phase 1 Config Changes (src/nanodistill/config.py)

**Add to DistillationConfig class (9 new fields):**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `temperature` | float | 0.7 | Synthetic generation temp (was hardcoded) |
| `lora_rank` | int | 8 | LoRA adapter rank (was undocumented getattr) |
| `lora_layers` | int | 4 | LoRA target layers (was undocumented getattr) |
| `val_split` | float | 0.2 | Validation/train split ratio (NEW) |
| `max_memory_gb` | int | auto | Memory pause threshold (NEW, auto-detect) |
| `memory_hard_limit_gb` | int | auto | Memory hard stop (NEW, auto-detect) |
| `cpu_capacity_percent` | float | 0.8 | CPU threshold (NEW) |
| `litellm_kwargs` | Dict[str, Any] | {} | ANY LiteLLM parameter (NEW - generic) |
| `mlx_lm_kwargs` | Dict[str, Any] | {} | ANY MLX-LM parameter (NEW - generic) |

**Add methods:**
- `get_system_defaults()` - static method for auto-detection with psutil
- `get_litellm_synthesis_kwargs()` - merge explicit config + litellm_kwargs
- `get_mlx_lm_cli_args()` - convert mlx_lm_kwargs dict to CLI args

---

### Phase 1 Code Changes (By File)

#### src/nanodistill/teacher/client.py

| Line | Current | Change | Why |
|------|---------|--------|-----|
| 87-95 | `completion(..., max_retries=...)` | Add `**self.config.get_litellm_synthesis_kwargs()` | FIX: Hardcoded temp, allow all LiteLLM params |
| 262 | `temperature=0.7` | Use `**self.config.get_litellm_synthesis_kwargs()` | FIX: Hardcoded |
| 320 | `temperature=0.7` | Use `**self.config.get_litellm_synthesis_kwargs()` | FIX: Hardcoded |

#### src/nanodistill/distiller/trainer.py

| Line | Current | Change | Why |
|------|---------|--------|-----|
| 123-124 | `getattr(self.config, "lora_rank", 8)` | `self.config.lora_rank` | FIX: Now in config |
| 211-212 | `getattr(self.config, "lora_layers", 4)` | `self.config.lora_layers` | FIX: Now in config |
| 219 | `split_idx = int(len(train_data) * 0.8)` | Use `1 - self.config.val_split` | NEW: Configurable |
| 278-290 | Missing max_seq_length CLI arg | Add `"--max-seq-length", str(max_seq_length)` | **CRITICAL FIX** |
| 151-152 | Manual truncation | Remove (MLX-LM handles via --max-seq-length) | FIX: Redundant |
| 177-206 | Hardcoded 12GB/80% thresholds | Use config values | NEW: Configurable |

#### src/nanodistill/requirements.txt

- Add `psutil>=5.8.0` for system auto-detection

---

## Summary: The Clean kwargs Approach

**Why this is better**:

```python
# ❌ Before (if we added to signature):
def distill(name, seed, instruction, teacher=..., student=..., augment_factor=...,
            output_dir=..., response_model=...,
            temperature=..., lora_rank=..., lora_layers=..., val_split=...,
            max_memory_gb=..., memory_hard_limit_gb=..., cpu_capacity_percent=...):
    # 15+ parameters! Signature is intimidating

# ✅ After (using kwargs):
def distill(name, seed, instruction, teacher=..., student=..., augment_factor=...,
            output_dir=..., response_model=..., **kwargs):
    # Clean! All the power via kwargs
```

**Benefits**:
- 📦 Signature stays clean (8 explicit + flexible kwargs)
- 📚 All parameters documented in one place
- 🚀 Easy to add future parameters
- 💪 Power users get all the control they need
- 🎯 Simple users see 3 required params, optional rest

---

## Next Steps - Ready for Implementation

1. ✅ Analysis complete - approach approved
2. 📝 **Implement Phase 1**:
   - Add 9 new fields to `DistillationConfig` (~40 lines)
   - Add 3 helper methods (~50 lines)
   - Fix 3 critical bugs (~10 lines)
   - Update requirements.txt
3. 📖 Write configuration documentation
4. 🧪 Update tests for new config parameters
5. ✨ Verify all library parameters work via kwargs dicts
6. 📢 Release v0.2.0 with new configuration system

---

## Summary: Clean, Flexible, Maintainable

### The Final Approach

**Explicit Parameters** (what we define):
- 7 core fields: `temperature`, `lora_rank/layers`, `val_split`, memory/cpu config
- 2 kwargs dicts: `litellm_kwargs`, `mlx_lm_kwargs`
- 3 helper methods: merge config + kwargs and convert to CLI args

**User Experience**:
```python
# Simple
distill(name="m", seed=[...], instruction="...")

# Customized
distill(name="m", seed=[...], instruction="...",
    temperature=0.5,
    lora_rank=16,
    litellm_kwargs={"top_p": 0.9, "timeout": 120},
    mlx_lm_kwargs={"warmup_steps": 500, "seed": 42},
)

# Any LiteLLM/MLX-LM parameter supported!
```

**Benefits**:
- ✅ No parameter bloat (9 fields + 2 dicts, not 30+ fields)
- ✅ Fixes all 3 critical bugs
- ✅ Supports ANY library parameter
- ✅ Easy to extend (no code changes needed for new params)
- ✅ Clean function signature (stays the same!)
- ✅ Backward compatible (all defaults unchanged)
- ✅ Maintainable (helper methods handle merging/conversion)

### Parameters Now Supported (via kwargs dicts)

**LiteLLM** (via `litellm_kwargs`):
```
temperature, top_p, top_k, max_tokens, timeout, seed,
api_base, fallback_function, context_window_fallback, ...
(ALL LiteLLM parameters supported!)
```

**MLX-LM** (via `mlx_lm_kwargs`):
```
lora_dropout, gradient_checkpointing, warmup_steps,
weight_decay, seed, use_dora, adam_beta_1, adam_beta_2, ...
(ALL MLX-LM parameters supported!)
```

**HuggingFace/Instructor** (future expansion):
- Could add `tokenizer_kwargs`, `instructor_kwargs` if needed

---

