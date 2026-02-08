"""Inference runners for teacher and student models."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import DistillationConfig
from ..teacher.client import TeacherClient
from ..teacher.schemas import ThinkingTrace

logger = logging.getLogger(__name__)


def run_teacher_inference(
    examples: List[Dict[str, str]],
    teacher_model: str,
    instruction: str,
    config: DistillationConfig,
) -> List[ThinkingTrace]:
    """Run teacher model inference on validation examples.

    Uses TeacherClient to generate Chain-of-Thought traces.

    Args:
        examples: List of dicts with 'input' field
        teacher_model: Teacher model name (e.g., "claude-sonnet-4-5")
        instruction: Task instruction for the teacher
        config: DistillationConfig with synthesis parameters

    Returns:
        List of ThinkingTrace objects from teacher

    Raises:
        RuntimeError: If teacher inference fails
    """
    logger.info(f"Running teacher inference with {teacher_model}")

    try:
        client = TeacherClient(teacher_model, config=config)
        traces = client.synthesize_cot(examples, instruction)
        logger.info(f"Generated {len(traces)} teacher traces")
        return traces

    except Exception as e:
        logger.error(f"Teacher inference failed: {e}")
        raise RuntimeError(f"Teacher inference failed: {str(e)}") from e


def run_student_inference(
    examples: List[Dict[str, str]],
    student_model: str,
    adapter_path: Optional[Path],
    instruction: str,
) -> List[Dict[str, str]]:
    """Run student model inference with optional trained adapters.

    Uses MLX to run the student model with LoRA adapters on inputs.
    If adapter_path is None, runs base model without adapters.

    Args:
        examples: List of dicts with 'input' field
        student_model: Student model ID (e.g., "Qwen/Qwen2.5-7B-Instruct-MLX-4bit")
        adapter_path: Path to saved LoRA adapters (None for base model)
        instruction: Task instruction (for context/prompt engineering)

    Returns:
        List of dicts with 'thinking' and 'output' fields

    Raises:
        RuntimeError: If student inference fails
    """
    model_type = "base model" if adapter_path is None else "fine-tuned model"
    logger.info(f"Running {model_type} inference with {student_model}")

    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        # Load model with adapters (or base model if adapter_path is None)
        logger.info(f"Loading model from {student_model}")
        if adapter_path is not None and adapter_path.exists():
            model, tokenizer = load(student_model, adapter_path=str(adapter_path))
            logger.info(f"Loaded adapters from {adapter_path}")
        else:
            if adapter_path is None:
                logger.info("Loading base model (no adapters)")
            else:
                logger.warning(f"Adapter path not found: {adapter_path}, loading base model")
            model, tokenizer = load(student_model)

        # Create sampler for deterministic inference
        sampler = make_sampler(temp=0.0)

        results = []

        for i, example in enumerate(examples):
            if i % max(1, len(examples) // 10) == 0:
                logger.info(f"Processing example {i + 1}/{len(examples)}")

            # Format prompt with chat template if available
            if hasattr(tokenizer, "apply_chat_template") and hasattr(tokenizer, "chat_template"):
                messages = [{"role": "user", "content": example["input"]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # Fallback to simple formatting
                prompt = f"Input: {example['input']}\n\nThinking:"

            # Generate response
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=512,
                sampler=sampler,
            )

            # Parse thinking and output
            thinking, output = parse_student_output(response)

            results.append({
                "thinking": thinking,
                "output": output,
            })

        logger.info(f"Generated {len(results)} student outputs")
        return results

    except ImportError as e:
        logger.error("MLX not installed")
        raise RuntimeError("MLX not installed. Install with: pip install mlx mlx-lm") from e
    except Exception as e:
        logger.error(f"Student inference failed: {e}")
        raise RuntimeError(f"Student inference failed: {str(e)}") from e


def parse_student_output(raw: str) -> Tuple[str, str]:
    """Extract thinking and output from student model response.

    Handles multiple output formats:
    - "<thinking>...</thinking> JSON/text" (thinking tags + JSON/text)
    - "Thinking: ... Output: ..." (explicit labels)
    - "<thinking>...</thinking> <answer>...</answer>" (tags)
    - Plain text with no structure (no thinking, all output)

    Args:
        raw: Raw model response string

    Returns:
        Tuple of (thinking, output) strings
    """
    import json

    thinking = ""
    output = ""

    # Extract thinking tags if present
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Get text after thinking tags (if any)
    remainder = raw
    if thinking_match:
        remainder = raw[thinking_match.end():].strip()

    # Try to extract and parse JSON from remainder
    # Look for { and try to find matching }
    if "{" in remainder:
        try:
            # Find the first { and try to extract JSON
            start_idx = remainder.index("{")
            # Try progressively longer substrings to find valid JSON
            for end_idx in range(len(remainder), start_idx, -1):
                potential_json = remainder[start_idx:end_idx].strip()
                # Remove trailing periods, commas, etc that might be added
                potential_json = potential_json.rstrip(".,;:")
                try:
                    parsed = json.loads(potential_json)
                    # Successfully parsed JSON
                    output = json.dumps(parsed)  # Normalize it
                    return thinking, output
                except (json.JSONDecodeError, ValueError):
                    continue
        except (ValueError, IndexError):
            pass

    # Try answer tags
    answer_match = re.search(r"<answer>(.*?)</answer>", remainder, re.DOTALL)
    if answer_match:
        output = answer_match.group(1).strip()
    else:
        # Try explicit labels (only if no thinking was found)
        if not thinking:
            thinking_match_label = re.search(
                r"Thinking:\s*(.*?)(?:\n\n|Output:)", raw, re.DOTALL
            )
            output_match = re.search(r"Output:\s*(.*?)$", raw, re.DOTALL)

            if thinking_match_label:
                thinking = thinking_match_label.group(1).strip()
            if output_match:
                output = output_match.group(1).strip()

        # Fallback: use remainder or full text as output
        if not output:
            output = remainder if remainder else raw.strip()

    return thinking, output
