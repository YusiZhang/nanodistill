"""Core orchestrator for baseline evaluation.

Compares base and fine-tuned student models against ground truth teacher outputs
(no API calls - uses cached training data).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DistillationConfig
from ..data.loader import load_traces_from_jsonl
from .html_report import generate_html_report
from .inference import run_student_inference
from .metrics import (
    aggregate_metrics,
    compare_json_fields,
    compute_diff_html,
    compute_similarity,
)
from .schemas import BaselineResult, ComparisonExample

logger = logging.getLogger(__name__)


def evaluate_baseline(
    run_name: str,
    output_dir: str = "./outputs",
    max_examples: Optional[int] = None,
) -> BaselineResult:
    """Run baseline evaluation comparing base and fine-tuned models vs ground truth.

    Uses ground truth from training data (no API calls to teacher model).

    Workflow:
    1. Load validation data from outputs/{run_name}/traces_amplified.jsonl
       (ground truth = teacher outputs already in training data)
    2. Run base model (no adapters) inference
    3. Run fine-tuned model (with adapters) inference
    4. Compare both to ground truth, show improvement
    5. Generate interactive HTML report
    6. Return aggregated results

    Args:
        run_name: Distillation run identifier (used for paths)
        output_dir: Base output directory (default: "./outputs")
        max_examples: Limit evaluation to N examples (default: None = all)

    Returns:
        BaselineResult with metrics, examples, and HTML path

    Raises:
        ValueError: If validation data not found
        RuntimeError: If inference fails
    """
    output_base = Path(output_dir) / run_name
    amplified_path = output_base / "traces_amplified.jsonl"
    cache_path = output_base / "baseline_cache.jsonl"
    html_path = output_base / "baseline_report.html"

    logger.info(f"Starting baseline evaluation for {run_name}")

    # Load amplified traces (includes validation set + ground truth)
    if not amplified_path.exists():
        raise ValueError(f"Amplified data not found: {amplified_path}")

    all_traces = load_traces_from_jsonl(str(amplified_path))
    logger.info(f"Loaded {len(all_traces)} total traces from {amplified_path}")

    # Use validation split from config to extract validation examples
    config = _load_config(output_base)
    val_split = config.val_split
    split_idx = int(len(all_traces) * (1 - val_split))
    validation_traces = all_traces[split_idx:]  # Validation set is after split

    logger.info(
        f"Split at index {split_idx} (val_split={val_split}): "
        f"Loading {len(validation_traces)} validation examples"
    )

    # Limit examples if requested
    if max_examples:
        validation_traces = validation_traces[:max_examples]
        logger.info(f"Limited to {max_examples} examples")

    # Load or generate inference results
    if cache_path.exists():
        logger.info(f"Loading cached inference from {cache_path}")
        cache_data = _load_cache(cache_path)
        base_results = cache_data["base_results"]
        finetuned_results = cache_data["finetuned_results"]
        logger.info(
            f"Loaded {len(base_results)} base and "
            f"{len(finetuned_results)} fine-tuned results from cache"
        )
    else:
        logger.info("Running model inference (cache not found)")

        # Base model inference (no adapters)
        logger.info("Running base model inference...")
        examples_for_inference = [{"input": t.input} for t in validation_traces]
        logger.info(f"Prepared {len(examples_for_inference)} examples for inference")
        base_results = run_student_inference(
            examples=examples_for_inference,
            student_model=config.student,
            adapter_path=None,  # No adapters = base model
            instruction=config.instruction,
        )
        logger.info(f"Generated {len(base_results)} base model results")

        # Fine-tuned model inference (with adapters)
        logger.info("Running fine-tuned model inference...")
        finetuned_results = run_student_inference(
            examples=examples_for_inference,
            student_model=config.student,
            adapter_path=output_base / "adapters",
            instruction=config.instruction,
        )
        logger.info(f"Generated {len(finetuned_results)} fine-tuned model results")

        # Save cache
        _save_cache(cache_path, base_results, finetuned_results, config)
        logger.info(f"Saved cache to {cache_path}")

    # Compute comparisons and metrics
    logger.info("Computing comparison metrics...")
    logger.info(
        f"Starting comparison loop: {len(base_results)} base, "
        f"{len(finetuned_results)} fine-tuned, "
        f"{len(validation_traces)} ground truth"
    )
    comparisons = []
    try:
        comparison_count = 0
        for i, (base_result, finetuned_result, ground_truth) in enumerate(
            zip(base_results, finetuned_results, validation_traces)
        ):
            comparison_count += 1
            # Ground truth from training data
            ground_truth_output = ground_truth.output
            ground_truth_thinking = ground_truth.thinking

            # Base model outputs
            base_output = base_result.get("output", "")
            base_thinking = base_result.get("thinking", "")

            # Fine-tuned model outputs
            finetuned_output = finetuned_result.get("output", "")
            finetuned_thinking = finetuned_result.get("thinking", "")

            # Check if outputs are JSON and compare field-by-field
            is_json, student_field_matches = compare_json_fields(
                ground_truth_output, finetuned_output
            )
            is_json_base, base_field_matches = compare_json_fields(
                ground_truth_output, base_output
            )

            # If JSON outputs, use field match accuracy; otherwise 0 if no JSON
            if is_json:
                # Field-by-field comparison for fine-tuned
                if student_field_matches:
                    field_match_rate = sum(
                        student_field_matches.values()
                    ) / len(student_field_matches)
                    exact_match = field_match_rate == 1.0
                else:
                    exact_match = False
            else:
                # No JSON - set accuracy to 0
                exact_match = False

            if is_json_base:
                # Field-by-field comparison for base
                if base_field_matches:
                    base_field_match_rate = sum(
                        base_field_matches.values()
                    ) / len(base_field_matches)
                    base_exact_match = base_field_match_rate == 1.0
                else:
                    base_exact_match = False
            else:
                # No JSON - set accuracy to 0
                base_exact_match = False

            # Still compute text similarity as fallback metric
            similarity = compute_similarity(ground_truth_output, finetuned_output)
            base_similarity = compute_similarity(ground_truth_output, base_output)
            diff_html = compute_diff_html(ground_truth_output, finetuned_output)

            comparison = ComparisonExample(
                input=ground_truth.input,
                teacher_thinking=ground_truth_thinking,
                teacher_output=ground_truth_output,
                student_thinking=finetuned_thinking,
                student_output=finetuned_output,
                exact_match=exact_match,
                similarity_score=similarity,
                diff_html=diff_html,
                base_output=base_output,
                base_thinking=base_thinking,
                base_exact_match=base_exact_match,
                base_similarity=base_similarity,
                student_field_matches=student_field_matches,
                base_field_matches=base_field_matches,
                is_json_output=is_json,
            )
            comparisons.append(comparison)
            if (i + 1) % max(1, len(validation_traces) // 5) == 0:
                logger.info(f"Processed {i + 1}/{len(validation_traces)} comparisons")

        logger.info(
            f"Created {len(comparisons)} total comparisons "
            f"(iterated {comparison_count} times)"
        )
    except Exception as e:
        logger.error(f"Error creating comparisons: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create comparisons: {str(e)}") from e

    # Aggregate metrics
    metrics = aggregate_metrics(comparisons)

    # Generate HTML report
    logger.info("Generating HTML report...")
    config_dict = _load_config_dict(output_base)
    result = BaselineResult(
        run_name=run_name,
        metrics=metrics,
        examples=comparisons,
        html_path=html_path,
        config=config_dict,
    )
    generate_html_report(result, html_path)
    logger.info(f"Report saved to {html_path}")

    return result


def _load_config(output_base: Path) -> DistillationConfig:
    """Load config from summary.json or create minimal config.

    Note: seed and instruction are populated with dummy values satisfying
    validation, since they're not needed for evaluation (we load cached data).
    """
    summary_path = output_base / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            # Reconstruct config from summary
            config_dict = summary.get("config", {})
            return DistillationConfig(
                name=summary.get("name", "unknown"),
                seed=[{"input": "dummy", "output": "dummy"}],  # Satisfies validation
                instruction="Evaluation mode - using cached data",  # Satisfies validation
                teacher=summary.get("model", {}).get("teacher", "claude-sonnet-4-5"),
                student=summary.get("model", {}).get(
                    "student", "mlx-community/Llama-3-8B-Instruct-4bit"
                ),
                val_split=config_dict.get("val_split", 0.2),
            )

    # Fallback: return minimal config with dummy values
    return DistillationConfig(
        name="unknown",
        seed=[{"input": "dummy", "output": "dummy"}],  # Satisfies validation
        instruction="Evaluation mode - using cached data",  # Satisfies validation
        teacher="claude-sonnet-4-5",
        student="mlx-community/Llama-3-8B-Instruct-4bit",
        val_split=0.2,
    )


def _load_config_dict(output_base: Path) -> Dict[str, Any]:
    """Load config as dictionary from summary.json."""
    summary_path = output_base / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    return {}


def _save_cache(
    cache_path: Path,
    base_results: List[Dict[str, str]],
    finetuned_results: List[Dict[str, str]],
    config: DistillationConfig,
) -> None:
    """Save inference results to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {
        "timestamp": datetime.now().isoformat(),
        "base_results": base_results,
        "finetuned_results": finetuned_results,
        "config": {
            "student": config.student,
            "instruction_length": len(config.instruction),
        },
    }

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load inference results from cache."""
    with open(cache_path) as f:
        cache: Dict[str, Any] = json.load(f)

    result: Dict[str, Any] = {
        "base_results": cache.get("base_results", []),
        "finetuned_results": cache.get("finetuned_results", []),
    }
    return result
