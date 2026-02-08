"""Data classes for baseline evaluation results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ComparisonExample:
    """Single comparison: base model vs fine-tuned vs ground truth."""

    input: str
    teacher_thinking: str  # Ground truth thinking
    teacher_output: str  # Ground truth output
    student_thinking: str  # Fine-tuned thinking
    student_output: str  # Fine-tuned output
    exact_match: bool  # Fine-tuned vs ground truth
    similarity_score: float  # Fine-tuned vs ground truth
    diff_html: str  # Fine-tuned vs ground truth diff
    base_output: str = ""  # Base model output (for comparison)
    base_thinking: str = ""  # Base model thinking
    base_exact_match: bool = False  # Base vs ground truth
    base_similarity: float = 0.0  # Base vs ground truth
    student_field_matches: Dict[str, bool] = field(
        default_factory=dict
    )  # Fine-tuned: field-by-field comparison
    base_field_matches: Dict[str, bool] = field(
        default_factory=dict
    )  # Base: field-by-field comparison
    is_json_output: bool = False  # Whether output is valid JSON


@dataclass
class BaselineMetrics:
    """Aggregated metrics from baseline evaluation."""

    total_examples: int
    exact_matches: int
    exact_match_rate: float
    avg_similarity: float
    teacher_avg_length: int
    student_avg_length: int


@dataclass
class BaselineResult:
    """Complete result of baseline evaluation."""

    run_name: str
    metrics: BaselineMetrics
    examples: List[ComparisonExample]
    html_path: Path
    config: Dict[str, Any]
