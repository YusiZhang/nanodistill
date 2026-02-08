"""Metrics computation for baseline evaluation.

Includes exact match, similarity scoring, and diff generation.
Uses only standard library (difflib, no external dependencies).
"""

import difflib
import json
from typing import Dict, List, Set, Tuple

from .schemas import BaselineMetrics, ComparisonExample


def compute_exact_match(output1: str, output2: str) -> bool:
    """Check if outputs match exactly after normalization.

    Performs case-insensitive comparison with whitespace normalization.

    Args:
        output1: First output string (teacher)
        output2: Second output string (student)

    Returns:
        True if outputs match (normalized)
    """
    return output1.strip().lower() == output2.strip().lower()


def compute_similarity(output1: str, output2: str) -> float:
    """Compute similarity using character 3-gram Jaccard similarity.

    Measures overlap of character sequences to capture output similarity
    without requiring exact matches.

    Args:
        output1: First output string
        output2: Second output string

    Returns:
        Similarity score (0.0-1.0, where 1.0 is identical)
    """

    def ngrams(s: str, n: int = 3) -> Set[str]:
        """Generate character n-grams from string."""
        s = s.lower().strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ng1 = ngrams(output1)
    ng2 = ngrams(output2)

    # Handle edge cases
    if not ng1 and not ng2:
        return 1.0  # Both empty
    if not ng1 or not ng2:
        return 0.0  # One empty, one not

    # Jaccard similarity: |intersection| / |union|
    intersection = len(ng1 & ng2)
    union = len(ng1 | ng2)

    return intersection / union if union > 0 else 0.0


def compare_json_fields(
    json_str1: str, json_str2: str
) -> Tuple[bool, Dict[str, bool]]:
    """Compare two JSON strings field by field.

    Attempts to parse both JSON strings and compare their fields.
    Returns success flag and field-level comparison results.

    Args:
        json_str1: First JSON string (ground truth)
        json_str2: Second JSON string (model output)

    Returns:
        Tuple of (is_valid_json, field_matches)
        - is_valid_json: True if both strings are valid JSON
        - field_matches: Dict mapping field names to match results
          (only for overlapping fields)
    """
    try:
        obj1 = json.loads(json_str1)
        obj2 = json.loads(json_str2)

        if not isinstance(obj1, dict) or not isinstance(obj2, dict):
            return False, {}

        # Compare overlapping fields
        field_matches = {}
        for key in obj1.keys():
            if key in obj2:
                # Compare field values (normalized for strings)
                val1 = obj1[key]
                val2 = obj2[key]

                # Normalize string comparison
                if isinstance(val1, str) and isinstance(val2, str):
                    match = val1.strip().lower() == val2.strip().lower()
                else:
                    match = val1 == val2

                field_matches[key] = match

        return True, field_matches

    except (json.JSONDecodeError, ValueError, TypeError):
        return False, {}


def compute_diff_html(output1: str, output2: str) -> str:
    """Generate HTML diff table comparing two outputs.

    Uses difflib.HtmlDiff to create a clean side-by-side comparison
    with highlighting of differences.

    Args:
        output1: Teacher output
        output2: Student output

    Returns:
        HTML table string showing differences
    """
    differ = difflib.HtmlDiff()

    # Split by lines for better diff
    lines1 = output1.strip().split("\n")
    lines2 = output2.strip().split("\n")

    # Generate HTML diff table
    html = differ.make_table(
        lines1,
        lines2,
        fromdesc="Teacher",
        todesc="Student",
        context=True,
        numlines=0,
    )

    # Minimal CSS cleanup: ensure table is readable in report
    html = html.replace('class="diff"', 'class="diff-table"')

    return html


def aggregate_metrics(comparisons: List[ComparisonExample]) -> BaselineMetrics:
    """Aggregate individual comparison metrics.

    Computes statistics over all comparisons: exact match rate,
    average similarity, output length statistics.

    Args:
        comparisons: List of comparison examples

    Returns:
        BaselineMetrics with aggregated statistics
    """
    if not comparisons:
        return BaselineMetrics(
            total_examples=0,
            exact_matches=0,
            exact_match_rate=0.0,
            avg_similarity=0.0,
            teacher_avg_length=0,
            student_avg_length=0,
        )

    total = len(comparisons)
    exact_matches = sum(1 for c in comparisons if c.exact_match)
    avg_similarity = sum(c.similarity_score for c in comparisons) / total

    teacher_avg_length = sum(len(c.teacher_output.split()) for c in comparisons) // total
    student_avg_length = sum(len(c.student_output.split()) for c in comparisons) // total

    return BaselineMetrics(
        total_examples=total,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / total,
        avg_similarity=avg_similarity,
        teacher_avg_length=teacher_avg_length,
        student_avg_length=student_avg_length,
    )
