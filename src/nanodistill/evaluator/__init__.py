"""Baseline evaluation module for NanoDistill.

Generates interactive HTML reports comparing teacher and student outputs
on validation data.
"""

from .baseline import evaluate_baseline

__all__ = ["evaluate_baseline"]
