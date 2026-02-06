"""Data amplification pipeline for NanoDistill.

Orchestrates policy extraction and synthetic example generation to expand
a small seed dataset into a large training dataset.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Type

from pydantic import BaseModel

from ..data.loader import append_traces_to_jsonl, load_traces_from_jsonl, save_traces_to_jsonl
from ..teacher.client import TeacherClient
from ..teacher.schemas import TaskPolicy, ThinkingTrace


class AmplificationPipeline:
    """Pipeline for amplifying seed data through synthetic generation.

    Two-phase approach:
    1. Extract task policy from seed data and CoT traces
    2. Generate new synthetic examples constrained by policy

    Attributes:
        teacher_client: Teacher client for API calls
    """

    def __init__(self, teacher_client: TeacherClient):
        """Initialize amplification pipeline.

        Args:
            teacher_client: Initialized TeacherClient for API calls
        """
        self.teacher = teacher_client

    def amplify(
        self,
        seed_examples: List[Dict[str, str]],
        cot_traces: List[ThinkingTrace],
        instruction: str,
        augment_factor: int,
        output_path: Optional[Path] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Generator[Tuple[int, int, int, int], None, Tuple[List[ThinkingTrace], TaskPolicy]]:
        """Amplify seed data into larger training dataset with incremental checkpointing.

        Generates synthetic examples in batches (one batch per augment unit),
        saving progress after each batch. Resumes from checkpoint if interrupted.

        Args:
            seed_examples: Original seed examples
            cot_traces: Generated Chain-of-Thought traces for seeds
            instruction: Task instruction / system prompt
            augment_factor: Target multiplication factor (e.g., 50x)
            output_path: Optional path to checkpoint file (traces_amplified.jsonl)
            response_model: Optional Pydantic model to enforce schema on synthetic outputs

        Yields:
            Tuple of (batch_num, total_batches, current_synthetic, total_synthetic)
            - batch_num: Current batch number (1-indexed)
            - total_batches: Total number of batches needed
            - current_synthetic: Total synthetic examples generated so far
            - total_synthetic: Total synthetic examples needed

        Returns:
            Tuple of (amplified traces, extracted policy)
            - List of original + synthetic ThinkingTrace objects
            - TaskPolicy describing the learned task pattern
        """
        # Start with original traces
        amplified_traces = cot_traces.copy()

        # Phase 1: Extract policy from seeds
        policy = self._extract_policy(seed_examples, cot_traces, instruction)

        # Calculate batch parameters
        seed_count = len(seed_examples)
        total_synthetic = seed_count * (augment_factor - 1)
        batch_size = seed_count
        num_batches = augment_factor - 1

        # Skip if no synthetic examples needed
        if num_batches == 0:
            return amplified_traces, policy

        # Check for existing progress if checkpoint path provided
        existing_synthetic = 0
        completed_batches = 0
        if output_path and Path(output_path).exists():
            try:
                all_traces = load_traces_from_jsonl(output_path)
                existing_synthetic = max(0, len(all_traces) - seed_count)
                amplified_traces = all_traces.copy()
                completed_batches = existing_synthetic // batch_size
            except Exception:
                # If checkpoint is corrupted, start fresh
                existing_synthetic = 0
                completed_batches = 0

        # Calculate which batches to generate
        start_batch = completed_batches + 1

        # Phase 2: Generate synthetic examples in batches
        for batch_num in range(start_batch, num_batches + 1):
            # Generate batch of synthetic examples
            batch_examples = self._generate_synthetic_examples(
                policy, batch_size, instruction, seed_count, response_model
            )

            # Synthesize CoT for batch
            batch_traces = self._synthesize_cot_for_synthetic(batch_examples, instruction)

            # Save checkpoint
            if output_path:
                output_path = Path(output_path)
                if batch_num == 1 and completed_batches == 0:
                    # First batch: write seeds + first batch
                    save_traces_to_jsonl(amplified_traces + batch_traces, output_path)
                else:
                    # Subsequent batches: append only
                    append_traces_to_jsonl(batch_traces, output_path)

            # Update in-memory list
            amplified_traces.extend(batch_traces)

            # Calculate progress
            current_synthetic = existing_synthetic + (batch_num - completed_batches) * batch_size

            # Yield progress
            yield batch_num, num_batches, current_synthetic, total_synthetic

        # Return final results
        return amplified_traces, policy

    def _extract_policy(
        self,
        seed_examples: List[Dict[str, str]],
        cot_traces: List[ThinkingTrace],
        instruction: str,
    ) -> TaskPolicy:
        """Extract task policy from seed data.

        Args:
            seed_examples: Original seed examples
            cot_traces: Chain-of-Thought traces
            instruction: Task instruction

        Returns:
            TaskPolicy describing the task pattern
        """
        policy = self.teacher.extract_policy(seed_examples, cot_traces, instruction)
        return policy

    def _generate_synthetic_examples(
        self,
        policy: TaskPolicy,
        num_examples: int,
        instruction: str,
        seed_count: int,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples matching the policy.

        Args:
            policy: Task policy extracted from seed data
            num_examples: Number of examples to generate
            instruction: Task instruction
            seed_count: Number of original seed examples
            response_model: Optional Pydantic model to enforce schema

        Returns:
            List of generated examples with 'input' and 'output' fields
        """
        examples = self.teacher.generate_synthetic_examples(
            policy, num_examples, instruction, seed_count, response_model=response_model
        )
        return examples

    def _synthesize_cot_for_synthetic(
        self,
        synthetic_examples: List[Dict[str, str]],
        instruction: str,
    ) -> List[ThinkingTrace]:
        """Generate Chain-of-Thought traces for synthetic examples.

        Args:
            synthetic_examples: Generated synthetic examples
            instruction: Task instruction

        Returns:
            List of ThinkingTrace objects
        """
        traces = self.teacher.synthesize_cot(synthetic_examples, instruction)
        return traces
