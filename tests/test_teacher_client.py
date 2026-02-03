"""Unit tests for TeacherClient parsing logic."""

from unittest.mock import patch

import pytest

from nanodistill.teacher.client import TeacherClient


@pytest.fixture
def teacher_client():
    """Create a TeacherClient with mocked API key validation."""
    with patch("nanodistill.teacher.client.validate_teacher_api_key"):
        client = TeacherClient(model="claude-sonnet-4-5")
    return client


class TestParseSyntheticExamples:
    """Tests for _parse_synthetic_examples method."""

    def test_parse_synthetic_examples_single_line(self, teacher_client):
        """Verify parsing when output is on the same line as '- Output:'."""
        response_text = """
- Input: What is 15 + 8?
- Output: 23

- Input: What is 20 - 7?
- Output: 13
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 2
        assert examples[0]["input"] == "What is 15 + 8?"
        assert examples[0]["output"] == "23"
        assert examples[1]["input"] == "What is 20 - 7?"
        assert examples[1]["output"] == "13"

    def test_parse_synthetic_examples_multi_line(self, teacher_client):
        """Verify parsing when output starts on line after '- Output:'."""
        response_text = """
- Input: What is 15 + 8?
- Output:
Step-by-Step Thinking Process:
First, I need to add 15 and 8.
15 + 8 = 23
Final Answer: 23
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 1
        assert examples[0]["input"] == "What is 15 + 8?"
        assert "Step-by-Step Thinking Process:" in examples[0]["output"]
        assert "Final Answer: 23" in examples[0]["output"]

    def test_parse_synthetic_examples_mixed_format(self, teacher_client):
        """Verify parsing with multiple examples in mixed formats."""
        response_text = """
- Input: What is 5 + 3?
- Output: 8

- Input: Explain why the sky is blue.
- Output:
The sky appears blue due to Rayleigh scattering.
When sunlight enters Earth's atmosphere, it collides with gas molecules.
Blue light has a shorter wavelength and scatters more than other colors.

- Input: What is 100 / 4?
- Output: 25
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 3
        # First example - single line
        assert examples[0]["input"] == "What is 5 + 3?"
        assert examples[0]["output"] == "8"
        # Second example - multi-line
        assert examples[1]["input"] == "Explain why the sky is blue."
        assert "Rayleigh scattering" in examples[1]["output"]
        assert "shorter wavelength" in examples[1]["output"]
        # Third example - single line
        assert examples[2]["input"] == "What is 100 / 4?"
        assert examples[2]["output"] == "25"

    def test_parse_synthetic_examples_with_example_markers(self, teacher_client):
        """Verify multi-line output stops at next '**Example N**' marker."""
        response_text = """
**Example 1**
- Input: Describe the process of photosynthesis.
- Output:
Photosynthesis is the process by which plants convert light energy into chemical energy.
The process occurs in chloroplasts and involves:
1. Light absorption by chlorophyll
2. Water splitting to release oxygen
3. Carbon dioxide fixation to produce glucose

**Example 2**
- Input: What is the capital of France?
- Output: Paris

**Example 3**
- Input: Explain gravity briefly.
- Output:
Gravity is a fundamental force that attracts objects with mass toward each other.
On Earth, it gives weight to physical objects.
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 3

        # First example - multi-line output should stop before Example 2
        assert examples[0]["input"] == "Describe the process of photosynthesis."
        assert "Photosynthesis is the process" in examples[0]["output"]
        assert "Carbon dioxide fixation" in examples[0]["output"]
        # Should not contain content from Example 2
        assert "Paris" not in examples[0]["output"]
        assert "**Example 2**" not in examples[0]["output"]

        # Second example - single line
        assert examples[1]["input"] == "What is the capital of France?"
        assert examples[1]["output"] == "Paris"

        # Third example - multi-line
        assert examples[2]["input"] == "Explain gravity briefly."
        assert "fundamental force" in examples[2]["output"]

    def test_parse_synthetic_examples_with_numbered_example_markers(self, teacher_client):
        """Verify parsing stops at 'Example N:' style markers."""
        response_text = """
Example 1:
- Input: What is machine learning?
- Output:
Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without being explicitly programmed.
Common applications include image recognition and natural language processing.

Example 2:
- Input: Define API.
- Output: An API (Application Programming Interface) is a set of protocols for building apps.
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 2
        # First example should not include content from second example
        assert examples[0]["input"] == "What is machine learning?"
        assert "subset of artificial intelligence" in examples[0]["output"]
        assert "API" not in examples[0]["output"]

    def test_parse_synthetic_examples_empty_response_raises_error(self, teacher_client):
        """Verify that empty/unparseable response raises TeacherAPIError."""
        from nanodistill.utils.errors import TeacherAPIError

        response_text = "This is just random text with no structure."

        with pytest.raises(TeacherAPIError) as exc_info:
            teacher_client._parse_synthetic_examples(response_text)

        assert "Could not parse any examples" in str(exc_info.value)

    def test_parse_synthetic_examples_json_format(self, teacher_client):
        """Verify parsing when response contains JSON array."""
        response_text = """
Here are the generated examples:
[
    {"input": "What is 2 + 2?", "output": "4"},
    {"input": "What is 10 - 3?", "output": "7"}
]
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 2
        assert examples[0]["input"] == "What is 2 + 2?"
        assert examples[0]["output"] == "4"
        assert examples[1]["input"] == "What is 10 - 3?"
        assert examples[1]["output"] == "7"

    def test_parse_synthetic_examples_question_answer_format(self, teacher_client):
        """Verify parsing with Question/Answer format instead of Input/Output."""
        response_text = """
- Question: What color is the sun?
- Answer: The sun appears yellow or white.

- Question: How many days in a week?
- Answer: 7
"""
        examples = teacher_client._parse_synthetic_examples(response_text)

        assert len(examples) == 2
        assert examples[0]["input"] == "What color is the sun?"
        assert examples[0]["output"] == "The sun appears yellow or white."
        assert examples[1]["input"] == "How many days in a week?"
        assert examples[1]["output"] == "7"
