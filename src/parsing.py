"""Shared parsing utilities for SKILL.md classification.

Provides constants and functions used by both analyze_corpus.py and
validate_models.py to avoid code duplication.
"""

import json
from pathlib import Path

CLASSIFICATION_FIELDS = [
    "object",
    "primary_intent",
    "secondary_intent",
    "discretion",
    "decision_count",
    "constraint_count",
]

FIXED_CATEGORY_FIELDS = [
    "primary_intent",
    "secondary_intent",
    "discretion",
]

NUMERIC_FIELDS = [
    "decision_count",
    "constraint_count",
]


def load_prompt(path: Path) -> str:
    """Read a prompt template from a file."""
    return path.read_text().strip()


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_classification(response_text: str) -> dict | None:
    """Parse the JSON classification from the model response.

    Handles responses that may include markdown code fences or extra text
    around the JSON object.

    Returns:
        Dict with classification fields, or None if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    # Validate that all required fields are present
    for field in CLASSIFICATION_FIELDS:
        if field not in parsed:
            return None

    return {field: parsed[field] for field in CLASSIFICATION_FIELDS}
