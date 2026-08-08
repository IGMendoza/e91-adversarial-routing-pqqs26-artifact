"""Mechanical faithfulness scorer — no LLM judge required.

A model output is faithful if every numeric value it asserts appears
verbatim in the input prompt.  This is a conservative lower bound:
it does not catch semantic hallucination of non-numeric claims, but it
catches the most dangerous physics-value hallucinations (invented QBER,
fidelity, bottleneck scores, hop counts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FaithfulnessResult:
    score: float
    grounded: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    total_numeric_claims: int = 0


def extract_numbers(text: str) -> list[str]:
    """Return all distinct numeric strings (int or float) found in text."""
    return re.findall(r"\b\d+\.?\d*\b", text)


def score_faithfulness(response: str, prompt: str) -> FaithfulnessResult:
    """Check that every number in response appears verbatim in prompt."""
    nums = extract_numbers(response)
    if not nums:
        return FaithfulnessResult(score=1.0, total_numeric_claims=0)

    grounded: list[str] = []
    hallucinated: list[str] = []
    for n in nums:
        (grounded if n in prompt else hallucinated).append(n)

    score = len(grounded) / len(nums)
    return FaithfulnessResult(
        score=score,
        grounded=grounded,
        hallucinated=hallucinated,
        total_numeric_claims=len(nums),
    )
