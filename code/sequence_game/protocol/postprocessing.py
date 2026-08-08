"""Classical post-processing: basis sifting, QBER estimation, accept/abort.

Scope: toy classical bookkeeping over already-recorded basis/outcome strings.
This module performs no quantum measurement modelling and makes no security
claim. Deliberate toy simplifications, each of which a literature-scoped
treatment must replace:

- TODO(scientific): the QBER estimate is computed over the *entire* sifted
  string instead of a disclosed error-estimation subset, so "estimate" here is
  the exact mismatch fraction of the toy data.
- TODO(scientific): no finite-key analysis, no privacy amplification, no
  error correction, no CHSH/Bell-inequality testing.

The accept/abort threshold and minimum sample size are explicit configuration
with no defaults, so no silent protocol choice is made here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .transcript import TrialTranscript


class PostprocessingError(ValueError):
    """Invalid sifting/QBER inputs or configuration."""


@dataclass(frozen=True)
class SiftingConfig:
    qber_threshold: float
    min_sifted_samples: int
    basis_labels: tuple[str, ...] = ("Z", "X")
    outcome_labels: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if not 0.0 <= self.qber_threshold <= 1.0:
            raise PostprocessingError(
                f"qber_threshold must be in [0, 1], got {self.qber_threshold}")
        if self.min_sifted_samples < 1:
            raise PostprocessingError("min_sifted_samples must be >= 1")
        if len(set(self.basis_labels)) < 2:
            raise PostprocessingError("need at least 2 distinct basis labels")
        if len(set(self.outcome_labels)) < 2:
            raise PostprocessingError("need at least 2 distinct outcome labels")

    def to_dict(self) -> dict:
        return {
            "qber_threshold": self.qber_threshold,
            "min_sifted_samples": self.min_sifted_samples,
            "basis_labels": list(self.basis_labels),
            "outcome_labels": list(self.outcome_labels),
        }


@dataclass(frozen=True)
class AcceptDecision:
    accepted: bool
    reason: str
    qber_estimate: Optional[float]
    sifted_count: int


def sift_indices(alice_bases: Sequence[str], bob_bases: Sequence[str],
                 config: SiftingConfig) -> tuple[int, ...]:
    """Indices where Alice and Bob used the same (valid) basis."""
    if len(alice_bases) != len(bob_bases):
        raise PostprocessingError(
            f"basis strings differ in length: {len(alice_bases)} vs {len(bob_bases)}")
    valid = set(config.basis_labels)
    for side, bases in (("alice", alice_bases), ("bob", bob_bases)):
        bad = set(bases) - valid
        if bad:
            raise PostprocessingError(f"{side} has unknown basis labels {sorted(bad)}")
    return tuple(i for i, (a, b) in enumerate(zip(alice_bases, bob_bases)) if a == b)


def extract_sifted_bits(outcomes: Sequence[int], indices: Sequence[int],
                        config: SiftingConfig) -> tuple[int, ...]:
    valid = set(config.outcome_labels)
    bad = set(outcomes) - valid
    if bad:
        raise PostprocessingError(f"unknown outcome labels {sorted(bad)}")
    try:
        return tuple(outcomes[i] for i in indices)
    except IndexError:
        raise PostprocessingError(
            f"sifted index out of range for outcome string of length {len(outcomes)}") from None


def estimate_qber(alice_bits: Sequence[int], bob_bits: Sequence[int]) -> float:
    """Mismatch fraction of the given bit strings (toy: full disclosure)."""
    if len(alice_bits) != len(bob_bits):
        raise PostprocessingError(
            f"bit strings differ in length: {len(alice_bits)} vs {len(bob_bits)}")
    if not alice_bits:
        raise PostprocessingError("cannot estimate QBER from zero sifted bits")
    mismatches = sum(1 for a, b in zip(alice_bits, bob_bits) if a != b)
    return mismatches / len(alice_bits)


def decide_accept(qber: Optional[float], sifted_count: int,
                  config: SiftingConfig) -> AcceptDecision:
    """Accept iff enough sifted samples and QBER <= threshold (boundary accepts)."""
    if sifted_count < config.min_sifted_samples:
        return AcceptDecision(False, "insufficient_sifted_samples", qber, sifted_count)
    if qber is None:
        raise PostprocessingError("qber required once sample size is sufficient")
    if qber > config.qber_threshold:
        return AcceptDecision(False, "qber_above_threshold", qber, sifted_count)
    return AcceptDecision(True, "accepted", qber, sifted_count)


def apply_postprocessing(transcript: TrialTranscript,
                         config: SiftingConfig) -> TrialTranscript:
    """Fill sifting/QBER/accept fields of a transcript in place and return it."""
    indices = sift_indices(transcript.alice_bases, transcript.bob_bases, config)
    transcript.sifted_indices = indices
    qber: Optional[float] = None
    if indices:
        alice_bits = extract_sifted_bits(transcript.alice_outcomes, indices, config)
        bob_bits = extract_sifted_bits(transcript.bob_outcomes, indices, config)
        qber = estimate_qber(alice_bits, bob_bits)
    decision = decide_accept(qber, len(indices), config)
    transcript.qber_estimate = decision.qber_estimate
    transcript.accepted = decision.accepted
    transcript.abort_reason = None if decision.accepted else decision.reason
    return transcript
