"""Toy single-trial generator for the game environment.

Scope: toy, game-mechanics only. This is NOT a quantum-state simulation and
makes no physical or security claim. The data-generation rule is:

- Alice and Bob each draw uniform random bases from the configured labels.
- Alice's outcomes are uniform random bits.
- Where bases match, Bob's outcome equals Alice's, flipped with the explicitly
  configured ``matched_flip_probability`` (pure game-mechanics noise knob).
- Where bases differ, Bob's outcome is an independent uniform random bit.

TODO(scientific): replace with transcripts produced by a SeQUeNCe-backed
protocol run (entanglement generation/swapping + measurement under a stated
formalism) once the physical models are literature-scoped. Latency is left
``None`` here because no channel/timing model is wired (TODO(scientific)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..claims import CONTROL_GAME
from ..eve.actions import AttackEffect
from ..routing.route import Route
from ..topology.ir import TopologyIR
from .postprocessing import PostprocessingError, SiftingConfig, apply_postprocessing
from .transcript import TrialTranscript

DISRUPTED_REASON = "route_disrupted"


@dataclass(frozen=True)
class ToyTrialConfig:
    num_pairs: int
    matched_flip_probability: float
    sifting: SiftingConfig
    scope: str = CONTROL_GAME

    def __post_init__(self) -> None:
        if self.num_pairs < 1:
            raise PostprocessingError("num_pairs must be >= 1")
        if not 0.0 <= self.matched_flip_probability <= 1.0:
            raise PostprocessingError(
                f"matched_flip_probability must be in [0, 1], got "
                f"{self.matched_flip_probability}")
        if self.scope != CONTROL_GAME:
            raise PostprocessingError(
                "ToyTrialConfig is CONTROL_GAME-only; other scopes need a real protocol backend")

    def to_dict(self) -> dict:
        return {
            "num_pairs": self.num_pairs,
            "matched_flip_probability": self.matched_flip_probability,
            "sifting": self.sifting.to_dict(),
            "scope": self.scope,
        }


def route_is_disrupted(route: Route, effect: AttackEffect) -> bool:
    """Resolved privately by the environment; Eve never calls this."""
    if effect.is_null:
        return False
    if set(route.edge_ids) & effect.disabled_edges:
        return True
    return bool(set(route.path) & effect.disabled_nodes)


def run_toy_trial(topology: TopologyIR, route: Route, config: ToyTrialConfig,
                  effect: AttackEffect, rng: np.random.Generator,
                  trial_id: str) -> TrialTranscript:
    """Produce one toy transcript for the given route under the given effect."""
    transcript = TrialTranscript(
        trial_id=trial_id,
        route_id=route.route_id,
        route_path=route.path,
        scope=config.scope,
    )
    if route_is_disrupted(route, effect):
        transcript.generation_attempts = config.num_pairs
        transcript.generation_successes = 0
        transcript.accepted = False
        transcript.abort_reason = DISRUPTED_REASON
        return transcript

    n = config.num_pairs
    basis_labels = config.sifting.basis_labels
    outcome_labels = config.sifting.outcome_labels
    alice_bases = tuple(basis_labels[i] for i in rng.integers(len(basis_labels), size=n))
    bob_bases = tuple(basis_labels[i] for i in rng.integers(len(basis_labels), size=n))
    alice_outcomes = tuple(outcome_labels[i]
                           for i in rng.integers(len(outcome_labels), size=n))
    bob_outcomes = []
    for i in range(n):
        if alice_bases[i] == bob_bases[i]:
            flip = rng.random() < config.matched_flip_probability
            if flip:
                others = [o for o in outcome_labels if o != alice_outcomes[i]]
                bob_outcomes.append(others[0])
            else:
                bob_outcomes.append(alice_outcomes[i])
        else:
            bob_outcomes.append(outcome_labels[int(rng.integers(len(outcome_labels)))])

    transcript.generation_attempts = n
    transcript.generation_successes = n
    transcript.alice_bases = alice_bases
    transcript.bob_bases = bob_bases
    transcript.alice_outcomes = alice_outcomes
    transcript.bob_outcomes = tuple(bob_outcomes)
    return apply_postprocessing(transcript, config.sifting)
