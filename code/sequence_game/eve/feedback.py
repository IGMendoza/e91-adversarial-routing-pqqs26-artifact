"""Observable, non-oracular feedback for Eve's learning policies.

Target membership and simulator-side attack activation are causal diagnostics,
not observations.  A disruption-oriented Eve can observe the authenticated
protocol decision and the public/heralded service yield, plus records produced
by her own physical instrument.  She cannot identify from one trial whether a
failure was caused by her action; the learner must infer efficacy statistically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


EVE_FEEDBACK_MODEL = "public_heralded_service_complement_v1"


@dataclass(frozen=True)
class EveFeedback:
    """Eve-visible evidence and scalar disruption utility for one turn."""

    reward: float
    gross_disruption_reward: float
    attack_cost: float
    public_outcome: str
    public_alice_utility: float
    local_observation_kind: str = "none"
    local_observation_count: int = 0
    causal_attribution: str = "not_identifiable_from_single_turn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": EVE_FEEDBACK_MODEL,
            "reward": self.reward,
            "gross_disruption_reward": self.gross_disruption_reward,
            "attack_cost": self.attack_cost,
            "public_outcome": self.public_outcome,
            "public_alice_utility": self.public_alice_utility,
            "local_observation_kind": self.local_observation_kind,
            "local_observation_count": self.local_observation_count,
            "causal_attribution": self.causal_attribution,
            "target_overlap_disclosed": False,
        }


def eve_feedback_from_public_utility(
    *,
    public_outcome: str,
    alice_utility: float,
    attack_cost: float = 0.0,
    local_observation_kind: str = "none",
    local_observation_count: int = 0,
) -> EveFeedback:
    """Build feedback from a public Alice utility in ``[0, 1]``."""

    utility = _unit_interval("alice_utility", alice_utility)
    cost = float(attack_cost)
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("attack_cost must be finite and >= 0")
    count = int(local_observation_count)
    if count < 0:
        raise ValueError("local_observation_count must be >= 0")
    kind = str(local_observation_kind).strip() or "none"
    if count == 0:
        kind = "none"
    gross = 1.0 - utility
    return EveFeedback(
        reward=gross - cost,
        gross_disruption_reward=gross,
        attack_cost=cost,
        public_outcome=str(public_outcome),
        public_alice_utility=utility,
        local_observation_kind=kind,
        local_observation_count=count,
    )


def eve_feedback_from_heralded_service(
    *,
    public_outcome: str,
    requested_pair_count: int,
    delivered_pair_count: int,
    attack_cost: float = 0.0,
    local_observation_kind: str = "none",
    local_observation_count: int = 0,
) -> EveFeedback:
    """Use Bell-qualified public service yield as Eve's disruption signal.

    An accepted turn has utility ``delivered/requested``.  Any public protocol
    abort has zero usable service utility even if physical pairs arrived.  The
    resulting reward is continuous when an accepted request is only partially
    served, and a background abort can reward an off-route action.  Both are
    intentional: no counterfactual target-hit oracle is available to Eve.
    """

    requested = int(requested_pair_count)
    delivered = int(delivered_pair_count)
    if requested < 1:
        raise ValueError("requested_pair_count must be >= 1")
    if delivered < 0 or delivered > requested:
        raise ValueError("delivered_pair_count must be in [0, requested_pair_count]")
    accepted = str(public_outcome) == "accepted"
    utility = (delivered / requested) if accepted else 0.0
    return eve_feedback_from_public_utility(
        public_outcome=public_outcome,
        alice_utility=utility,
        attack_cost=attack_cost,
        local_observation_kind=local_observation_kind,
        local_observation_count=local_observation_count,
    )


def _unit_interval(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return result
