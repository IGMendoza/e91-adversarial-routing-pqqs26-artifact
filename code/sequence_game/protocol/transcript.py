"""Trial transcript structures with explicit public/private separation.

Scope: toy. These structures record what happened in one protocol trial at the
classical bookkeeping level. Which fields count as "public" is a modelling
choice of this game, not a security claim:

- public: trial id, accept/abort outcome and reason, announced QBER estimate,
  latency, sifted-sample count, and (optionally, filtered later by the Eve
  observation config) the route id.
- private: measurement bases, outcomes, sifted indices, and the route path.

TODO(scientific): a literature-scoped treatment must specify exactly which
classical announcements the modelled protocol makes (basis announcement,
disclosed error-estimation subset, etc.) before any field is re-classified.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from ..claims import CONTROL_GAME

#: Field names that must never appear in any public view.
PRIVATE_FIELDS = frozenset({
    "route_path",
    "alice_bases",
    "bob_bases",
    "alice_outcomes",
    "bob_outcomes",
    "sifted_indices",
})


@dataclass(frozen=True)
class PublicTranscript:
    """The protocol-public slice of one trial."""

    trial_id: str
    route_id: Optional[str]
    accepted: Optional[bool]
    abort_reason: Optional[str]
    qber_estimate: Optional[float]
    latency_ps: Optional[int]
    sifted_count: Optional[int]
    scope: str = CONTROL_GAME

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class TrialTranscript:
    """Full internal record of one trial; never given to Eve directly."""

    trial_id: str
    route_id: Optional[str] = None
    route_path: tuple[str, ...] = ()
    generation_attempts: int = 0
    generation_successes: int = 0
    swap_attempts: int = 0
    swap_successes: int = 0
    alice_bases: tuple[str, ...] = ()
    bob_bases: tuple[str, ...] = ()
    alice_outcomes: tuple[int, ...] = ()
    bob_outcomes: tuple[int, ...] = ()
    sifted_indices: tuple[int, ...] = ()
    qber_estimate: Optional[float] = None
    accepted: Optional[bool] = None
    abort_reason: Optional[str] = None
    latency_ps: Optional[int] = None
    notes: str = ""
    scope: str = CONTROL_GAME
    extra: dict[str, Any] = field(default_factory=dict)

    def public_view(self) -> PublicTranscript:
        return PublicTranscript(
            trial_id=self.trial_id,
            route_id=self.route_id,
            accepted=self.accepted,
            abort_reason=self.abort_reason,
            qber_estimate=self.qber_estimate,
            latency_ps=self.latency_ps,
            sifted_count=len(self.sifted_indices) if self.sifted_indices else 0,
            scope=self.scope,
        )

    def alice_private_view(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "route_id": self.route_id,
            "route_path": self.route_path,
            "bases": self.alice_bases,
            "outcomes": self.alice_outcomes,
            "sifted_indices": self.sifted_indices,
        }

    def bob_private_view(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "bases": self.bob_bases,
            "outcomes": self.bob_outcomes,
            "sifted_indices": self.sifted_indices,
        }

    def debug_view(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["extra"] = dict(self.extra)
        return out
