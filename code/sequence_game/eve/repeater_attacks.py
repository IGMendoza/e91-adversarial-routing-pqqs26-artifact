"""Literature-scoped repeater-infrastructure attack specs.

This module only describes Eve's selected corruption target and vector. The
SeQUeNCe-specific effect is applied by the repeater E91 builder/trial runner.
Endpoint detector side channels and information-gain rewards are intentionally
out of scope for this pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REPEATER_ATTACK_KINDS = (
    "none",
    "edge_intercept_resend",
    "swap_denial",
    "memory_degradation",
    "repeater_memory_measurement",
    "repeater_memory_dephasing_probe",
)
SUPPORTED_REPEATER_GAME_KINDS = (
    "swap_denial",
    "memory_degradation",
)


class RepeaterAttackError(ValueError):
    """Invalid repeater-infrastructure attack specification."""


@dataclass(frozen=True)
class RepeaterAttackSpec:
    """A trial-scoped corruption of one repeater node on the fixed path."""

    kind: str = "none"
    target_node: str = ""
    target_edge: str = ""
    target_u: str = ""
    target_v: str = ""
    memory_fidelity_multiplier: float = 0.5
    basis: str = ""
    theta: float | None = None
    theta_label: str = ""
    attack_id: str = ""
    cost: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in REPEATER_ATTACK_KINDS:
            raise RepeaterAttackError(
                f"unknown repeater attack kind {self.kind!r}; {REPEATER_ATTACK_KINDS}")
        if self.kind == "edge_intercept_resend":
            if not (self.target_u and self.target_v):
                raise RepeaterAttackError("edge_intercept_resend needs target_u and target_v")
            if self.target_u == self.target_v:
                raise RepeaterAttackError("edge_intercept_resend target edge cannot be a self-loop")
        elif self.kind != "none" and not self.target_node:
            raise RepeaterAttackError(f"{self.kind} needs a target repeater node")
        if self.kind == "repeater_memory_measurement" and self.basis not in {"Z", "X"}:
            raise RepeaterAttackError("repeater_memory_measurement basis must be Z or X")
        if self.kind == "repeater_memory_dephasing_probe" and self.theta is None:
            raise RepeaterAttackError("repeater_memory_dephasing_probe needs theta")
        if not 0.0 <= self.memory_fidelity_multiplier <= 1.0:
            raise RepeaterAttackError("memory_fidelity_multiplier must be in [0, 1]")
        if self.cost < 0:
            raise RepeaterAttackError("cost must be >= 0")

    @property
    def is_active(self) -> bool:
        return self.kind != "none"

    def validate_on_path(self, path: tuple[str, ...]) -> None:
        if not self.is_active:
            return
        if self.kind == "edge_intercept_resend":
            edge = frozenset((self.target_u, self.target_v))
            path_edges = {
                frozenset((u, v))
                for u, v in zip(path, path[1:])
            }
            if edge not in path_edges:
                raise RepeaterAttackError(
                    f"target edge {self.target_u!r}-{self.target_v!r} "
                    f"is not on path {path}")
            return
        interior = set(path[1:-1])
        if self.target_node not in interior:
            raise RepeaterAttackError(
                f"target_node {self.target_node!r} is not an interior repeater "
                f"on path {path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_node": self.target_node,
            "target_edge": self.target_edge,
            "target_u": self.target_u,
            "target_v": self.target_v,
            "memory_fidelity_multiplier": self.memory_fidelity_multiplier,
            "basis": self.basis,
            "theta": self.theta,
            "theta_label": self.theta_label,
            "attack_id": self.attack_id,
            "cost": self.cost,
        }
