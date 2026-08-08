"""Eve action interface and the first concrete actions.

Actions never touch the protocol or simulator directly. ``apply`` returns an
``AttackEffect`` (which network elements are disabled for the trial) plus an
``EveActionResult`` (cost, public/private result fields). The game environment
privately resolves how the effect interacts with Alice's route, so Eve never
observes the route through this interface.

Scopes:

- ``NoAttackAction``: trivially safe bookkeeping, scope ``CONTROL_GAME``.
- ``DenialAttackAction``: scope ``CONTROL_GAME``. A game-mechanics disruption of one
  eve-eligible edge or node. It is not a parameterized physical attack and no
  claim is made about how a real adversary would realize it.
- ``MeasurementAttackPlaceholder``: fail-closed. Running it requires explicit
  user-supplied scientific inputs; until then it raises ``NotImplementedError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..claims import BLOCKED, CONTROL_GAME, SECURITY_REFERENCE
from ..topology.ir import TopologyIR

TARGET_TYPES = ("none", "node", "edge", "route", "detector", "memory", "swap", "protocol")

ACTION_SCOPES = (CONTROL_GAME, BLOCKED, SECURITY_REFERENCE)


class EveActionError(ValueError):
    """Invalid Eve action configuration or target."""


@dataclass(frozen=True)
class AttackEffect:
    """Trial-scoped effect of an action on network availability."""

    disabled_edges: frozenset[str] = frozenset()
    disabled_nodes: frozenset[str] = frozenset()

    @property
    def is_null(self) -> bool:
        return not self.disabled_edges and not self.disabled_nodes


@dataclass(frozen=True)
class EveActionResult:
    action_id: str
    success: bool
    cost: float
    public_fields: dict[str, Any] = field(default_factory=dict)
    private_fields: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def public_view(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "cost": self.cost,
            **self.public_fields,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "cost": self.cost,
            "public_fields": dict(self.public_fields),
            "private_fields": dict(self.private_fields),
            "notes": self.notes,
        }


class EveAction(ABC):
    """One selectable Eve action."""

    action_id: str = "abstract"
    target_type: str = "none"
    scope: str = CONTROL_GAME
    cost: float = 0.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.target_type not in TARGET_TYPES:
            raise EveActionError(f"invalid target_type {cls.target_type!r}")
        if cls.scope not in ACTION_SCOPES:
            raise EveActionError(f"invalid scope {cls.scope!r}")

    def validate_target(self, topology: TopologyIR) -> None:
        """Raise EveActionError if the action's target is not allowed."""

    @abstractmethod
    def apply(self, topology: TopologyIR, *,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[AttackEffect, EveActionResult]:
        ...

    def metadata(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target_type": self.target_type,
            "scope": self.scope,
            "cost": self.cost,
        }


class NoAttackAction(EveAction):
    action_id = "no_attack"
    target_type = "none"
    scope = CONTROL_GAME
    cost = 0.0

    def apply(self, topology: TopologyIR, *,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[AttackEffect, EveActionResult]:
        return AttackEffect(), EveActionResult(self.action_id, True, 0.0)


class DenialAttackAction(EveAction):
    """Toy denial/disruption of one eve-eligible edge or node for one trial.

    Game-mechanics only: the targeted element is unavailable for the trial.
    Not a physical attack model; no loss/noise parameters are claimed.
    """

    scope = CONTROL_GAME

    def __init__(self, target_type: str, target_id: str, cost: float = 1.0):
        if target_type not in ("edge", "node"):
            raise EveActionError(
                f"DenialAttackAction supports edge/node targets, got {target_type!r}")
        if cost < 0:
            raise EveActionError("cost must be >= 0")
        self.target_type = target_type
        self.target_id = target_id
        self.cost = float(cost)
        self.action_id = f"denial_{target_type}_{target_id}"

    def validate_target(self, topology: TopologyIR) -> None:
        if self.target_type == "edge":
            matching = [e for e in topology.edges if e.edge_id == self.target_id]
            if not matching:
                raise EveActionError(f"unknown edge target {self.target_id!r}")
            if not matching[0].eve_eligible:
                raise EveActionError(f"edge {self.target_id!r} is not eve_eligible")
        else:
            record = topology.nodes.get(self.target_id)
            if record is None:
                raise EveActionError(f"unknown node target {self.target_id!r}")
            if "eve_eligible" not in record.roles:
                raise EveActionError(f"node {self.target_id!r} is not eve_eligible")

    def apply(self, topology: TopologyIR, *,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[AttackEffect, EveActionResult]:
        self.validate_target(topology)
        if self.target_type == "edge":
            effect = AttackEffect(disabled_edges=frozenset({self.target_id}))
        else:
            effect = AttackEffect(disabled_nodes=frozenset({self.target_id}))
        result = EveActionResult(
            action_id=self.action_id,
            success=True,
            cost=self.cost,
            private_fields={"target_type": self.target_type, "target_id": self.target_id},
            notes="modeled denial effect; route overlap is resolved privately",
        )
        return effect, result


#: Scientific inputs required before activating this legacy measurement adapter.
#: Primary histogram interventions are implemented under ``histograms/``.
MEASUREMENT_ATTACK_REQUIRED_INPUTS = (
    "target_subsystem",
    "basis_selection_rule",
    "timing_event_hook",
    "state_update_rule",
    "information_gain_metric",
    "disturbance_qber_effect",
    "reference_tag",
)


class MeasurementAttackPlaceholder(EveAction):
    """Unsupported legacy adapter for measurement/intercept-style actions.

    TODO(scientific): implement only with source-grounded definitions for every
    item in ``MEASUREMENT_ATTACK_REQUIRED_INPUTS``.
    """

    action_id = "measurement_attack_placeholder"
    target_type = "protocol"
    scope = BLOCKED

    def __init__(self, supplied_inputs: Optional[dict[str, Any]] = None):
        self.supplied_inputs = dict(supplied_inputs or {})

    def missing_inputs(self) -> tuple[str, ...]:
        return tuple(k for k in MEASUREMENT_ATTACK_REQUIRED_INPUTS
                     if k not in self.supplied_inputs)

    def apply(self, topology: TopologyIR, *,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[AttackEffect, EveActionResult]:
        missing = self.missing_inputs()
        raise NotImplementedError(
            "measurement/intercept attack is not implemented: missing user-supplied "
            f"scientific inputs {list(missing) or '(inputs present but no model wired)'}; "
            "supply equations, units, and reference before literature-scope use")

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta["required_inputs"] = list(MEASUREMENT_ATTACK_REQUIRED_INPUTS)
        meta["missing_inputs"] = list(self.missing_inputs())
        meta["reference_tag"] = self.supplied_inputs.get(
            "reference_tag", "TODO(user-citation)")
        return meta
