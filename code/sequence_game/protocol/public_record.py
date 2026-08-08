"""Shared public step records for graph-game replay and learning boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..claims import (
    CONTROL_GAME,
    ClaimGuardError,
    assert_no_private_public_fields,
    assert_public_eve_observation,
    validate_scope_label,
)

PUBLIC_STEP_FIELDS = (
    "step",
    "mode",
    "scope_label",
    "alice_route_id",
    "alice_path",
    "eve_action",
    "public_outcome",
    "collision_or_delivery_failure",
    "alice_reward",
    "eve_reward",
    "route_features",
    "quality_metrics",
)


class PublicStepRecordError(ValueError):
    """Invalid public step record."""


@dataclass(frozen=True)
class PublicStepRecord:
    step: int
    mode: str
    scope_label: str
    alice_route_id: str
    alice_path: tuple[str, ...]
    eve_action: Mapping[str, Any]
    public_outcome: str
    collision_or_delivery_failure: bool
    alice_reward: float
    eve_reward: float
    route_features: Mapping[str, Any] = field(default_factory=dict)
    quality_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            validate_scope_label(self.scope_label)
            object.__setattr__(self, "step", int(self.step))
            object.__setattr__(self, "alice_path", tuple(str(n) for n in self.alice_path))
            object.__setattr__(self, "eve_action", dict(self.eve_action))
            object.__setattr__(self, "alice_reward", float(self.alice_reward))
            object.__setattr__(self, "eve_reward", float(self.eve_reward))
            object.__setattr__(self, "route_features", dict(self.route_features))
            object.__setattr__(self, "quality_metrics", dict(self.quality_metrics))
            assert_no_private_public_fields(self.to_dict())
        except (ClaimGuardError, TypeError, ValueError) as exc:
            raise PublicStepRecordError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "mode": self.mode,
            "scope_label": self.scope_label,
            "alice_route_id": self.alice_route_id,
            "alice_path": list(self.alice_path),
            "eve_action": dict(self.eve_action),
            "public_outcome": self.public_outcome,
            "collision_or_delivery_failure": self.collision_or_delivery_failure,
            "alice_reward": self.alice_reward,
            "eve_reward": self.eve_reward,
            "route_features": dict(self.route_features),
            "quality_metrics": dict(self.quality_metrics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublicStepRecord":
        extra = set(data) - set(PUBLIC_STEP_FIELDS)
        if extra:
            raise PublicStepRecordError(f"unexpected public step fields: {sorted(extra)}")
        missing = set(PUBLIC_STEP_FIELDS) - set(data)
        if missing:
            raise PublicStepRecordError(f"missing public step fields: {sorted(missing)}")
        return cls(
            step=data["step"],
            mode=str(data["mode"]),
            scope_label=str(data["scope_label"]),
            alice_route_id=str(data["alice_route_id"]),
            alice_path=tuple(str(n) for n in data["alice_path"]),
            eve_action=dict(data["eve_action"]),
            public_outcome=str(data["public_outcome"]),
            collision_or_delivery_failure=bool(data["collision_or_delivery_failure"]),
            alice_reward=float(data["alice_reward"]),
            eve_reward=float(data["eve_reward"]),
            route_features=dict(data["route_features"]),
            quality_metrics=dict(data["quality_metrics"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "PublicStepRecord":
        return cls.from_dict(json.loads(text))


def make_control_step_record(**kwargs: Any) -> PublicStepRecord:
    """Convenience constructor for CONTROL_GAME records."""

    kwargs.setdefault("scope_label", CONTROL_GAME)
    return PublicStepRecord(**kwargs)


def eve_public_state_from_step(record: PublicStepRecord,
                               previous_action: str,
                               previous_public_outcome: str,
                               *,
                               oracle_control: bool = False) -> dict[str, Any]:
    """Build Eve's tabular-learning state from public history.

    Normal Eve receives only previous action and previous public outcome.  The
    route-aware oracle control additionally receives the current route id and is
    explicitly marked as privileged control.
    """

    state = {
        "previous_action": previous_action,
        "previous_public_outcome": previous_public_outcome,
    }
    if oracle_control:
        state.update({
            "scope_label": CONTROL_GAME,
            "oracle_control": True,
            "alice_route_id": record.alice_route_id,
        })
    assert_public_eve_observation(state, oracle_control=oracle_control)
    return state
