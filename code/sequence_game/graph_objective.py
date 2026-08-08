"""Fail-closed graph objective candidate interface.

Research context checked before this slice:

- Acin, Cirac, and Lewenstein, "Entanglement Percolation in Quantum Networks"
  (arXiv:quant-ph/0612167) motivates percolation-style connectivity objectives.
- Abane et al., "Entanglement Routing in Quantum Networks: A Comprehensive
  Survey" (arXiv:2408.01234) summarizes routing objectives that depend on
  fidelity, swapping probability, memory lifetime, and request constraints.

This module does not implement either as a scientific objective.  It only
defines the tests that must pass before a graph objective can be emitted as more
than a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .claims import (
    BLOCKED_PENDING_MAPPING,
    BLOCKED_PENDING_SECURITY_QUANTITY,
    BLOCKED_PENDING_SOURCE,
    DESIGNED_NOT_IMPLEMENTED,
    GRAPH_OBJECTIVE_CANDIDATE,
    ClaimGuardError,
    require_graph_objective_test,
)


class GraphObjectiveError(ValueError):
    """Graph objective is missing source, mapping, or comparator prerequisites."""


@dataclass(frozen=True)
class GraphObjectiveSpec:
    name: str
    formula: str
    source: str
    required_features: tuple[str, ...]

    def validate(self) -> None:
        missing = []
        if not self.name:
            missing.append("name")
        if not self.formula:
            missing.append("formula")
        if not self.source:
            missing.append("source")
        if not self.required_features:
            missing.append("required_features")
        if missing:
            raise GraphObjectiveError(f"graph objective missing {missing}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_label": GRAPH_OBJECTIVE_CANDIDATE,
            "claim_status": DESIGNED_NOT_IMPLEMENTED,
            "name": self.name,
            "formula": self.formula,
            "source": self.source,
            "required_features": list(self.required_features),
        }


def blocked_objective_report(reason: str,
                             *,
                             claim_status: str = BLOCKED_PENDING_SOURCE) -> dict[str, Any]:
    return {
        "blocked": True,
        "blocked_reason": reason,
        "scope_label": GRAPH_OBJECTIVE_CANDIDATE,
        "claim_status": claim_status,
        "goal_1_claim": False,
    }


def validate_objective_spec(spec: GraphObjectiveSpec) -> dict[str, Any]:
    try:
        spec.validate()
    except GraphObjectiveError as exc:
        return blocked_objective_report(str(exc), claim_status=BLOCKED_PENDING_SOURCE)
    return spec.to_dict()


def validate_feature_mapping(spec: GraphObjectiveSpec,
                             feature_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spec_result = validate_objective_spec(spec)
    if spec_result.get("blocked"):
        return spec_result
    if not feature_rows:
        return blocked_objective_report(
            "graph objective requires measurable graph feature rows",
            claim_status=BLOCKED_PENDING_MAPPING,
        )
    available = set().union(*(set(row) for row in feature_rows))
    missing = [feature for feature in spec.required_features if feature not in available]
    if missing:
        return blocked_objective_report(
            f"graph objective features are not mapped: {missing}",
            claim_status=BLOCKED_PENDING_MAPPING,
        )
    return {
        "blocked": False,
        "scope_label": GRAPH_OBJECTIVE_CANDIDATE,
        "claim_status": DESIGNED_NOT_IMPLEMENTED,
        "mapped_features": list(spec.required_features),
        "goal_1_claim": False,
    }


def validate_control_baselines(random_baseline: Mapping[str, Any] | None,
                               oracle_baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    if not random_baseline or not oracle_baseline:
        return blocked_objective_report(
            "graph objective requires random and oracle controls",
            claim_status=BLOCKED_PENDING_SECURITY_QUANTITY,
        )
    return {
        "blocked": False,
        "scope_label": GRAPH_OBJECTIVE_CANDIDATE,
        "claim_status": DESIGNED_NOT_IMPLEMENTED,
        "controls": {
            "random": dict(random_baseline),
            "oracle": dict(oracle_baseline),
        },
        "goal_1_claim": False,
    }


def compare_strategy_to_objective(*,
                                  spec: GraphObjectiveSpec,
                                  feature_rows: Sequence[Mapping[str, Any]],
                                  random_baseline: Mapping[str, Any] | None,
                                  oracle_baseline: Mapping[str, Any] | None,
                                  graph_objective_test_id: str | None = None
                                  ) -> dict[str, Any]:
    """Return candidate status or a blocked reason; no hardness claim is made."""

    mapping = validate_feature_mapping(spec, feature_rows)
    if mapping.get("blocked"):
        return mapping
    controls = validate_control_baselines(random_baseline, oracle_baseline)
    if controls.get("blocked"):
        return controls
    result = {
        "blocked": True,
        "blocked_reason": (
            "Eve strategy is not comparable to the graph objective until a "
            "source-scoped security or operational quantity is implemented"
        ),
        "scope_label": GRAPH_OBJECTIVE_CANDIDATE,
        "claim_status": BLOCKED_PENDING_SECURITY_QUANTITY,
        "graph_objective_test_id": graph_objective_test_id,
        "goal_1_claim": False,
    }
    try:
        require_graph_objective_test(result)
    except ClaimGuardError as exc:
        result["blocked_reason"] = str(exc)
    return result
