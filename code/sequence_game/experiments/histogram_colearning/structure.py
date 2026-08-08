"""Topology-derived route and Eve action feature embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sequence_game.topology import TopologyIR

from .corpus import HistogramGameCase


EVE_FEATURE_FAMILY = "family"
EVE_FEATURE_FAMILY_TARGET_TOPOLOGY = "family_target_topology"
EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY = "family_by_target_topology"
EVE_FEATURE_ACTION_IDENTITY = "action_identity"
EVE_FEATURE_MODES = (
    EVE_FEATURE_FAMILY,
    EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
    EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY,
    EVE_FEATURE_ACTION_IDENTITY,
)

_ATTACK_KIND_BY_FAMILY = {
    "edge_dos": "edge",
    "edge_information_probe": "edge",
    "swap_denial": "node",
    "memory_degradation": "node",
    "repeater_memory_measure_Z": "node",
    "repeater_memory_measure_X": "node",
}


@dataclass(frozen=True)
class RouteResourceEmbedding:
    """Cycle-free route incidence compressed to its numerical column space."""

    raw_labels: tuple[str, ...]
    raw_matrix: np.ndarray
    embedding: np.ndarray
    rank: int
    singular_values: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_feature_count": len(self.raw_labels),
            "raw_feature_labels": list(self.raw_labels),
            "route_count": int(self.raw_matrix.shape[0]),
            "rank": int(self.rank),
            "singular_values": self.singular_values.tolist(),
            "compression": (
                "left singular-vector basis of [constant, edge incidence, "
                "internal-node incidence]; preserves its route-action linear span"
            ),
            "payoff_used_to_construct_features": False,
        }


@dataclass(frozen=True)
class EveActionEmbedding:
    """Declared Eve action semantics compressed to their numerical row span."""

    mode: str
    raw_labels: tuple[str, ...]
    raw_matrix: np.ndarray
    embedding: np.ndarray
    rank: int
    singular_values: np.ndarray
    feature_group_counts: dict[str, int]
    target_descriptor_labels: tuple[str, ...]
    action_cost_in_features: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "raw_feature_count": len(self.raw_labels),
            "raw_feature_labels": list(self.raw_labels),
            "action_count": int(self.raw_matrix.shape[0]),
            "rank": int(self.rank),
            "singular_values": self.singular_values.tolist(),
            "feature_group_counts": dict(self.feature_group_counts),
            "target_descriptor_labels": list(self.target_descriptor_labels),
            "compression": (
                "exact one-hot action basis; no cross-action compression"
                if self.mode == EVE_FEATURE_ACTION_IDENTITY
                else (
                    "left singular-vector basis of the declared Eve feature "
                    "matrix; preserves its action-feature linear span"
                )
            ),
            "online_availability": (
                "all fields are fixed by the declared action universe, normalized "
                "topology, and cycle-free route universe before play; the selected "
                "Alice route and all payoff observations are excluded"
            ),
            "action_cost_in_features": self.action_cost_in_features,
            "payoff_used_to_construct_features": False,
        }


def route_resource_embedding(case: HistogramGameCase) -> RouteResourceEmbedding:
    """Build a topology-only feature basis for Alice's routes."""

    edges = sorted({
        str(edge)
        for route in case.routes
        for edge in route.get("edge_ids") or []
    })
    nodes = sorted({
        str(node)
        for route in case.routes
        for node in route.get("internal_nodes") or []
    })
    labels = (
        "constant",
        *(f"edge:{edge}" for edge in edges),
        *(f"node:{node}" for node in nodes),
    )
    raw = np.asarray([
        [
            1.0,
            *(float(edge in set(map(str, route.get("edge_ids") or []))) for edge in edges),
            *(
                float(node in set(map(str, route.get("internal_nodes") or [])))
                for node in nodes
            ),
        ]
        for route in case.routes
    ], dtype=float)
    embedding, rank, singular_values = _left_span_embedding(
        raw, label="route-resource"
    )
    return RouteResourceEmbedding(
        raw_labels=tuple(labels),
        raw_matrix=raw,
        embedding=embedding,
        rank=rank,
        singular_values=singular_values,
    )


def eve_action_embedding(
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
    *,
    mode: str = EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
) -> EveActionEmbedding:
    """Build one auditable Eve action feature basis.

    ``family_target_topology`` adds the action family and target descriptors.
    ``family_by_target_topology`` additionally conditions every target descriptor
    on its intervention family. ``action_identity`` is the no-sharing reference
    ceiling and is not a candidate structured representation.
    """

    if mode not in EVE_FEATURE_MODES:
        raise ValueError(f"unsupported Eve feature mode {mode!r}")
    normalized = validate_eve_action_contract(case, actions)
    attack_families = sorted({
        row["attack_type"]
        for row in normalized
        if row["attack_type"] != "no_attack"
    })
    family_labels = tuple(
        ["action_family:no_attack"]
        + [f"action_family:{family}" for family in attack_families]
    )
    family_index = {
        label.split(":", 1)[1]: index
        for index, label in enumerate(family_labels)
    }

    descriptor_labels, descriptor_matrix = _eve_target_descriptors(case, normalized)
    if mode == EVE_FEATURE_ACTION_IDENTITY:
        labels = tuple(f"action_identity:{row['id']}" for row in normalized)
        raw = np.eye(len(normalized), dtype=float)
        group_counts = {"action_identity": len(labels)}
    else:
        family_matrix = np.zeros((len(normalized), len(family_labels)), dtype=float)
        for action_index, row in enumerate(normalized):
            family_matrix[action_index, family_index[row["attack_type"]]] = 1.0
        labels = family_labels
        raw = family_matrix
        group_counts = {"action_family": len(family_labels)}
        if mode == EVE_FEATURE_FAMILY_TARGET_TOPOLOGY:
            labels = (*labels, *descriptor_labels)
            raw = np.column_stack((raw, descriptor_matrix))
            group_counts["target_topology"] = len(descriptor_labels)
        elif mode == EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY:
            attack_families = tuple(
                label.split(":", 1)[1]
                for label in family_labels
                if label != "action_family:no_attack"
            )
            interaction_labels: list[str] = []
            interaction_blocks: list[np.ndarray] = []
            for family in attack_families:
                family_mask = np.asarray(
                    [float(row["attack_type"] == family) for row in normalized],
                    dtype=float,
                )
                interaction_blocks.append(descriptor_matrix * family_mask[:, np.newaxis])
                interaction_labels.extend(
                    f"family_target:{family}*{label}" for label in descriptor_labels
                )
            if interaction_blocks:
                raw = np.column_stack((raw, *interaction_blocks))
                labels = (*labels, *interaction_labels)
            group_counts["family_by_target_topology"] = len(interaction_labels)

    if mode == EVE_FEATURE_ACTION_IDENTITY:
        embedding = raw.copy()
        rank = len(normalized)
        singular_values = np.ones(rank, dtype=float)
    else:
        embedding, rank, singular_values = _left_span_embedding(
            raw, label=f"Eve {mode}"
        )
    return EveActionEmbedding(
        mode=mode,
        raw_labels=tuple(labels),
        raw_matrix=raw,
        embedding=embedding,
        rank=rank,
        singular_values=singular_values,
        feature_group_counts=group_counts,
        target_descriptor_labels=(
            descriptor_labels
            if mode in {
                EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
                EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY,
            }
            else ()
        ),
    )


def eve_reward_embedding(
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
    *,
    mode: str = EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
) -> EveActionEmbedding:
    """Add Eve's declared cost coordinate to the pre-play action features.

    The public disruption reward is an affine transformation of Alice's public
    utility and the declared action cost. The cost is known before play and is
    therefore an admissible action feature. Keeping it explicit preserves the
    online reward span if future configurations assign different costs to
    targets within one intervention family.
    """

    base = eve_action_embedding(case, actions, mode=mode)
    costs = np.asarray(
        [float(action.get("cost", 0.0)) for action in actions],
        dtype=float,
    )
    raw = np.column_stack((base.raw_matrix, costs))
    embedding, rank, singular_values = _left_span_embedding(
        raw,
        label=f"Eve {mode} reward",
    )
    group_counts = dict(base.feature_group_counts)
    group_counts["declared_action_cost"] = 1
    return EveActionEmbedding(
        mode=base.mode,
        raw_labels=(*base.raw_labels, "declared_action_cost"),
        raw_matrix=raw,
        embedding=embedding,
        rank=rank,
        singular_values=singular_values,
        feature_group_counts=group_counts,
        target_descriptor_labels=base.target_descriptor_labels,
        action_cost_in_features=True,
    )


def route_action_incidence(
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Whether each declared Eve target lies on each cycle-free route."""

    incidence = np.zeros((len(case.routes), len(actions)), dtype=bool)
    for route_index, route in enumerate(case.routes):
        resources = {
            *map(str, route.get("edge_ids") or []),
            *map(str, route.get("internal_nodes") or []),
        }
        for action_index, action in enumerate(actions):
            incidence[route_index, action_index] = bool(
                str(action.get("attack_type") or "no_attack") != "no_attack"
                and str(action.get("target") or "") in resources
            )
    return incidence


def validate_eve_action_contract(
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate and normalize the complete declared Eve action universe."""

    if not actions:
        raise ValueError("Eve action universe must be nonempty")
    edge_targets = {
        str(edge)
        for route in case.routes
        for edge in route.get("edge_ids") or []
    }
    node_targets = {
        str(node)
        for route in case.routes
        for node in route.get("internal_nodes") or []
    }
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    no_attack_count = 0
    for index, action in enumerate(actions):
        missing = {"id", "attack_type", "kind", "target", "cost"} - set(action)
        if missing:
            raise ValueError(
                f"Eve action {index} is missing required fields {sorted(missing)}"
            )
        row = {
            "id": str(action["id"]),
            "attack_type": str(action["attack_type"]),
            "kind": str(action["kind"]),
            "target": str(action["target"]),
            "cost": float(action["cost"]),
        }
        if not row["id"] or row["id"] in ids:
            raise ValueError(f"Eve action IDs must be unique and nonempty: {row['id']!r}")
        ids.add(row["id"])
        if not np.isfinite(row["cost"]) or row["cost"] < 0.0:
            raise ValueError(f"Eve action {row['id']!r} has invalid cost")
        if row["attack_type"] == "no_attack":
            no_attack_count += 1
            if row["kind"] != "none" or row["target"]:
                raise ValueError("no_attack must have kind='none' and an empty target")
        else:
            expected_kind = _ATTACK_KIND_BY_FAMILY.get(row["attack_type"])
            if expected_kind is None:
                raise ValueError(
                    f"Eve action {row['id']!r} has unsupported family "
                    f"{row['attack_type']!r}"
                )
            if row["kind"] != expected_kind:
                raise ValueError(
                    f"Eve family {row['attack_type']!r} requires kind={expected_kind!r}"
                )
            valid_targets = edge_targets if expected_kind == "edge" else node_targets
            if row["target"] not in valid_targets:
                raise ValueError(
                    f"Eve action {row['id']!r} targets a resource outside the "
                    "declared route universe"
                )
        normalized.append(row)
    if no_attack_count != 1:
        raise ValueError("Eve action universe must contain exactly one no_attack action")
    return tuple(normalized)


def _eve_target_descriptors(
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
) -> tuple[tuple[str, ...], np.ndarray]:
    route_ids = tuple(str(route["route_id"]) for route in case.routes)
    labels = (
        "target_kind:edge",
        "target_kind:node",
        "target_route_support_fraction",
        "target_edge_length_scaled",
        "target_node_degree_scaled",
        "target_edge_endpoint_degree_mean_scaled",
        *(f"target_on_route:{route_id}" for route_id in route_ids),
        *(f"target_distance_from_alice:{route_id}" for route_id in route_ids),
    )
    ir = TopologyIR.from_dict(case.ir_dict)
    adjacency = ir.adjacency()
    maximum_degree = max((len(neighbors) for neighbors in adjacency.values()), default=1)
    edge_by_label = {
        _edge_label(edge.u, edge.v): edge for edge in ir.edges
    }
    route_edge_targets = {
        str(edge)
        for route in case.routes
        for edge in route.get("edge_ids") or []
    }
    maximum_edge_length = max(
        (
            float(edge_by_label[target].length_m)
            for target in route_edge_targets
            if target in edge_by_label
        ),
        default=1.0,
    )
    rows: list[list[float]] = []
    for action in actions:
        target = action["target"]
        kind = action["kind"]
        incidence: list[float] = []
        positions: list[float] = []
        for route in case.routes:
            present, position = _target_position_on_route(
                route, target=target, kind=kind
            )
            incidence.append(float(present))
            positions.append(position)
        support_fraction = float(np.mean(incidence)) if incidence else 0.0
        edge_length = 0.0
        node_degree = 0.0
        endpoint_degree = 0.0
        if kind == "edge":
            edge = edge_by_label.get(target)
            if edge is None:
                raise ValueError(f"cannot resolve Eve edge target {target!r}")
            edge_length = float(edge.length_m) / maximum_edge_length
            endpoint_degree = (
                len(adjacency[edge.u]) + len(adjacency[edge.v])
            ) / (2.0 * maximum_degree)
        elif kind == "node":
            node_degree = len(adjacency[target]) / maximum_degree
        rows.append([
            float(kind == "edge"),
            float(kind == "node"),
            support_fraction,
            edge_length,
            node_degree,
            endpoint_degree,
            *incidence,
            *positions,
        ])
    return tuple(labels), np.asarray(rows, dtype=float)


def _target_position_on_route(
    route: dict[str, Any],
    *,
    target: str,
    kind: str,
) -> tuple[bool, float]:
    edge_ids = tuple(map(str, route.get("edge_ids") or []))
    edge_lengths = np.asarray(route.get("edge_lengths_m") or [], dtype=float)
    path = tuple(map(str, route.get("path") or []))
    if edge_lengths.shape != (len(edge_ids),):
        raise ValueError(
            f"route {route.get('route_id')!r} has inconsistent edge IDs/lengths"
        )
    total_length = float(np.sum(edge_lengths))
    if total_length <= 0.0:
        raise ValueError(f"route {route.get('route_id')!r} has nonpositive length")
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    if kind == "edge" and target in edge_ids:
        index = edge_ids.index(target)
        return True, float(
            (cumulative[index] + 0.5 * edge_lengths[index]) / total_length
        )
    if kind == "node" and target in path[1:-1]:
        index = path.index(target)
        return True, float(cumulative[index] / total_length)
    return False, 0.0


def _edge_label(u: str, v: str) -> str:
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


def _left_span_embedding(
    raw: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            f"{label} feature matrix must be nonempty and two-dimensional"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} feature matrix contains non-finite values")
    left, singular_values, _right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        max(matrix.shape) * np.finfo(float).eps * singular_values[0]
        if singular_values.size else 0.0
    )
    rank = int(np.sum(singular_values > tolerance))
    if rank < 1:
        raise ValueError(f"{label} feature matrix has zero rank")
    return left[:, :rank], rank, singular_values[:rank]
