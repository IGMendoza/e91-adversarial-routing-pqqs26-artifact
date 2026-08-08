"""Outcome-independent case partitions for histogram learner evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np


SELECTION_FEATURE_SCHEMA = "histogram_topology_structural_features.v1"

_CONTINUOUS_FEATURES = (
    "log1p_graph_nodes",
    "log1p_graph_edges",
    "log1p_route_count",
    "log_alice_compression",
    "log1p_eve_action_count",
    "log_eve_compression",
    "route_hops_minimum",
    "route_hops_mean",
    "route_hops_maximum",
    "mean_internal_node_jaccard",
    "mean_edge_jaccard",
    "node_cut_size",
    "node_cut_present",
    "edge_cut_size",
    "edge_cut_present",
    "route_resource_incidence_density",
    "log1p_cache_mib",
    "dag_compatible",
)


def freeze_case_partitions(
    census: dict[str, Any],
    *,
    development_case_ids: Sequence[str],
    calibration_count: int = 5,
    evaluation_count: int = 5,
) -> dict[str, Any]:
    """Freeze graph-disjoint development, calibration, and evaluation cases.

    Selection uses only topology, route-incidence, learner-feature dimension,
    and declared resource-demand fields produced without physical histograms.
    The sequential maximin rule spreads the selected cases across that measured
    structural space. Alternating the selected order assigns calibration and
    evaluation roles without consulting any outcome.
    """

    if calibration_count < 1 or evaluation_count < 1:
        raise ValueError("calibration_count and evaluation_count must be positive")
    rows = list(census.get("cases") or [])
    by_id = {str(row["case_id"]): row for row in rows}
    development_ids = tuple(str(case_id) for case_id in development_case_ids)
    missing = sorted(set(development_ids) - set(by_id))
    if missing:
        raise ValueError(f"development cases absent from census: {missing}")
    development_rows = [by_id[case_id] for case_id in development_ids]
    development_graphs = {str(row["graph_id"]) for row in development_rows}

    candidates = [
        row
        for row in rows
        if bool(row.get("selection_eligible"))
        and str(row["graph_id"]) not in development_graphs
    ]
    requested = calibration_count + evaluation_count
    if len({str(row["graph_id"]) for row in candidates}) < requested:
        raise ValueError("not enough eligible, graph-distinct cases for partitions")

    feature_spec = build_selection_feature_matrix(candidates)
    selected = _select_structural_maximin(
        candidates,
        feature_spec["standardized_matrix"],
        count=requested,
    )
    calibration_selected = selected[0::2][:calibration_count]
    evaluation_selected = selected[1::2][:evaluation_count]
    if len(calibration_selected) != calibration_count or len(evaluation_selected) != evaluation_count:
        raise RuntimeError("alternating partition assignment produced wrong counts")

    calibration = _partition_rows(
        "calibration", calibration_selected, candidates, feature_spec
    )
    evaluation = _partition_rows(
        "evaluation", evaluation_selected, candidates, feature_spec
    )
    development = [
        _case_record(row, selection_order=None, maximin_distance=None)
        for row in development_rows
    ]
    selection_core = {
        "schema": "histogram_topology_case_manifest.v1",
        "census_schema": census.get("schema"),
        "corpus": dict(census.get("corpus") or {}),
        "selection_method": {
            "name": "deterministic_structural_maximin",
            "first_case": (
                "eligible case nearest the component-wise median in standardized "
                "feature space"
            ),
            "subsequent_cases": (
                "eligible case maximizing distance to its nearest already-selected "
                "case, with one assignment per graph"
            ),
            "tie_break": "lexicographic case_id",
            "role_assignment": (
                "alternate maximin order between calibration and evaluation"
            ),
            "feature_schema": SELECTION_FEATURE_SCHEMA,
            "continuous_features": list(_CONTINUOUS_FEATURES),
            "categorical_features": feature_spec["categorical_features"],
            "scaling": (
                "continuous columns centered at the median and divided by IQR; "
                "zero-IQR columns divided by one; categorical columns are one-hot"
            ),
        },
        "information_boundary": {
            "allowed": [
                "graph size and assignment classification",
                "complete cycle-free route geometry",
                "Alice route-resource feature rank",
                "Eve topology/action feature rank",
                "declared action-surface cache and audit demand",
                "bounded DAG compatibility",
            ],
            "excluded": [
                "compiled histogram outcomes",
                "service rate, delivery probability, QBER, and CHSH",
                "Alice utility and Eve reward",
                "learner trajectories, regret, exploitability, and Nash gap",
                "bilinear payoff fit and residual",
            ],
            "payoff_residual_status": "not_measured_before_case_selection",
        },
        "graph_disjointness": {
            "development_vs_calibration": True,
            "development_vs_evaluation": True,
            "calibration_vs_evaluation": True,
            "one_selected_assignment_per_graph": True,
        },
        "eligible_candidate_count": len(candidates),
        "eligible_candidate_graph_count": len({row["graph_id"] for row in candidates}),
        "execution_envelope": {
            "default_concurrent_cases": 1,
            "selected_case_cache_limit_mib": float(
                census["config"]["selected_case_cache_limit_mib"]
            ),
            "selected_case_audit_draw_limit": int(
                census["config"]["selected_case_audit_draw_limit"]
            ),
            "host_available_ram_used_to_select_cases": False,
            "parallel_execution_boundary": (
                "multiple case processes require a fresh RAM preflight using the "
                "sum of their planned cache demand plus process and learner state; "
                "runtime backend diagnostics remain authoritative"
            ),
            "cache_reuse_boundary": (
                "compile cost is paid once per case configuration when one physics "
                "pool is reused across matched learner comparisons"
            ),
            "partition_resource_totals": {
                "calibration": _partition_resource_summary(calibration),
                "evaluation": _partition_resource_summary(evaluation),
            },
        },
        "partitions": {
            "development_reference": development,
            "calibration": calibration,
            "evaluation": evaluation,
        },
    }
    _assert_partition_contract(selection_core)
    selection_core["selection_digest_blake2b_128"] = _stable_digest(selection_core)
    return selection_core


def build_selection_feature_matrix(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return auditable raw and standardized structural selection features."""

    if not rows:
        raise ValueError("selection feature matrix requires at least one row")
    ordered_rows = list(rows)
    continuous = np.asarray(
        [[_continuous_value(row, name) for name in _CONTINUOUS_FEATURES] for row in ordered_rows],
        dtype=float,
    )
    medians = np.median(continuous, axis=0)
    q25 = np.quantile(continuous, 0.25, axis=0)
    q75 = np.quantile(continuous, 0.75, axis=0)
    scales = q75 - q25
    scales[scales == 0.0] = 1.0
    standardized_continuous = (continuous - medians) / scales

    categorical_fields = ("assignment_policy", "strategy_validity")
    categories = {
        field: tuple(sorted({str(row[field]) for row in ordered_rows}))
        for field in categorical_fields
    }
    categorical_names = [
        f"{field}={value}"
        for field in categorical_fields
        for value in categories[field]
    ]
    categorical = np.asarray(
        [
            [
                float(str(row[field]) == value)
                for field in categorical_fields
                for value in categories[field]
            ]
            for row in ordered_rows
        ],
        dtype=float,
    )
    matrix = np.column_stack((standardized_continuous, categorical))
    return {
        "case_ids": [str(row["case_id"]) for row in ordered_rows],
        "raw_continuous_matrix": continuous,
        "standardized_matrix": matrix,
        "continuous_medians": medians,
        "continuous_scales": scales,
        "continuous_features": list(_CONTINUOUS_FEATURES),
        "categorical_features": categorical_names,
    }


def selection_coordinates(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Project the audited selection matrix to two deterministic SVD axes."""

    features = build_selection_feature_matrix(rows)
    matrix = np.asarray(features["standardized_matrix"], dtype=float)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    width = min(2, u.shape[1])
    coordinates = u[:, :width] * singular_values[:width]
    if width == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(rows))))
    total_variance = float(np.sum(singular_values**2))
    explained = [
        float(value**2 / total_variance) if total_variance else 0.0
        for value in singular_values[:2]
    ]
    while len(explained) < 2:
        explained.append(0.0)
    return {
        "case_ids": features["case_ids"],
        "coordinates": coordinates,
        "explained_variance_fraction": explained,
    }


def _select_structural_maximin(
    rows: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    *,
    count: int,
) -> list[dict[str, Any]]:
    if matrix.shape[0] != len(rows):
        raise ValueError("feature rows do not match candidate rows")
    median_point = np.median(matrix, axis=0)
    center_distances = np.linalg.norm(matrix - median_point, axis=1)
    first_index = min(
        range(len(rows)),
        key=lambda index: (float(center_distances[index]), str(rows[index]["case_id"])),
    )
    chosen = [first_index]
    chosen_graphs = {str(rows[first_index]["graph_id"])}
    selection_distances = [float(center_distances[first_index])]
    while len(chosen) < count:
        available = [
            index
            for index, row in enumerate(rows)
            if index not in chosen and str(row["graph_id"]) not in chosen_graphs
        ]
        if not available:
            raise RuntimeError("structural maximin exhausted graph-distinct candidates")
        scored = []
        for index in available:
            distance = min(
                float(np.linalg.norm(matrix[index] - matrix[selected]))
                for selected in chosen
            )
            scored.append((distance, str(rows[index]["case_id"]), index))
        distance, _, next_index = min(
            scored,
            key=lambda item: (-item[0], item[1]),
        )
        chosen.append(next_index)
        chosen_graphs.add(str(rows[next_index]["graph_id"]))
        selection_distances.append(distance)
    return [
        {
            "row": rows[index],
            "selection_order": order,
            "maximin_distance": selection_distances[order],
        }
        for order, index in enumerate(chosen)
    ]


def _partition_rows(
    role: str,
    selected: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    feature_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    index_by_id = {
        str(case_id): index for index, case_id in enumerate(feature_spec["case_ids"])
    }
    records = []
    for item in selected:
        row = item["row"]
        index = index_by_id[str(row["case_id"])]
        raw = feature_spec["raw_continuous_matrix"][index]
        values = {
            name: float(value)
            for name, value in zip(_CONTINUOUS_FEATURES, raw)
        }
        record = _case_record(
            row,
            selection_order=int(item["selection_order"]),
            maximin_distance=float(item["maximin_distance"]),
        )
        record["partition_role"] = role
        record["selection_continuous_features"] = values
        records.append(record)
    return records


def _case_record(
    row: dict[str, Any],
    *,
    selection_order: int | None,
    maximin_distance: float | None,
) -> dict[str, Any]:
    keys = (
        "case_id",
        "graph_id",
        "assignment_id",
        "family",
        "assignment_policy",
        "strategy_validity",
        "alice",
        "bob",
        "graph_node_count",
        "graph_edge_count",
        "route_count",
        "route_hops_minimum",
        "route_hops_mean",
        "route_hops_maximum",
        "node_cut_size",
        "edge_cut_size",
        "alice_feature_rank",
        "eve_action_count",
        "eve_feature_rank",
        "canonical_cache_entries",
        "estimated_cache_mib",
        "audit_draws_per_seed",
        "estimated_first_configuration_physics_seconds",
        "exact_bounded_dag_compatible",
    )
    return {
        **{key: row.get(key) for key in keys},
        "selection_order": selection_order,
        "maximin_distance": maximin_distance,
    }


def _partition_resource_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "maximum_single_case_cache_mib": max(
            float(row["estimated_cache_mib"]) for row in rows
        ),
        "sum_cache_mib_if_all_cases_resident": sum(
            float(row["estimated_cache_mib"]) for row in rows
        ),
        "estimated_first_configuration_physics_seconds_serial": sum(
            float(row["estimated_first_configuration_physics_seconds"])
            for row in rows
        ),
        "maximum_single_case_audit_draws_per_seed": max(
            int(row["audit_draws_per_seed"]) for row in rows
        ),
    }


def _continuous_value(row: dict[str, Any], name: str) -> float:
    mapping = {
        "log1p_graph_nodes": math.log1p(float(row["graph_node_count"])),
        "log1p_graph_edges": math.log1p(float(row["graph_edge_count"])),
        "log1p_route_count": math.log1p(float(row["route_count"])),
        "log_alice_compression": math.log(float(row["alice_compression_ratio"])),
        "log1p_eve_action_count": math.log1p(float(row["eve_action_count"])),
        "log_eve_compression": math.log(float(row["eve_compression_ratio"])),
        "route_hops_minimum": float(row["route_hops_minimum"]),
        "route_hops_mean": float(row["route_hops_mean"]),
        "route_hops_maximum": float(row["route_hops_maximum"]),
        "mean_internal_node_jaccard": float(row["mean_internal_node_jaccard"]),
        "mean_edge_jaccard": float(row["mean_edge_jaccard"]),
        "node_cut_size": float(row["node_cut_size"] or 0),
        "node_cut_present": float(row["node_cut_size"] is not None),
        "edge_cut_size": float(row["edge_cut_size"] or 0),
        "edge_cut_present": float(row["edge_cut_size"] is not None),
        "route_resource_incidence_density": float(row["route_resource_incidence_density"]),
        "log1p_cache_mib": math.log1p(float(row["estimated_cache_mib"])),
        "dag_compatible": float(bool(row["exact_bounded_dag_compatible"])),
    }
    return mapping[name]


def _assert_partition_contract(manifest: dict[str, Any]) -> None:
    partitions = manifest["partitions"]
    graph_sets = {
        name: {str(row["graph_id"]) for row in rows}
        for name, rows in partitions.items()
    }
    names = tuple(graph_sets)
    for left_index, left in enumerate(names):
        if len(graph_sets[left]) != len(partitions[left]):
            raise RuntimeError(f"partition {left!r} repeats a graph")
        for right in names[left_index + 1 :]:
            overlap = graph_sets[left] & graph_sets[right]
            if overlap:
                raise RuntimeError(
                    f"partitions {left!r} and {right!r} share graphs: {sorted(overlap)}"
                )


def _stable_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()
