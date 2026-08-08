"""Topology-held-out checks for Exp3 sequence decision-tree summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .dt_rows import ACTION_FEATURE_NAMES, GRAPH_FEATURE_NAMES, ROUTE_FEATURE_NAMES


@dataclass(frozen=True)
class DTHoldoutTarget:
    display_name: str
    rows_key: str
    target_key: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class DTHoldoutResult:
    target: str
    rows: int
    topologies: int
    train_r2: float
    topology_heldout_r2: float
    topology_heldout_mae: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "rows": self.rows,
            "topologies": self.topologies,
            "train_r2": self.train_r2,
            "topology_heldout_r2": self.topology_heldout_r2,
            "topology_heldout_mae": self.topology_heldout_mae,
        }


ORACLE_DT_HOLDOUT_TARGETS = (
    DTHoldoutTarget(
        display_name="Oracle graph retention",
        rows_key="graph_rows",
        target_key="oracle_retention",
        feature_names=tuple(GRAPH_FEATURE_NAMES),
    ),
    DTHoldoutTarget(
        display_name="Oracle Eve action probability",
        rows_key="action_rows",
        target_key="oracle_eve_strategy_prob",
        feature_names=tuple(ACTION_FEATURE_NAMES),
    ),
    DTHoldoutTarget(
        display_name="Expected denial vs. oracle Alice",
        rows_key="action_rows",
        target_key="action_expected_denial_under_oracle_alice",
        feature_names=tuple(ACTION_FEATURE_NAMES),
    ),
    DTHoldoutTarget(
        display_name="Oracle Alice route probability",
        rows_key="route_rows",
        target_key="oracle_alice_strategy_prob",
        feature_names=tuple(ROUTE_FEATURE_NAMES),
    ),
)


def evaluate_dt_holdout_payload(
        payload: dict[str, Any],
        *,
        max_depth: int = 3,
) -> list[DTHoldoutResult]:
    """Evaluate the oracle DT targets with pooled topology-held-out metrics."""
    return [
        evaluate_dt_holdout_target(
            payload.get(spec.rows_key, []),
            spec,
            max_depth=max_depth,
        )
        for spec in ORACLE_DT_HOLDOUT_TARGETS
    ]


def evaluate_dt_holdout_target(
        rows: list[dict[str, Any]],
        spec: DTHoldoutTarget,
        *,
        max_depth: int = 3,
) -> DTHoldoutResult:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.tree import DecisionTreeRegressor

    valid = _valid_rows(rows, spec.target_key)
    if not valid:
        raise ValueError(f"no valid rows for target {spec.target_key!r}")

    X = np.asarray([row["features"] for row in valid], dtype=float)
    y = np.asarray([row[spec.target_key] for row in valid], dtype=float)
    groups = np.asarray([row["graph_id"] for row in valid], dtype=object)
    topologies = int(len(set(groups.tolist())))
    if topologies < 2:
        raise ValueError(
            f"need at least two graph_id groups for target {spec.target_key!r}"
        )

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X, y)
    train_r2 = _r2(y, tree.predict(X))

    oof_pred = np.empty_like(y, dtype=float)
    splitter = LeaveOneGroupOut()
    for train_index, test_index in splitter.split(X, y, groups):
        fold_tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        fold_tree.fit(X[train_index], y[train_index])
        oof_pred[test_index] = fold_tree.predict(X[test_index])

    return DTHoldoutResult(
        target=spec.display_name,
        rows=int(y.size),
        topologies=topologies,
        train_r2=float(train_r2),
        topology_heldout_r2=float(_r2(y, oof_pred)),
        topology_heldout_mae=float(np.mean(np.abs(y - oof_pred))),
    )


def markdown_table(results: list[DTHoldoutResult]) -> str:
    lines = [
        "| Target | Rows | Topologies | Train R2 | Topology-held-out R2 | Topology-held-out MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.target} | {result.rows} | {result.topologies} | "
            f"{result.train_r2:.4f} | {result.topology_heldout_r2:.4f} | "
            f"{result.topology_heldout_mae:.4f} |"
        )
    return "\n".join(lines)


def strong_generalization_drops(
        results: list[DTHoldoutResult],
        *,
        min_drop: float = 0.25,
) -> list[DTHoldoutResult]:
    return [
        result for result in results
        if result.topology_heldout_r2 < 0.0
        or result.train_r2 - result.topology_heldout_r2 >= min_drop
    ]


def _valid_rows(rows: list[dict[str, Any]], target_key: str) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in rows:
        target = row.get(target_key)
        features = row.get("features")
        graph_id = row.get("graph_id")
        if target is None or features is None or graph_id is None:
            continue
        target_value = float(target)
        feature_values = [float(value) for value in features]
        if not math.isfinite(target_value):
            continue
        if not all(math.isfinite(value) for value in feature_values):
            continue
        clean = dict(row)
        clean[target_key] = target_value
        clean["features"] = feature_values
        valid.append(clean)
    return valid


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
