"""Decision-tree surrogate for Q-table policy explanation.

Builds a sklearn DT that mimics the Q-table's greedy policy from
(state_encoding + route_features) → action.  Depth is capped at 4
to keep paths human-readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_NAMES = [
    "prev_outcome_is_failure",
    "prev_action_targeted_r0",       # attack_surface: swap_denial_r0 / memory_degradation_r0
    "prev_action_targeted_r1",       # attack_surface: swap_denial_r1 / memory_degradation_r1
    "prev_action_was_denial",        # attack_surface: swap_denial_*
    "prev_action_was_node_attack",   # binary_collision: node_*
    "prev_action_was_edge_attack",   # binary_collision: edge_*
    "prev_action_was_no_attack",     # any mode: none / no_attack
    "r0_bottleneck_score",
    "r1_bottleneck_score",
    "r0_hops",
    "r1_hops",
]


@dataclass
class RouteFeatures:
    bottleneck_score: float
    hops: int
    length_m: float
    overlap_with_other: int


@dataclass
class SurrogateDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    action_names: list[str]


_FAILURE_OUTCOMES_SURROGATE = {"delivery_failure", "hit", "blocked", "qber_abort"}


def encode_state(prev_action: str, prev_outcome: str) -> list[float]:
    # "hit" = binary_collision: Eve hit Alice's route = delivery failed
    # "miss" = binary_collision: Eve missed = Alice delivered (not a failure)
    is_failure = prev_outcome in _FAILURE_OUTCOMES_SURROGATE
    return [
        1.0 if is_failure else 0.0,
        1.0 if "r0" in prev_action else 0.0,          # attack_surface repeater names
        1.0 if "r1" in prev_action else 0.0,          # attack_surface repeater names
        1.0 if "denial" in prev_action else 0.0,      # attack_surface swap_denial
        1.0 if prev_action.startswith("node_") else 0.0,   # binary_collision node attack
        1.0 if prev_action.startswith("edge_") else 0.0,   # binary_collision edge attack
        1.0 if prev_action in {"none", "no_attack"} else 0.0,
    ]


def build_dataset(
    q_table_entries: list[dict[str, Any]],
    action_names: list[str],
    r0_features: RouteFeatures,
    r1_features: RouteFeatures,
) -> SurrogateDataset:
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for entry in q_table_entries:
        state = entry["state"]
        q_vals = entry["q"]
        best_idx = max(range(len(q_vals)), key=lambda i: (q_vals[i], -i))
        state_feats = encode_state(state[0], state[1])
        route_feats = [
            r0_features.bottleneck_score,
            r1_features.bottleneck_score,
            float(r0_features.hops),
            float(r1_features.hops),
        ]
        X_rows.append(state_feats + route_feats)
        y_rows.append(best_idx)
    return SurrogateDataset(
        X=np.array(X_rows, dtype=float),
        y=np.array(y_rows, dtype=int),
        feature_names=FEATURE_NAMES,
        action_names=action_names,
    )


def fit_surrogate_dt(dataset: SurrogateDataset, max_depth: int = 4):
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(dataset.X, dataset.y)
    return dt


@dataclass
class DTPathStep:
    feature: str
    threshold: float
    value: float
    went_left: bool
    depth: int


@dataclass
class DTPath:
    steps: list[DTPathStep]
    predicted_action: str
    predicted_action_idx: int
    fidelity: float


def extract_decision_path(dt, dataset: SurrogateDataset, query_X: np.ndarray) -> DTPath:
    from sklearn.tree import _tree
    tree_ = dt.tree_
    node = 0
    steps: list[DTPathStep] = []
    while tree_.feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = int(tree_.feature[node])
        thresh = float(tree_.threshold[node])
        val = float(query_X[0, feat_idx])
        went_left = val <= thresh
        steps.append(DTPathStep(
            feature=dataset.feature_names[feat_idx],
            threshold=round(thresh, 4),
            value=round(val, 4),
            went_left=went_left,
            depth=len(steps),
        ))
        node = int(tree_.children_left[node] if went_left else tree_.children_right[node])

    # sklearn internally remaps class labels; dt.classes_ maps internal index → original label
    internal_idx = int(np.argmax(tree_.value[node]))
    predicted_idx = int(dt.classes_[internal_idx])
    fidelity = float(np.mean(dt.predict(dataset.X) == dataset.y))
    return DTPath(
        steps=steps,
        predicted_action=dataset.action_names[predicted_idx],
        predicted_action_idx=predicted_idx,
        fidelity=fidelity,
    )


def format_path_text(path: DTPath) -> str:
    lines: list[str] = []
    for step in path.steps:
        indent = "  " * step.depth
        op = "<=" if step.went_left else ">"
        lines.append(f"{indent}IF {step.feature} {op} {step.threshold}  [actual={step.value}]")
    lines.append(f"{'  ' * len(path.steps)}→ PREDICT: {path.predicted_action}")
    lines.append(f"(DT fidelity on training set: {path.fidelity:.2%})")
    return "\n".join(lines)
