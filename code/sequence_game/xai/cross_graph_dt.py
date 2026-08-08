"""Cross-graph structural DT (Path B-lite).

Pools structural rows from one corpus emission class and fits a single
DecisionTreeRegressor.  The resulting DT emits rules like
"IF in_node_hs=1 AND route_coverage > 0.4 → high Eve Q-value" that hold
across graph families within the same experiment design because the features
are structural, not node-ID-specific.

Scientific scope: toy — DT fidelity reflects learned Q-values, not security claims.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from sequence_game.corpus.runner import STRUCTURAL_FEATURE_NAMES


@dataclass
class StructuralDTResult:
    dt: Any                        # sklearn DecisionTreeRegressor
    feature_names: list[str]
    target_key: str
    num_rows: int
    num_topologies: int            # unique graph topologies (< num assignments)
    r2_score: float
    rules_text: str


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_cross_graph_dt(
    all_rows: list[dict[str, Any]],
    *,
    max_depth: int = 4,
    target_key: str = "q_rank",
    feature_names: list[str] | None = None,
) -> StructuralDTResult:
    """Fit a pooled DT on structural rows from all corpus graphs.

    `all_rows` is the union of a single emission class across all accepted
    graphs. Each row has keys: features (list[float]) and `target_key` (float).
    """
    from sklearn.tree import DecisionTreeRegressor

    if not all_rows:
        raise ValueError("No structural rows to fit — run corpus pipeline first.")
    if target_key not in all_rows[0]:
        raise ValueError(f"Structural rows do not contain target {target_key!r}.")

    X = np.array([r["features"] for r in all_rows], dtype=float)
    y = np.array([r[target_key] for r in all_rows], dtype=float)
    names = feature_names or STRUCTURAL_FEATURE_NAMES

    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X, y)

    y_pred = dt.predict(X)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    num_topologies = len({r["graph_id"] for r in all_rows})
    rules = _extract_rules_text(dt, names, _leaf_label(target_key))

    return StructuralDTResult(
        dt=dt,
        feature_names=names,
        target_key=target_key,
        num_rows=len(all_rows),
        num_topologies=num_topologies,
        r2_score=round(r2, 4),
        rules_text=rules,
    )


# ── Rule extraction ───────────────────────────────────────────────────────────

def _extract_rules_text(dt, feature_names: list[str], leaf_label: str) -> str:
    """Export DT as human-readable IF/THEN rules (one per leaf, depth-first)."""
    from sklearn.tree import _tree

    tree_ = dt.tree_
    lines: list[str] = []

    def _recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            mean_q = float(tree_.value[node].ravel().mean())
            n_samples = int(tree_.n_node_samples[node])
            prefix = " AND ".join(conditions) if conditions else "TRUE"
            lines.append(
                f"IF {prefix}\n"
                f"  → {leaf_label} = {mean_q:.3f}  (n={n_samples})"
            )
            return
        feat = feature_names[int(tree_.feature[node])]
        thresh = float(tree_.threshold[node])
        _recurse(int(tree_.children_left[node]),
                 conditions + [f"{feat} <= {thresh:.3f}"])
        _recurse(int(tree_.children_right[node]),
                 conditions + [f"{feat} > {thresh:.3f}"])

    _recurse(0, [])
    return "\n\n".join(lines)


def _leaf_label(target_key: str) -> str:
    if target_key == "q_rank":
        return "avg_q_rank"
    return f"avg_{target_key}"


# ── Aggregate stats ───────────────────────────────────────────────────────────

def aggregate_corpus_stats(
    per_graph_records: list[dict[str, Any]],
    dt_result: StructuralDTResult,
) -> dict[str, Any]:
    """Compute per-family summary stats over all per-graph game records."""
    from collections import defaultdict

    by_family: dict[str, list[dict]] = defaultdict(list)
    for rec in per_graph_records:
        by_family[rec["family"]].append(rec)

    family_stats = {}
    for fam, recs in sorted(by_family.items()):
        win_rates = [r["eve_win_rate"] for r in recs]
        final_rates = [r["final_win_rate"] for r in recs]
        Ns = [r["graph_N"] for r in recs if r.get("graph_N") is not None]
        Es = [r["graph_E"] for r in recs if r.get("graph_E") is not None]
        family_stats[fam] = {
            "count": len(recs),
            "eve_win_rate_mean": round(float(np.mean(win_rates)), 4),
            "eve_win_rate_std": round(float(np.std(win_rates)), 4),
            "final_win_rate_mean": round(float(np.mean(final_rates)), 4),
            "final_win_rate_std": round(float(np.std(final_rates)), 4),
            "N_mean": round(float(np.mean(Ns)), 2) if Ns else None,
            "E_mean": round(float(np.mean(Es)), 2) if Es else None,
        }

    # Top structural predictors from feature importances
    importances = dt_result.dt.feature_importances_
    ranked = sorted(
        zip(dt_result.feature_names, importances),
        key=lambda x: -x[1],
    )
    top_features = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in ranked
        if imp > 0.01
    ]

    all_win = [r["eve_win_rate"] for r in per_graph_records]
    num_topologies = len({r["graph_id"] for r in per_graph_records})
    return {
        "total_assignments": len(per_graph_records),
        "total_topologies": num_topologies,
        "families": family_stats,
        "overall_eve_win_rate_mean": round(float(np.mean(all_win)), 4),
        "overall_eve_win_rate_std": round(float(np.std(all_win)), 4),
        "structural_dt": {
            "num_rows": dt_result.num_rows,
            "num_topologies": dt_result.num_topologies,
            "target_key": dt_result.target_key,
            "r2_score": dt_result.r2_score,
            "top_features": top_features,
            "rules_text": dt_result.rules_text,
        },
    }
