"""Decision-tree artifacts for the histogram co-learning paper bundle.

These trees are compact descriptions of one fixed four-topology experiment.
They are not security models and, with so few independent topologies, they do
not support an out-of-sample topology-generalization claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


DECISION_TREE_SCHEMA = "histogram_colearning_decision_tree_v1"
VALIDATION_SCOPE = (
    "in_sample_descriptive_with_exploratory_leave_one_topology_out_diagnostic"
)


@dataclass(frozen=True)
class PaperDecisionTreeBundle:
    """Written decision-tree files plus in-memory prompt context."""

    manifest: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class _TreeSpec:
    name: str
    title: str
    question: str
    unit_of_analysis: str
    target: str
    target_label: str
    feature_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    max_depth: int


def write_decision_tree_artifacts(
    out_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    source_schema: str,
    source_backend: str,
) -> PaperDecisionTreeBundle:
    """Fit and export three histogram-specific descriptive decision trees."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = _tree_specs(tables)
    artifacts: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    entries: list[dict[str, Any]] = []

    for spec in specs:
        artifact, estimator = _fit_tree(
            spec,
            source_schema=source_schema,
            source_backend=source_backend,
        )
        artifacts[spec.name] = artifact
        stem = output / spec.name
        json_path = stem.with_suffix(".json")
        rules_path = stem.with_suffix(".txt")
        dot_path = stem.with_suffix(".dot")
        png_path = stem.with_suffix(".png")
        pdf_path = stem.with_suffix(".pdf")

        _write_json(json_path, artifact)
        rules_path.write_text(_rules_document(artifact), encoding="utf-8")
        _write_dot(dot_path, estimator, spec)
        _write_tree_figure(png_path, estimator, spec)
        _write_tree_figure(pdf_path, estimator, spec)
        tree_paths = [json_path, rules_path, dot_path, png_path, pdf_path]
        paths.extend(str(path) for path in tree_paths)
        entries.append({
            "name": spec.name,
            "title": spec.title,
            "unit_of_analysis": spec.unit_of_analysis,
            "target": spec.target,
            "sample_count": len(spec.rows),
            "case_count": len({str(row["case_slug"]) for row in spec.rows}),
            "files": [path.name for path in tree_paths],
            "in_sample_r2": artifact["metrics"]["in_sample_r2"],
            "exploratory_leave_one_topology_out_r2": artifact["metrics"][
                "exploratory_leave_one_topology_out_r2"
            ],
        })

    manifest_path = output / "manifest.json"
    manifest = {
        "schema": DECISION_TREE_SCHEMA,
        "source_schema": source_schema,
        "source_backend": source_backend,
        "tree_count": len(entries),
        "validation_scope": VALIDATION_SCOPE,
        "generalization_claim": "none",
        "independent_topology_count": len(tables["case_summary"]),
        "limitation": (
            "Only four fixed Topology Bench assignments were evaluated. Tree "
            "fits and leave-one-topology-out scores are descriptive diagnostics, "
            "not validated predictors for unseen networks."
        ),
        "trees": entries,
    }
    _write_json(manifest_path, manifest)
    paths.append(str(manifest_path))
    return PaperDecisionTreeBundle(
        manifest=manifest,
        artifacts=artifacts,
        paths=tuple(paths),
    )


def _tree_specs(tables: dict[str, list[dict[str, Any]]]) -> tuple[_TreeSpec, ...]:
    graph_rows = _graph_rows(tables)
    route_rows = _route_rows(tables)
    eve_rows, eve_features = _eve_rows(tables)
    return (
        _TreeSpec(
            name="graph_alice_utility",
            title="Graph structure and observed Alice utility",
            question=(
                "Which declared topology and clean-route physics features partition "
                "the observed mean Alice public utility?"
            ),
            unit_of_analysis="fixed_topology_bench_assignment",
            target="mean_alice_utility_mean",
            target_label="Mean Alice utility",
            feature_names=(
                "route_count",
                "maximum_route_depth",
                "max_route_distance_km",
                "edge_disjoint_pair_fraction",
                "mean_clean_service_rate_hz",
                "minimum_clean_chsh_s",
                "maximum_clean_qber",
            ),
            rows=tuple(graph_rows),
            max_depth=2,
        ),
        _TreeSpec(
            name="alice_route_strategy",
            title="Clean route physics and learned Alice route mass",
            question=(
                "Which pre-decision route and case features partition Alice's "
                "empirical 100k-turn route-selection mass?"
            ),
            unit_of_analysis="route_aggregated_across_three_learner_seeds",
            target="learned_mass_mean",
            target_label="Alice learned route mass",
            feature_names=(
                "hop_count",
                "distance_km",
                "service_rate_hz",
                "clean_expected_chsh_s",
                "clean_expected_qber",
                "route_availability",
                "service_rate_ratio_within_case",
                "distance_ratio_to_case_shortest",
                "route_count",
                "edge_disjoint_pair_fraction",
            ),
            rows=tuple(route_rows),
            max_depth=3,
        ),
        _TreeSpec(
            name="eve_attack_family_strategy",
            title="Case structure and learned Eve attack-family mass",
            question=(
                "Which declared case features and attack-family identities partition "
                "Eve's learned family-level action mass?"
            ),
            unit_of_analysis="attack_family_by_case_aggregated_across_three_learner_seeds",
            target="learned_mass_mean",
            target_label="Eve learned family mass",
            feature_names=tuple(eve_features),
            rows=tuple(eve_rows),
            max_depth=3,
        ),
    )


def _graph_rows(
    tables: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    calibration_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in tables["route_calibration"]:
        calibration_by_case.setdefault(str(row["case_slug"]), []).append(row)
    output = []
    for case in tables["case_summary"]:
        slug = str(case["case_slug"])
        calibrations = calibration_by_case[slug]
        features = {
            "route_count": float(case["route_count"]),
            "maximum_route_depth": float(case["maximum_route_depth"]),
            "max_route_distance_km": float(case["max_route_distance_km"]),
            "edge_disjoint_pair_fraction": float(case["edge_disjoint_pair_fraction"]),
            "mean_clean_service_rate_hz": float(np.mean([
                float(row["service_rate_hz"]) for row in calibrations
            ])),
            "minimum_clean_chsh_s": float(min(
                float(row["clean_expected_chsh_s"]) for row in calibrations
            )),
            "maximum_clean_qber": float(max(
                float(row["clean_expected_qber"]) for row in calibrations
            )),
        }
        output.append({
            "case_slug": slug,
            "family": str(case["family"]),
            "features": features,
            "mean_alice_utility_mean": float(case["mean_alice_utility_mean"]),
        })
    return output


def _route_rows(
    tables: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    cases = {str(row["case_slug"]): row for row in tables["case_summary"]}
    calibration = {
        (str(row["case_slug"]), str(row["route_id"])): row
        for row in tables["route_calibration"]
    }
    grouped = _mean_group(
        tables["alice_route_strategy"],
        keys=("case_slug", "route_index", "route_id"),
        values=("learned_mass", "prior_mass", "oracle_mass"),
    )
    max_service: dict[str, float] = {}
    min_distance: dict[str, float] = {}
    for (slug, _route_id), row in calibration.items():
        max_service[slug] = max(max_service.get(slug, 0.0), float(row["service_rate_hz"]))
        distance = float(row["total_distance_km"])
        min_distance[slug] = min(min_distance.get(slug, distance), distance)

    output = []
    for row in grouped:
        slug = str(row["case_slug"])
        route_id = str(row["route_id"])
        physical = calibration[(slug, route_id)]
        case = cases[slug]
        distance = float(physical["total_distance_km"])
        service = float(physical["service_rate_hz"])
        features = {
            "hop_count": float(physical["hop_count"]),
            "distance_km": distance,
            "service_rate_hz": service,
            "clean_expected_chsh_s": float(physical["clean_expected_chsh_s"]),
            "clean_expected_qber": float(physical["clean_expected_qber"]),
            "route_availability": float(physical["route_availability"]),
            "service_rate_ratio_within_case": service / max_service[slug],
            "distance_ratio_to_case_shortest": distance / min_distance[slug],
            "route_count": float(case["route_count"]),
            "edge_disjoint_pair_fraction": float(case["edge_disjoint_pair_fraction"]),
        }
        output.append({
            "case_slug": slug,
            "route_index": int(row["route_index"]),
            "route_id": route_id,
            "features": features,
            "learned_mass_mean": float(row["learned_mass_mean"]),
            "learned_mass_sd": float(row["learned_mass_sd"]),
            "prior_mass_mean": float(row["prior_mass_mean"]),
            "oracle_mass_mean": _optional_float(row["oracle_mass_mean"]),
        })
    return output


def _eve_rows(
    tables: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    cases = {str(row["case_slug"]): row for row in tables["case_summary"]}
    grouped = _mean_group(
        tables["eve_family_strategy"],
        keys=("case_slug", "attack_family"),
        values=("target_action_count", "prior_mass", "learned_mass", "oracle_mass"),
    )
    families = sorted({str(row["attack_family"]) for row in grouped})
    feature_names = [
        "target_action_count",
        "prior_mass",
        "route_count",
        "min_hops",
        "maximum_route_depth",
        "max_route_distance_km",
        "edge_disjoint_pair_fraction",
    ] + [f"family_is_{family}" for family in families]
    output = []
    for row in grouped:
        slug = str(row["case_slug"])
        family = str(row["attack_family"])
        case = cases[slug]
        features = {
            "target_action_count": float(row["target_action_count_mean"]),
            "prior_mass": float(row["prior_mass_mean"]),
            "route_count": float(case["route_count"]),
            "min_hops": float(case["min_hops"]),
            "maximum_route_depth": float(case["maximum_route_depth"]),
            "max_route_distance_km": float(case["max_route_distance_km"]),
            "edge_disjoint_pair_fraction": float(case["edge_disjoint_pair_fraction"]),
            **{
                f"family_is_{candidate}": float(family == candidate)
                for candidate in families
            },
        }
        output.append({
            "case_slug": slug,
            "attack_family": family,
            "features": features,
            "learned_mass_mean": float(row["learned_mass_mean"]),
            "learned_mass_sd": float(row["learned_mass_sd"]),
            "prior_mass_mean": float(row["prior_mass_mean"]),
            "oracle_mass_mean": _optional_float(row["oracle_mass_mean"]),
        })
    return output, feature_names


def _fit_tree(
    spec: _TreeSpec,
    *,
    source_schema: str,
    source_backend: str,
) -> tuple[dict[str, Any], Any]:
    from sklearn.tree import DecisionTreeRegressor, export_text

    if not spec.rows:
        raise ValueError(f"decision tree {spec.name!r} has no rows")
    x = np.asarray([
        [float(row["features"][name]) for name in spec.feature_names]
        for row in spec.rows
    ], dtype=float)
    y = np.asarray([float(row[spec.target]) for row in spec.rows], dtype=float)
    estimator = DecisionTreeRegressor(max_depth=spec.max_depth, random_state=42)
    estimator.fit(x, y)
    predicted = estimator.predict(x)
    holdout = _leave_one_topology_out(spec, x, y)
    rules = export_text(
        estimator,
        feature_names=list(spec.feature_names),
        decimals=4,
        show_weights=True,
    ).rstrip()
    feature_importances = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(spec.feature_names, estimator.feature_importances_),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]
    artifact = {
        "schema": DECISION_TREE_SCHEMA,
        "source_schema": source_schema,
        "source_backend": source_backend,
        "deprecated_sequence_backend_data_used": False,
        "name": spec.name,
        "title": spec.title,
        "question": spec.question,
        "unit_of_analysis": spec.unit_of_analysis,
        "target": spec.target,
        "target_label": spec.target_label,
        "feature_names": list(spec.feature_names),
        "sample_count": len(spec.rows),
        "independent_topology_count": len({str(row["case_slug"]) for row in spec.rows}),
        "max_depth": spec.max_depth,
        "validation_scope": VALIDATION_SCOPE,
        "generalization_claim": "none",
        "scientific_boundary": (
            "The target describes learned behavior or public utility in this fixed "
            "histogram-backend run. It is not a physical law, attack oracle, finite-key "
            "security result, or validated predictor for unseen topologies."
        ),
        "metrics": {
            "in_sample_r2": _r2(y, predicted),
            "in_sample_mean_absolute_error": float(np.mean(np.abs(y - predicted))),
            **holdout["metrics"],
        },
        "exploratory_leave_one_topology_out_predictions": holdout["predictions"],
        "feature_importances": feature_importances,
        "rules_text": rules,
        "tree": _serialize_tree(estimator, spec.feature_names),
        "rows": [
            {
                **{key: value for key, value in row.items() if key != "features"},
                "features": dict(row["features"]),
                "in_sample_prediction": float(prediction),
            }
            for row, prediction in zip(spec.rows, predicted)
        ],
    }
    return artifact, estimator


def _leave_one_topology_out(
    spec: _TreeSpec,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    from sklearn.tree import DecisionTreeRegressor

    groups = np.asarray([str(row["case_slug"]) for row in spec.rows])
    unique = sorted(set(groups))
    if len(unique) < 2:
        return {
            "metrics": {
                "exploratory_leave_one_topology_out_r2": None,
                "exploratory_leave_one_topology_out_mean_absolute_error": None,
            },
            "predictions": [],
        }
    predictions = np.empty_like(y)
    records = []
    for held_out in unique:
        train = groups != held_out
        test = ~train
        estimator = DecisionTreeRegressor(max_depth=spec.max_depth, random_state=42)
        estimator.fit(x[train], y[train])
        fold = estimator.predict(x[test])
        predictions[test] = fold
        for index, predicted in zip(np.flatnonzero(test), fold):
            records.append({
                "case_slug": held_out,
                "row_index": int(index),
                "observed": float(y[index]),
                "predicted": float(predicted),
                "absolute_error": float(abs(y[index] - predicted)),
            })
    return {
        "metrics": {
            "exploratory_leave_one_topology_out_r2": _r2(y, predictions),
            "exploratory_leave_one_topology_out_mean_absolute_error": float(
                np.mean(np.abs(y - predictions))
            ),
        },
        "predictions": records,
    }


def _serialize_tree(estimator: Any, feature_names: Sequence[str]) -> dict[str, Any]:
    tree = estimator.tree_

    def visit(node: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "node_id": int(node),
            "sample_count": int(tree.n_node_samples[node]),
            "weighted_sample_count": float(tree.weighted_n_node_samples[node]),
            "value": float(tree.value[node].ravel()[0]),
            "squared_error": float(tree.impurity[node]),
        }
        feature_index = int(tree.feature[node])
        if feature_index < 0:
            result["leaf"] = True
            return result
        result.update({
            "leaf": False,
            "feature": str(feature_names[feature_index]),
            "threshold": float(tree.threshold[node]),
            "left": visit(int(tree.children_left[node])),
            "right": visit(int(tree.children_right[node])),
        })
        return result

    return visit(0)


def _write_dot(path: Path, estimator: Any, spec: _TreeSpec) -> None:
    from sklearn.tree import export_graphviz

    dot = export_graphviz(
        estimator,
        out_file=None,
        feature_names=list(spec.feature_names),
        filled=True,
        rounded=True,
        special_characters=False,
        precision=4,
    )
    path.write_text(str(dot), encoding="utf-8")


def _write_tree_figure(path: Path, estimator: Any, spec: _TreeSpec) -> None:
    from sklearn.tree import plot_tree

    width = 15 if spec.max_depth <= 2 else 20
    height = 7 if spec.max_depth <= 2 else 10
    with plt.rc_context({"font.size": 8}):
        figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
        plot_tree(
            estimator,
            ax=axis,
            feature_names=list(spec.feature_names),
            filled=True,
            rounded=True,
            precision=4,
            proportion=False,
        )
        axis.set_title(spec.title)
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if path.suffix.lower() == ".pdf"
            else None
        )
        figure.savefig(path, dpi=220, bbox_inches="tight", metadata=metadata)
        plt.close(figure)


def _rules_document(artifact: dict[str, Any]) -> str:
    metrics = artifact["metrics"]
    return "\n".join([
        artifact["title"],
        "=" * len(artifact["title"]),
        "",
        f"Backend: {artifact['source_backend']}",
        f"Unit: {artifact['unit_of_analysis']}",
        f"Target: {artifact['target']}",
        f"Rows: {artifact['sample_count']}",
        f"Independent topologies: {artifact['independent_topology_count']}",
        f"Validation scope: {artifact['validation_scope']}",
        "Generalization claim: none",
        f"In-sample R2: {_metric_text(metrics['in_sample_r2'])}",
        "Exploratory leave-one-topology-out R2: "
        f"{_metric_text(metrics['exploratory_leave_one_topology_out_r2'])}",
        "",
        artifact["scientific_boundary"],
        "",
        "Rules",
        "-----",
        artifact["rules_text"],
        "",
    ])


def _mean_group(
    rows: Sequence[dict[str, Any]],
    *,
    keys: Sequence[str],
    values: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_key, selected in sorted(grouped.items(), key=lambda item: item[0]):
        record = dict(zip(keys, group_key))
        for value in values:
            data = [float(row[value]) for row in selected if row.get(value) is not None]
            record[f"{value}_mean"] = float(np.mean(data)) if data else None
            record[f"{value}_sd"] = (
                float(np.std(data, ddof=1)) if len(data) > 1 else 0.0 if data else None
            )
        output.append(record)
    return output


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= 0.0:
        return None
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _metric_text(value: float | None) -> str:
    return "not available" if value is None else f"{value:.4f}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
