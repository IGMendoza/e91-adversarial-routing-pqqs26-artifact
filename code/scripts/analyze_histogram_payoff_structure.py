#!/usr/bin/env python3
"""Audit Alice/Eve feature structure in a histogram payoff bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning.corpus import (  # noqa: E402
    load_topology_bench_case,
)
from sequence_game.experiments.histogram_colearning.payoff_structure import (  # noqa: E402
    analyze_payoff_structure,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    bundle = args.bundle.resolve()
    run_paths = sorted((bundle / "runs").glob("*/seed_*.json"))
    if not run_paths:
        raise ValueError(f"no paper run records found under {bundle / 'runs'}")
    records = []
    cases: dict[tuple[str, str], Any] = {}
    for run_path in run_paths:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        case_row = payload["case"]
        config = payload["config"]
        key = (
            str(case_row["graph_id"]),
            str(case_row["assignment_id"]),
        )
        if key not in cases:
            cases[key] = load_topology_bench_case(
                case_row["corpus_db_path"],
                graph_id=key[0],
                assignment_id=key[1],
            )
        empirical = payload["empirical_game"]
        if empirical.get("sampling_design") != "stratified_independent_posthoc":
            raise ValueError(f"{run_path} has no independent post-hoc payoff audit")
        analysis = analyze_payoff_structure(
            payoff=np.asarray(empirical["alice_public_utility_matrix"], dtype=float),
            cell_standard_errors=np.asarray(empirical["cell_standard_errors"], dtype=float),
            case=cases[key],
            actions=payload["actions"],
        )
        analysis["case_slug"] = str(payload["paper_run"]["case_slug"])
        analysis["seed"] = int(payload["paper_run"]["seed"])
        analysis["source_run"] = str(run_path.relative_to(bundle))
        records.append(analysis)
    result = {
        "schema": "histogram_payoff_structure_bundle.v2",
        "source_bundle": str(bundle),
        "record_count": len(records),
        "records": records,
        "aggregate": _aggregate(records),
        "online_feedback_boundary": (
            "this is a post-run matrix audit; the online learners receive only "
            "their selected-action scalar feedback, and this command neither "
            "constructs nor enables a structured Eve learner"
        ),
    }
    output = args.out or (bundle / "analysis" / "payoff_structure.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} ({len(records)} run diagnostics)")
    return result


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for slug in sorted({str(row["case_slug"]) for row in records}):
        selected = [row for row in records if row["case_slug"] == slug]
        model_names = sorted(selected[0]["models"])
        result[slug] = {
            "replicates": len(selected),
            "route_count": selected[0]["matrix_shape"][0],
            "action_count": selected[0]["matrix_shape"][1],
            "resource_rank": selected[0]["route_resource_embedding"]["rank"],
            "eve_feature_ranks": {
                mode: int(embedding["rank"])
                for mode, embedding in selected[0]["eve_action_embeddings"].items()
            },
            "models": {
                model: _aggregate_model(selected, model)
                for model in model_names
            },
        }
    return result


def _aggregate_model(
    records: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    result = {
        metric: _mean_sd([
            float(row["models"][model][metric]) for row in records
        ])
        for metric in (
            "parameter_to_cell_ratio",
            "rmse",
            "rmse_to_rms_cell_standard_error",
            "r_squared",
            "absolute_equilibrium_value_error",
            "predicted_equilibrium_exploitability_in_true_game",
        )
    }
    first = records[0]["models"][model]
    if "bilinear" not in first:
        return result
    result["bilinear"] = {
        axis: {
            metric: _mean_sd([
                float(row["models"][model]["bilinear"][axis][metric])
                for row in records
            ])
            for metric in (
                "rmse",
                "baseline_axis_mean_rmse",
                "relative_rmse_reduction_vs_axis_mean",
                "rmse_to_rms_cell_standard_error",
                "mean_group_feature_identifiable_energy",
                "fully_identifiable_group_fraction",
            )
        }
        for axis in ("leave_one_route_out", "leave_one_target_out")
    }
    return result


def _mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


if __name__ == "__main__":
    main()
