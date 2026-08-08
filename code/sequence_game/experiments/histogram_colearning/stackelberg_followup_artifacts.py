"""Publication artifact for the 105-topology Stackelberg follow-up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .stackelberg_followup_corpus import CALIBRATION_GRAPH_IDS


_TSALLIS = "tsallis_inf_rv_eta1"
_LINEAR = "linear_exp2_goptimal_eta2"
_LABELS = {_TSALLIS: "Tsallis-INF", _LINEAR: "LinearExp2"}
_COLORS = {_TSALLIS: "#0072B2", _LINEAR: "#D55E00"}


def build_stackelberg_followup_artifacts(
    run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Summarize completed paired runs and render the paper figure."""

    source = Path(run_dir)
    runs = [
        json.loads(line)
        for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paired = _paired_rows(runs)
    destination = Path(output_path)
    _plot(paired, destination)
    summary = _summary(paired, runs)
    summary_path = destination.with_name(destination.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _paired_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[float, Mapping[str, Any]]] = {}
    for row in runs:
        key = (str(row["graph_id"]), str(row["defender_label"]), int(row["seed"]))
        margin = float(row["steering_config"]["response_margin"])
        grouped.setdefault(key, {})[margin] = row
    paired: list[dict[str, Any]] = []
    for (graph_id, defender, seed), conditions in sorted(grouped.items()):
        if set(conditions) != {0.0, 0.01}:
            raise ValueError(f"unpaired stationary conditions for {graph_id}")
        boundary = conditions[0.0]
        strict = conditions[0.01]
        boundary_mass = float(boundary["intended_route_empirical_mass"])
        strict_mass = float(strict["intended_route_empirical_mass"])
        paired.append({
            "graph_id": graph_id,
            "family": str(strict["family"]),
            "defender_label": defender,
            "seed": seed,
            "boundary_mass": boundary_mass,
            "strict_mass": strict_mass,
            "mass_change": strict_mass - boundary_mass,
            "strict_audit_best_response": bool(
                strict["intended_route_is_audit_best_response"]
            ),
        })
    return paired


def _plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(3.25, 3.25))
    markers = {_TSALLIS: "o", _LINEAR: "^"}
    for learner in (_TSALLIS, _LINEAR):
        values = [row for row in rows if row["defender_label"] == learner]
        axis.scatter(
            [float(row["boundary_mass"]) for row in values],
            [float(row["strict_mass"]) for row in values],
            color=_COLORS[learner],
            marker=markers[learner],
            s=15,
            alpha=0.58,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
    axis.plot(
        (0.0, 1.0),
        (0.0, 1.0),
        color="#555555",
        linestyle="--",
        linewidth=0.75,
        zorder=1,
    )
    axis.text(
        0.075,
        0.105,
        "no change",
        rotation=45,
        rotation_mode="anchor",
        color="#555555",
        fontsize=6.0,
        ha="left",
        va="bottom",
    )
    axis.set_xlabel(r"Target-route mass at $d=0$")
    axis.set_ylabel(r"Target-route mass at $d=0.01$")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.20, linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=markers[learner],
                linestyle="",
                color=_COLORS[learner],
                label=_LABELS[learner],
                markersize=4.5,
            )
            for learner in (_TSALLIS, _LINEAR)
        ],
        loc="lower right",
        frameon=True,
        fontsize=6.2,
        borderpad=0.25,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.96)
    _save(figure, path)


def _summary(
    rows: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_learner: dict[str, Any] = {}
    calibration_ids = set(CALIBRATION_GRAPH_IDS)
    for learner in (_TSALLIS, _LINEAR):
        selected = [row for row in rows if row["defender_label"] == learner]
        noncalibration = [
            row for row in selected if row["graph_id"] not in calibration_ids
        ]
        by_learner[learner] = {
            "paired_graph_count": len(selected),
            "boundary_mass_median": float(
                np.median([row["boundary_mass"] for row in selected])
            ),
            "strict_mass_median": float(
                np.median([row["strict_mass"] for row in selected])
            ),
            "mass_change_median": float(
                np.median([row["mass_change"] for row in selected])
            ),
            "strict_audit_best_response_count": sum(
                bool(row["strict_audit_best_response"]) for row in selected
            ),
            "noncalibration": {
                "paired_graph_count": len(noncalibration),
                "boundary_mass_median": float(np.median([
                    row["boundary_mass"] for row in noncalibration
                ])),
                "strict_mass_median": float(np.median([
                    row["strict_mass"] for row in noncalibration
                ])),
                "mass_change_median": float(np.median([
                    row["mass_change"] for row in noncalibration
                ])),
                "strict_audit_best_response_count": sum(
                    bool(row["strict_audit_best_response"])
                    for row in noncalibration
                ),
            },
        }
    noncalibration_rows = [
        row for row in rows if row["graph_id"] not in calibration_ids
    ]
    return {
        "paired_topology_learner_count": len(rows),
        "condition_run_count": len(runs),
        "combined": _combined_summary(rows),
        "combined_noncalibration": _combined_summary(noncalibration_rows),
        "by_learner": by_learner,
    }


def _combined_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changes = np.asarray([row["mass_change"] for row in rows], dtype=float)
    audit_by_graph: dict[str, bool] = {}
    for row in rows:
        graph_id = str(row["graph_id"])
        value = bool(row["strict_audit_best_response"])
        previous = audit_by_graph.setdefault(graph_id, value)
        if previous != value:
            raise ValueError(
                f"learners disagree on the audit commitment for {graph_id}"
            )
    return {
        "paired_topology_learner_count": len(rows),
        "boundary_mass_median": float(np.median([
            row["boundary_mass"] for row in rows
        ])),
        "strict_mass_median": float(np.median([
            row["strict_mass"] for row in rows
        ])),
        "mass_change_median": float(np.median(changes)),
        "mass_change_positive_count": int(np.sum(changes > 0.0)),
        "mass_change_zero_count": int(np.sum(changes == 0.0)),
        "mass_change_negative_count": int(np.sum(changes < 0.0)),
        "strict_audit_best_response_count": sum(
            bool(row["strict_audit_best_response"]) for row in rows
        ),
        "strict_audit_graph_count": sum(audit_by_graph.values()),
        "strict_audit_graph_total": len(audit_by_graph),
    }


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        destination = path.with_suffix(suffix)
        figure.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
