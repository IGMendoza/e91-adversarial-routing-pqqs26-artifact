"""Paper figures and tables for the five-topology Tsallis-INF refinement."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .calibration_artifacts import TOPOLOGY_LABELS, _save_figure


PAIR_LABELS = {
    "one_standard_error_selected_pair": "RV: Alice eta 1 / Eve eta 1",
    "raw_minimum_pair": (
        r"RV raw minima: Alice eta $1/\sqrt{2}$ / Eve eta $\sqrt{2}$"
    ),
}
PAIR_COLORS = {
    "one_standard_error_selected_pair": "#0072B2",
    "raw_minimum_pair": "#D55E00",
}
ESTIMATOR_COLORS = {"rv": "#0072B2", "iw": "#D55E00"}


def render_tsallis_refinement_artifacts(
    source_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate the completed run and render its committed discussion bundle."""

    source = Path(source_dir)
    destination = Path(output_dir)
    figure_dir = destination / "figures"
    table_dir = destination / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    screen = _read_json(source / "screening_summary.json")
    confirmation = _read_json(source / "confirmation_summary.json")
    plan = _read_json(source / "tsallis_refinement_plan.json")
    screen_rows = _read_jsonl(source / "screening_rows.jsonl")
    confirmation_rows = _read_jsonl(source / "confirmation_rows.jsonl")
    _validate_complete(screen, confirmation, screen_rows, confirmation_rows)

    role_rows = _role_profile_rows(screen)
    pair_rows = _screen_pair_rows(screen)
    confirmation_summary_rows = _confirmation_summary_rows(confirmation)
    topology_rows = _confirmation_topology_rows(confirmation)
    exploration_rows = _exploration_rows(confirmation_rows)
    contrast_rows = _paired_contrast_rows(confirmation_rows)
    reproducibility_rows = _matched_reproducibility_rows(
        confirmation_rows,
        source.parent / "histogram_step7_calibration" / "confirmation_rows.jsonl",
    )
    table_paths = [
        _write_rows(table_dir / "screening_role_profiles.csv", role_rows),
        _write_rows(table_dir / "screening_pair_matrix.csv", pair_rows),
        _write_rows(
            table_dir / "confirmation_pair_summary.csv",
            confirmation_summary_rows,
        ),
        _write_rows(
            table_dir / "confirmation_by_topology.csv",
            topology_rows,
        ),
        _write_rows(
            table_dir / "confirmation_paired_contrasts.csv",
            contrast_rows,
        ),
        _write_rows(
            table_dir / "confirmation_exploration_dynamics.csv",
            exploration_rows,
        ),
        _write_rows(
            table_dir / "matched_rv1_reproducibility.csv",
            reproducibility_rows,
        ),
    ]
    figure_paths = [
        *_plot_eta_response(
            role_rows,
            figure_dir / "tsallis_eta_response",
        ),
        *_plot_joint_screen(
            pair_rows,
            figure_dir / "tsallis_joint_screen_nash_gap",
        ),
        *_plot_confirmation_comparison(
            topology_rows,
            confirmation_summary_rows,
            figure_dir / "tsallis_confirmation_comparison",
        ),
        *_plot_exploration_dynamics(
            exploration_rows,
            figure_dir / "tsallis_exploration_dynamics",
        ),
    ]
    report_path = destination / "TSALLIS_REFINEMENT_REPORT.md"
    report_path.write_text(
        _report(
            plan,
            screen,
            confirmation,
            role_rows=role_rows,
            contrast_rows=contrast_rows,
            reproducibility_rows=reproducibility_rows,
        ),
        encoding="utf-8",
    )
    files = [*table_paths, *figure_paths, report_path]
    manifest = {
        "schema": "histogram_tsallis_refinement_artifacts.v1",
        "source_screen_schema": screen["schema"],
        "source_confirmation_schema": confirmation["schema"],
        "screening_run_count": screen["run_count"],
        "confirmation_run_count": confirmation["run_count"],
        "evaluation_partition_loaded": False,
        "files": [
            {
                "path": str(path.relative_to(destination)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(files)
        ],
    }
    manifest_path = destination / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _validate_complete(
    screen: dict[str, Any],
    confirmation: dict[str, Any],
    screen_rows: Sequence[dict[str, Any]],
    confirmation_rows: Sequence[dict[str, Any]],
) -> None:
    if screen.get("complete") is not True:
        raise ValueError("Tsallis refinement screen is incomplete")
    if confirmation.get("complete") is not True:
        raise ValueError("Tsallis refinement confirmation is incomplete")
    if len(screen_rows) != int(screen["expected_run_count"]):
        raise ValueError("screening row count does not match its contract")
    if len(confirmation_rows) != int(confirmation["expected_run_count"]):
        raise ValueError("confirmation row count does not match its contract")
    required = {
        f"{player}_policy_{suffix}"
        for player in ("alice", "eve")
        for suffix in (
            "minimum_action_probability",
            "uniform_coverage_mass",
            "nonmodal_probability_mass",
            "effective_action_count",
            "effective_action_fraction",
        )
    }
    for row in confirmation_rows:
        trajectory = row.get("trajectory")
        if not trajectory:
            raise ValueError("confirmation row has no policy trajectory")
        missing = required - set(trajectory[-1])
        if missing:
            raise ValueError(
                f"confirmation row lacks exploration telemetry: {sorted(missing)}"
            )


def _role_profile_rows(screen: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for player in ("alice", "eve"):
        selection = screen["selection"][player]["selected"]
        selected_label = selection["selected"]["candidate"]["label"]
        raw_label = selection["raw_minimum"]["candidate"]["label"]
        eligible = set(selection["eligible_labels"])
        for summary in screen["selection"][player]["candidate_summaries"]:
            rows.append({
                "player": player,
                "estimator": summary["estimator"],
                "eta_scale": summary["eta_scale"],
                "mean_external_regret_per_turn": summary[
                    "mean_external_regret_per_turn"
                ],
                "standard_error": summary["standard_error"],
                "worst_case_mean_external_regret_per_turn": summary[
                    "worst_case_mean_external_regret_per_turn"
                ],
                "selected": summary["candidate"]["label"] == selected_label,
                "raw_minimum": summary["candidate"]["label"] == raw_label,
                "one_standard_error_eligible": (
                    summary["candidate"]["label"] in eligible
                ),
            })
    return rows


def _screen_pair_rows(screen: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {}
    for player in ("alice", "eve"):
        labels[player] = {
            row["candidate"]["label"]: (row["estimator"], row["eta_scale"])
            for row in screen["selection"][player]["candidate_summaries"]
        }
    return [
        {
            **row,
            "alice_estimator": labels["alice"][row["alice_label"]][0],
            "alice_eta_scale": labels["alice"][row["alice_label"]][1],
            "eve_estimator": labels["eve"][row["eve_label"]][0],
            "eve_eta_scale": labels["eve"][row["eve_label"]][1],
        }
        for row in screen["pair_summaries"]
    ]


def _confirmation_summary_rows(
    confirmation: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = (
        "mean_alice_external_regret_per_turn",
        "standard_error_alice_external_regret_per_turn",
        "mean_eve_external_regret_per_turn",
        "standard_error_eve_external_regret_per_turn",
        "mean_exploitability",
        "standard_error_exploitability",
        "mean_mean_alice_reward",
        "standard_error_mean_alice_reward",
        "mean_mean_bell_qualified_rate_hz",
        "standard_error_mean_bell_qualified_rate_hz",
        "mean_false_signal_rate",
        "standard_error_false_signal_rate",
        "mean_false_signal_policy_total_variation",
        "standard_error_false_signal_policy_total_variation",
        "mean_uncontended_turns_per_second",
    )
    return [
        {
            "pair_id": row["pair_id"],
            "alice_label": row["alice_label"],
            "eve_label": row["eve_label"],
            "run_count": row["run_count"],
            **{field: row[field] for field in fields},
        }
        for row in confirmation["aggregate"]
    ]


def _confirmation_topology_rows(
    confirmation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for aggregate in confirmation["aggregate"]:
        for case_id, metrics in aggregate["case_metrics"].items():
            rows.append({
                "pair_id": aggregate["pair_id"],
                "case_id": case_id,
                "topology": _topology_label(case_id),
                **metrics,
            })
    return rows


def _exploration_rows(
    confirmation_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = {}
    for row in confirmation_rows:
        pair_id = str(row["pair_id"])
        for player in ("alice", "eve"):
            action_count = int(row[f"{player}_runtime"]["action_count"])
            for point in row["trajectory"]:
                values = {
                    "eta": float(point[f"{player}_eta"]),
                    "uniform_coverage_mass": float(
                        point[f"{player}_policy_uniform_coverage_mass"]
                    ),
                    "nonmodal_probability_mass": float(
                        point[f"{player}_policy_nonmodal_probability_mass"]
                    ),
                    "effective_action_fraction": float(
                        point[f"{player}_policy_effective_action_fraction"]
                    ),
                    "normalized_policy_entropy": (
                        float(point[f"{player}_policy_entropy"])
                        / math.log(action_count)
                        if action_count > 1 else 0.0
                    ),
                }
                for metric, value in values.items():
                    grouped.setdefault(
                        (pair_id, player, int(point["turn"]), metric),
                        [],
                    ).append(value)
    rows = []
    for (pair_id, player, turn, metric), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        rows.append({
            "pair_id": pair_id,
            "player": player,
            "turn": turn,
            "metric": metric,
            "sample_count": int(array.size),
            "q25": float(np.quantile(array, 0.25)),
            "median": float(np.median(array)),
            "q75": float(np.quantile(array, 0.75)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        })
    return rows


def _paired_contrast_rows(
    confirmation_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = "one_standard_error_selected_pair"
    raw = "raw_minimum_pair"
    if {row["pair_id"] for row in confirmation_rows} != {selected, raw}:
        raise ValueError("expected selected and raw-minimum confirmation pairs")
    metrics = (
        "alice_external_regret_per_turn",
        "eve_external_regret_per_turn",
        "exploitability",
        "mean_alice_reward",
        "mean_bell_qualified_rate_hz",
        "false_signal_rate",
        "false_signal_policy_total_variation",
    )
    rows = []
    seeds = sorted({int(row["seed"]) for row in confirmation_rows})
    for metric in metrics:
        differences = []
        for seed in seeds:
            values = {}
            for pair_id in (selected, raw):
                values[pair_id] = float(np.mean([
                    row[metric] for row in confirmation_rows
                    if row["pair_id"] == pair_id and int(row["seed"]) == seed
                ]))
            differences.append(values[raw] - values[selected])
        mean = float(np.mean(differences))
        standard_error = (
            float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
            if len(differences) > 1 else 0.0
        )
        critical = 4.302652729911275 if len(differences) == 3 else 1.96
        rows.append({
            "metric": metric,
            "contrast": "raw_minimum_pair_minus_selected_pair",
            "paired_seed_count": len(differences),
            "mean_difference": mean,
            "standard_error": standard_error,
            "confidence_95_lower": mean - critical * standard_error,
            "confidence_95_upper": mean + critical * standard_error,
        })
    return rows


def _matched_reproducibility_rows(
    confirmation_rows: Sequence[dict[str, Any]],
    previous_path: Path,
) -> list[dict[str, Any]]:
    if not previous_path.exists():
        raise FileNotFoundError(
            f"missing preceding Step 7 confirmation rows {previous_path}"
        )
    previous_rows = _read_jsonl(previous_path)
    previous = {
        (row["case_id"], int(row["seed"])): row
        for row in previous_rows
        if row["alice_label"] == "tsallis_rv_eta1"
        and row["eve_label"] == "tsallis_rv_eta1"
    }
    current = {
        (row["case_id"], int(row["seed"])): row
        for row in confirmation_rows
        if row["pair_id"] == "one_standard_error_selected_pair"
    }
    if set(previous) != set(current) or len(current) != 15:
        raise ValueError("RV1 reproducibility rows do not form 15 matched runs")
    scalar_fields = (
        "mean_alice_reward",
        "mean_eve_bandit_reward",
        "mean_bell_qualified_rate_hz",
        "alice_external_regret_per_turn",
        "eve_external_regret_per_turn",
        "exploitability",
        "false_signal_rate",
        "false_signal_policy_total_variation",
    )
    rows = [
        {
            "field": field,
            "matched_run_count": len(current),
            "maximum_absolute_difference": max(
                abs(float(previous[key][field]) - float(current[key][field]))
                for key in current
            ),
        }
        for field in scalar_fields
    ]
    for strategy_field in ("alice_strategy", "eve_strategy"):
        rows.append({
            "field": strategy_field,
            "matched_run_count": len(current),
            "maximum_absolute_difference": max(
                float(np.max(np.abs(
                    np.asarray(previous[key][strategy_field], dtype=float)
                    - np.asarray(current[key][strategy_field], dtype=float)
                )))
                for key in current
            ),
        })
    if any(row["maximum_absolute_difference"] != 0.0 for row in rows):
        raise ValueError("exploration telemetry changed matched RV1 behavior")
    return rows


def _plot_eta_response(rows: Sequence[dict[str, Any]], stem: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), sharey=False)
    for axis, player in zip(axes, ("alice", "eve")):
        selected_rows = [row for row in rows if row["player"] == player]
        for estimator in ("rv", "iw"):
            points = sorted(
                (row for row in selected_rows if row["estimator"] == estimator),
                key=lambda row: row["eta_scale"],
            )
            axis.errorbar(
                [row["eta_scale"] for row in points],
                [row["mean_external_regret_per_turn"] for row in points],
                yerr=[row["standard_error"] for row in points],
                color=ESTIMATOR_COLORS[estimator],
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=estimator.upper(),
            )
        for row in selected_rows:
            if row["selected"]:
                axis.scatter(
                    row["eta_scale"],
                    row["mean_external_regret_per_turn"],
                    s=150,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.8,
                    zorder=5,
                    label="one-SE selection",
                )
            if row["raw_minimum"]:
                axis.scatter(
                    row["eta_scale"],
                    row["mean_external_regret_per_turn"],
                    marker="*",
                    s=130,
                    color="black",
                    zorder=6,
                    label="raw minimum",
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({row["eta_scale"] for row in selected_rows}))
        axis.set_xticklabels([
            f"{value:.3g}" for value in sorted({
                row["eta_scale"] for row in selected_rows
            })
        ], rotation=35)
        axis.set_title(f"{player.title()} own external regret")
        axis.set_xlabel("eta multiplier")
        axis.set_ylabel("external regret per turn")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    figure.suptitle(
        "Finer two-sided Tsallis-INF calibration on five Topology Bench graphs",
        fontsize=14,
    )
    figure.tight_layout()
    return _save_figure(figure, stem)


def _plot_joint_screen(rows: Sequence[dict[str, Any]], stem: Path) -> list[Path]:
    eta_values = sorted({float(row["alice_eta_scale"]) for row in rows})
    vmin = min(float(row["mean_nash_gap"]) for row in rows)
    vmax = max(float(row["mean_nash_gap"]) for row in rows)
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 11.0), sharex=True, sharey=True)
    image = None
    for row_index, alice_estimator in enumerate(("rv", "iw")):
        for column, eve_estimator in enumerate(("rv", "iw")):
            axis = axes[row_index, column]
            matrix = np.full((len(eta_values), len(eta_values)), np.nan)
            for row in rows:
                if (
                    row["alice_estimator"] == alice_estimator
                    and row["eve_estimator"] == eve_estimator
                ):
                    i = eta_values.index(float(row["alice_eta_scale"]))
                    j = eta_values.index(float(row["eve_eta_scale"]))
                    matrix[i, j] = float(row["mean_nash_gap"])
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(
                f"Alice {alice_estimator.upper()} / Eve {eve_estimator.upper()}"
            )
            axis.set_xticks(range(len(eta_values)))
            axis.set_yticks(range(len(eta_values)))
            axis.set_xticklabels([f"{value:.3g}" for value in eta_values], rotation=45)
            axis.set_yticklabels([f"{value:.3g}" for value in eta_values])
            axis.set_xlabel("Eve eta multiplier")
            axis.set_ylabel("Alice eta multiplier")
            if alice_estimator == "rv" and eve_estimator == "rv":
                axis.scatter(
                    eta_values.index(1.0),
                    eta_values.index(1.0),
                    s=150,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=2.0,
                    label="one-SE selection",
                )
                axis.scatter(
                    min(
                        range(len(eta_values)),
                        key=lambda index: abs(
                            eta_values[index] - math.sqrt(2.0)
                        ),
                    ),
                    min(
                        range(len(eta_values)),
                        key=lambda index: abs(
                            eta_values[index] - 1.0 / math.sqrt(2.0)
                        ),
                    ),
                    marker="*",
                    s=130,
                    color="white",
                    edgecolors="black",
                    linewidths=0.7,
                    label="raw-minimum pair",
                )
                axis.legend(loc="upper right", frameon=True, fontsize=8)
    assert image is not None
    color_axis = figure.add_axes((0.885, 0.16, 0.018, 0.68))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("mean 10k Nash gap")
    figure.suptitle(
        "Full 18 x 18 Tsallis-INF screening matrix",
        fontsize=14,
    )
    figure.subplots_adjust(
        top=0.92,
        bottom=0.10,
        left=0.08,
        right=0.86,
        hspace=0.23,
        wspace=0.20,
    )
    return _save_figure(figure, stem)


def _plot_confirmation_comparison(
    topology_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    stem: Path,
) -> list[Path]:
    topology_order = ["Memorex", "Hibernia Ireland", "Layer42", "KAREN", "CERNET"]
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
    for pair_id in PAIR_LABELS:
        rows = {row["topology"]: row for row in topology_rows if row["pair_id"] == pair_id}
        axes[0].plot(
            range(len(topology_order)),
            [rows[name]["exploitability"] for name in topology_order],
            color=PAIR_COLORS[pair_id],
            marker="o",
            linewidth=2.0,
            label=PAIR_LABELS[pair_id],
        )
        axes[2].plot(
            range(len(topology_order)),
            [rows[name]["mean_bell_qualified_rate_hz"] for name in topology_order],
            color=PAIR_COLORS[pair_id],
            marker="o",
            linewidth=2.0,
            label=PAIR_LABELS[pair_id],
        )
    for axis in (axes[0], axes[2]):
        axis.set_xticks(range(len(topology_order)))
        axis.set_xticklabels(topology_order, rotation=35, ha="right")
        axis.grid(alpha=0.25)
    axes[0].set_title("Fresh 100k Nash gap by topology")
    axes[0].set_ylabel("Nash gap")
    summary = {row["pair_id"]: row for row in summary_rows}
    x = np.arange(2)
    width = 0.36
    for index, pair_id in enumerate(PAIR_LABELS):
        row = summary[pair_id]
        axes[1].bar(
            x + (index - 0.5) * width,
            [
                row["mean_alice_external_regret_per_turn"],
                row["mean_eve_external_regret_per_turn"],
            ],
            width,
            color=PAIR_COLORS[pair_id],
            label=PAIR_LABELS[pair_id],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Alice", "Eve"])
    axes[1].set_ylabel("external regret per turn")
    axes[1].set_title("Five-topology regret decomposition")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].set_yscale("log")
    axes[2].set_title("Bell-qualified service")
    axes[2].set_ylabel("pairs/s (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Selected schedule versus screen raw minima", fontsize=14)
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    return _save_figure(figure, stem)


def _plot_exploration_dynamics(
    rows: Sequence[dict[str, Any]],
    stem: Path,
) -> list[Path]:
    metrics = (
        ("eta", "algorithm-internal eta", True),
        (
            "uniform_coverage_mass",
            r"uniform coverage $K\,\min_i p_{t,i}$",
            True,
        ),
        (
            "nonmodal_probability_mass",
            r"nonmodal mass $1-\max_i p_{t,i}$",
            False,
        ),
        ("effective_action_fraction", "effective action fraction", False),
        ("normalized_policy_entropy", "normalized policy entropy", False),
    )
    figure, axes = plt.subplots(5, 2, figsize=(14.8, 17.8), sharex=True)
    for column, player in enumerate(("alice", "eve")):
        for row_index, (metric, title, log_y) in enumerate(metrics):
            axis = axes[row_index, column]
            for pair_id in PAIR_LABELS:
                points = sorted(
                    (
                        row for row in rows
                        if row["player"] == player
                        and row["pair_id"] == pair_id
                        and row["metric"] == metric
                    ),
                    key=lambda row: row["turn"],
                )
                turns = np.asarray([row["turn"] for row in points], dtype=float)
                median = np.asarray([row["median"] for row in points], dtype=float)
                q25 = np.asarray([row["q25"] for row in points], dtype=float)
                q75 = np.asarray([row["q75"] for row in points], dtype=float)
                axis.plot(
                    turns,
                    median,
                    color=PAIR_COLORS[pair_id],
                    linewidth=2.0,
                    label=PAIR_LABELS[pair_id],
                )
                axis.fill_between(
                    turns,
                    q25,
                    q75,
                    color=PAIR_COLORS[pair_id],
                    alpha=0.14,
                )
            axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
            axis.set_title(f"{player.title()}: {title}")
            axis.grid(alpha=0.24)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("turn")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(
        "Realized Tsallis-INF learning and exploration during fresh 100k runs",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    return _save_figure(figure, stem)


def _report(
    plan: dict[str, Any],
    screen: dict[str, Any],
    confirmation: dict[str, Any],
    *,
    role_rows: Sequence[dict[str, Any]],
    contrast_rows: Sequence[dict[str, Any]],
    reproducibility_rows: Sequence[dict[str, Any]],
) -> str:
    selected = {
        player: screen["selection"][player]["selected"]
        for player in ("alice", "eve")
    }
    aggregate = {row["pair_id"]: row for row in confirmation["aggregate"]}
    chosen = aggregate["one_standard_error_selected_pair"]
    raw = aggregate["raw_minimum_pair"]
    contrast = {row["metric"]: row for row in contrast_rows}
    maximum_reproduction_difference = max(
        row["maximum_absolute_difference"] for row in reproducibility_rows
    )
    return "\n".join([
        "# Five-topology Tsallis-INF refinement",
        "",
        "Status: complete",
        "Updated: 2026-07-15",
        "",
        "## Result",
        "",
        "The complete 18-by-18 two-sided screen retains reduced-variance "
        "Tsallis-INF at eta multiplier 1 for both Alice and Eve under the "
        "declared one-standard-error rule. Alice's raw screen minimum is RV "
        "eta `0.707107`; Eve's is RV eta `1.41421`. The raw minima are close "
        "to eta 1 in the 10k bank screen, but their paired fresh 100k "
        f"combination has Nash gap `{raw['mean_exploitability']:.6f}`, versus "
        f"`{chosen['mean_exploitability']:.6f}` for RV1/RV1.",
        "",
        "![Eta response](figures/tsallis_eta_response.png)",
        "",
        "The paired raw-minus-selected Nash-gap difference is "
        f"`{contrast['exploitability']['mean_difference']:.6f}` with a "
        "three-seed 95% interval of "
        f"`[{contrast['exploitability']['confidence_95_lower']:.6f}, "
        f"{contrast['exploitability']['confidence_95_upper']:.6f}]`. The "
        "interval is wide with three seeds, but the fresh point estimate does "
        "not support replacing the conservative selection with the raw "
        "screen minima.",
        "",
        "![Confirmation comparison](figures/tsallis_confirmation_comparison.png)",
        "",
        "## Calibration design",
        "",
        f"The screen contains `{screen['run_count']:,}` 10k-turn games on the "
        "five frozen calibration graphs. Both roles cross RV and IW with eta "
        "multipliers `0.5, 0.7071, 1, 1.4142, 2, 2.8284, 4, 5.6569, 8`. "
        "Selection minimizes the role's own external regret after equal "
        "macro-averaging across cases and all opposing settings. The five "
        "evaluation graphs remain unloaded.",
        "",
        "![Joint screen](figures/tsallis_joint_screen_nash_gap.png)",
        "",
        "Alice's one-standard-error eligible set is `" + ", ".join(
            selected["alice"]["eligible_labels"]
        ) + "`. Eve's is `" + ", ".join(
            selected["eve"]["eligible_labels"]
        ) + "`. Neither overall eligible set contains IW. IW has an interior "
        "minimum for both roles in this full two-sided screen, resolving the "
        "earlier eta-2 boundary observed against the smaller opponent-anchor "
        "set.",
        "",
        "## Realized exploration",
        "",
        "The confirmation checkpoints now measure Tsallis exploration "
        "directly from each realized policy. `K min_i p_i` is the maximum "
        "uniform component contained in the policy; `1-max_i p_i` measures "
        "mass outside the modal action; entropy and effective action fraction "
        "capture broader support. These values are diagnostics and do not "
        "enter either learner's updates.",
        "",
        "![Exploration dynamics](figures/tsallis_exploration_dynamics.png)",
        "",
        "Reducing Alice's eta to `0.707107` yields visibly broader Alice "
        "coverage, while increasing Eve's eta to `1.41421` yields a more "
        "concentrated Eve policy. This confirms that the eta changes have the "
        "expected policy-level effect even though the combined raw-minimum "
        "pair does not improve the fresh aggregate Nash gap.",
        "",
        "## Reproducibility and claim boundary",
        "",
        "The selected RV1/RV1 confirmation uses the same five cases and seeds "
        "as the preceding Step 7 run. All 15 matched rows reproduce Alice "
        "reward, Eve reward, Bell-qualified rate, both regrets, Nash gap, "
        "false-signal rate, policy-attribution distance, and both empirical "
        "strategies exactly. The maximum checked absolute difference is "
        f"`{maximum_reproduction_difference:.1f}`. The new telemetry therefore "
        "did not alter learning behavior.",
        "",
        f"The selected pair runs at a mean uncontended "
        f"`{chosen['mean_uncontended_turns_per_second']:.0f}` turns/s across "
        "the five topologies. This is a calibration result, not an evaluation-"
        "partition estimate or a claim of a unique globally optimal eta. The "
        "appropriate frozen setting for the evaluation pass is RV eta 1 for "
        "both roles.",
        "",
    ])


def _topology_label(case_id: str) -> str:
    family = case_id.split("_topology_bench_", 1)[1].split("_seed_", 1)[0]
    return TOPOLOGY_LABELS[family].replace("\n", " ").split(" topology", 1)[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"table {path.name} has inconsistent columns")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path
