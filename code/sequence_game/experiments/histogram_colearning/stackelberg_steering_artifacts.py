"""Publication figures for strict Stackelberg route steering."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .corpus_scale import load_corpus_scale_case


STACKELBERG_STEERING_ARTIFACT_SCHEMA = (
    "histogram_stackelberg_steering_artifacts.v1"
)
DEFENDER_ORDER = (
    "tsallis_inf_rv_eta1",
    "linear_exp2_goptimal_eta2",
)
DEFENDER_LABELS = {
    "tsallis_inf_rv_eta1": "Tsallis-INF (RV, $\\eta{=}1$)",
    "linear_exp2_goptimal_eta2": "LinearExp2 (G-optimal, $\\eta{=}2$)",
}
DEFENDER_COLORS = {
    "tsallis_inf_rv_eta1": "#0072B2",
    "linear_exp2_goptimal_eta2": "#D55E00",
}
GRAPH_LABELS = {
    "memorex": "Memorex",
    "hiberniaireland": "Hibernia Ireland",
    "layer42": "Layer42",
    "karen": "Karen",
    "cernet": "Cernet",
}
GRAPH_TICK_LABELS = {
    "memorex": "Memorex",
    "hiberniaireland": "Hibernia\nIreland",
    "layer42": "Layer42",
    "karen": "Karen",
    "cernet": "Cernet",
}
GRAPH_ORDER = tuple(GRAPH_LABELS)
COMPONENT_COLORS = {
    "pessimism": "#CC79A7",
    "bank": "#F0E442",
    "follower": "#009E73",
}
COMPONENT_LABELS = {
    "pessimism": "$V_0-V_d$",
    "bank": "$V_d-V_z$",
    "follower": "$V_z-U_E$",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.transparent": False,
})


def write_stackelberg_steering_artifacts(
    strict_run_dir: str | Path,
    boundary_run_dir: str | Path,
    corpus_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render strict-response calibration, evaluation, and commitments."""

    strict_source = Path(strict_run_dir)
    boundary_source = Path(boundary_run_dir)
    corpus_source = Path(corpus_run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(strict_source / "run_manifest.json")
    corpus_manifest = _load_json(corpus_source / "run_manifest.json")
    if (
        corpus_manifest["selection_digest_blake2b_128"]
        != manifest["source_corpus_selection_digest_blake2b_128"]
    ):
        raise ValueError(
            "steering and corpus selection digests do not match"
        )
    strict_calibration = _load_jsonl(
        strict_source / "calibration_results.jsonl"
    )
    strict_rows = _load_jsonl(strict_source / "results.jsonl")
    boundary_rows = _load_jsonl(boundary_source / "results.jsonl")
    analysis = analyze_stackelberg_steering_results(
        manifest,
        strict_calibration,
        strict_rows,
        boundary_rows,
    )

    files: list[Path] = []
    frontier = _plot_margin_frontier(strict_calibration)
    files.extend(
        _save_publication_figure(
            frontier,
            target / "stackelberg_margin_frontier",
        )
    )
    plt.close(frontier)

    evaluation = _plot_strict_evaluation(strict_rows)
    files.extend(
        _save_publication_figure(
            evaluation,
            target / "stackelberg_strict_evaluation",
        )
    )
    plt.close(evaluation)

    target_mass = _plot_target_route_mass(strict_rows, boundary_rows)
    files.extend(
        _save_publication_figure(
            target_mass,
            target / "stackelberg_target_route_mass",
        )
    )
    plt.close(target_mass)

    comparison = _plot_strict_vs_zero(strict_rows, boundary_rows)
    files.extend(
        _save_publication_figure(
            comparison,
            target / "stackelberg_strict_vs_zero",
        )
    )
    plt.close(comparison)

    commitments = _plot_physical_commitments(
        corpus_manifest,
        strict_rows,
    )
    files.extend(
        _save_publication_figure(
            commitments,
            target / "stackelberg_physical_commitments",
        )
    )
    plt.close(commitments)

    report_path = target / "STACKELBERG_STEERING_REPORT.md"
    report_path.write_text(
        _report_markdown(analysis),
        encoding="utf-8",
    )
    files.append(report_path)

    artifact_manifest = {
        "schema": STACKELBERG_STEERING_ARTIFACT_SCHEMA,
        "strict_source_run": str(strict_source),
        "boundary_source_run": str(boundary_source),
        "corpus_source_run": str(corpus_source),
        "source_selection_digest_blake2b_128": manifest[
            "source_corpus_selection_digest_blake2b_128"
        ],
        "analysis": analysis,
        "scientific_boundary": {
            "leader_information": (
                "declared feedback payoff surface and physical Eve action "
                "costs; no Alice policy or weight access"
            ),
            "commitment": (
                "stationary mixture over declared physical interventions "
                "with a positive Alice route-response margin"
            ),
            "audit": (
                "fresh evaluation seeds and disjoint equal-coverage payoff "
                "bank"
            ),
            "claim": (
                "finite-time route induction under the tested learners and "
                "five frozen topology cases"
            ),
        },
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "blake2b_128": _file_digest(path),
            }
            for path in files
        ],
    }
    artifact_manifest_path = target / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_manifest


def analyze_stackelberg_steering_results(
    manifest: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    strict_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the exact aggregate quantities used in the figures."""

    expected_evaluations = int(manifest["expected_evaluation_run_count"])
    if len(strict_rows) != expected_evaluations:
        raise ValueError("strict evaluation row count does not match manifest")
    strict_by_key = {_run_key(row): row for row in strict_rows}
    boundary_by_key = {_run_key(row): row for row in boundary_rows}
    if set(strict_by_key) != set(boundary_by_key):
        raise ValueError("strict and boundary evaluation identities differ")
    selected_margins = {
        float(row["steering_config"]["response_margin"])
        for row in strict_rows
    }
    if not selected_margins or min(selected_margins) <= 0.0:
        raise ValueError("strict evaluation contains a nonpositive margin")
    if any(
        float(row["steering_config"]["response_margin"]) != 0.0
        for row in boundary_rows
    ):
        raise ValueError("boundary comparison contains a positive margin")

    by_defender: dict[str, dict[str, Any]] = {}
    for defender in DEFENDER_ORDER:
        selected = [
            row for row in strict_rows
            if row["defender_label"] == defender
        ]
        if not selected:
            raise ValueError(f"missing strict rows for {defender}")
        paired_boundary = [boundary_by_key[_run_key(row)] for row in selected]
        by_defender[defender] = {
            "run_count": len(selected),
            "mean_intended_route_empirical_mass": _mean(
                selected,
                "intended_route_empirical_mass",
            ),
            "median_intended_route_empirical_mass": _median(
                selected,
                "intended_route_empirical_mass",
            ),
            "mean_intended_route_current_policy_mass": _mean(
                selected,
                "intended_route_current_policy_mass",
            ),
            "mean_per_turn_stackelberg_regret": _mean(
                selected,
                "per_turn_unconstrained_stackelberg_regret",
            ),
            "mean_pessimism_price_on_audit": _mean(
                selected,
                "pessimism_price_on_audit",
            ),
            "mean_feedback_to_audit_value_loss": _mean(
                selected,
                "feedback_to_audit_commitment_value_loss",
            ),
            "mean_finite_follower_value_loss": _mean(
                selected,
                "finite_follower_value_loss",
            ),
            "mean_route_mass_change_from_boundary": float(np.mean([
                float(row["intended_route_empirical_mass"])
                - float(boundary["intended_route_empirical_mass"])
                for row, boundary in zip(selected, paired_boundary)
            ])),
            "mean_eve_utility_change_from_boundary": float(np.mean([
                float(row["mean_eve_public_utility"])
                - float(boundary["mean_eve_public_utility"])
                for row, boundary in zip(selected, paired_boundary)
            ])),
        }

    calibration_margins = sorted({
        float(row["steering_config"]["response_margin"])
        for row in calibration_rows
    })
    return {
        "graph_count": len({
            str(row["family"]) for row in strict_rows
        }),
        "strict_evaluation_run_count": len(strict_rows),
        "calibration_run_count": len(calibration_rows),
        "calibration_margins": calibration_margins,
        "selected_margins": sorted(selected_margins),
        "audit_best_response_count": sum(
            bool(row["intended_route_is_audit_best_response"])
            for row in strict_rows
        ),
        "median_intended_route_empirical_mass": float(np.median([
            float(row["intended_route_empirical_mass"])
            for row in strict_rows
        ])),
        "minimum_intended_route_empirical_mass": min(
            float(row["intended_route_empirical_mass"])
            for row in strict_rows
        ),
        "median_turns_per_second": float(np.median([
            float(row["turns_per_second"]) for row in strict_rows
        ])),
        "mean_decomposition_identity_error": float(np.mean([
            abs(_decomposition_identity_error(row))
            for row in strict_rows
        ])),
        "by_defender": by_defender,
    }


def _plot_margin_frontier(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    grouped = _calibration_aggregate(calibration_rows)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 6.5),
        sharex=True,
        constrained_layout=True,
    )
    metrics = (
        (
            "intended_route_empirical_mass",
            "Mean intended-route mass",
            (0.0, 1.03),
        ),
        (
            "per_turn_unconstrained_stackelberg_regret",
            "Per-turn Stackelberg regret",
            None,
        ),
    )
    graph_colors = plt.get_cmap("turbo")(
        np.linspace(0.08, 0.92, len(GRAPH_ORDER))
    )
    for column, defender in enumerate(DEFENDER_ORDER):
        for row_index, (metric, ylabel, ylim) in enumerate(metrics):
            axis = axes[row_index, column]
            for graph_index, family in enumerate(GRAPH_ORDER):
                rows = grouped[(family, defender)]
                margins = np.asarray(
                    [row["response_margin"] for row in rows],
                    dtype=float,
                )
                means = np.asarray(
                    [row[f"mean_{metric}"] for row in rows],
                    dtype=float,
                )
                standard_errors = np.asarray(
                    [row[f"se_{metric}"] for row in rows],
                    dtype=float,
                )
                axis.errorbar(
                    margins,
                    means,
                    yerr=standard_errors,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.2,
                    capsize=2,
                    color=graph_colors[graph_index],
                    label=GRAPH_LABELS[family],
                )
            axis.axvline(
                0.01,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
                label=(
                    "selected $d=0.01$"
                    if row_index == 0 and column == 0 else None
                ),
            )
            if metric.endswith("regret"):
                axis.axhline(0.0, color="0.25", linewidth=0.8)
            axis.grid(True, color="0.9", linewidth=0.6)
            axis.set_ylabel(ylabel)
            if ylim is not None:
                axis.set_ylim(*ylim)
            if row_index == 0:
                axis.set_title(DEFENDER_LABELS[defender])
            if row_index == 1:
                axis.set_xlabel("Required Alice response margin $d$")
    axes[0, 0].legend(
        ncol=2,
        frameon=False,
        loc="lower right",
    )
    figure.suptitle(
        "Strict response margins trade leader value for faster route induction",
        fontsize=13,
    )
    return figure


def _plot_strict_evaluation(
    rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    figure = plt.figure(
        figsize=(3.45, 6.05),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 1.16, 0.34),
    )
    mass_axis = figure.add_subplot(grid[0, 0])
    value_axis = figure.add_subplot(grid[1, 0])
    shift_axis = figure.add_subplot(grid[2, 0], sharex=value_axis)
    x = np.arange(len(GRAPH_ORDER), dtype=float)
    offsets = (-0.16, 0.16)
    for offset, defender in zip(offsets, DEFENDER_ORDER):
        grouped = [
            [
                row for row in rows
                if row["family"] == family
                and row["defender_label"] == defender
            ]
            for family in GRAPH_ORDER
        ]
        means = np.asarray([
            np.mean([
                float(row["intended_route_empirical_mass"])
                for row in selected
            ])
            for selected in grouped
        ])
        lower_errors = np.asarray([
            means[index] - min(
                float(row["intended_route_empirical_mass"])
                for row in selected
            )
            for index, selected in enumerate(grouped)
        ])
        upper_errors = np.asarray([
            max(
                float(row["intended_route_empirical_mass"])
                for row in selected
            ) - means[index]
            for index, selected in enumerate(grouped)
        ])
        mass_axis.errorbar(
            x + offset,
            means,
            yerr=np.vstack((lower_errors, upper_errors)),
            marker="o" if defender == DEFENDER_ORDER[0] else "^",
            markersize=6,
            linestyle="none",
            capsize=3,
            color=DEFENDER_COLORS[defender],
            label=(
                "Tsallis-INF"
                if defender == DEFENDER_ORDER[0]
                else "LinearExp2"
            ),
        )
        for graph_index, selected in enumerate(grouped):
            seed_values = [
                float(row["intended_route_empirical_mass"])
                for row in selected
            ]
            jitter = np.linspace(-0.035, 0.035, len(seed_values))
            mass_axis.scatter(
                np.full(len(seed_values), x[graph_index] + offset) + jitter,
                seed_values,
                s=12,
                color=DEFENDER_COLORS[defender],
                alpha=0.45,
                linewidths=0,
            )
    mass_axis.set_title("(a) Target-route empirical mass", loc="left")
    mass_axis.set_ylabel("Target-route mass")
    mass_axis.set_xticks(
        x,
        [GRAPH_TICK_LABELS[value] for value in GRAPH_ORDER],
    )
    mass_axis.tick_params(axis="x", labelsize=7.0)
    mass_axis.set_ylim(0.65, 1.01)
    mass_axis.grid(True, axis="y", color="0.9")
    mass_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        fontsize=7.2,
        ncol=2,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    bar_width = 0.33
    audit_shift_by_defender: dict[str, list[float]] = {
        defender: [] for defender in DEFENDER_ORDER
    }
    for graph_index, family in enumerate(GRAPH_ORDER):
        for defender_index, defender in enumerate(DEFENDER_ORDER):
            selected = [
                row for row in rows
                if row["family"] == family
                and row["defender_label"] == defender
            ]
            position = (
                graph_index
                + (defender_index - 0.5) * (bar_width + 0.04)
            )
            components = {
                "pessimism": np.mean([
                    float(row["pessimism_price_on_audit"])
                    for row in selected
                ]),
                "bank": np.mean([
                    float(row["feedback_to_audit_commitment_value_loss"])
                    for row in selected
                ]),
                "follower": np.mean([
                    float(row["finite_follower_value_loss"])
                    for row in selected
                ]),
            }
            audit_shift_by_defender[defender].append(
                float(components["bank"])
            )
            positive_bottom = 0.0
            negative_bottom = 0.0
            for component in ("pessimism", "bank", "follower"):
                value = float(components[component])
                bottom = positive_bottom if value >= 0.0 else negative_bottom
                value_axis.bar(
                    position,
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=COMPONENT_COLORS[component],
                    edgecolor=(
                        "#6B6100" if component == "bank" else "white"
                    ),
                    linewidth=0.7 if component == "bank" else 0.4,
                    label=(
                        COMPONENT_LABELS[component]
                        if graph_index == 0 and defender_index == 0 else None
                    ),
                    zorder=3 if component == "bank" else 2,
                )
                if value >= 0.0:
                    positive_bottom += value
                else:
                    negative_bottom += value
            total = sum(components.values())
            value_axis.scatter(
                [position],
                [total],
                color=DEFENDER_COLORS[defender],
                marker="o" if defender == DEFENDER_ORDER[0] else "^",
                s=25,
                zorder=6,
            )
    value_axis.axhline(0.0, color="black", linewidth=0.9)
    value_axis.set_title(
        "(b) Components of $V_0-U_E$",
        loc="left",
    )
    value_axis.set_ylabel(
        "Per-turn\n$V_0-U_E$"
    )
    value_axis.set_xticks(
        np.arange(len(GRAPH_ORDER)),
        [GRAPH_TICK_LABELS[value] for value in GRAPH_ORDER],
    )
    value_axis.tick_params(axis="x", labelbottom=False)
    value_axis.grid(True, axis="y", color="0.9")
    value_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        fontsize=7.2,
        ncol=3,
        columnspacing=0.75,
        handlelength=1.6,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    scaled_audit_shifts: list[float] = []
    shift_bar_width = 0.25
    for offset, defender in zip(offsets, DEFENDER_ORDER):
        scaled = (
            1.0e4
            * np.asarray(
                audit_shift_by_defender[defender],
                dtype=float,
            )
        )
        scaled_audit_shifts.extend(scaled.tolist())
        shift_axis.bar(
            x + offset,
            scaled,
            width=shift_bar_width,
            color=COMPONENT_COLORS["bank"],
            edgecolor=DEFENDER_COLORS[defender],
            linewidth=1.15,
            zorder=3,
        )
    shift_axis.axhline(0.0, color="black", linewidth=0.8)
    shift_axis.set_ylabel(
        "$10^4(V_d-V_z)$",
        fontsize=7.5,
        labelpad=2,
    )
    shift_axis.set_xticks(
        x,
        [GRAPH_TICK_LABELS[value] for value in GRAPH_ORDER],
    )
    shift_axis.tick_params(axis="x", labelsize=7.0)
    shift_axis.tick_params(axis="y", labelsize=6.8)
    shift_axis.grid(True, axis="y", color="0.91", linewidth=0.6)
    shift_min = min(scaled_audit_shifts)
    shift_max = max(scaled_audit_shifts)
    shift_axis.set_ylim(
        min(-1.0, shift_min - 0.35),
        max(3.1, shift_max + 0.35),
    )
    return figure


def _plot_physical_commitments(
    corpus_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    by_family = _one_commitment_per_family(rows)
    case_entries = {
        str(row["family"]).lower(): row
        for row in corpus_manifest["cases"]
    }
    figure, axes = plt.subplots(
        len(GRAPH_ORDER),
        1,
        figsize=(11.2, 8.2),
        constrained_layout=True,
    )
    family_colors = {
        "edge_dos": "#D55E00",
        "swap_denial": "#CC79A7",
        "memory_degradation": "#009E73",
        "edge_information_probe": "#0072B2",
        "repeater_memory_measure_X": "#56B4E9",
        "repeater_memory_measure_Z": "#F0E442",
    }
    for axis, family in zip(axes, GRAPH_ORDER):
        row = by_family[family]
        commitment = row["commitment"]
        case = load_corpus_scale_case(
            case_entries[family],
            corpus_manifest,
        )
        route_index = int(commitment["intended_alice_route_index"])
        route = case.routes[route_index]
        support = sorted(
            commitment["support_actions"],
            key=lambda value: float(value["probability"]),
        )
        labels = [
            _display_action_id(str(value["action_id"]))
            for value in support
        ]
        probabilities = [
            float(value["probability"]) for value in support
        ]
        colors = [
            family_colors.get(
                str(value["action_id"]).split(":", 1)[0],
                "0.5",
            )
            for value in support
        ]
        y = np.arange(len(support))
        axis.barh(
            y,
            probabilities,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set_yticks(y, labels)
        axis.set_xlim(0.0, 1.0)
        axis.grid(True, axis="x", color="0.9")
        path = "–".join(map(str, route["path"]))
        axis.set_title(
            (
                f"{GRAPH_LABELS[family]}: induce route {path} "
                f"({route['hop_count']} hops, "
                f"{route['total_length_m'] / 1000.0:.2f} km)"
            ),
            loc="left",
        )
        for index, probability in enumerate(probabilities):
            axis.text(
                probability + 0.012,
                index,
                f"{probability:.3f}",
                va="center",
                fontsize=8,
            )
    axes[-1].set_xlabel("Eve commitment probability")
    figure.suptitle(
        "The selected Stackelberg commitments use declared physical interventions",
        fontsize=13,
    )
    return figure


def _plot_strict_vs_zero(
    strict_rows: Sequence[Mapping[str, Any]],
    zero_rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    strict_by_key = {_run_key(row): row for row in strict_rows}
    zero_by_key = {_run_key(row): row for row in zero_rows}
    if set(strict_by_key) != set(zero_by_key):
        raise ValueError("strict and zero-margin comparison identities differ")
    figure, (mass_axis, utility_axis) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.55),
        constrained_layout=True,
    )
    x = np.arange(len(GRAPH_ORDER), dtype=float)
    offsets = (-0.16, 0.16)
    for offset, defender in zip(offsets, DEFENDER_ORDER):
        strict_means = []
        zero_means = []
        utility_changes = []
        utility_lower = []
        utility_upper = []
        for family in GRAPH_ORDER:
            selected = [
                row for row in strict_rows
                if row["family"] == family
                and row["defender_label"] == defender
            ]
            matched = [zero_by_key[_run_key(row)] for row in selected]
            strict_means.append(np.mean([
                float(row["intended_route_empirical_mass"])
                for row in selected
            ]))
            zero_means.append(np.mean([
                float(row["intended_route_empirical_mass"])
                for row in matched
            ]))
            deltas = np.asarray([
                float(row["mean_eve_public_utility"])
                - float(zero["mean_eve_public_utility"])
                for row, zero in zip(selected, matched)
            ])
            mean_delta = float(np.mean(deltas))
            utility_changes.append(mean_delta)
            utility_lower.append(mean_delta - float(np.min(deltas)))
            utility_upper.append(float(np.max(deltas)) - mean_delta)
        positions = x + offset
        for position, zero_mass, strict_mass in zip(
            positions,
            zero_means,
            strict_means,
        ):
            mass_axis.annotate(
                "",
                xy=(position, strict_mass),
                xytext=(position, zero_mass),
                arrowprops={
                    "arrowstyle": "->",
                    "color": DEFENDER_COLORS[defender],
                    "linewidth": 1.2,
                    "alpha": 0.75,
                },
            )
        mass_axis.scatter(
            positions,
            zero_means,
            facecolors="white",
            edgecolors=DEFENDER_COLORS[defender],
            marker="o" if defender == DEFENDER_ORDER[0] else "^",
            s=45,
            linewidths=1.3,
            label=(
                "Tsallis-INF: $d=0$"
                if defender == DEFENDER_ORDER[0]
                else "LinearExp2: $d=0$"
            ),
        )
        mass_axis.scatter(
            positions,
            strict_means,
            color=DEFENDER_COLORS[defender],
            marker="o" if defender == DEFENDER_ORDER[0] else "^",
            s=45,
            label=(
                "Tsallis-INF: $d=.01$"
                if defender == DEFENDER_ORDER[0]
                else "LinearExp2: $d=.01$"
            ),
        )
        utility_axis.errorbar(
            positions,
            utility_changes,
            yerr=np.vstack((utility_lower, utility_upper)),
            color=DEFENDER_COLORS[defender],
            marker="o" if defender == DEFENDER_ORDER[0] else "^",
            linestyle="none",
            capsize=3,
            markersize=6,
            label=(
                "Tsallis-INF"
                if defender == DEFENDER_ORDER[0]
                else "LinearExp2"
            ),
        )
    mass_axis.set_title(
        "(a) Target-route empirical mass",
        loc="left",
    )
    mass_axis.set_ylabel("Target-route mass")
    mass_axis.set_xticks(
        x,
        [GRAPH_TICK_LABELS[value] for value in GRAPH_ORDER],
    )
    mass_axis.tick_params(axis="x", labelsize=7.0)
    mass_axis.set_ylim(0.1, 1.02)
    mass_axis.grid(True, axis="y", color="0.9")
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=DEFENDER_COLORS[DEFENDER_ORDER[0]],
            markeredgecolor=DEFENDER_COLORS[DEFENDER_ORDER[0]],
            markersize=5.5,
            label="Tsallis-INF",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markerfacecolor=DEFENDER_COLORS[DEFENDER_ORDER[1]],
            markeredgecolor=DEFENDER_COLORS[DEFENDER_ORDER[1]],
            markersize=5.5,
            label="LinearExp2",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.30",
            markeredgewidth=1.1,
            markersize=5.5,
            label="$d=0$",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="0.30",
            markeredgecolor="0.30",
            markersize=5.5,
            label="$d=0.01$",
        ),
    ]

    utility_axis.axhline(0.0, color="black", linewidth=0.9)
    utility_axis.set_title(
        "(b) Change in Eve utility",
        loc="left",
    )
    utility_axis.set_ylabel("$\\Delta U_E$ per turn")
    utility_axis.set_xticks(
        x,
        [GRAPH_TICK_LABELS[value] for value in GRAPH_ORDER],
    )
    utility_axis.tick_params(axis="x", labelsize=7.0)
    utility_axis.grid(True, axis="y", color="0.9")

    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        framealpha=0.94,
        fontsize=7.0,
        columnspacing=1.1,
        handletextpad=0.4,
        borderpad=0.3,
    )
    return figure


def _plot_target_route_mass(
    strict_rows: Sequence[Mapping[str, Any]],
    zero_rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    """Plot only the target-route mass change induced by the strict margin."""

    strict_by_key = {_run_key(row): row for row in strict_rows}
    zero_by_key = {_run_key(row): row for row in zero_rows}
    if set(strict_by_key) != set(zero_by_key):
        raise ValueError("strict and zero-margin comparison identities differ")

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(3.45, 2.55),
        constrained_layout=True,
    )
    x = np.arange(len(GRAPH_ORDER), dtype=float)
    offsets = (-0.16, 0.16)
    for offset, defender in zip(offsets, DEFENDER_ORDER):
        strict_means = []
        zero_means = []
        for family in GRAPH_ORDER:
            selected = [
                row for row in strict_rows
                if row["family"] == family
                and row["defender_label"] == defender
            ]
            matched = [zero_by_key[_run_key(row)] for row in selected]
            strict_means.append(np.mean([
                float(row["intended_route_empirical_mass"])
                for row in selected
            ]))
            zero_means.append(np.mean([
                float(row["intended_route_empirical_mass"])
                for row in matched
            ]))
        positions = x + offset
        for position, zero_mass, strict_mass in zip(
            positions,
            zero_means,
            strict_means,
        ):
            axis.annotate(
                "",
                xy=(position, strict_mass),
                xytext=(position, zero_mass),
                arrowprops={
                    "arrowstyle": "->",
                    "color": DEFENDER_COLORS[defender],
                    "linewidth": 1.2,
                    "alpha": 0.75,
                },
            )
        marker = "o" if defender == DEFENDER_ORDER[0] else "^"
        axis.scatter(
            positions,
            zero_means,
            facecolors="white",
            edgecolors=DEFENDER_COLORS[defender],
            marker=marker,
            s=42,
            linewidths=1.3,
        )
        axis.scatter(
            positions,
            strict_means,
            color=DEFENDER_COLORS[defender],
            marker=marker,
            s=42,
        )

    axis.set_ylabel("Target-route mass")
    axis.set_xticks(
        x,
        [GRAPH_TICK_LABELS[value] for value in GRAPH_ORDER],
    )
    axis.tick_params(axis="x", labelsize=7.0)
    axis.set_ylim(0.1, 1.02)
    axis.grid(True, axis="y", color="0.9")
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=DEFENDER_COLORS[DEFENDER_ORDER[0]],
                markeredgecolor=DEFENDER_COLORS[DEFENDER_ORDER[0]],
                markersize=5.0,
                label="Tsallis-INF",
            ),
            Line2D(
                [],
                [],
                marker="^",
                linestyle="none",
                markerfacecolor=DEFENDER_COLORS[DEFENDER_ORDER[1]],
                markeredgecolor=DEFENDER_COLORS[DEFENDER_ORDER[1]],
                markersize=5.0,
                label="LinearExp2",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="0.30",
                markeredgewidth=1.1,
                markersize=5.0,
                label="$d=0$",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="0.30",
                markeredgecolor="0.30",
                markersize=5.0,
                label="$d=0.01$",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    return figure


def _calibration_aggregate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, float]]]:
    values: dict[
        tuple[str, str, float],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        values[(
            str(row["family"]),
            str(row["defender_label"]),
            float(row["steering_config"]["response_margin"]),
        )].append(row)
    result: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (family, defender, margin), selected in values.items():
        record = {"response_margin": margin}
        for metric in (
            "intended_route_empirical_mass",
            "per_turn_unconstrained_stackelberg_regret",
        ):
            array = np.asarray(
                [float(row[metric]) for row in selected],
                dtype=float,
            )
            record[f"mean_{metric}"] = float(np.mean(array))
            record[f"se_{metric}"] = (
                float(np.std(array, ddof=1) / np.sqrt(array.size))
                if array.size > 1 else 0.0
            )
        result[(family, defender)].append(record)
    for key in result:
        result[key].sort(key=lambda row: row["response_margin"])
    return result


def _one_commitment_per_family(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for family in GRAPH_ORDER:
        selected = [row for row in rows if row["family"] == family]
        if not selected:
            raise ValueError(f"missing evaluation rows for {family}")
        reference = json.dumps(
            selected[0]["commitment"],
            sort_keys=True,
        )
        if any(
            json.dumps(row["commitment"], sort_keys=True) != reference
            for row in selected[1:]
        ):
            raise ValueError(f"{family} commitment differs across defenders")
        result[family] = selected[0]
    return result


def _decomposition_identity_error(row: Mapping[str, Any]) -> float:
    return (
        float(row["pessimism_price_on_audit"])
        + float(row["feedback_to_audit_commitment_value_loss"])
        + float(row["finite_follower_value_loss"])
        - float(row["per_turn_unconstrained_stackelberg_regret"])
    )


def _report_markdown(analysis: Mapping[str, Any]) -> str:
    tsallis = analysis["by_defender"][DEFENDER_ORDER[0]]
    linear = analysis["by_defender"][DEFENDER_ORDER[1]]
    return "\n".join([
        "# Strict Stackelberg route-steering result",
        "",
        "## Experiment",
        "",
        (
            f"The study evaluated {analysis['graph_count']} normalized Real "
            f"Topologies, {analysis['calibration_run_count']} 100k calibration "
            f"runs, and {analysis['strict_evaluation_run_count']} fresh 500k "
            "runs. The selected response margin was "
            f"`{analysis['selected_margins'][0]:.2f}` for both Alice learners "
            "on every graph."
        ),
        "",
        "## Route induction",
        "",
        (
            f"The intended Alice route remained an audit-bank best response in "
            f"`{analysis['audit_best_response_count']}/"
            f"{analysis['strict_evaluation_run_count']}` runs. Its median "
            f"empirical mass was "
            f"`{analysis['median_intended_route_empirical_mass']:.6f}` and the "
            f"minimum across all graph/learner/seed runs was "
            f"`{analysis['minimum_intended_route_empirical_mass']:.6f}`."
        ),
        "",
        (
            "Relative to the zero-margin boundary control, the positive "
            f"commitment increased intended-route mass by a mean "
            f"`{tsallis['mean_route_mass_change_from_boundary']:+.6f}` for "
            "Tsallis-INF and "
            f"`{linear['mean_route_mass_change_from_boundary']:+.6f}` for "
            "LinearExp2."
        ),
        "",
        "## Leader value",
        "",
        (
            "The mean per-turn unconstrained Stackelberg regret was "
            f"`{tsallis['mean_per_turn_stackelberg_regret']:.6f}` against "
            "Tsallis-INF and "
            f"`{linear['mean_per_turn_stackelberg_regret']:.6f}` against "
            "LinearExp2. The plotted decomposition is exact up to mean absolute "
            f"numerical error `{analysis['mean_decomposition_identity_error']:.3e}`."
        ),
        "",
        (
            "The positive response margin deliberately gives up rational-"
            "follower leader value. During finite learning, Alice often plays "
            "routes that are worse for her than the induced best response, so "
            "the signed finite-follower term can temporarily increase Eve's "
            "realized value."
        ),
        "",
        "## Information boundary",
        "",
        (
            "Eve uses the declared histogram feedback payoff surface and "
            "physical action costs to choose one stationary commitment. She "
            "does not observe Alice's live policy, weights, audit outcomes, or "
            "future feedback draws. The audit bank and evaluation seeds are "
            "disjoint from commitment construction and margin calibration."
        ),
        "",
        "## Figures",
        "",
        "- `stackelberg_margin_frontier`: finite-time margin/value tradeoff.",
        "- `stackelberg_strict_evaluation`: route induction and value decomposition.",
        "- `stackelberg_target_route_mass`: target-route mass at `d=0` and `d=0.01`.",
        "- `stackelberg_strict_vs_zero`: paired strict/boundary comparison.",
        "- `stackelberg_physical_commitments`: exact intervention mixtures and target routes.",
        "",
        "## Primary sources",
        "",
        (
            "- Deng, Schneider, and Sivan, *Strategizing against No-regret "
            "Learners*, NeurIPS 2019."
        ),
        (
            "- Zhang, Ma, and Mazumdar, *Learning to Steer Learners in Games*, "
            "arXiv:2502.20770, 2025."
        ),
        "",
    ])


def _display_action_id(action_id: str) -> str:
    family, target = action_id.split(":", 1)
    return f"{family.replace('_', ' ')}: {target}"


def _run_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["family"]),
        str(row["defender_label"]),
        int(row["seed"]),
    )


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _save_publication_figure(
    figure: plt.Figure,
    stem: Path,
) -> list[Path]:
    paths = [
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
    ]
    figure.savefig(
        paths[0],
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        paths[1],
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        paths[2],
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    _normalize_svg_whitespace(paths[2])
    return paths


def _normalize_svg_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
