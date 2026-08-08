"""Publication-oriented figures for the histogram co-learning paper bundle."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
ATTACK_LABELS = {
    "no_attack": "No attack",
    "edge_dos": "Edge DoS",
    "edge_information_probe": "Edge information probe",
    "swap_denial": "Swap denial",
    "memory_degradation": "Memory degradation",
    "repeater_memory_measure_Z": "Memory measure Z",
    "repeater_memory_measure_X": "Memory measure X",
}


def write_paper_figures(
    figure_dir: str | Path,
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Write every figure as vector PDF plus high-resolution PNG."""

    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    writers = (
        ("learning_convergence", _plot_learning_convergence),
        ("posthoc_exploitability", _plot_exploitability),
        ("alice_false_signal_influence", _plot_false_signal_influence),
        ("route_physics_calibration", _plot_route_physics),
        ("alice_route_strategies", _plot_alice_route_strategies),
        ("eve_attack_family_strategies", _plot_eve_family_strategies),
        ("public_outcome_mix", _plot_outcome_mix),
        ("payoff_by_attack_family", _plot_payoff_by_family),
    )
    paths: list[str] = []
    for stem, writer in writers:
        figure = writer(payloads, tables)
        paths.extend(_save_both(figure, output / stem))
        plt.close(figure)
    return paths


def _plot_learning_convergence(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["learning_curve"]
    cases = _case_order(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for case_index, slug in enumerate(cases):
        color = CASE_COLORS[case_index % len(CASE_COLORS)]
        selected = [row for row in rows if row["case_slug"] == slug]
        for axis, metric, ylabel in (
            (axes[0], "alice_utility_mean", "Mean Alice service utility"),
            (axes[1], "bell_qualified_rate_hz_mean", "Bell-qualified rate (pairs/s)"),
        ):
            turns, mean, sd = _trajectory_mean_sd(selected, metric)
            axis.plot(turns, mean, color=color, linewidth=2.0, label=slug)
            axis.fill_between(
                turns,
                np.maximum(0.0, mean - sd),
                np.minimum(1.0, mean + sd) if metric == "alice_utility_mean" else mean + sd,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            axis.set_xlabel("Co-learning turn")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
    axes[0].set_title("Alice utility convergence")
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_title("Timing-aware usable service")
    axes[0].legend(frameon=False)
    figure.suptitle("Histogram co-learning: mean ± sample SD across seeds", y=1.02)
    figure.tight_layout()
    return figure


def _plot_exploitability(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["learning_curve"]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for case_index, slug in enumerate(_case_order(rows)):
        selected = [row for row in rows if row["case_slug"] == slug]
        turns, mean, sd = _trajectory_mean_sd(selected, "posthoc_exploitability")
        color = CASE_COLORS[case_index % len(CASE_COLORS)]
        axis.plot(turns, mean, color=color, linewidth=2, label=slug)
        axis.fill_between(turns, np.maximum(0.0, mean - sd), mean + sd,
                          color=color, alpha=0.16, linewidth=0)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Co-learning turn")
    axis.set_ylabel("Nash gap")
    axis.set_title("Post-hoc exploitability against stratified empirical game")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _plot_false_signal_influence(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    learning = tables["learning_curve"]
    cases = tables["case_summary"]
    order = [str(row["case_slug"]) for row in cases]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for case_index, slug in enumerate(order):
        selected = [row for row in learning if row["case_slug"] == slug]
        color = CASE_COLORS[case_index % len(CASE_COLORS)]
        for axis, metric in (
            (axes[0, 0], "alice_false_signal_rate_all_turns"),
            (axes[0, 1], "alice_false_signal_policy_total_variation"),
        ):
            turns, mean, sd = _trajectory_mean_sd(selected, metric)
            axis.plot(turns, mean, color=color, linewidth=2, label=slug)
            axis.fill_between(turns, np.maximum(0.0, mean - sd), mean + sd,
                              color=color, alpha=0.16, linewidth=0)
    axes[0, 0].set_title("False public quality-signal rate")
    axes[0, 0].set_ylabel("False signals / all turns")
    axes[0, 1].set_title("Policy mass attributable to false signals")
    axes[0, 1].set_ylabel("Factual vs corrected policy TV distance")
    for axis in axes[0]:
        axis.set_xlabel("Co-learning turn")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)

    x = np.arange(len(order))
    false_accept = np.asarray([
        row["alice_false_accept_rate_all_turns_mean"] for row in cases
    ])
    false_abort = np.asarray([
        row["alice_false_abort_rate_all_turns_mean"] for row in cases
    ])
    axes[1, 0].bar(x, false_accept, label="False accept", color="#E69F00")
    axes[1, 0].bar(x, false_abort, bottom=false_accept,
                   label="False abort", color="#56B4E9")
    axes[1, 0].set_ylabel("Fraction of all turns")
    axes[1, 0].set_title("Final false-signal decomposition")
    axes[1, 0].legend(frameon=False)

    tv = np.asarray([
        row["alice_false_signal_policy_total_variation_mean"] for row in cases
    ])
    tv_sd = np.asarray([
        row["alice_false_signal_policy_total_variation_sd"] for row in cases
    ])
    axes[1, 1].bar(
        x,
        tv,
        yerr=np.vstack([np.minimum(tv, tv_sd), tv_sd]),
        capsize=3,
        color="#009E73",
    )
    axes[1, 1].set_ylabel("Total-variation distance")
    axes[1, 1].set_title("Final fixed-behavior policy sensitivity")
    for axis in axes[1]:
        axis.set_xticks(x, order, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Alice signal fidelity and learned-policy attribution\n"
        "Latent comparator: expected QBER/CHSH under the same public thresholds",
        y=1.015,
    )
    figure.tight_layout()
    return figure


def _plot_route_physics(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["route_calibration"]
    cases = _case_order(rows)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for case_index, slug in enumerate(cases):
        selected = [row for row in rows if row["case_slug"] == slug]
        color = CASE_COLORS[case_index % len(CASE_COLORS)]
        distance = [row["total_distance_km"] for row in selected]
        rate = [row["service_rate_hz"] for row in selected]
        hops = [row["hop_count"] for row in selected]
        chsh = [row["clean_expected_chsh_s"] for row in selected]
        qber = [row["clean_expected_qber"] for row in selected]
        axes[0].scatter(distance, rate, s=np.asarray(hops) * 9,
                        color=color, label=slug, alpha=0.85)
        axes[1].scatter(hops, chsh, s=55, color=color, label=slug, alpha=0.85)
        axes[2].scatter(hops, qber, s=55, color=color, label=slug, alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("End-to-end route distance (km)")
    axes[0].set_ylabel("Rolling service rate (pairs/s, log scale)")
    axes[0].set_title("Route-specific controller timing")
    axes[1].axhline(2.0, color="black", linestyle="--", linewidth=1,
                    label="CHSH threshold")
    axes[1].set_xlabel("Hop count")
    axes[1].set_ylabel("Clean expected CHSH S")
    axes[1].set_title("Bell quality after composed swaps")
    axes[2].axhline(0.11, color="black", linestyle="--", linewidth=1,
                    label="QBER threshold")
    axes[2].set_xlabel("Hop count")
    axes[2].set_ylabel("Clean expected QBER")
    axes[2].set_title("Key-basis quality after composed swaps")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    return figure


def _plot_alice_route_strategies(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["alice_route_strategy"]
    cases = _case_order(rows)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for axis, slug in zip(axes.flat, cases):
        selected = [row for row in rows if row["case_slug"] == slug]
        route_indices = sorted({int(row["route_index"]) for row in selected})
        x = np.arange(len(route_indices))
        width = 0.25
        factual, factual_sd = zip(*[
            _mean_sd(selected, "factual_current_mass", route=i)
            for i in route_indices
        ])
        corrected_rows = [
            _mean_sd(selected, "false_signal_corrected_current_mass", route=i)
            for i in route_indices
        ]
        corrected, corrected_sd = zip(*corrected_rows)
        oracle_rows = [_mean_sd(selected, "oracle_mass", route=i) for i in route_indices]
        oracle, oracle_sd = zip(*oracle_rows)
        axis.bar(x - width, factual, width, yerr=factual_sd, capsize=2,
                 label="Factual current", color="#0072B2")
        axis.bar(x, corrected, width, yerr=corrected_sd, capsize=2,
                 label="False-signal corrected", color="#009E73")
        axis.bar(x + width, oracle, width, yerr=oracle_sd, capsize=2,
                 label="Empirical oracle", color="#D55E00")
        labels = []
        for index in route_indices:
            first = next(row for row in selected if row["route_index"] == index)
            labels.append(f"r{index}\n{first['hop_count']}h/{first['distance_km']:.1f}km")
        axis.set_xticks(x, labels)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Route probability")
        axis.set_title(slug)
        axis.grid(axis="y", alpha=0.22)
    axes.flat[0].legend(frameon=False, ncol=3, loc="upper center",
                        bbox_to_anchor=(1.1, 1.25))
    figure.suptitle("Alice learned route policies and false-signal sensitivity", y=1.01)
    figure.tight_layout()
    return figure


def _plot_eve_family_strategies(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["eve_family_strategy"]
    cases = _case_order(rows)
    families = _family_order(rows)
    learned = np.zeros((len(cases), len(families)))
    oracle = np.zeros_like(learned)
    for case_index, slug in enumerate(cases):
        selected = [row for row in rows if row["case_slug"] == slug]
        for family_index, family in enumerate(families):
            group = [row for row in selected if row["attack_family"] == family]
            learned[case_index, family_index] = np.mean([row["learned_mass"] for row in group])
            oracle[case_index, family_index] = np.mean([row["oracle_mass"] for row in group])
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.7),
        sharey=True,
        layout="constrained",
    )
    for axis, matrix, title in (
        (axes[0], learned, "Learned empirical play"),
        (axes[1], oracle, "Post-hoc empirical oracle"),
    ):
        image = axis.imshow(matrix, aspect="auto", vmin=0.0,
                            vmax=max(0.25, float(max(learned.max(), oracle.max()))),
                            cmap="viridis")
        axis.set_xticks(range(len(families)),
                        [ATTACK_LABELS.get(family, family) for family in families],
                        rotation=38, ha="right")
        axis.set_yticks(range(len(cases)), cases)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.2f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if matrix[row, column] > image.norm.vmax * 0.45 else "black")
    figure.colorbar(
        image,
        ax=axes,
        label="Attack-family probability mass",
        shrink=0.82,
        pad=0.025,
    )
    figure.suptitle("Eve strategy by physical attack family")
    return figure


def _plot_outcome_mix(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["case_summary"]
    cases = [row["case_slug"] for row in rows]
    x = np.arange(len(cases))
    metrics = (
        ("accepted_fraction_mean", "Accepted", "#009E73"),
        ("delivery_failure_fraction_mean", "Delivery failure", "#999999"),
        ("qber_abort_fraction_mean", "QBER abort", "#E69F00"),
        ("chsh_abort_fraction_mean", "CHSH abort", "#D55E00"),
    )
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    bottom = np.zeros(len(cases))
    for metric, label, color in metrics:
        values = np.asarray([row[metric] for row in rows])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x, cases, rotation=22, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of turns")
    axis.set_title("Public finite-transcript outcomes")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    return figure


def _plot_payoff_by_family(
    payloads: Sequence[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> plt.Figure:
    del payloads
    rows = tables["payoff_by_attack_family"]
    cases = _case_order(rows)
    families = _family_order(rows)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 7.2),
        layout="constrained",
    )
    image = None
    for axis, slug in zip(axes.flat, cases):
        selected = [row for row in rows if row["case_slug"] == slug]
        routes = sorted({int(row["route_index"]) for row in selected})
        matrix = np.zeros((len(routes), len(families)))
        for route_position, route in enumerate(routes):
            for family_position, family in enumerate(families):
                item = next(
                    row for row in selected
                    if int(row["route_index"]) == route
                    and row["attack_family"] == family
                )
                matrix[route_position, family_position] = item[
                    "uniform_target_mean_utility_mean"
                ]
        image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0,
                            cmap="cividis")
        axis.set_xticks(range(len(families)),
                        [ATTACK_LABELS.get(family, family) for family in families],
                        rotation=38, ha="right", fontsize=7)
        axis.set_yticks(range(len(routes)), [f"route {route}" for route in routes])
        axis.set_title(slug)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[row, column] < 0.42 else "black",
                )
    for axis in axes[0]:
        axis.tick_params(axis="x", labelbottom=False)
    assert image is not None
    figure.colorbar(
        image,
        ax=axes,
        label="Alice utility (uniform target within family)",
        shrink=0.82,
        pad=0.025,
    )
    figure.suptitle("Empirical route payoff by attack family")
    return figure


def _trajectory_mean_sd(
    rows: Sequence[dict[str, Any]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            by_turn[int(row["turn"])].append(float(value))
    turns = np.asarray(sorted(by_turn), dtype=float)
    mean = np.asarray([np.mean(by_turn[int(turn)]) for turn in turns])
    sd = np.asarray([
        np.std(by_turn[int(turn)], ddof=1) if len(by_turn[int(turn)]) > 1 else 0.0
        for turn in turns
    ])
    return turns, mean, sd


def _case_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(row["case_slug"]) for row in rows))


def _family_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    present = {str(row["attack_family"]) for row in rows}
    canonical = list(ATTACK_LABELS)
    return [family for family in canonical if family in present] + sorted(present - set(canonical))


def _mean_sd(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    route: int,
) -> tuple[float, float]:
    values = [
        float(row[metric]) for row in rows
        if int(row["route_index"]) == int(route) and row.get(metric) is not None
    ]
    if not values:
        return 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    )


def _save_both(figure: plt.Figure, base: Path) -> list[str]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    figure.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(png, dpi=240)
    return [str(pdf), str(png)]
