"""Figures and tables for the normalized Real Topologies 500k study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.transparent": False,
})


_TSALLIS_LABEL = "tsallis_rv_eta1_vs_tsallis_rv_eta1"
_LINEAR_LABEL = "linear_goptimal_eta2_vs_tsallis_rv_eta1"
_DISPLAY = {
    _TSALLIS_LABEL: "Tsallis-INF / Tsallis-INF",
    _LINEAR_LABEL: "LinearExp2 / Tsallis-INF",
}
_COLORS = {
    _TSALLIS_LABEL: "#0072B2",
    _LINEAR_LABEL: "#D55E00",
}


def write_corpus_scale_artifacts(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    source = Path(run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(source / "results.jsonl")
    expected = len(manifest["cases"]) * len(manifest["pairings"])
    if require_complete and len(rows) != expected:
        raise ValueError(f"corpus run is incomplete: {len(rows)}/{expected} rows")
    _validate_rows(manifest, rows)
    by_graph = _paired_rows(rows)
    files = []
    for stem, render in (
        ("nash_gap_summary", _plot_nash_gap_summary),
        ("topology_nash_gap_heatmaps", _plot_topology_nash_gap_heatmaps),
        ("pairing_exploitability", _plot_pairing_exploitability),
        ("topology_paired_exploitability", _plot_topology_pairs),
        ("action_geometry_and_gap", _plot_geometry_and_gap),
        ("learning_dynamics", _plot_learning_dynamics),
        ("service_and_signal", _plot_service_and_signal),
        ("corpus_case_geometry", _plot_case_geometry),
    ):
        path = target / f"{stem}.png"
        render(manifest, rows, by_graph, path)
        files.extend(path.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))
    csv_path = target / "corpus_pairing_results.csv"
    _write_results_csv(rows, csv_path)
    files.append(csv_path)
    report_path = target / "CORPUS_500K_REPORT.md"
    report_path.write_text(
        _report_markdown(manifest, rows, by_graph),
        encoding="utf-8",
    )
    files.append(report_path)
    artifact_manifest = {
        "schema": "histogram_corpus_scale_artifacts.v1",
        "selection_digest_blake2b_128": manifest[
            "selection_digest_blake2b_128"
        ],
        "complete_run_required": require_complete,
        "run_count": len(rows),
        "expected_run_count": expected,
        "graph_count": len(by_graph),
        "publication_rendering": {
            "formats": ["pdf", "svg", "png"],
            "png_dpi": 400,
            "vector_text": True,
            "pdf_font_embedding": "TrueType",
            "primary_figure_width_inches": 7.1,
            "color_encoding": "Okabe-Ito-derived with marker/line redundancy",
            "uncertainty_boundary": (
                "topology interquartile ranges and finite-corpus paired "
                "topology bootstrap intervals; no repeated-seed per-graph claim"
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
    manifest_path = target / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_manifest


def _plot_nash_gap_summary(
    _manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    """Show the corpus-average Nash gap and its matched-pair difference explicitly."""

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    labels = (_TSALLIS_LABEL, _LINEAR_LABEL)
    x = np.arange(len(labels))
    left_limits = []
    for index, label in enumerate(labels):
        values = np.asarray([
            float(row["exploitability"])
            for row in rows
            if row["pairing_label"] == label
        ])
        mean = float(np.mean(values))
        median = float(np.median(values))
        mean_lower, mean_upper = _bootstrap_statistic_interval(
            values,
            statistic="mean",
            seed=20_260_716 + index,
        )
        median_lower, median_upper = _bootstrap_statistic_interval(
            values,
            statistic="median",
            seed=20_260_726 + index,
        )
        left_limits.extend((mean_lower, mean_upper, median_lower, median_upper))
        axes[0].errorbar(
            index - 0.08,
            mean,
            yerr=np.asarray([[mean - mean_lower], [mean_upper - mean]]),
            marker="s",
            markersize=6.0,
            color=_COLORS[label],
            markerfacecolor=_COLORS[label],
            capsize=3,
            linewidth=1.2,
            label="Mean" if index == 0 else None,
        )
        axes[0].errorbar(
            index + 0.08,
            median,
            yerr=np.asarray([[median - median_lower], [median_upper - median]]),
            marker="o",
            markersize=6.0,
            color=_COLORS[label],
            markerfacecolor="white",
            capsize=3,
            linewidth=1.2,
            label="Median" if index == 0 else None,
        )
        axes[0].annotate(
            f"mean {mean:.5f}",
            (index - 0.08, mean_upper),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=_COLORS[label],
        )
    axes[0].set(
        xticks=x,
        xticklabels=("Tsallis / Tsallis", "Linear / Tsallis"),
        ylabel="Nash gap",
        title="(a) Across 105 normalized Real Topologies",
    )
    left_span = max(left_limits) - min(left_limits)
    axes[0].set_ylim(
        min(left_limits) - 0.07 * left_span,
        max(left_limits) + 0.16 * left_span,
    )
    axes[0].title.set_fontsize(8.3)
    axes[0].tick_params(axis="x", labelsize=7.0)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].grid(axis="y", alpha=0.22)

    differences = np.asarray([
        float(pair[_LINEAR_LABEL]["exploitability"])
        - float(pair[_TSALLIS_LABEL]["exploitability"])
        for pair in by_graph.values()
    ])
    difference_statistics = (
        ("Mean", float(np.mean(differences)), "mean", "s"),
        ("Median", float(np.median(differences)), "median", "o"),
    )
    difference_limits = [0.0]
    for index, (label, value, statistic, marker) in enumerate(difference_statistics):
        lower, upper = _bootstrap_statistic_interval(
            differences,
            statistic=statistic,
            seed=20_260_736 + index,
        )
        difference_limits.extend((lower, upper))
        axes[1].errorbar(
            index,
            value,
            yerr=np.asarray([[value - lower], [upper - value]]),
            marker=marker,
            markersize=6.5,
            color="#5E3C99",
            markerfacecolor="#5E3C99" if marker == "s" else "white",
            capsize=3,
            linewidth=1.3,
        )
        axes[1].annotate(
            f"{value:+.5f}",
            (index, upper),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#5E3C99",
        )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set(
        xticks=(0, 1),
        xticklabels=("Paired mean", "Paired median"),
        ylabel="Paired Nash-gap difference\n(Linear/Tsallis − Tsallis/Tsallis)",
        title="(b) Matched-topology difference",
    )
    difference_span = max(difference_limits) - min(difference_limits)
    axes[1].set_ylim(
        min(difference_limits) - 0.05 * difference_span,
        max(difference_limits) + 0.14 * difference_span,
    )
    axes[1].title.set_fontsize(8.3)
    axes[1].tick_params(axis="x", labelsize=7.0)
    axes[1].grid(axis="y", alpha=0.22)
    lower_count = int(np.sum(differences < 0.0))
    equal_count = int(np.sum(np.isclose(differences, 0.0)))
    axes[1].text(
        0.98,
        0.04,
        f"Linear/Tsallis lower on {lower_count}/{len(differences)}\n"
        f"equal on {equal_count}/{len(differences)}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
    )
    figure.suptitle(
        "Final audit-matrix Nash gap\n"
        "error bars: finite-topology bootstrap 95% intervals; one seed per topology",
        y=1.015,
        fontsize=10.0,
    )
    _save_publication_figure(figure, path)


def _plot_topology_nash_gap_heatmaps(
    _manifest: Mapping[str, Any],
    _rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    """Render all topology-resolved gaps without hiding the 105 graph labels."""

    ordered = sorted(
        by_graph.items(),
        key=lambda item: float(item[1][_TSALLIS_LABEL]["exploitability"]),
    )
    graph_labels = _unique_family_labels(ordered)
    absolute = np.asarray([
        (
            float(pair[_TSALLIS_LABEL]["exploitability"]),
            float(pair[_LINEAR_LABEL]["exploitability"]),
        )
        for _graph_id, pair in ordered
    ])
    block_count = min(3, len(ordered))
    index_blocks = [
        block
        for block in np.array_split(np.arange(len(ordered)), block_count)
        if block.size
    ]
    figure, axes = plt.subplots(
        1,
        block_count,
        figsize=(7.1, 4.55),
        squeeze=False,
    )
    axes = axes.ravel()
    absolute_min = float(np.min(absolute))
    absolute_max = float(np.max(absolute))
    if np.isclose(absolute_min, absolute_max):
        absolute_max = absolute_min + max(abs(absolute_min) * 0.05, 1e-9)
    absolute_image = None
    for block_index, indices in enumerate(index_blocks):
        absolute_axis = axes[block_index]
        absolute_image = absolute_axis.imshow(
            absolute[indices],
            aspect="auto",
            interpolation="nearest",
            cmap="turbo",
            vmin=absolute_min,
            vmax=absolute_max,
        )
        positions = np.arange(len(indices))
        absolute_axis.set_yticks(
            positions,
            [graph_labels[index] for index in indices],
            fontsize=6.0,
        )
        absolute_axis.set_xticks(
            (0, 1),
            ("Tsallis /\nTsallis", "Linear /\nTsallis"),
            fontsize=6.2,
        )
        rank_start = int(indices[0]) + 1
        rank_stop = int(indices[-1]) + 1
        absolute_axis.set_title(
            f"Ranks {rank_start}\N{EN DASH}{rank_stop}",
            fontsize=7.0,
            pad=2,
        )
        absolute_axis.tick_params(axis="y", length=0, pad=2)
        absolute_axis.set_yticks(
            np.arange(-0.5, len(indices), 1),
            minor=True,
        )
        absolute_axis.grid(
            which="minor",
            axis="y",
            color="white",
            linewidth=0.25,
        )
        absolute_axis.tick_params(which="minor", left=False)
    figure.subplots_adjust(
        left=0.115,
        right=0.895,
        bottom=0.075,
        top=0.97,
        wspace=1.10,
    )
    absolute_colorbar = figure.colorbar(
        absolute_image,
        ax=list(axes),
        orientation="vertical",
        fraction=0.026,
        pad=0.025,
        aspect=35,
        shrink=0.98,
    )
    absolute_colorbar.set_label(
        "Nash gap = Alice best-response gap + Eve best-response gap",
        fontsize=7.0,
        labelpad=3,
    )
    absolute_colorbar.ax.tick_params(labelsize=6.4)
    figure._skip_tight_layout = True
    _save_publication_figure(figure, path)


def _plot_pairing_exploitability(
    _manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for label, linestyle in (
        (_TSALLIS_LABEL, "-"),
        (_LINEAR_LABEL, "--"),
    ):
        values = np.sort([
            float(row["exploitability"])
            for row in rows
            if row["pairing_label"] == label
        ])
        axes[0].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            linewidth=2.2,
            color=_COLORS[label],
            linestyle=linestyle,
            label=_DISPLAY[label],
        )
    differences = np.asarray([
        float(pair[_LINEAR_LABEL]["exploitability"])
        - float(pair[_TSALLIS_LABEL]["exploitability"])
        for pair in by_graph.values()
    ])
    lower_ci, upper_ci = _bootstrap_median_interval(differences)
    axes[1].hist(
        differences,
        bins=max(12, int(np.sqrt(len(differences)) * 2)),
        color="#5E3C99",
        alpha=0.82,
        edgecolor="white",
        hatch="///",
    )
    axes[1].axvline(0.0, color="black", linewidth=1.2)
    axes[1].axvline(float(np.median(differences)), color="#E66101", linewidth=2)
    axes[0].set(
        xlabel="Final audit-matrix exploitability",
        ylabel="Fraction of topologies",
        title="Corpus distribution",
    )
    axes[1].set(
        xlabel="Linear/Tsallis minus Tsallis/Tsallis exploitability",
        ylabel="Topology count",
        title=(
            f"(b) Paired difference: median {np.median(differences):+.4f}\n"
            f"topology bootstrap 95% interval [{lower_ci:+.4f}, {upper_ci:+.4f}]"
        ),
    )
    axes[0].set_title("(a) Finite-corpus empirical CDF")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Strategic convergence across 105 normalized Real Topologies", y=1.02)
    _save_publication_figure(figure, path)


def _plot_topology_pairs(
    _manifest: Mapping[str, Any],
    _rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    ordered = sorted(
        by_graph.items(),
        key=lambda item: float(item[1][_TSALLIS_LABEL]["exploitability"]),
    )
    x = np.arange(len(ordered))
    first = np.asarray([
        float(pair[_TSALLIS_LABEL]["exploitability"])
        for _graph, pair in ordered
    ])
    second = np.asarray([
        float(pair[_LINEAR_LABEL]["exploitability"])
        for _graph, pair in ordered
    ])
    figure, axis = plt.subplots(figsize=(7.1, 3.15))
    for index, (left, right) in enumerate(zip(first, second)):
        axis.plot(
            (index, index),
            (left, right),
            color="#BDBDBD",
            linewidth=0.8,
            zorder=1,
        )
    axis.scatter(
        x, first, s=14, marker="o", facecolors="none",
        edgecolors=_COLORS[_TSALLIS_LABEL], linewidths=0.8,
        label=_DISPLAY[_TSALLIS_LABEL],
    )
    axis.scatter(
        x, second, s=15, marker="^", color=_COLORS[_LINEAR_LABEL],
        linewidths=0.6, label=_DISPLAY[_LINEAR_LABEL],
    )
    axis.set(
        xlabel="Topology, sorted by Tsallis/Tsallis exploitability",
        ylabel="Final audit-matrix exploitability",
        title="Each vertical segment is one matched topology and seed",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    _save_publication_figure(figure, path)


def _plot_geometry_and_gap(
    manifest: Mapping[str, Any],
    _rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    cases = {str(row["graph_id"]): row for row in manifest["cases"]}
    graph_ids = sorted(by_graph)
    routes = np.asarray([float(cases[graph]["route_count"]) for graph in graph_ids])
    actions = np.asarray([float(cases[graph]["eve_action_count"]) for graph in graph_ids])
    delta = np.asarray([
        float(by_graph[graph][_LINEAR_LABEL]["exploitability"])
        - float(by_graph[graph][_TSALLIS_LABEL]["exploitability"])
        for graph in graph_ids
    ])
    maximum_tier = np.asarray([
        float(cases[graph]["estimated_cache_mib"]) > float(
            manifest["plan"]["case_cache_limit_mib"]
        )
        for graph in graph_ids
    ])
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    color_limit = max(abs(delta.min()), abs(delta.max()), 1e-12)
    for axis, values, label in (
        (axes[0], routes, "Alice route count"),
        (axes[1], actions, "Eve action count"),
    ):
        axis.scatter(
            values[~maximum_tier],
            delta[~maximum_tier],
            c=delta[~maximum_tier],
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            s=30,
            alpha=0.8,
        )
        axis.scatter(
            values[maximum_tier],
            delta[maximum_tier],
            marker="D",
            facecolors="none",
            edgecolors="black",
            s=64,
            linewidths=1.2,
            label="maximum cache tier",
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Paired exploitability difference")
    axes[0].set_title("(a) Alice action geometry")
    axes[1].set_title("(b) Eve action geometry")
    axes[1].legend(frameon=False)
    figure.suptitle("Action-space geometry and structured-Alice difference", y=1.02)
    _save_publication_figure(figure, path)


def _plot_learning_dynamics(
    _manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    _by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    fields = (
        ("alice_utility_mean", "Mean Alice public-service reward"),
        (
            "alice_policy_effective_action_fraction",
            "Alice normalized order-one Hill diversity",
        ),
        (
            "eve_policy_effective_action_fraction",
            "Eve normalized order-one Hill diversity",
        ),
        (
            "alice_false_signal_policy_total_variation",
            "TV distance from fixed-action reward replay",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.25), sharex=True)
    for axis, (field, title) in zip(axes.flat, fields):
        for label, linestyle in (
            (_TSALLIS_LABEL, "-"),
            (_LINEAR_LABEL, "--"),
        ):
            selected = [row for row in rows if row["pairing_label"] == label]
            turns = np.asarray([point["turn"] for point in selected[0]["trajectory"]])
            values = np.asarray([
                [float(point[field]) for point in row["trajectory"]]
                for row in selected
            ])
            median = np.median(values, axis=0)
            lower = np.quantile(values, 0.25, axis=0)
            upper = np.quantile(values, 0.75, axis=0)
            axis.plot(
                turns, median, color=_COLORS[label], linewidth=1.8,
                linestyle=linestyle, label=_DISPLAY[label],
            )
            axis.fill_between(turns, lower, upper, color=_COLORS[label], alpha=0.16)
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[1, 0].set_xlabel("Game turn")
    axes[1, 1].set_xlabel("Game turn")
    axes[0, 0].legend(frameon=False)
    for panel, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.set_title(f"{panel} {axis.get_title()}")
    _save_publication_figure(figure, path)


def _plot_service_and_signal(
    _manifest: Mapping[str, Any],
    _rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    metrics = (
        ("mean_alice_reward", "Mean Alice public-service reward"),
        ("mean_bell_qualified_rate_hz", "Mean Bell-qualified service rate (pairs/s)"),
        ("false_signal_rate", "False public-signal rate"),
        ("false_signal_policy_total_variation", "False-signal policy attribution (TV)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.0))
    for axis, (field, title) in zip(axes.flat, metrics):
        left = np.asarray([
            float(pair[_TSALLIS_LABEL][field]) for pair in by_graph.values()
        ])
        right = np.asarray([
            float(pair[_LINEAR_LABEL][field]) for pair in by_graph.values()
        ])
        bound_min = min(left.min(), right.min())
        bound_max = max(left.max(), right.max())
        if np.isclose(bound_min, bound_max):
            padding = max(abs(bound_min) * 0.05, 1e-6)
            bound_min -= padding
            bound_max += padding
        axis.scatter(
            left, right, s=20, alpha=0.75, marker="o",
            facecolors="none", edgecolors="#009E73", linewidths=0.8,
        )
        axis.plot((bound_min, bound_max), (bound_min, bound_max), color="black", linewidth=1)
        axis.set(
            xlabel="Tsallis/Tsallis",
            ylabel="Linear/Tsallis",
            title=title,
        )
        axis.grid(alpha=0.22)
    for panel, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.set_title(f"{panel} {axis.get_title()}")
    figure.suptitle("Matched service and finite-transcript outcomes", y=1.01)
    _save_publication_figure(figure, path)


def _plot_case_geometry(
    manifest: Mapping[str, Any],
    _rows: Sequence[Mapping[str, Any]],
    _by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: Path,
) -> None:
    cases = manifest["cases"]
    fields = (
        ("route_count", "Alice routes", True),
        ("eve_action_count", "Eve actions", True),
        ("estimated_cache_mib", "Estimated resident cache (MiB)", True),
        ("nominal_route_action_cells", "Route/action cells", True),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))
    for axis, (field, title, log_x) in zip(axes.flat, fields):
        values = np.asarray([float(row[field]) for row in cases])
        positive = values[values > 0.0]
        bins = np.geomspace(positive.min(), positive.max(), 18) if (
            log_x and positive.max() > positive.min()
        ) else 10
        axis.hist(values, bins=bins, color="#56B4E9", edgecolor="white")
        if log_x and positive.max() > positive.min():
            axis.set_xscale("log")
        axis.set_title(title)
        axis.set_ylabel("Topology count")
        axis.grid(axis="y", alpha=0.22)
    for panel, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.set_title(f"{panel} {axis.get_title()}")
    figure.suptitle("Frozen 105-topology action and cache geometry", y=1.01)
    _save_publication_figure(figure, path)


def _write_results_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "graph_id",
        "family",
        "assignment_id",
        "pairing_label",
        "seed",
        "turns",
        "route_count",
        "eve_action_count",
        "exploitability",
        "alice_external_regret_per_turn",
        "eve_external_regret_per_turn",
        "mean_alice_reward",
        "mean_bell_qualified_rate_hz",
        "false_signal_rate",
        "false_signal_policy_total_variation",
        "information_observation_rate",
        "turns_per_second",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["graph_id"]), str(item["pairing_label"]))):
            writer.writerow({field: row[field] for field in fields})


def _save_publication_figure(figure: Any, png_path: Path) -> None:
    """Write a consistent raster preview and two editable vector formats."""

    if not getattr(figure, "_skip_tight_layout", False):
        figure.tight_layout()
    figure.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        png_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        png_path.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    draws: int = 20_000,
    seed: int = 20_260_716,
) -> tuple[float, float]:
    """Finite-corpus topology bootstrap interval for a paired median."""

    return _bootstrap_statistic_interval(
        values,
        statistic="median",
        draws=draws,
        seed=seed,
    )


def _bootstrap_statistic_interval(
    values: np.ndarray,
    *,
    statistic: str,
    draws: int = 20_000,
    seed: int = 20_260_716,
) -> tuple[float, float]:
    """Finite-corpus topology bootstrap interval for a mean or median."""

    observed = np.asarray(values, dtype=float)
    if observed.ndim != 1 or observed.size < 1 or not np.all(np.isfinite(observed)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if statistic not in {"mean", "median"}:
        raise ValueError("bootstrap statistic must be 'mean' or 'median'")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    # Block the resampling to avoid allocating draws x topology_count at once.
    block = 1_000
    for start in range(0, draws, block):
        stop = min(draws, start + block)
        indices = rng.integers(0, observed.size, size=(stop - start, observed.size))
        if statistic == "mean":
            estimates[start:stop] = np.mean(observed[indices], axis=1)
        else:
            estimates[start:stop] = np.median(observed[indices], axis=1)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _report_markdown(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    by_graph: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    lines = [
        "# Normalized Real Topologies 500k learner results",
        "",
        f"Selection digest: `{manifest['selection_digest_blake2b_128']}`.",
        "",
        "Both pairings use 500,000 fresh compiled-histogram transcript turns per topology. Strategic metrics use the independent equal-coverage audit matrix.",
        "",
        "| Pairing | Graphs | Median exploitability | Mean exploitability | Median Alice regret | Median Eve regret | Median turns/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (_TSALLIS_LABEL, _LINEAR_LABEL):
        selected = [row for row in rows if row["pairing_label"] == label]
        values = lambda field: np.asarray([float(row[field]) for row in selected])
        lines.append(
            f"| {_DISPLAY[label]} | {len(selected)} | "
            f"{np.median(values('exploitability')):.6f} | "
            f"{np.mean(values('exploitability')):.6f} | "
            f"{np.median(values('alice_external_regret_per_turn')):.6f} | "
            f"{np.median(values('eve_external_regret_per_turn')):.6f} | "
            f"{np.median(values('turns_per_second')):.1f} |"
        )
    differences = np.asarray([
        float(pair[_LINEAR_LABEL]["exploitability"])
        - float(pair[_TSALLIS_LABEL]["exploitability"])
        for pair in by_graph.values()
    ])
    lower_ci, upper_ci = _bootstrap_median_interval(differences)
    lines.extend([
        "",
        "## Paired corpus comparison",
        "",
        f"The median Linear/Tsallis minus Tsallis/Tsallis exploitability is {np.median(differences):+.6f}; its finite-corpus topology-bootstrap 95% interval is [{lower_ci:+.6f}, {upper_ci:+.6f}]. Linear/Tsallis is lower on {int(np.sum(differences < 0.0))} of {len(differences)} topologies, equal on {int(np.sum(np.isclose(differences, 0.0)))}.",
        "",
        "This is one matched seed per topology. Corpus-level paired differences do not establish repeated-seed uncertainty for an individual graph.",
        "",
        "## Figures",
        "",
        "- `nash_gap_summary.png`: explicit corpus mean/median Nash gaps and matched-topology differences with finite-topology bootstrap intervals.",
        "- `topology_nash_gap_heatmaps.png`: every selected topology's two final Nash gaps, ranked by Tsallis/Tsallis gap in three consecutive columns.",
        "- `pairing_exploitability.png`: corpus CDFs and paired differences.",
        "- `topology_paired_exploitability.png`: all matched graph results.",
        "- `action_geometry_and_gap.png`: route/action dimensions versus the paired learner difference.",
        "- `learning_dynamics.png`: median policy, reward, and finite-sample decision-mismatch attribution trajectories.",
        "- `service_and_signal.png`: matched service and finite-transcript outcomes.",
        "- `corpus_case_geometry.png`: the frozen action/cache geometry.",
        "",
    ])
    return "\n".join(lines)


def _paired_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["graph_id"]), {})[str(row["pairing_label"])] = row
    return {
        graph: pair
        for graph, pair in result.items()
        if {_TSALLIS_LABEL, _LINEAR_LABEL} <= set(pair)
    }


def _unique_family_labels(
    ordered: Sequence[tuple[str, Mapping[str, Mapping[str, Any]]]],
) -> list[str]:
    families = [
        str(pair[_TSALLIS_LABEL].get("family", graph_id))
        for graph_id, pair in ordered
    ]
    normalized_counts: dict[str, int] = {}
    for family in families:
        key = family.casefold()
        normalized_counts[key] = normalized_counts.get(key, 0) + 1
    labels = []
    for (graph_id, _pair), family in zip(ordered, families):
        readable = family.replace("_", " ")
        if normalized_counts[family.casefold()] > 1:
            readable = f"{readable} [{graph_id.split('_', 1)[0]}]"
        labels.append(readable)
    return labels


def _validate_rows(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    digest = manifest["selection_digest_blake2b_128"]
    keys = set()
    for row in rows:
        if row.get("selection_digest_blake2b_128") != digest:
            raise ValueError("result row selection digest mismatch")
        key = str(row["run_key"])
        if key in keys:
            raise ValueError(f"duplicate result row {key}")
        keys.add(key)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
