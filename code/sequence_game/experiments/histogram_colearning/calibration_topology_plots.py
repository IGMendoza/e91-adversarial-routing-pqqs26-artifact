"""Topology context figures for the Step 7 learner calibration.

The network panels show the exact frozen Topology Bench graphs and every
cycle-free Alice--Bob route. Geographic
coordinates, when present, determine layout only; the histogram backend uses
the separately normalized edge lengths stored in each topology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np

from .calibration_candidates import LEARNER_FAMILIES
from .paper_topology_plots import (
    ALICE_COLOR,
    BOB_COLOR,
    CONTEXT_EDGE_COLOR,
    CONTEXT_NODE_COLOR,
    TopologyFigureCase,
    _CaseView,
    _build_case_view,
    _edge_key,
    _save_both,
)


CALIBRATION_FAMILY_LABELS = {
    "flat_exp3": "Exp3",
    "tsallis_inf": "Tsallis-INF",
    "exp3_plus_plus": "EXP3++",
    "linear_exp2": "LinearExp2",
}

TOPOLOGY_LABELS = {
    "cernet": "CERNET",
    "hiberniaireland": "Hibernia Ireland",
    "karen": "KAREN",
    "layer42": "Layer42",
    "memorex": "Memorex topology",
}

TOPOLOGY_ORDER = {
    "cernet": 0,
    "hiberniaireland": 1,
    "karen": 2,
    "layer42": 3,
    "memorex": 4,
}


def write_step7_calibration_topology_figures(
    figure_dir: str | Path,
    cases: Sequence[TopologyFigureCase],
    confirmation: dict[str, Any],
) -> list[str]:
    """Write a five-network overview and a network-aligned Nash-gap figure."""

    if not cases:
        raise ValueError("Step 7 topology figures require at least one case")
    views = tuple(sorted(
        (_build_case_view(entry) for entry in cases),
        key=lambda view: TOPOLOGY_ORDER.get(view.entry.case.family, 999),
    ))
    results = _calibration_results(views, confirmation)
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    overview = _plot_calibration_topology_overview(views, results)
    paths.extend(_save_both(
        overview,
        output / "calibration_topology_overview",
    ))
    plt.close(overview)

    comparison = _plot_network_aligned_nash_gap(views, results)
    paths.extend(_save_both(
        comparison,
        output / "confirmation_topology_network_context",
    ))
    plt.close(comparison)
    return paths


def _calibration_results(
    views: Sequence[_CaseView],
    confirmation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    aggregates = confirmation.get("aggregate")
    if not isinstance(aggregates, list):
        raise ValueError("confirmation has no aggregate cross-play rows")
    expected_pairs = {
        (alice, eve)
        for alice in LEARNER_FAMILIES
        for eve in LEARNER_FAMILIES
    }
    lookup = {
        (str(row["alice_family"]), str(row["eve_family"])): row
        for row in aggregates
    }
    if set(lookup) != expected_pairs:
        raise ValueError("confirmation does not contain the complete 4-by-4 cross-play")

    results: dict[str, dict[str, Any]] = {}
    for view in views:
        case_id = _case_id(view)
        gaps = {
            pair: float(row["case_means_exploitability"][case_id])
            for pair, row in lookup.items()
        }
        best_pair = min(gaps, key=gaps.__getitem__)
        results[case_id] = {
            "gaps": gaps,
            "best_pair": best_pair,
            "best_gap": gaps[best_pair],
            "baseline_gap": gaps[("flat_exp3", "flat_exp3")],
        }
    return results


def _plot_calibration_topology_overview(
    views: Sequence[_CaseView],
    results: dict[str, dict[str, Any]],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 10.4), squeeze=False)
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.90,
        bottom=0.14,
        hspace=0.29,
        wspace=0.10,
    )
    norm, maximum_coverage = _coverage_norm(views)
    cmap = matplotlib.colormaps["viridis"]
    for axis, view in zip(axes.ravel(), views):
        _draw_calibration_network(axis, view, norm=norm, cmap=cmap, compact=False)
        result = results[_case_id(view)]
        alice_family, eve_family = result["best_pair"]
        case = view.entry.case
        axis.set_title(
            f"{_topology_label(case.family)}\n"
            f"|V|={view.graph.number_of_nodes()}, |E|={view.graph.number_of_edges()}, "
            f"routes={len(case.routes)}, Eve actions={len(case.actions('attack_surface_all'))}\n"
            f"lowest gap: {CALIBRATION_FAMILY_LABELS[alice_family]} / "
            f"{CALIBRATION_FAMILY_LABELS[eve_family]} = {result['best_gap']:.4f}",
            fontsize=10,
        )

    information_axis = axes.ravel()[-1]
    information_axis.set_axis_off()
    information_axis.text(
        0.03,
        0.96,
        "How to read the networks",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=information_axis.transAxes,
    )
    information_axis.text(
        0.03,
        0.82,
        "Gray link\n"
        "  present in the Topology Bench graph but absent from\n"
        "  this Alice/Bob assignment's candidate routes\n\n"
        "Colored link\n"
        "  used by at least one candidate route; color and width\n"
        "  encode the exact number of candidate routes using it\n\n"
        "Green circle / orange square\n"
        "  Alice / Bob endpoints\n\n"
        "The displayed layout is geographic context when corpus\n"
        "coordinates exist. It is not a fiber-length scale.",
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.25,
        color="#333333",
        transform=information_axis.transAxes,
    )
    figure.suptitle(
        "Five frozen Topology Bench calibration networks",
        fontsize=17,
        fontweight="bold",
    )
    _add_coverage_colorbar(
        figure,
        norm=norm,
        cmap=cmap,
        maximum=maximum_coverage,
        rect=(0.34, 0.065, 0.34, 0.020),
    )
    figure.legend(
        handles=_endpoint_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.105,
        "Candidate routes are all cycle-free Alice--Bob paths. No service-quality filter is applied.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return figure


def _plot_network_aligned_nash_gap(
    views: Sequence[_CaseView],
    results: dict[str, dict[str, Any]],
) -> plt.Figure:
    figure = plt.figure(figsize=(22.0, 12.6), constrained_layout=False)
    grid = figure.add_gridspec(
        len(views),
        2,
        width_ratios=(1.05, 4.65),
        left=0.025,
        right=0.965,
        top=0.90,
        bottom=0.18,
        hspace=0.03,
        wspace=0.08,
    )
    norm, maximum_coverage = _coverage_norm(views)
    route_cmap = matplotlib.colormaps["viridis"]
    for row_index, view in enumerate(views):
        axis = figure.add_subplot(grid[row_index, 0])
        _draw_calibration_network(
            axis,
            view,
            norm=norm,
            cmap=route_cmap,
            compact=True,
        )
        case = view.entry.case
        axis.set_title(
            f"{_topology_label(case.family)}  "
            f"(|V|={view.graph.number_of_nodes()}, |E|={view.graph.number_of_edges()}, "
            f"routes={len(case.routes)}, attacks={len(case.actions('attack_surface_all'))})",
            fontsize=8.5,
            pad=1.5,
        )

    pairs = [
        (alice, eve)
        for alice in LEARNER_FAMILIES
        for eve in LEARNER_FAMILIES
    ]
    matrix = np.asarray([
        [results[_case_id(view)]["gaps"][pair] for pair in pairs]
        for view in views
    ])
    heatmap_axis = figure.add_subplot(grid[:, 1])
    image = heatmap_axis.imshow(matrix, cmap="turbo", aspect="auto")
    heatmap_axis.set_xticks(
        np.arange(len(pairs)),
        [
            f"A: {CALIBRATION_FAMILY_LABELS[alice]}\n"
            f"E: {CALIBRATION_FAMILY_LABELS[eve]}"
            for alice, eve in pairs
        ],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )
    heatmap_axis.set_yticks(
        np.arange(len(views)),
        [_topology_label(view.entry.case.family) for view in views],
        fontsize=10,
    )
    heatmap_axis.set_title(
        "100k calibration Nash gap by Alice/Eve learner pair",
        fontsize=14,
        pad=10,
    )
    for row in range(matrix.shape[0]):
        best_column = int(np.argmin(matrix[row]))
        heatmap_axis.add_patch(Rectangle(
            (best_column - 0.5, row - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=2.4,
        ))
        for column in range(matrix.shape[1]):
            rgba = image.cmap(image.norm(matrix[row, column]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            heatmap_axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="black" if luminance > 0.52 else "white",
            )
    for boundary in (3.5, 7.5, 11.5):
        heatmap_axis.axvline(boundary, color="white", linewidth=1.8, alpha=0.9)
    heatmap_axis.set_xlabel(
        "Alice learner / Eve learner (black frame = lowest gap for that topology)",
        fontsize=10,
        labelpad=12,
    )
    figure.colorbar(
        image,
        ax=heatmap_axis,
        fraction=0.022,
        pad=0.015,
        label="Nash gap (lower is better)",
    )
    figure.suptitle(
        "Network structure beside the topology-resolved learner comparison",
        fontsize=17,
        fontweight="bold",
    )
    _add_coverage_colorbar(
        figure,
        norm=norm,
        cmap=route_cmap,
        maximum=maximum_coverage,
        rect=(0.055, 0.075, 0.20, 0.018),
    )
    figure.legend(
        handles=_endpoint_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.035, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.text(
        0.36,
        0.055,
        "Network color/width is candidate-route multiplicity. Gray links are outside the selected "
        "Alice/Bob route corridor. Layout geometry is not the physical fiber-length scale.",
        ha="left",
        fontsize=8.5,
        color="#444444",
    )
    return figure


def _draw_calibration_network(
    axis: plt.Axes,
    view: _CaseView,
    *,
    norm: Normalize,
    cmap: Any,
    compact: bool,
) -> None:
    graph = view.graph
    positions = view.geographic_positions
    route_edges = set(view.route_edge_coverage)
    route_nodes = set(view.route_node_coverage)
    context_edges = [
        (left, right)
        for left, right in graph.edges
        if _edge_key(left, right) not in route_edges
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=context_edges,
        ax=axis,
        edge_color=CONTEXT_EDGE_COLOR,
        width=0.7 if compact else 0.9,
        alpha=0.72,
    )
    coverage_edges = [
        (left, right)
        for left, right in graph.edges
        if _edge_key(left, right) in route_edges
    ]
    coverage_values = [
        view.route_edge_coverage[_edge_key(left, right)]
        for left, right in coverage_edges
    ]
    maximum = max(view.route_edge_coverage.values(), default=1)
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=coverage_edges,
        ax=axis,
        edge_color=[cmap(norm(value)) for value in coverage_values],
        width=[
            (1.2 if compact else 1.7)
            + (2.2 if compact else 3.0) * np.sqrt(value / maximum)
            for value in coverage_values
        ],
        alpha=0.96,
    )
    node_scale = 0.72 if compact else 1.0
    context_nodes = [node for node in graph.nodes if node not in route_nodes]
    internal_nodes = sorted(
        route_nodes - {view.entry.case.alice, view.entry.case.bob}
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=context_nodes,
        ax=axis,
        node_color=CONTEXT_NODE_COLOR,
        edgecolors="#A8AFBA",
        linewidths=0.25,
        node_size=node_scale * (13 if graph.number_of_nodes() > 50 else 24),
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=internal_nodes,
        ax=axis,
        node_color="white",
        edgecolors="#30343B",
        linewidths=0.65,
        node_size=node_scale * 45,
    )
    endpoint_size = node_scale * 105
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[view.entry.case.alice],
        ax=axis,
        node_color=ALICE_COLOR,
        edgecolors="#1B1B1B",
        linewidths=0.9,
        node_size=endpoint_size,
        node_shape="o",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[view.entry.case.bob],
        ax=axis,
        node_color=BOB_COLOR,
        edgecolors="#1B1B1B",
        linewidths=0.9,
        node_size=endpoint_size,
        node_shape="s",
    )
    if not compact:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={
                view.entry.case.alice: "A",
                view.entry.case.bob: "B",
            },
            ax=axis,
            font_size=7,
            font_weight="bold",
            font_color="#111111",
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.10 if compact else 0.08)
    axis.set_axis_off()


def _coverage_norm(views: Sequence[_CaseView]) -> tuple[Normalize, int]:
    maximum = max(
        max(view.route_edge_coverage.values(), default=1)
        for view in views
    )
    if maximum <= 1:
        return Normalize(vmin=0.0, vmax=1.0), 1
    return LogNorm(vmin=1.0, vmax=float(maximum)), int(maximum)


def _add_coverage_colorbar(
    figure: plt.Figure,
    *,
    norm: Normalize,
    cmap: Any,
    maximum: int,
    rect: tuple[float, float, float, float],
) -> None:
    axis = figure.add_axes(rect)
    ticks = sorted({
        value for value in (1, 2, 5, 10, 20, maximum)
        if value <= maximum
    })
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=axis,
        orientation="horizontal",
        ticks=ticks,
    )
    colorbar.set_label("candidate routes using link", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _endpoint_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=CONTEXT_EDGE_COLOR, lw=1.5, label="non-route link"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ALICE_COLOR,
            markeredgecolor="#1B1B1B",
            markersize=7,
            label="Alice",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=BOB_COLOR,
            markeredgecolor="#1B1B1B",
            markersize=7,
            label="Bob",
        ),
    ]


def _case_id(view: _CaseView) -> str:
    case = view.entry.case
    return f"{case.graph_id}::{case.assignment_id}"


def _topology_label(family: str) -> str:
    return TOPOLOGY_LABELS.get(family, family)
