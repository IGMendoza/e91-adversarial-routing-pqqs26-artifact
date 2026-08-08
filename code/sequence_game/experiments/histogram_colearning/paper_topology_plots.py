"""Publication figures for the exact histogram paper-run topologies.

The geographic panels use corpus coordinates only as a layout.  The route
schematics label the normalized edge lengths that the histogram physics
actually consumed; map geometry is never treated as a fiber-length scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

from sequence_game.topology import TopologyIR

from .corpus import HistogramGameCase


CONTEXT_EDGE_COLOR = "#C5CAD3"
CONTEXT_NODE_COLOR = "#D9DDE4"
ROUTE_EDGE_COLORS = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#CC79A7",
    4: "#D55E00",
}
ALICE_COLOR = "#009E73"
BOB_COLOR = "#D55E00"


@dataclass(frozen=True)
class TopologyFigureCase:
    """Display metadata paired with one exact histogram game case."""

    slug: str
    structural_regime: str
    case: HistogramGameCase


@dataclass(frozen=True)
class _CaseView:
    entry: TopologyFigureCase
    graph: nx.Graph
    geographic_positions: dict[str, np.ndarray]
    route_edge_coverage: dict[tuple[str, str], int]
    route_node_coverage: dict[str, int]
    edge_lengths_km: dict[tuple[str, str], float]
    corridor: nx.Graph
    corridor_positions: dict[str, np.ndarray]
    coordinate_layout_used: bool


def write_topology_figures(
    figure_dir: str | Path,
    cases: Sequence[TopologyFigureCase],
) -> list[str]:
    """Write a four-case overview and one detailed figure per case."""

    if not cases:
        raise ValueError("topology figures require at least one paper case")
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    views = tuple(_build_case_view(entry) for entry in cases)
    paths: list[str] = []

    overview = _plot_overview(views)
    paths.extend(_save_both(overview, output / "topology_overview"))
    plt.close(overview)
    for view in views:
        figure = _plot_case_detail(view)
        paths.extend(_save_both(figure, output / f"topology_{view.entry.slug}"))
        plt.close(figure)
    return paths


def topology_figure_metadata(entry: TopologyFigureCase) -> dict[str, Any]:
    """Return auditable topology/route values used in a topology figure."""

    view = _build_case_view(entry)
    shared_counts: dict[int, int] = {}
    for coverage in view.route_edge_coverage.values():
        shared_counts[coverage] = shared_counts.get(coverage, 0) + 1
    return {
        "case_slug": entry.slug,
        "graph_id": entry.case.graph_id,
        "assignment_id": entry.case.assignment_id,
        "family": entry.case.family,
        "node_count": view.graph.number_of_nodes(),
        "edge_count": view.graph.number_of_edges(),
        "route_count": len(entry.case.routes),
        "route_hops": [int(route["hop_count"]) for route in entry.case.routes],
        "route_lengths_km": [
            float(route["total_length_m"]) / 1000.0
            for route in entry.case.routes
        ],
        "route_edge_count_by_multiplicity": {
            str(key): shared_counts[key] for key in sorted(shared_counts)
        },
        "coordinate_layout_used": view.coordinate_layout_used,
        "edge_length_semantics": "normalized_topology_bench_physics_input",
    }


def _build_case_view(entry: TopologyFigureCase) -> _CaseView:
    ir = TopologyIR.from_dict(entry.case.ir_dict)
    graph = nx.Graph()
    graph.add_nodes_from(ir.nodes)
    edge_lengths: dict[tuple[str, str], float] = {}
    for edge in ir.edges:
        pair = _edge_key(edge.u, edge.v)
        graph.add_edge(edge.u, edge.v, length_m=float(edge.length_m))
        edge_lengths[pair] = float(edge.length_m) / 1000.0

    edge_coverage: dict[tuple[str, str], int] = {}
    node_coverage: dict[str, int] = {}
    for route in entry.case.routes:
        path = [str(node) for node in route["path"]]
        for node in path:
            node_coverage[node] = node_coverage.get(node, 0) + 1
        for left, right in zip(path, path[1:]):
            pair = _edge_key(left, right)
            if not graph.has_edge(left, right):
                raise ValueError(
                    f"route {route['route_id']!r} references absent edge {left!r}-{right!r}"
                )
            edge_coverage[pair] = edge_coverage.get(pair, 0) + 1

    corridor = nx.Graph()
    corridor.add_nodes_from(sorted(node_coverage))
    corridor.add_edges_from(sorted(edge_coverage))
    geographic, used_coordinates = _geographic_positions(ir, graph)
    corridor_positions = _corridor_positions(
        corridor,
        alice=entry.case.alice,
        bob=entry.case.bob,
    )
    return _CaseView(
        entry=entry,
        graph=graph,
        geographic_positions=geographic,
        route_edge_coverage=edge_coverage,
        route_node_coverage=node_coverage,
        edge_lengths_km=edge_lengths,
        corridor=corridor,
        corridor_positions=corridor_positions,
        coordinate_layout_used=used_coordinates,
    )


def _geographic_positions(
    ir: TopologyIR,
    graph: nx.Graph,
) -> tuple[dict[str, np.ndarray], bool]:
    if all(ir.nodes[node].coordinates is not None for node in graph.nodes):
        latitudes = np.asarray([
            float(ir.nodes[node].coordinates[0]) for node in graph.nodes
        ])
        longitudes = np.asarray([
            float(ir.nodes[node].coordinates[1]) for node in graph.nodes
        ])
        if np.ptp(latitudes) > 1e-9 or np.ptp(longitudes) > 1e-9:
            longitude_scale = float(np.cos(np.deg2rad(np.mean(latitudes))))
            return {
                str(node): np.asarray([
                    float(ir.nodes[node].coordinates[1]) * longitude_scale,
                    float(ir.nodes[node].coordinates[0]),
                ])
                for node in graph.nodes
            }, True
    return {
        str(node): np.asarray(position)
        for node, position in nx.spring_layout(
            graph,
            seed=42,
            weight=None,
        ).items()
    }, False


def _corridor_positions(
    graph: nx.Graph,
    *,
    alice: str,
    bob: str,
) -> dict[str, np.ndarray]:
    initial = nx.spring_layout(graph, seed=42, weight=None)
    initial[alice] = np.asarray([-1.0, 0.0])
    initial[bob] = np.asarray([1.0, 0.0])
    positions = nx.spring_layout(
        graph,
        seed=42,
        weight=None,
        pos=initial,
        fixed=[alice, bob],
        iterations=200,
    )
    x_values = np.asarray([position[0] for position in positions.values()])
    y_values = np.asarray([position[1] for position in positions.values()])
    x_center = float((x_values.max() + x_values.min()) / 2.0)
    y_center = float((y_values.max() + y_values.min()) / 2.0)
    x_scale = max(float(np.ptp(x_values)) / 2.0, 1e-9)
    y_scale = max(float(np.ptp(y_values)) / 2.0, 1e-9)
    return {
        str(node): np.asarray([
            (float(position[0]) - x_center) / x_scale,
            (float(position[1]) - y_center) / y_scale,
        ])
        for node, position in positions.items()
    }


def _plot_overview(views: Sequence[_CaseView]) -> plt.Figure:
    columns = 2
    rows = int(np.ceil(len(views) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 5.5 * rows),
        constrained_layout=False,
        squeeze=False,
    )
    figure.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.12, hspace=0.24, wspace=0.12)
    for axis, view in zip(axes.ravel(), views):
        _draw_full_topology(axis, view, overview=True)
        case = view.entry.case
        hops = [int(route["hop_count"]) for route in case.routes]
        axis.set_title(
            f"{view.entry.slug} — {case.family}\n"
            f"|V|={view.graph.number_of_nodes()}, |E|={view.graph.number_of_edges()}, "
            f"routes={len(case.routes)}, hops={min(hops)}–{max(hops)}",
            fontsize=10,
        )
    for axis in axes.ravel()[len(views):]:
        axis.set_visible(False)
    figure.suptitle(
        "Normalized Topology Bench assignments used by the 100k histogram learner",
        fontsize=14,
        fontweight="bold",
    )
    figure.legend(
        handles=_legend_handles(max(_max_coverage(view) for view in views)),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    figure.text(
        0.5,
        0.025,
        "Coordinate geometry is layout context only. Highlight width/color gives candidate-route edge multiplicity; "
        "the histogram backend uses separately normalized fiber lengths.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    return figure


def _plot_case_detail(view: _CaseView) -> plt.Figure:
    figure = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.15, 1.0),
        height_ratios=(1.0, 0.44),
        left=0.04,
        right=0.98,
        top=0.88,
        bottom=0.08,
        hspace=0.08,
        wspace=0.14,
    )
    full_axis = figure.add_subplot(grid[:, 0])
    corridor_axis = figure.add_subplot(grid[0, 1])
    route_axis = figure.add_subplot(grid[1, 1])
    _draw_full_topology(full_axis, view, overview=False)
    _draw_corridor(corridor_axis, view)
    _draw_route_table(route_axis, view)

    case = view.entry.case
    assignment_label = case.assignment_id.replace("__", " / ").replace("_", " ")
    figure.suptitle(
        f"{view.entry.slug}: {case.family} — {assignment_label}",
        fontsize=14,
        fontweight="bold",
    )
    full_axis.set_title(
        "Full topology with candidate-route corridor\n"
        f"{view.graph.number_of_nodes()} nodes, {view.graph.number_of_edges()} links",
        fontsize=10,
    )
    corridor_axis.set_title(
        "Candidate-route corridor schematic (not to scale)\n"
        "edge labels are normalized fiber lengths in km",
        fontsize=10,
    )
    figure.legend(
        handles=_legend_handles(_max_coverage(view)),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    return figure


def _draw_full_topology(
    axis: plt.Axes,
    view: _CaseView,
    *,
    overview: bool,
) -> None:
    graph = view.graph
    positions = view.geographic_positions
    route_nodes = set(view.route_node_coverage)
    context_nodes = [node for node in graph.nodes if node not in route_nodes]
    route_edges = set(view.route_edge_coverage)
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
        width=0.8 if overview else 1.0,
        alpha=0.65,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=context_nodes,
        ax=axis,
        node_color=CONTEXT_NODE_COLOR,
        edgecolors="#A8AFBA",
        linewidths=0.3,
        node_size=13 if graph.number_of_nodes() > 50 else 28,
        alpha=0.9,
    )
    _draw_coverage_edges(axis, graph, positions, view.route_edge_coverage, scale=0.85 if overview else 1.0)
    internal = sorted(route_nodes - {view.entry.case.alice, view.entry.case.bob})
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=internal,
        ax=axis,
        node_color="white",
        edgecolors="#30343B",
        linewidths=0.8,
        node_size=45 if overview else 65,
    )
    _draw_endpoints(axis, graph, positions, view.entry.case, overview=overview)
    if (
        not overview
        and len(route_nodes) <= 25
        and graph.number_of_nodes() <= 50
    ):
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node: str(node) for node in internal},
            ax=axis,
            font_size=6,
            font_color="#222222",
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.08)
    axis.set_axis_off()


def _draw_corridor(axis: plt.Axes, view: _CaseView) -> None:
    graph = view.corridor
    positions = view.corridor_positions
    _draw_coverage_edges(axis, graph, positions, view.route_edge_coverage, scale=1.25)
    internal = sorted(set(graph.nodes) - {view.entry.case.alice, view.entry.case.bob})
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=internal,
        ax=axis,
        node_color="white",
        edgecolors="#30343B",
        linewidths=1.0,
        node_size=170,
    )
    _draw_endpoints(axis, graph, positions, view.entry.case, overview=False, size=260)
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: str(node) for node in internal},
        ax=axis,
        font_size=7,
        font_color="#222222",
    )
    labels = {
        (left, right): f"{view.edge_lengths_km[_edge_key(left, right)]:.1f}"
        for left, right in graph.edges
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=labels,
        ax=axis,
        font_size=5.8,
        rotate=False,
        label_pos=0.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.2},
    )
    axis.set_axis_off()
    axis.margins(0.18)


def _draw_route_table(axis: plt.Axes, view: _CaseView) -> None:
    axis.set_axis_off()
    case = view.entry.case
    lines = [
        f"Structural regime: {view.entry.structural_regime}",
        "",
    ]
    for index, route in enumerate(case.routes, start=1):
        path = "–".join(str(node) for node in route["path"])
        header = (
            f"R{index}: {int(route['hop_count'])} hops, "
            f"{float(route['total_length_m']) / 1000.0:.2f} km"
        )
        lines.append(header)
        lines.extend(f"  {part}" for part in textwrap.wrap(path, width=64))
    lines.extend([
        "",
        "Lengths are normalized Topology Bench inputs consumed by the histogram physics;",
        "geographic coordinates are retained only to place the full-network context.",
    ])
    axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=7.3,
        family="monospace",
        linespacing=1.2,
        transform=axis.transAxes,
    )


def _draw_coverage_edges(
    axis: plt.Axes,
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    coverage: dict[tuple[str, str], int],
    *,
    scale: float,
) -> None:
    for multiplicity in sorted(set(coverage.values())):
        edges = [
            (left, right)
            for left, right in graph.edges
            if coverage.get(_edge_key(left, right)) == multiplicity
        ]
        if not edges:
            continue
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            ax=axis,
            edge_color=_coverage_color(multiplicity),
            width=scale * (2.2 + 0.8 * multiplicity),
            alpha=0.95,
        )


def _draw_endpoints(
    axis: plt.Axes,
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    case: HistogramGameCase,
    *,
    overview: bool,
    size: float | None = None,
) -> None:
    node_size = size or (95 if overview else 140)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[case.alice],
        ax=axis,
        node_color=ALICE_COLOR,
        edgecolors="#1B1B1B",
        linewidths=1.0,
        node_size=node_size,
        node_shape="o",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[case.bob],
        ax=axis,
        node_color=BOB_COLOR,
        edgecolors="#1B1B1B",
        linewidths=1.0,
        node_size=node_size,
        node_shape="s",
    )
    labels = {
        case.alice: f"A:{case.alice}",
        case.bob: f"B:{case.bob}",
    }
    offset = 0.025 if overview else 0.035
    label_positions = {
        node: np.asarray([position[0], position[1] + offset * _position_span(positions)])
        for node, position in positions.items()
        if node in labels
    }
    nx.draw_networkx_labels(
        graph,
        label_positions,
        labels=labels,
        ax=axis,
        font_size=7 if overview else 8,
        font_weight="bold",
        font_color="#111111",
    )


def _legend_handles(max_coverage: int) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=CONTEXT_EDGE_COLOR, lw=1.5, label="non-candidate link"),
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
    handles.extend(
        Line2D(
            [0],
            [0],
            color=_coverage_color(multiplicity),
            lw=2.2 + 0.8 * multiplicity,
            label=f"used by {multiplicity} route{'s' if multiplicity != 1 else ''}",
        )
        for multiplicity in range(1, max_coverage + 1)
    )
    return handles


def _coverage_color(multiplicity: int) -> str:
    return ROUTE_EDGE_COLORS.get(multiplicity, "#7A1F5C")


def _max_coverage(view: _CaseView) -> int:
    return max(view.route_edge_coverage.values(), default=1)


def _position_span(positions: dict[str, np.ndarray]) -> float:
    x_values = [float(position[0]) for position in positions.values()]
    y_values = [float(position[1]) for position in positions.values()]
    return max(max(x_values) - min(x_values), max(y_values) - min(y_values), 1.0)


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _save_both(figure: plt.Figure, stem: Path) -> list[str]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    return [str(pdf), str(png)]
