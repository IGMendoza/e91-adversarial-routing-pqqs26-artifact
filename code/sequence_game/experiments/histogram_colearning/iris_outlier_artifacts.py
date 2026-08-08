"""Publication artifacts for the Iris corpus outlier.

The analysis is intentionally tied to the frozen 500k corpus manifest and its
matched Tsallis-INF/Tsallis-INF and LinearExp2/Tsallis-INF result rows.  It does
not rerun physics or infer route quality from geographic drawing geometry.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

from sequence_game.topology import TopologyIR

from .corpus import HistogramGameCase
from .corpus_scale import load_corpus_scale_case
from .structure import route_resource_embedding


TSALLIS_LABEL = "tsallis_rv_eta1_vs_tsallis_rv_eta1"
LINEAR_LABEL = "linear_goptimal_eta2_vs_tsallis_rv_eta1"
IRIS_FAMILY = "iris"

TSALLIS_COLOR = "#0072B2"
LINEAR_COLOR = "#D55E00"
LOCKED_ROUTE_COLOR = "#E69F00"
EQUILIBRIUM_DETOUR_COLOR = "#0072B2"
DIRECT_ROUTE_COLOR = "#009E73"
ALICE_COLOR = "#009E73"
BOB_COLOR = "#D55E00"
CONTEXT_EDGE_COLOR = "#D3D7DE"
CONTEXT_NODE_COLOR = "#E4E7EC"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.3,
    "axes.labelsize": 8.4,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.transparent": False,
})


def write_iris_outlier_artifacts(
    run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render the frozen Iris topology and learner-outlier diagnosis."""

    source = Path(run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (source / "run_manifest.json").read_text(encoding="utf-8")
    )
    rows = _load_jsonl(source / "results.jsonl")
    iris_entry = _unique_iris_entry(manifest)
    case = load_corpus_scale_case(iris_entry, manifest)
    paired = _unique_iris_rows(rows)
    analysis = _analyze(manifest, rows, iris_entry, case, paired)

    files: list[Path] = []
    network_figure = _plot_network(case, analysis)
    files.extend(
        _save_publication_figure(
            network_figure,
            target / "iris_outlier_network",
        )
    )
    plt.close(network_figure)

    dynamics_figure = _plot_dynamics(rows, paired, analysis)
    files.extend(
        _save_publication_figure(
            dynamics_figure,
            target / "iris_outlier_dynamics",
        )
    )
    plt.close(dynamics_figure)

    report_path = target / "IRIS_OUTLIER_REPORT.md"
    report_path.write_text(
        _report_markdown(analysis),
        encoding="utf-8",
    )
    files.append(report_path)

    artifact_manifest = {
        "schema": "histogram_iris_outlier_artifacts.v1",
        "source_run": str(source),
        "source_selection_digest_blake2b_128": manifest[
            "selection_digest_blake2b_128"
        ],
        "graph_id": case.graph_id,
        "assignment_id": case.assignment_id,
        "analysis": analysis,
        "scientific_boundary": {
            "network_geometry": (
                "corpus coordinates provide layout only; normalized edge "
                "lengths in the frozen case drive histogram physics"
            ),
            "learner_claim": (
                "one matched seed identifies an Alice-side outlier and its "
                "route concentration; repeated Iris seeds are required to "
                "estimate graph-specific failure probability"
            ),
            "theorem_claim": (
                "no formal Tsallis-INF assumption violation is identified; "
                "the measured run is a single realized audited external-"
                "regret outcome, whereas the source theorem bounds expected "
                "pseudo-regret"
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


def _analyze(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    case: HistogramGameCase,
    paired: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    tsallis = paired[TSALLIS_LABEL]
    linear = paired[LINEAR_LABEL]
    tsallis_strategy = _array(tsallis["alice_strategy"])
    linear_strategy = _array(linear["alice_strategy"])
    oracle_strategy = _array(tsallis["oracle_alice_strategy"])
    if not (
        len(case.routes)
        == tsallis_strategy.size
        == linear_strategy.size
        == oracle_strategy.size
    ):
        raise ValueError("Iris route count and strategy dimensions disagree")

    learned_index = int(np.argmax(tsallis_strategy))
    oracle_indices = [
        int(index)
        for index, probability in enumerate(oracle_strategy)
        if probability > 1e-12
    ]
    if not oracle_indices:
        raise ValueError("Iris audit equilibrium has empty Alice support")
    equilibrium_detour_index = max(
        (
            index
            for index in oracle_indices
            if int(case.routes[index]["hop_count"]) > 1
        ),
        key=lambda index: float(oracle_strategy[index]),
        default=max(oracle_indices, key=lambda index: float(oracle_strategy[index])),
    )
    direct_index = min(
        oracle_indices,
        key=lambda index: int(case.routes[index]["hop_count"]),
    )

    route_edge_sets = [_route_edges(route) for route in case.routes]
    common_detour_edges = set.intersection(
        *[
            edges
            for route, edges in zip(case.routes, route_edge_sets)
            if int(route["hop_count"]) > 1
        ]
    )
    eve_ids = [str(value) for value in tsallis["eve_runtime"]["action_ids"]]
    oracle_eve = _array(tsallis["oracle_eve_strategy"])
    oracle_eve_support = [
        {
            "action_index": int(index),
            "action_id": eve_ids[index],
            "probability": float(probability),
        }
        for index, probability in enumerate(oracle_eve)
        if probability > 1e-12
    ]

    tsallis_values = np.asarray([
        float(row["exploitability"])
        for row in rows
        if row["pairing_label"] == TSALLIS_LABEL
    ])
    linear_values = np.asarray([
        float(row["exploitability"])
        for row in rows
        if row["pairing_label"] == LINEAR_LABEL
    ])
    route_embedding = route_resource_embedding(case)
    return {
        "corpus_graph_count": int(len(manifest["cases"])),
        "graph_node_count": int(entry["graph_node_count"]),
        "graph_edge_count": int(entry["graph_edge_count"]),
        "route_count": int(len(case.routes)),
        "route_resource_raw_feature_count": int(
            len(route_embedding.raw_labels)
        ),
        "route_resource_feature_rank": int(route_embedding.rank),
        "eve_action_count": int(entry["eve_action_count"]),
        "route_hops_maximum": max(
            int(route["hop_count"]) for route in case.routes
        ),
        "route_enumeration_complete": bool(
            entry["route_universe"]["enumeration_complete"]
        ),
        "quality_based_route_pruning": bool(
            entry["route_universe"]["quality_based_route_pruning"]
        ),
        "common_non_direct_edges": [
            list(edge) for edge in sorted(common_detour_edges)
        ],
        "learned_route": _route_record(
            case,
            learned_index,
            tsallis_probability=float(tsallis_strategy[learned_index]),
            linear_probability=float(linear_strategy[learned_index]),
            oracle_probability=float(oracle_strategy[learned_index]),
        ),
        "equilibrium_detour_route": _route_record(
            case,
            equilibrium_detour_index,
            tsallis_probability=float(tsallis_strategy[equilibrium_detour_index]),
            linear_probability=float(linear_strategy[equilibrium_detour_index]),
            oracle_probability=float(oracle_strategy[equilibrium_detour_index]),
        ),
        "direct_route": _route_record(
            case,
            direct_index,
            tsallis_probability=float(tsallis_strategy[direct_index]),
            linear_probability=float(linear_strategy[direct_index]),
            oracle_probability=float(oracle_strategy[direct_index]),
        ),
        "oracle_eve_support": oracle_eve_support,
        "tsallis_tsallis": _result_record(
            tsallis,
            tsallis_values,
        ),
        "linear_tsallis": _result_record(
            linear,
            linear_values,
        ),
        "paired_nash_gap_change_linear_minus_tsallis": float(
            linear["exploitability"] - tsallis["exploitability"]
        ),
        "tsallis_source": {
            "paper": (
                "Zimmert and Seldin, Tsallis-INF: An Optimal Algorithm for "
                "Stochastic and Adversarial Bandits, JMLR 22(28), 2021"
            ),
            "url": "https://www.jmlr.org/papers/v22/19-753.html",
            "formal_scope": (
                "bounded finite-arm bandit; adaptive adversarial losses may "
                "depend on past actions but not the current sampled action; "
                "the theorem bounds expected pseudo-regret"
            ),
        },
    }


def _result_record(
    row: Mapping[str, Any],
    corpus_values: np.ndarray,
) -> dict[str, Any]:
    value = float(row["exploitability"])
    return {
        "seed": int(row["seed"]),
        "turns": int(row["turns"]),
        "nash_gap": value,
        "alice_best_response_gap": float(row["alice_best_response_gap"]),
        "eve_best_response_gap": float(row["eve_best_response_gap"]),
        "alice_external_regret_per_turn": float(
            row["alice_external_regret_per_turn"]
        ),
        "eve_external_regret_per_turn": float(
            row["eve_external_regret_per_turn"]
        ),
        "mean_alice_reward": float(row["mean_alice_reward"]),
        "mean_bell_qualified_rate_hz": float(
            row["mean_bell_qualified_rate_hz"]
        ),
        "false_signal_rate": float(row["false_signal_rate"]),
        "rank_descending": int(1 + np.sum(corpus_values > value)),
        "empirical_percentile": float(100.0 * np.mean(corpus_values <= value)),
        "corpus_median": float(np.median(corpus_values)),
        "corpus_95th_percentile": float(np.quantile(corpus_values, 0.95)),
        "corpus_maximum": float(np.max(corpus_values)),
    }


def _plot_network(
    case: HistogramGameCase,
    analysis: Mapping[str, Any],
) -> plt.Figure:
    ir = TopologyIR.from_dict(case.ir_dict)
    graph = nx.Graph()
    graph.add_nodes_from(ir.nodes)
    for edge in ir.edges:
        graph.add_edge(edge.u, edge.v, length_km=float(edge.length_m) / 1000.0)
    positions = _positions(ir, graph)
    coverage = Counter(
        edge
        for route in case.routes
        for edge in _route_edges(route)
    )
    learned_index = int(analysis["learned_route"]["route_index"])
    detour_index = int(
        analysis["equilibrium_detour_route"]["route_index"]
    )
    direct_index = int(analysis["direct_route"]["route_index"])

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 4.0))
    figure.subplots_adjust(
        left=0.035,
        right=0.93,
        top=0.82,
        bottom=0.25,
        wspace=0.09,
    )
    _draw_full_network(axes[0], case, graph, positions, coverage)
    axes[0].set_title(
        "(a) Iris topology and all 20 cycle-free routes\n"
        "edge color = number of admitted routes",
    )

    _draw_key_corridor(
        axes[1],
        case,
        graph,
        positions,
        learned_index=learned_index,
        detour_index=detour_index,
        direct_index=direct_index,
    )
    axes[1].set_title(
        "(b) The learned corridor versus audited equilibrium support",
    )

    maximum = max(coverage.values(), default=1)
    colorbar = figure.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=1, vmax=maximum),
            cmap="turbo",
        ),
        ax=axes[0],
        fraction=0.04,
        pad=0.01,
    )
    colorbar.set_label("Candidate routes containing edge")
    figure.legend(
        handles=[
            Line2D(
                [0], [0], color=LOCKED_ROUTE_COLOR, linewidth=3.0,
                label=_route_legend_label(
                    "Tsallis empirical mode",
                    analysis["learned_route"],
                ),
            ),
            Line2D(
                [0], [0], color=EQUILIBRIUM_DETOUR_COLOR, linewidth=3.0,
                linestyle="--",
                label=_route_legend_label(
                    "Audit detour",
                    analysis["equilibrium_detour_route"],
                ),
            ),
            Line2D(
                [0], [0], color=DIRECT_ROUTE_COLOR, linewidth=3.0,
                label=_route_legend_label(
                    "Audit direct route",
                    analysis["direct_route"],
                ),
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=6.6,
    )
    figure.suptitle(
        "Why Iris is structurally difficult for independent route arms",
        fontsize=11.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Coordinates provide layout only. The histogram backend uses the "
        "normalized fiber lengths stored on the edges.",
        ha="center",
        fontsize=6.8,
        color="#444444",
    )
    return figure


def _draw_full_network(
    axis: plt.Axes,
    case: HistogramGameCase,
    graph: nx.Graph,
    positions: Mapping[str, np.ndarray],
    coverage: Mapping[tuple[str, str], int],
) -> None:
    corridor_edges = set(coverage)
    context_edges = [
        edge
        for edge in graph.edges
        if _edge_key(*edge) not in corridor_edges
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=context_edges,
        edge_color=CONTEXT_EDGE_COLOR,
        width=0.7,
        alpha=0.65,
        ax=axis,
    )
    maximum = max(coverage.values(), default=1)
    normalized = Normalize(vmin=1, vmax=maximum)
    for left, right in graph.edges:
        edge = _edge_key(left, right)
        if edge not in coverage:
            continue
        count = int(coverage[edge])
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(left, right)],
            edge_color=[plt.get_cmap("turbo")(normalized(count))],
            width=0.8 + 0.12 * count,
            alpha=0.9,
            ax=axis,
        )
    route_nodes = {
        node for route in case.routes for node in map(str, route["path"])
    }
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[node for node in graph if node not in route_nodes],
        node_color=CONTEXT_NODE_COLOR,
        edgecolors="#B1B7C1",
        linewidths=0.25,
        node_size=13,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=sorted(route_nodes - {case.alice, case.bob}),
        node_color="white",
        edgecolors="#30343B",
        linewidths=0.55,
        node_size=27,
        ax=axis,
    )
    _draw_endpoints(axis, graph, positions, case, size=85)
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.06)


def _draw_key_corridor(
    axis: plt.Axes,
    case: HistogramGameCase,
    graph: nx.Graph,
    positions: Mapping[str, np.ndarray],
    *,
    learned_index: int,
    detour_index: int,
    direct_index: int,
) -> None:
    key_indices = (learned_index, detour_index, direct_index)
    key_nodes = {
        node
        for index in key_indices
        for node in map(str, case.routes[index]["path"])
    }
    key_edges = {
        edge
        for index in key_indices
        for edge in _route_edges(case.routes[index])
    }
    corridor = graph.subgraph(key_nodes).copy()
    nx.draw_networkx_edges(
        corridor,
        positions,
        edgelist=list(corridor.edges),
        edge_color="#D8DCE3",
        width=1.0,
        ax=axis,
    )
    route_specs = (
        (learned_index, LOCKED_ROUTE_COLOR, "-", 5.4),
        (detour_index, EQUILIBRIUM_DETOUR_COLOR, "--", 3.0),
        (direct_index, DIRECT_ROUTE_COLOR, "-", 4.0),
    )
    for index, color, linestyle, width in route_specs:
        route_edges = [
            (str(left), str(right))
            for left, right in zip(
                case.routes[index]["path"],
                case.routes[index]["path"][1:],
            )
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=route_edges,
            edge_color=color,
            width=width,
            style=linestyle,
            alpha=0.92,
            ax=axis,
        )
    internal = sorted(key_nodes - {case.alice, case.bob})
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=internal,
        node_color="white",
        edgecolors="#30343B",
        linewidths=0.8,
        node_size=95,
        ax=axis,
    )
    _draw_endpoints(axis, graph, positions, case, size=150)
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: str(node) for node in key_nodes},
        font_size=6.7,
        font_color="#222222",
        ax=axis,
    )
    edge_labels = {
        (left, right): f"{graph[left][right]['length_km']:.1f}"
        for left, right in graph.edges
        if _edge_key(left, right) in key_edges
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        rotate=False,
        font_size=5.2,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 0.15,
        },
        ax=axis,
    )
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.12)


def _plot_dynamics(
    rows: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.35))
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.88,
        bottom=0.09,
        hspace=0.38,
        wspace=0.28,
    )
    _draw_corpus_rank(axes[0, 0], rows, paired)
    _draw_gap_decomposition(axes[0, 1], paired)
    route_indices = (
        int(analysis["learned_route"]["route_index"]),
        int(analysis["equilibrium_detour_route"]["route_index"]),
        int(analysis["direct_route"]["route_index"]),
    )
    _draw_route_dynamics(
        axes[1, 0],
        paired[TSALLIS_LABEL],
        route_indices,
        title="(c) Tsallis-INF Alice cumulative route frequencies",
    )
    _draw_route_dynamics(
        axes[1, 1],
        paired[LINEAR_LABEL],
        route_indices,
        title="(d) LinearExp2 Alice cumulative route frequencies",
    )
    figure.suptitle(
        "Iris outlier decomposition: the excess gap is Alice-side route lock-in",
        fontsize=11.5,
        fontweight="bold",
    )
    return figure


def _draw_corpus_rank(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]],
) -> None:
    for label, color, display in (
        (TSALLIS_LABEL, TSALLIS_COLOR, "Tsallis / Tsallis"),
        (LINEAR_LABEL, LINEAR_COLOR, "Linear / Tsallis"),
    ):
        values = np.sort(np.asarray([
            float(row["exploitability"])
            for row in rows
            if row["pairing_label"] == label
        ]))
        cumulative = np.arange(1, values.size + 1) / values.size
        axis.step(
            values,
            cumulative,
            where="post",
            color=color,
            label=display,
        )
        iris_value = float(paired[label]["exploitability"])
        iris_y = float(np.mean(values <= iris_value))
        axis.scatter(
            [iris_value],
            [iris_y],
            color=color,
            edgecolors="black",
            linewidths=0.5,
            s=34,
            zorder=4,
        )
        axis.annotate(
            f"Iris {iris_value:.4f}",
            (iris_value, iris_y),
            xytext=(-4, 7 if label == TSALLIS_LABEL else -12),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
            color=color,
        )
    axis.set(
        xlabel="Nash gap",
        ylabel="Corpus empirical CDF",
        title="(a) Iris is extreme only for Tsallis Alice",
    )
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.22)


def _draw_gap_decomposition(
    axis: plt.Axes,
    paired: Mapping[str, Mapping[str, Any]],
) -> None:
    labels = (TSALLIS_LABEL, LINEAR_LABEL)
    alice = np.asarray([
        float(paired[label]["alice_best_response_gap"]) for label in labels
    ])
    eve = np.asarray([
        float(paired[label]["eve_best_response_gap"]) for label in labels
    ])
    x = np.arange(2)
    axis.bar(
        x,
        alice,
        color="#56B4E9",
        edgecolor="white",
        label="Alice best-response gap",
    )
    axis.bar(
        x,
        eve,
        bottom=alice,
        color="#CC79A7",
        edgecolor="white",
        label="Eve best-response gap",
    )
    for index, total in enumerate(alice + eve):
        axis.annotate(
            f"{total:.4f}",
            (index, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
        )
    axis.set(
        xticks=x,
        xticklabels=("Tsallis /\nTsallis", "Linear /\nTsallis"),
        ylabel="Nash-gap contribution",
        title="(b) 88% of the Tsallis gap is Alice-side",
    )
    axis.legend(frameon=False, fontsize=6.5)
    axis.grid(axis="y", alpha=0.22)


def _draw_route_dynamics(
    axis: plt.Axes,
    row: Mapping[str, Any],
    route_indices: tuple[int, int, int],
    *,
    title: str,
) -> None:
    turns = np.asarray([int(point["turn"]) for point in row["trajectory"]])
    labels = (
        f"r{route_indices[0]} learned 5-hop",
        f"r{route_indices[1]} audit detour",
        f"r{route_indices[2]} direct",
    )
    for index, label, color, linestyle in zip(
        route_indices,
        labels,
        (
            LOCKED_ROUTE_COLOR,
            EQUILIBRIUM_DETOUR_COLOR,
            DIRECT_ROUTE_COLOR,
        ),
        ("-", "--", "-"),
    ):
        probabilities = np.asarray([
            float(point["alice_strategy"][index])
            for point in row["trajectory"]
        ])
        axis.plot(
            turns,
            probabilities,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axis.set(
        xlabel="Game turn",
        ylabel="Cumulative selection frequency",
        title=title,
        ylim=(-0.025, 1.025),
    )
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.legend(frameon=False, fontsize=6.2)
    axis.grid(alpha=0.22)


def _report_markdown(analysis: Mapping[str, Any]) -> str:
    tsallis = analysis["tsallis_tsallis"]
    linear = analysis["linear_tsallis"]
    learned = analysis["learned_route"]
    detour = analysis["equilibrium_detour_route"]
    direct = analysis["direct_route"]
    common = ", ".join(
        f"`{left}-{right}`"
        for left, right in analysis["common_non_direct_edges"]
    )
    eve_support = ", ".join(
        f"`{row['action_id']}` ({row['probability']:.3f})"
        for row in analysis["oracle_eve_support"]
    )
    return "\n".join([
        "# Iris outlier analysis",
        "",
        "## Result",
        "",
        (
            f"Iris is rank {tsallis['rank_descending']} of "
            f"{analysis['corpus_graph_count']} by Tsallis/Tsallis Nash gap: "
            f"`{tsallis['nash_gap']:.6f}` versus the corpus median "
            f"`{tsallis['corpus_median']:.6f}`. Its Alice contribution is "
            f"`{tsallis['alice_best_response_gap']:.6f}` and its Eve "
            f"contribution is `{tsallis['eve_best_response_gap']:.6f}`."
        ),
        "",
        (
            "With the same Tsallis-INF Eve and the same frozen Iris physics, "
            f"LinearExp2 Alice gives gap `{linear['nash_gap']:.6f}` (rank "
            f"{linear['rank_descending']}; empirical percentile "
            f"{linear['empirical_percentile']:.1f}). The matched change is "
            f"`{analysis['paired_nash_gap_change_linear_minus_tsallis']:+.6f}`."
        ),
        "",
        "## Network mechanism visible in this run",
        "",
        (
            f"The action set contains all {analysis['route_count']} cycle-free "
            "Alice--Bob routes. Enumeration completed and no route-quality "
            "pruning was used."
        ),
        "",
        (
            "Every non-direct route shares the structural corridor edges "
            f"{common}. The direct route is therefore a separate route family, "
            "while the other nineteen arms are alternative internal paths "
            "between the same two corridor boundaries."
        ),
        "",
        (
            f"Tsallis-INF Alice concentrated on route r{learned['route_index']} "
            f"(`{'-'.join(learned['path'])}`, {learned['hop_count']} hops, "
            f"{learned['length_km']:.2f} normalized km) with empirical mass "
            f"`{learned['tsallis_probability']:.3f}`. That route has zero "
            "audited equilibrium mass."
        ),
        "",
        (
            f"The audited equilibrium instead uses r{detour['route_index']} "
            f"(`{'-'.join(detour['path'])}`, {detour['hop_count']} hops, "
            f"{detour['length_km']:.2f} km) with mass "
            f"`{detour['oracle_probability']:.3f}` and the direct "
            f"r{direct['route_index']} route with mass "
            f"`{direct['oracle_probability']:.3f}`. Tsallis selected the "
            f"detour only `{detour['tsallis_probability']:.6f}` of turns; "
            f"LinearExp2 selected it `{detour['linear_probability']:.3f}`."
        ),
        "",
        (
            "The corresponding audited Eve support is "
            f"{eve_support}. This is the expected response to Iris's two "
            "effective route families: deny the shared repeater corridor or "
            "interrogate the direct edge."
        ),
        "",
        "## Does Iris violate Tsallis-INF assumptions?",
        "",
        (
            "No formal violation is visible in the experiment contract. Alice "
            "has a fixed finite arm set, rewards are bounded, she observes only "
            "her selected-arm scalar reward, and Eve samples simultaneously "
            "from a policy determined by prior history. The Tsallis-INF source "
            "permits adaptive adversarial losses that depend on actions through "
            "the preceding turn."
        ),
        "",
        (
            "The source guarantee is an expected pseudo-regret bound. The "
            "reported Iris value is one seed's audited external regret against "
            "the realized opponent-action sequence. It is therefore not a "
            "contradiction of the theorem and does not estimate how frequently "
            "this lock-in occurs."
        ),
        "",
        (
            "The important modeling distinction is structural information. "
            f"Tsallis-INF treats all {analysis['route_count']} route labels as "
            "independent arms and cannot transfer an observation from r4 to "
            "the structurally similar r18. LinearExp2 represents the "
            f"{analysis['route_count']} routes in the rank-"
            f"{analysis['route_resource_feature_rank']} span of "
            f"{analysis['route_resource_raw_feature_count']} declared "
            "constant/edge/node incidence coordinates. Its selected-route "
            "reward update therefore changes the estimates of routes sharing "
            "those resources. In this run it finds the audited two-route "
            "family. This is evidence that the Iris action geometry is a "
            "consequential invariant for Alice."
        ),
        "",
        "## Remaining graph-specific validation",
        "",
        (
            "Run repeated Iris seeds with the same frozen route/action surface "
            "and independent feedback banks. Report the distribution of Alice "
            "external regret, the probability of concentrating on r4 rather "
            "than r18, and time-to-transfer mass onto the audited support. That "
            "experiment distinguishes a rare one-seed realization from a "
            "reproducible independent-arm failure mode."
        ),
        "",
        "## Figures",
        "",
        "- `iris_outlier_network`: full Iris graph, route multiplicity, and exact key corridor routes.",
        "- `iris_outlier_dynamics`: corpus rank, Nash-gap decomposition, and the two Alice learners' cumulative route frequencies.",
        "",
        "## Primary learner reference",
        "",
        (
            "[Zimmert and Seldin, *Tsallis-INF: An Optimal Algorithm for "
            "Stochastic and Adversarial Bandits*, JMLR 22(28), 2021]"
            "(https://www.jmlr.org/papers/v22/19-753.html)."
        ),
        "",
    ])


def _route_record(
    case: HistogramGameCase,
    index: int,
    *,
    tsallis_probability: float,
    linear_probability: float,
    oracle_probability: float,
) -> dict[str, Any]:
    route = case.routes[index]
    return {
        "route_index": int(index),
        "route_id": str(route["route_id"]),
        "path": [str(node) for node in route["path"]],
        "hop_count": int(route["hop_count"]),
        "length_km": float(route["total_length_m"]) / 1000.0,
        "tsallis_probability": float(tsallis_probability),
        "linear_probability": float(linear_probability),
        "oracle_probability": float(oracle_probability),
    }


def _route_legend_label(prefix: str, route: Mapping[str, Any]) -> str:
    return (
        f"{prefix}: r{route['route_index']}, {route['hop_count']} hops, "
        f"{route['length_km']:.1f} km"
    )


def _positions(
    ir: TopologyIR,
    graph: nx.Graph,
) -> dict[str, np.ndarray]:
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
            }
    return {
        str(node): np.asarray(position)
        for node, position in nx.spring_layout(
            graph,
            seed=42,
            weight=None,
        ).items()
    }


def _draw_endpoints(
    axis: plt.Axes,
    graph: nx.Graph,
    positions: Mapping[str, np.ndarray],
    case: HistogramGameCase,
    *,
    size: float,
) -> None:
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[case.alice],
        node_color=ALICE_COLOR,
        edgecolors="black",
        linewidths=0.8,
        node_size=size,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[case.bob],
        node_color=BOB_COLOR,
        edgecolors="black",
        linewidths=0.8,
        node_size=size,
        ax=axis,
    )
    axis.annotate(
        f"Alice ({case.alice})",
        positions[case.alice],
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=6.5,
        fontweight="bold",
    )
    axis.annotate(
        f"Bob ({case.bob})",
        positions[case.bob],
        xytext=(4, -9),
        textcoords="offset points",
        fontsize=6.5,
        fontweight="bold",
    )


def _route_edges(route: Mapping[str, Any]) -> set[tuple[str, str]]:
    path = [str(node) for node in route["path"]]
    return {
        _edge_key(left, right)
        for left, right in zip(path, path[1:])
    }


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _array(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("strategy must be a finite one-dimensional vector")
    return result


def _unique_iris_entry(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    entries = [
        entry
        for entry in manifest["cases"]
        if str(entry["family"]).casefold() == IRIS_FAMILY
    ]
    if len(entries) != 1:
        raise ValueError(f"expected one Iris manifest entry, found {len(entries)}")
    return entries[0]


def _unique_iris_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    iris_rows = [
        row
        for row in rows
        if str(row["family"]).casefold() == IRIS_FAMILY
    ]
    result: dict[str, Mapping[str, Any]] = {}
    for label in (TSALLIS_LABEL, LINEAR_LABEL):
        selected = [
            row for row in iris_rows if str(row["pairing_label"]) == label
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one Iris result for {label}, found {len(selected)}"
            )
        result[label] = selected[0]
    return result


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
