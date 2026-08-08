"""Figures for retained historical Exp3 SeQUeNCe outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

import numpy as np

from .attack_cache import attack_cache_hop_statistics
from .baseline_cache import baseline_cache_hop_statistics
from .corpus import GraphCase
from .online import OnlineRunSummary
from .oracle import OracleSummary
from .payoff import PayoffEstimate


def write_figures(fig_dir: Path, cases: list[GraphCase],
                  oracles: dict[str, OracleSummary],
                  online: dict[str, dict[str, OnlineRunSummary]],
                  payoffs: dict[str, PayoffEstimate] | None = None,
                  attack_cache_db_path: Path | None = None,
                  baseline_cache_db_path: Path | None = None) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    paths.append(_retention_by_family(fig_dir, cases, oracles, plt))
    paths.append(_retention_vs_node_disjoint(fig_dir, cases, oracles, plt))
    paths.append(_oracle_vs_exp3(fig_dir, cases, oracles, online, plt))
    paths.append(_exploitability_curves(fig_dir, online, plt))
    selected = _selected_graph_convergence(fig_dir, cases, oracles, online, plt)
    if selected is not None:
        paths.append(selected)
    paths.append(_strategy_heatmap(fig_dir, online, "eve", plt))
    paths.append(_strategy_heatmap(fig_dir, online, "alice", plt))
    paths.append(_outcome_mix(fig_dir, cases, online, plt))
    if attack_cache_db_path is not None:
        path = _attack_cache_diagnostics(
            fig_dir,
            attack_cache_db_path,
            baseline_cache_db_path,
            plt,
        )
        if path is not None:
            paths.append(path)
    elif payoffs:
        paths.append(_attack_diagnostics_from_payoffs(fig_dir, payoffs, plt))
    return [str(path) for path in paths]


def _retention_by_family(fig_dir: Path, cases: list[GraphCase],
                         oracles: dict[str, OracleSummary], plt: Any) -> Path:
    by_family: dict[str, list[float]] = {}
    for case in cases:
        retention = oracles[case.graph_id].retention
        if retention is not None:
            by_family.setdefault(case.family, []).append(retention)
    labels = sorted(by_family, key=lambda label: float(np.mean(by_family[label])))
    values = [float(np.mean(by_family[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(labels))
    colors = _family_colors(labels)
    ax.bar(positions, values, color=[colors[label] for label in labels], alpha=0.82)
    for pos, label in zip(positions, labels):
        vals = by_family[label]
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(
            pos + np.asarray(jitter),
            vals,
            s=22,
            color="#1f1f1f",
            alpha=0.72,
            linewidths=0,
        )
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("oracle retention")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Complete-information SeQUeNCe oracle by graph family")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    path = fig_dir / "retention_by_family.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _retention_vs_node_disjoint(fig_dir: Path, cases: list[GraphCase],
                                oracles: dict[str, OracleSummary], plt: Any) -> Path:
    xs, ys, labels = [], [], []
    for case in cases:
        retention = oracles[case.graph_id].retention
        if retention is None:
            continue
        xs.append(case.features["node_disjoint_paths"])
        ys.append(retention)
        labels.append(case.family)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = _family_colors(sorted(set(labels)))
    for family in sorted(set(labels)):
        fx = [x for x, label in zip(xs, labels) if label == family]
        fy = [y for y, label in zip(ys, labels) if label == family]
        ax.scatter(fx, fy, s=42, label=family, color=colors[family], alpha=0.85)
    ideal_x = np.array([1, 2, 3, 4, 8, 16, 32], dtype=float)
    ideal_y = 1.0 - 1.0 / ideal_x
    ax.plot(ideal_x, ideal_y, color="#303030", linestyle=":", linewidth=1.6,
            label="1 - 1/N")
    ax.set_xlabel("node-disjoint paths")
    ax.set_ylabel("oracle retention")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    path = fig_dir / "retention_vs_node_disjoint_paths.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _oracle_vs_exp3(fig_dir: Path, cases: list[GraphCase],
                    oracles: dict[str, OracleSummary],
                    online: dict[str, dict[str, OnlineRunSummary]], plt: Any) -> Path:
    xs, ys, labels = [], [], []
    for case in cases:
        exp3 = online.get(case.graph_id, {}).get("exp3_vs_exp3")
        if exp3 is None or exp3.final_retention is None or oracles[case.graph_id].retention is None:
            continue
        xs.append(oracles[case.graph_id].retention)
        ys.append(exp3.final_retention)
        labels.append(case.family)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    colors = _family_colors(sorted(set(labels)))
    for family in sorted(set(labels)):
        fx = [x for x, label in zip(xs, labels) if label == family]
        fy = [y for y, label in zip(ys, labels) if label == family]
        ax.scatter(fx, fy, s=42, label=family, color=colors[family], alpha=0.85)
    ax.plot([0, 1], [0, 1], color="#333333", linestyle=":")
    ax.set_xlabel("oracle retention")
    ax.set_ylabel("Exp3 final retention")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if labels:
        ax.legend(fontsize=7, frameon=False)
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    path = fig_dir / "oracle_vs_exp3_retention.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _exploitability_curves(fig_dir: Path, online: dict[str, dict[str, OnlineRunSummary]],
                           plt: Any) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    curves = []
    for graph_id, by_condition in online.items():
        run = by_condition.get("exp3_vs_exp3")
        if run is None:
            continue
        xs = [row["turn"] for row in run.learning_curve]
        ys = [max(float(row["exploitability"]), 1e-6) for row in run.learning_curve]
        if xs:
            curves.append((xs, ys))
            ax.plot(xs, ys, linewidth=0.8, color="#9aa0a6", alpha=0.35)
    if curves:
        common_x = curves[0][0]
        if all(curve[0] == common_x for curve in curves):
            matrix = np.asarray([curve[1] for curve in curves], dtype=float)
            ax.plot(common_x, np.median(matrix, axis=0), color="#1f77b4",
                    linewidth=2.0, label="median")
            ax.fill_between(
                common_x,
                np.percentile(matrix, 10, axis=0),
                np.percentile(matrix, 90, axis=0),
                color="#1f77b4",
                alpha=0.16,
                label="10-90%",
            )
    ax.set_xlabel("turn")
    ax.set_ylabel("exploitability")
    ax.set_yscale("log")
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
    if curves:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = fig_dir / "exploitability_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _selected_graph_convergence(
        fig_dir: Path,
        cases: list[GraphCase],
        oracles: dict[str, OracleSummary],
        online: dict[str, dict[str, OnlineRunSummary]],
        plt: Any,
) -> Path | None:
    selected = _select_convergence_cases(cases, online, limit=4)
    if not selected:
        return None

    fig, (ax_gap, ax_retention) = plt.subplots(
        2, 1, figsize=(8.8, 6.4), sharex=True)
    palette = _family_colors([case.graph_id for case in selected])
    for case in selected:
        run = online.get(case.graph_id, {}).get("exp3_vs_exp3")
        oracle = oracles.get(case.graph_id)
        if run is None or oracle is None:
            continue
        curve = list(run.learning_curve)
        if not curve:
            continue
        turns = [int(row["turn"]) for row in curve]
        gaps = [
            max(float(row.get("exploitability", 0.0)), 1e-6)
            for row in curve
        ]
        retention = [
            _curve_retention(row, oracle)
            for row in curve
        ]
        color = palette[case.graph_id]
        label = (
            f"{case.graph_id} "
            f"(N={case.features['node_disjoint_paths']:.0f}, "
            f"h={case.features['longest_hops']:.0f})"
        )
        ax_gap.plot(turns, gaps, linewidth=1.9, color=color, label=label)
        ax_retention.plot(
            turns,
            retention,
            linewidth=1.9,
            color=color,
            label=label,
        )
        if oracle.retention is not None:
            ax_retention.axhline(
                oracle.retention,
                color=color,
                linestyle=":",
                linewidth=1.1,
                alpha=0.75,
            )

    ax_gap.set_ylabel("Nash gap")
    ax_gap.set_yscale("log")
    ax_gap.set_title("Exp3 convergence from cached SeQUeNCe game-turn draws")
    ax_retention.set_ylabel("sampled retention")
    ax_retention.set_xlabel("turn")
    ax_retention.set_ylim(-0.05, 1.05)
    for ax in (ax_gap, ax_retention):
        ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax_gap.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = fig_dir / "selected_graph_exp3_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _select_convergence_cases(
        cases: list[GraphCase],
        online: dict[str, dict[str, OnlineRunSummary]],
        *,
        limit: int,
) -> list[GraphCase]:
    available = [
        case for case in cases
        if online.get(case.graph_id, {}).get("exp3_vs_exp3") is not None
        and online[case.graph_id]["exp3_vs_exp3"].learning_curve
    ]
    by_id = {case.graph_id: case for case in available}
    selected: list[GraphCase] = []

    for graph_id in (
            "disjoint_parallel_2_v0",
            "wheatstone_chain_1_v0",
            "multi_bottleneck_2x4_v0",
            "layered_parallel_2x3_v0",
    ):
        case = by_id.get(graph_id)
        if case is not None:
            selected.append(case)
        if len(selected) >= limit:
            return selected

    def add_first(candidates: list[GraphCase]) -> None:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
                return

    add_first([case for case in available if "wheatstone" in case.family])
    add_first(sorted(
        available,
        key=lambda case: (
            -case.features.get("bottleneck_node_count", 0.0),
            -case.features.get("num_routes", 0.0),
        ),
    ))
    add_first(sorted(
        available,
        key=lambda case: (
            -case.features.get("node_disjoint_paths", 0.0),
            -case.features.get("num_routes", 0.0),
        ),
    ))
    add_first(sorted(
        available,
        key=lambda case: (
            -case.features.get("longest_hops", 0.0),
            -case.features.get("num_routes", 0.0),
        ),
    ))
    return selected[:limit]


def _curve_retention(row: dict[str, Any], oracle: OracleSummary) -> float:
    if row.get("retention_so_far") is not None:
        return float(row["retention_so_far"])
    baseline = float(oracle.baseline_rate)
    if baseline <= 0:
        return 0.0
    return float(row.get("key_rate_so_far", 0.0)) / baseline


def _strategy_heatmap(fig_dir: Path, online: dict[str, dict[str, OnlineRunSummary]],
                      side: str, plt: Any) -> Path:
    rows = []
    labels = []
    max_len = 0
    for graph_id, by_condition in online.items():
        run = by_condition.get("exp3_vs_exp3")
        if run is None:
            continue
        strategy = run.eve_strategy if side == "eve" else run.alice_strategy
        values = np.asarray(strategy, dtype=float)
        max_len = max(max_len, len(values))
        rows.append(values)
        labels.append(graph_id)
    if not rows:
        rows = [np.zeros(1)]
        labels = ["no online runs"]
        max_len = 1
    data = np.zeros((len(rows), max_len or 1))
    for idx, row in enumerate(rows):
        data[idx, :len(row)] = row
    fig, ax = plt.subplots(figsize=(8.4, max(2.8, 0.24 * len(rows))))
    image = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xlabel(f"{side} action index" if side == "eve" else "route index")
    ax.set_title(f"Exp3 empirical {side} strategy")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.026, pad=0.012)
    fig.tight_layout()
    path = fig_dir / f"{side}_strategy_heatmap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _outcome_mix(fig_dir: Path, cases: list[GraphCase],
                 online: dict[str, dict[str, OnlineRunSummary]], plt: Any) -> Path:
    outcomes = ["accepted", "chsh_abort", "qber_abort", "delivery_failure"]
    case_by_id = {case.graph_id: case for case in cases}
    counts_by_family: dict[str, dict[str, int]] = {}
    for graph_id, by_condition in online.items():
        case = case_by_id.get(graph_id)
        if case is None:
            continue
        run = by_condition.get("exp3_vs_exp3")
        if run is None:
            continue
        fam_counts = counts_by_family.setdefault(case.family, {})
        for outcome, count in run.outcome_counts.items():
            fam_counts[outcome] = fam_counts.get(outcome, 0) + int(count)
    labels = sorted(counts_by_family)
    rows = []
    for family in labels:
        fam_counts = counts_by_family[family]
        total = max(1, sum(fam_counts.values()))
        rows.append([fam_counts.get(outcome, 0) / total for outcome in outcomes])
    if not rows:
        rows = [[0.0, 0.0, 0.0, 0.0]]
        labels = ["no online runs"]
    data = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, max(3.2, 0.45 * len(labels))))
    left = np.zeros(len(labels))
    colors = ["#4c78a8", "#e45756", "#f58518", "#72b7b2"]
    for idx, outcome in enumerate(outcomes):
        vals = data[:, idx] if data.size else []
        ax.barh(labels, vals, left=left, label=outcome, color=colors[idx])
        left += vals
    ax.set_xlabel("share")
    ax.set_title("Exp3 outcome mix by graph family")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = fig_dir / "outcome_mix_by_family.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _attack_cache_diagnostics(
        fig_dir: Path,
        attack_db_path: Path,
        baseline_db_path: Path | None,
        plt: Any,
) -> Path | None:
    attack_rows = attack_cache_hop_statistics(attack_db_path)
    baseline_rows = (
        baseline_cache_hop_statistics(baseline_db_path)
        if baseline_db_path is not None else []
    )
    if not attack_rows and not baseline_rows:
        return None

    series: list[tuple[str, list[dict[str, Any]]]] = []
    if baseline_rows:
        series.append(("no_attack", baseline_rows))
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in attack_rows:
        by_kind.setdefault(str(row["attack_kind"]), []).append(row)
    for kind in sorted(by_kind):
        series.append((kind, by_kind[kind]))

    fig, (ax_s, ax_q) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    colors = {
        "no_attack": "#303030",
        "edge_intercept_resend": "#d45087",
        "memory_degradation": "#2f7ed8",
    }
    labels = {
        "no_attack": "no attack",
        "edge_intercept_resend": "intercept-resend",
        "memory_degradation": "memory degradation",
    }
    all_hops = sorted({
        int(row["hop_count"])
        for _kind, kind_rows in series
        for row in kind_rows
    })
    for kind, kind_rows in series:
        kind_rows = sorted(kind_rows, key=lambda row: int(row["hop_count"]))
        hops = [int(row["hop_count"]) for row in kind_rows]
        abs_s = [float(row["mean_abs_chsh_s"]) for row in kind_rows]
        abs_s_sd = [float(row["sd_abs_chsh_s"]) for row in kind_rows]
        qber = [float(row["mean_qber"]) for row in kind_rows]
        qber_sd = [float(row["sd_qber"]) for row in kind_rows]
        color = colors.get(kind, "#333333")
        label = labels.get(kind, kind.replace("_", " "))
        _plot_mean_band(ax_s, hops, abs_s, abs_s_sd, color=color, label=label)
        _plot_mean_band(ax_q, hops, qber, qber_sd, color=color, label=label)

    ax_s.axhline(2.0, color="#202124", linestyle=":", linewidth=1.2,
                 label="Bell bound |S|=2")
    ax_q.axhline(0.15, color="#202124", linestyle=":", linewidth=1.2,
                 label="QBER threshold=0.15")
    ax_s.set_ylabel("mean |CHSH S|")
    ax_s.set_title("Route-depth diagnostics from SQLite caches")
    ax_q.set_ylabel("mean QBER")
    ax_q.set_xlabel("route hops")
    if all_hops:
        ax_q.set_xticks(all_hops)
        ax_s.set_xlim(min(all_hops) - 0.35, max(all_hops) + 0.35)
    ax_s.set_ylim(bottom=0.0)
    ax_q.set_ylim(bottom=0.0)
    for ax in (ax_s, ax_q):
        ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
        ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    path = fig_dir / "attack_chsh_qber_by_hop.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_mean_band(
        ax: Any,
        hops: list[int],
        means: list[float],
        deviations: list[float],
        *,
        color: str,
        label: str,
) -> None:
    x = np.asarray(hops, dtype=float)
    y = np.asarray(means, dtype=float)
    sd = np.asarray(deviations, dtype=float)
    lower = np.maximum(0.0, y - sd)
    upper = y + sd
    ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
    ax.plot(
        x,
        y,
        marker="o",
        markersize=5.0,
        linewidth=2.0,
        color=color,
        label=f"{label} (band=1 sd)",
    )


def _attack_diagnostics_from_payoffs(
        fig_dir: Path,
        payoffs: dict[str, PayoffEstimate],
        plt: Any,
) -> Path:
    rows: dict[str, dict[int, dict[str, list[float]]]] = {}
    for payoff in payoffs.values():
        for cell in payoff.cells:
            timing = cell.sequence_timing
            if timing.get("payoff_model") != "attack_route_profile_cache":
                continue
            kind = str(cell.simulated_action_id).removeprefix("attack_cache:")
            hop_count = int(timing.get("attack_cache_hop_count", 0))
            bucket = rows.setdefault(kind, {}).setdefault(
                hop_count,
                {"chsh": [], "qber": []},
            )
            if cell.mean_chsh_s is not None:
                bucket["chsh"].append(float(cell.mean_chsh_s))
            if cell.mean_qber is not None:
                bucket["qber"].append(float(cell.mean_qber))

    fig, (ax_s, ax_q) = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    colors = {
        "edge_intercept_resend": "#e45756",
        "memory_degradation": "#4c78a8",
    }
    for kind in sorted(rows):
        hops = sorted(rows[kind])
        if not hops:
            continue
        chsh = [
            float(np.mean(rows[kind][hop]["chsh"]))
            if rows[kind][hop]["chsh"] else np.nan
            for hop in hops
        ]
        qber = [
            float(np.mean(rows[kind][hop]["qber"]))
            if rows[kind][hop]["qber"] else np.nan
            for hop in hops
        ]
        ax_s.plot(
            hops,
            chsh,
            marker="o",
            linewidth=1.8,
            color=colors.get(kind, "#333333"),
            label=kind,
        )
        ax_q.plot(
            hops,
            qber,
            marker="o",
            linewidth=1.8,
            color=colors.get(kind, "#333333"),
            label=kind,
        )
    ax_s.axhline(2.0, color="#333333", linestyle=":", linewidth=1.2)
    ax_s.set_ylabel("mean signed CHSH S")
    ax_s.set_title("Attack-cache diagnostics from payoff cells")
    ax_q.axhline(0.15, color="#333333", linestyle=":", linewidth=1.2)
    ax_q.set_ylabel("mean QBER")
    ax_q.set_xlabel("route hops")
    for ax in (ax_s, ax_q):
        ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
        if rows:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = fig_dir / "attack_chsh_qber_by_hop.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _family_colors(labels: list[str]) -> dict[str, str]:
    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    ]
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}
