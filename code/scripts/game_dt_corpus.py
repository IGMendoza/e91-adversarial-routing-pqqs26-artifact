"""Corpus decision tree on the co-learned equilibrium (not the DQN Q-rank).

For each graph we co-learn the zero-sum equilibrium by Exp3 bandit self-play, with
the minimax LP as the oracle it converges to, and record:
  * the equilibrium key-rate retention (Alice's retained rate under optimal Eve),
  * the exploitability trajectory (Nash gap of the running average -> 0, the
    no-regret convergence proof), and
  * structural graph features (node-disjoint a-b paths = min vertex cut, etc.).

Then we fit a decision tree ``structural features -> keyrate_retention`` -- a SOUND,
meaningful target (the network's usefulness to a smartly-routed Alice), replacing
the old ``-> DQN q_rank`` tree whose target was an exploitable, unstable policy.

Outputs (to --out-dir): game_dt_corpus.json (per-graph value/features/strategies),
learning_curves.png (exploitability -> 0 and value -> LP oracle).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sequence_game.game_theory import (  # noqa: E402
    bandit_selfplay_trajectory,
    characterize_equilibrium,
    solve_zero_sum_lp,
)
from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt  # noqa: E402

FEATURE_NAMES = ["node_disjoint_paths", "num_nodes", "num_edges", "num_routes",
                 "avg_route_hops", "max_route_overlap"]


# --------------------------------------------------------------------------- #
# Corpus: graphs spanning node-connectivity (a-b) from 1 (bottleneck) upward.
# --------------------------------------------------------------------------- #

def _graph(edges, routes):
    g = nx.Graph()
    g.add_edges_from(edges)
    return g, routes


def build_corpus():
    corpus = {}
    for n in range(1, 7):                       # n node-disjoint paths a->r_i->b
        edges = [("a", f"r{i}") for i in range(n)] + [(f"r{i}", "b") for i in range(n)]
        routes = [["a", f"r{i}", "b"] for i in range(n)]
        corpus[f"disjoint_{n}"] = _graph(edges, routes)
    # shared-bottleneck: 3 routes all through s -> connectivity 1.
    edges = [("a", "s")] + [("s", f"r{i}") for i in range(3)] + [(f"r{i}", "b") for i in range(3)]
    corpus["bottleneck_3"] = _graph(edges, [["a", "s", f"r{i}", "b"] for i in range(3)])
    # Braess/Wheatstone bridge: connectivity 2, plus a cross route sharing hops.
    corpus["braess"] = _graph(
        [("a", "m1"), ("a", "m2"), ("m1", "b"), ("m2", "b"), ("m1", "m2")],
        [["a", "m1", "b"], ["a", "m2", "b"], ["a", "m1", "m2", "b"]])
    # Partial overlap: 3 routes, two sharing an interior node -> connectivity 2.
    corpus["overlap_3"] = _graph(
        [("a", "p"), ("p", "b"), ("a", "q"), ("q", "b"), ("a", "u"), ("u", "v"), ("v", "b"), ("p", "v")],
        [["a", "p", "b"], ["a", "q", "b"], ["a", "u", "v", "b"]])
    return corpus


def structural_features(g, routes):
    interior = sorted({v for r in routes for v in r[1:-1]})
    node_disjoint = nx.node_connectivity(g, "a", "b")
    hops = [len(r) - 1 for r in routes]
    # Max pairwise interior-node overlap between routes.
    overlap = 0
    isets = [set(r[1:-1]) for r in routes]
    for i in range(len(isets)):
        for j in range(i + 1, len(isets)):
            overlap = max(overlap, len(isets[i] & isets[j]))
    feats = [float(node_disjoint), float(g.number_of_nodes()),
             float(g.number_of_edges()), float(len(routes)),
             float(np.mean(hops)), float(overlap)]
    return feats, interior, node_disjoint


def incidence_payoff(routes, interior):
    actions = ["no_attack"] + [f"swap_denial:{v}" for v in interior]
    P = np.ones((len(routes), len(actions)))
    for i, r in enumerate(routes):
        on = set(r[1:-1])
        for j, act in enumerate(actions):
            if act != "no_attack" and act.split(":")[1] in on:
                P[i, j] = 0.0
    return P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=60000)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/game_dt_corpus"))
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus()
    rows, records, trajectories = [], [], {}
    print(f"{'graph':<14}{'disjoint':>9}{'retention':>10}{'lp_value':>9}{'final_gap':>10}")
    for name, (g, routes) in corpus.items():
        feats, interior, node_disjoint = structural_features(g, routes)
        P = incidence_payoff(routes, interior)
        lp_value, _, _ = solve_zero_sum_lp(P)
        char = characterize_equilibrium(P, baseline_rate=1.0)
        traj = bandit_selfplay_trajectory(P, iterations=args.iterations, record_every=2000)
        trajectories[name] = traj
        rows.append({"features": feats, "keyrate_retention": char.keyrate_retention,
                     "graph_id": name})
        records.append({"graph": name, "node_disjoint_paths": node_disjoint,
                        "keyrate_retention": char.keyrate_retention, "lp_value": lp_value,
                        "eve_win": char.eve_win, "final_exploitability": float(traj.exploitability[-1]),
                        "features": dict(zip(FEATURE_NAMES, feats))})
        print(f"{name:<14}{node_disjoint:>9}{char.keyrate_retention:>10.3f}"
              f"{lp_value:>9.3f}{traj.exploitability[-1]:>10.4f}")

    dt = fit_cross_graph_dt(rows, max_depth=args.max_depth,
                            target_key="keyrate_retention", feature_names=FEATURE_NAMES)
    print(f"\nDT[keyrate_retention]  R2={dt.r2_score}  ({dt.num_topologies} graphs, {dt.num_rows} rows)")
    print(dt.rules_text)

    (args.out_dir / "game_dt_corpus.json").write_text(json.dumps({
        "records": records,
        "dt": {"target": "keyrate_retention", "r2": dt.r2_score,
               "feature_names": FEATURE_NAMES, "rules": dt.rules_text,
               "num_graphs": dt.num_topologies},
    }, indent=2))

    _plot_learning_curves(trajectories, records, args.out_dir / "learning_curves.png")
    print(f"\n-> {args.out_dir}/game_dt_corpus.json")
    print(f"-> {args.out_dir}/learning_curves.png")


def _plot_learning_curves(trajectories, records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lp = {r["graph"]: r["lp_value"] for r in records}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    names = list(trajectories)
    for k, name in enumerate(names):
        t = trajectories[name]
        c = cmap(k / max(1, len(names) - 1))
        ax1.plot(t.iters, np.maximum(t.exploitability, 1e-4), color=c, lw=1.6, label=name)
        ax2.plot(t.iters, t.value, color=c, lw=1.6)
        ax2.axhline(lp[name], color=c, ls=":", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Exp3 self-play iteration")
    ax1.set_ylabel("exploitability (Nash gap)")
    ax1.set_title("No-regret convergence: exploitability -> 0")
    ax2.set_xlabel("Exp3 self-play iteration")
    ax2.set_ylabel("running-average key-rate value")
    ax2.set_title("Value -> LP oracle (dotted)")
    ax1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
