"""1/N disjoint-path keyrate experiment.

Hypothesis (user): with N node-disjoint Alice->Bob paths, under optimal
targeted-Eve pressure a learned Alice loses at most 1/N of her keyrate, i.e.
retains >= (N-1)/N; a single-path (1-node) bottleneck -> 100% denial.

This is the routing/pursuit game realised on the SeQUeNCe E91 backend: each step
Eve picks a node/edge to attack and Alice picks a route; a collision denies the
key (delivery_failure / qber_abort / chsh_abort), otherwise E91 makes a key.

We build width-N graphs n0 -> {r0..r_{N-1}} -> sink (N node-disjoint 2-hop
routes) and measure Eve's denial (win) rate over the final window as a function
of N, for a learned (deep_q) Eve vs a learned (deep_q) Alice. Run with the fast
qber monitor by default; --chsh uses the physical Bell monitor (slow).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sequence_game.corpus.e91_runtime_game import (
    _is_eve_win_step,
    run_e91_graph_runtime_game,
)

EDGE_M = 10.0


def disjoint_path_graph(n: int) -> tuple[dict, list[dict]]:
    """n node-disjoint 2-hop routes: a -> r_i -> b."""
    nodes = {"a": {"node_id": "a", "roles": ["alice"], "coordinates": [0.0, 0.0]},
             "b": {"node_id": "b", "roles": ["bob"], "coordinates": [2.0, 0.0]}}
    edges = []
    routes = []
    for i in range(n):
        r = f"r{i}"
        nodes[r] = {"node_id": r, "roles": ["swap_candidate"],
                    "coordinates": [1.0, float(i) - (n - 1) / 2.0]}
        edges.append({"edge_id": f"a-{r}", "u": "a", "v": r, "length_m": EDGE_M})
        edges.append({"edge_id": f"{r}-b", "u": r, "v": "b", "length_m": EDGE_M})
        routes.append({
            "route_id": f"via_{r}", "path": ["a", r, "b"],
            "edge_ids": [f"a-{r}", f"{r}-b"], "total_length": 2 * EDGE_M,
            "total_length_m": 2 * EDGE_M, "hop_count": 2, "internal_nodes": [r],
        })
    ir = {"nodes": nodes, "edges": edges,
          "metadata": {"generator": f"disjoint_width_{n}", "params": {"width": n}, "seed": 7}}
    return ir, routes


def run_one(n, steps, seed, monitor, window):
    ir, routes = disjoint_path_graph(n)
    res = run_e91_graph_runtime_game(
        ir, "a", "b", routes=routes, game_mode="attack_surface_dos",
        eve_algo="deep_q", alice_algo="deep_q", num_steps=steps, seed=seed,
        final_window_size=window, security_monitor=monitor,
        repeater_memory_pairs_per_step=8)
    steps_list = res["steps"]
    wins = [1 if _is_eve_win_step(s) else 0 for s in steps_list]
    final = wins[-window:]
    denial_reasons = {}
    for s in steps_list[-window:]:
        dr = s.get("denial_reason")
        if dr:
            denial_reasons[dr] = denial_reasons.get(dr, 0) + 1
    return {
        "n": n,
        "eve_denial_final": sum(final) / len(final),
        "keyrate_retention_final": 1 - sum(final) / len(final),
        "bound_1_over_n": 1.0 / n,
        "denial_reasons_final": denial_reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--chsh", action="store_true", help="use the physical CHSH Bell monitor (slow)")
    args = ap.parse_args()
    monitor = "chsh" if args.chsh else "qber"

    print(f"monitor={monitor} steps={args.steps} window={args.window}")
    print(f"{'N':>2}  {'eve_denial':>10}  {'retention':>9}  {'<=1/N?':>7}  {'1/N':>5}  denial_reasons")
    t0 = time.perf_counter()
    for n in args.widths:
        r = run_one(n, args.steps, args.seed + n, monitor, args.window)
        ok = "yes" if r["eve_denial_final"] <= r["bound_1_over_n"] + 1e-9 else "NO"
        print(f"{n:>2}  {r['eve_denial_final']:>10.3f}  {r['keyrate_retention_final']:>9.3f}  "
              f"{ok:>7}  {r['bound_1_over_n']:>5.2f}  {r['denial_reasons_final']}")
    print(f"total wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
