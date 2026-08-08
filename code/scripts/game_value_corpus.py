"""Co-learned game value over a small structural corpus (the smoothness sweep).

For each graph we build the zero-sum payoff matrix ``P[route, attack]`` = Alice's
key rate under (route, attack), solve the equilibrium by adversarial co-learning
(Exp3 bandit self-play; RM+/LP as oracle), and report Alice's retained key rate
and the graded Eve-win verdict. Swept over a family of graphs that interpolates
from Alice-favoured (many node-disjoint routes) to Eve-favoured (a shared
bottleneck node), the retention traces a smooth, diminishing-returns curve
(retention = 1 - 1/(node-disjoint paths)) -- the "discrete L2-looking smoothness".

Two payoff sources:
  --idealized : P from route/attack incidence (0/1). Instant; the clean curve.
  (default)   : P estimated from SeQUeNCe physics trials per cell (real key rate,
                includes probabilistic CHSH/QBER partial denial).

Rerun / extend: edit ``build_corpus`` or pass --widths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sequence_game.game_theory import (  # noqa: E402
    characterize_equilibrium,
    solve_zero_sum_bandit_selfplay,
    solve_zero_sum_lp,
)

EDGE_M = 10.0


# --------------------------------------------------------------------------- #
# Corpus of small graphs spanning the structural spectrum.
# --------------------------------------------------------------------------- #

def disjoint_graph(n: int):
    """n node-disjoint 2-hop routes: a -> r_i -> b."""
    nodes = {"a": {"node_id": "a", "roles": ["alice"], "coordinates": [0.0, 0.0]},
             "b": {"node_id": "b", "roles": ["bob"], "coordinates": [2.0, 0.0]}}
    edges, routes = [], []
    for i in range(n):
        r = f"r{i}"
        nodes[r] = {"node_id": r, "roles": ["swap_candidate"],
                    "coordinates": [1.0, float(i)]}
        edges += [{"edge_id": f"a-{r}", "u": "a", "v": r, "length_m": EDGE_M},
                  {"edge_id": f"{r}-b", "u": r, "v": "b", "length_m": EDGE_M}]
        routes.append({"route_id": f"via_{r}", "path": ["a", r, "b"],
                       "edge_ids": [f"a-{r}", f"{r}-b"], "total_length": 2 * EDGE_M,
                       "total_length_m": 2 * EDGE_M, "hop_count": 2, "internal_nodes": [r]})
    ir = {"nodes": nodes, "edges": edges,
          "metadata": {"generator": f"disjoint_{n}", "params": {"width": n}, "seed": 7}}
    return ir, routes, n  # n node-disjoint paths


def bottleneck_graph(n: int):
    """n routes that all pass through one shared node s: a -> s -> r_i -> b."""
    nodes = {"a": {"node_id": "a", "roles": ["alice"], "coordinates": [0.0, 0.0]},
             "s": {"node_id": "s", "roles": ["swap_candidate"], "coordinates": [1.0, 0.0]},
             "b": {"node_id": "b", "roles": ["bob"], "coordinates": [3.0, 0.0]}}
    edges = [{"edge_id": "a-s", "u": "a", "v": "s", "length_m": EDGE_M}]
    routes = []
    for i in range(n):
        r = f"r{i}"
        nodes[r] = {"node_id": r, "roles": ["swap_candidate"], "coordinates": [2.0, float(i)]}
        edges += [{"edge_id": f"s-{r}", "u": "s", "v": r, "length_m": EDGE_M},
                  {"edge_id": f"{r}-b", "u": r, "v": "b", "length_m": EDGE_M}]
        routes.append({"route_id": f"via_{r}", "path": ["a", "s", r, "b"],
                       "edge_ids": [f"a-s", f"s-{r}", f"{r}-b"], "total_length": 3 * EDGE_M,
                       "total_length_m": 3 * EDGE_M, "hop_count": 3, "internal_nodes": ["s", r]})
    ir = {"nodes": nodes, "edges": edges,
          "metadata": {"generator": f"bottleneck_{n}", "params": {"width": n}, "seed": 7}}
    return ir, routes, 1  # all share s -> a single node-disjoint path


def build_corpus(widths):
    corpus = [("disjoint", w, disjoint_graph(w)) for w in widths]
    corpus.append(("bottleneck", 3, bottleneck_graph(3)))
    return corpus


# --------------------------------------------------------------------------- #
# Payoff matrices.
# --------------------------------------------------------------------------- #

def idealized_payoff(ir, routes):
    """P[route, attack] = 0 iff the attacked node/edge lies on the route, else 1.

    Attacks: no_attack + one swap_denial per internal node (the canonical
    interdiction set). Full denial on collision; no partial effects.
    """
    internal_nodes = sorted({n for r in routes for n in r["internal_nodes"]})
    actions = ["no_attack"] + [f"swap_denial:{v}" for v in internal_nodes]
    P = np.ones((len(routes), len(actions)))
    for i, r in enumerate(routes):
        on_route = set(r["internal_nodes"])
        for j, act in enumerate(actions):
            if act.startswith("swap_denial:") and act.split(":")[1] in on_route:
                P[i, j] = 0.0
    return P, [r["route_id"] for r in routes], actions


def physics_payoff(ir, routes, alice, bob, *, trials_per_cell, security_monitor, seed):
    """Estimate P[route, attack] = P(key accepted) on ONE consistent engine.

    Every cell runs the repeater trial (so baseline and denied cells share the
    same physics -- no bare-fiber/repeater engine mismatch). The shared attack
    set is no_attack + one swap_denial per interior node; swap_denial:v is active
    only when v lies on the route (else it is a no-op = baseline). A node shared
    by every route therefore zeros a whole column -> the single point of failure.
    """
    from sequence_game.corpus.e91_runtime_game import (
        _default_models, _multiplexed_memory_efficiency)
    from sequence_game.eve.repeater_attacks import RepeaterAttackSpec
    from sequence_game.protocol.postprocessing import SiftingConfig
    from sequence_game.protocol.repeater_trial import (
        RepeaterE91RunConfig, run_fixed_repeater_chsh_trial,
        run_fixed_repeater_e91_trial)
    from sequence_game.sequence_build.repeater_e91_builder import FixedRepeaterPath

    _, _, fiber, memory = _default_models()
    interior = sorted({n for r in routes for n in r["internal_nodes"]})
    actions = ["no_attack"] + [f"swap_denial:{v}" for v in interior]
    runner = (run_fixed_repeater_chsh_trial if security_monitor == "chsh"
              else run_fixed_repeater_e91_trial)
    pairs = 350 if security_monitor == "chsh" else 8
    min_sifted = 4 if security_monitor == "chsh" else 1
    cfg = RepeaterE91RunConfig(
        memory_pairs=pairs,
        sifting=SiftingConfig(qber_threshold=0.15, min_sifted_samples=min_sifted),
        request_fidelity=0.01, start_time_ps=1_000_000, end_time_ps=80_000_000,
        stop_time_ps=100_000_000, swapping_success_prob=1.0, swapping_degradation=1.0,
        memory_efficiency_override=_multiplexed_memory_efficiency(memory))
    P = np.zeros((len(routes), len(actions)))
    for i, route in enumerate(routes):
        path = FixedRepeaterPath(
            nodes=tuple(route["path"]),
            edge_lengths_m=tuple([EDGE_M] * (len(route["path"]) - 1)))
        on_route = set(route["internal_nodes"])
        for j, action in enumerate(actions):
            if action == "no_attack" or action.split(":")[1] not in on_route:
                spec = RepeaterAttackSpec()               # no-op / baseline
            else:
                spec = RepeaterAttackSpec(kind="swap_denial",
                                          target_node=action.split(":")[1])
            acc = 0
            for k in range(trials_per_cell):
                tr, _ = runner(path, memory_model=memory, fiber_model=fiber,
                               run_config=cfg, trial_id=f"cell_{i}_{j}_{k}",
                               seed=seed + 1000 * i + 10 * j + k, attack=spec)
                if tr.accepted:
                    acc += 1
            P[i, j] = acc / trials_per_cell
    return P, [r["route_id"] for r in routes], actions


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--idealized", action="store_true", help="incidence 0/1 payoff (instant)")
    ap.add_argument("--trials-per-cell", type=int, default=2)
    ap.add_argument("--monitor", choices=["qber", "chsh"], default="qber")
    ap.add_argument("--iterations", type=int, default=40000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    t0 = time.perf_counter()
    print(f"{'graph':<16}{'disjoint':>9}{'retention':>10}{'value':>8}{'eve_win':>10}{'exploit':>9}")
    for family, w, (ir, routes, disjoint_paths) in build_corpus(args.widths):
        if args.idealized:
            P, rids, aids = idealized_payoff(ir, routes)
        else:
            P, rids, aids = physics_payoff(
                ir, routes, "a", "b", trials_per_cell=args.trials_per_cell,
                security_monitor=args.monitor, seed=13)
        # Baseline = Alice's best UN-ATTACKED route rate (no_attack is column 0).
        # Retention then isolates the interdiction effect from the network's
        # natural (physics) delivery loss.
        baseline = float(P[:, 0].max()) if P[:, 0].max() > 0 else 1.0
        char = characterize_equilibrium(P, baseline_rate=baseline)
        sp = solve_zero_sum_bandit_selfplay(P, iterations=args.iterations, seed=5)
        lp_value, _, _ = solve_zero_sum_lp(P)
        label = f"{family}_{w}"
        rows.append({"graph": label, "family": family, "width": w,
                     "node_disjoint_paths": disjoint_paths,
                     "baseline_rate": baseline, "value": char.value,
                     "keyrate_retention": char.keyrate_retention,
                     "eve_win": char.eve_win, "exploitability": char.exploitability,
                     "bandit_value": sp.value, "lp_value": lp_value,
                     "expected_retention_1_minus_1_over_k": 1 - 1 / disjoint_paths})
        print(f"{label:<16}{disjoint_paths:>9}{char.keyrate_retention:>10.3f}"
              f"{char.value:>8.3f}{char.eve_win:>10}{char.exploitability:>9.3f}")
    print(f"total wall={time.perf_counter()-t0:.1f}s  "
          f"(payoff={'idealized' if args.idealized else args.monitor+' physics'})")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
