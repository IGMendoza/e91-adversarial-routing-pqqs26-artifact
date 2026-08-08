"""Tiny experiment CLI. All defaults are toy-scope and finish in seconds.

Usage examples:

    python -m sequence_game.experiments.cli gen-topology \
        --n 6 --p 0.6 --seed 1 --edge-length 1000 --out topo.json

    python -m sequence_game.experiments.cli train-eve \
        --config configs/experiments/toy_eve_training.json --out runs/demo

    python -m sequence_game.experiments.cli aggregate --runs runs --out agg.json

    python -m sequence_game.experiments.cli xai \
        --training-summary results/repeater_runtime/q_eve_two_route_training_summary.json \
        --outcome-table results/repeater_runtime/two_route_runtime_outcome_table.json \
        --out results/xai/benchmark.json

    python -m sequence_game.experiments.cli xai --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..analysis.metrics import write_json
from ..game import GameConfig, evaluate_strategic_game
from ..physical import load_models_from_dir
from ..protocol import E91RunConfig, SiftingConfig
from ..routing.baselines import ShortestHopPolicy, ShortestLengthPolicy
from ..topology import TopologyIR
from ..topology.er_generator import ERTopologyConfig, generate_er_topology
from ..topology.roles import RoleAssignmentConfig, assign_roles
from .runner import load_config, run_eve_experiment

_DEFAULT_TRAINING_SUMMARY = (
    "results/repeater_runtime/q_eve_two_route_training_summary.json"
)
_DEFAULT_OUTCOME_TABLE = (
    "results/repeater_runtime/two_route_runtime_outcome_table.json"
)
_DEFAULT_XAI_MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "qwen3:8b",
    "qwen3:14b",
    "phi4:14b",
    "gemma4:26b",
]

_POLICIES = {"shortest_hop": ShortestHopPolicy, "shortest_length": ShortestLengthPolicy}


def cmd_gen_topology(args: argparse.Namespace) -> int:
    assign = bool(args.assign_roles or args.alice or args.bob)
    # Random Alice/Bob selection needs a connected graph; force it when assigning
    # roles without explicit endpoints.
    require_connected = args.require_connected or (assign and not (args.alice and args.bob))
    config = ERTopologyConfig(
        n=args.n, p=args.p, fixed_edge_length_m=args.edge_length,
        require_connected=require_connected)
    ir = generate_er_topology(config, seed=args.seed)
    if assign:
        ir = assign_roles(ir, RoleAssignmentConfig(
            alice=args.alice, bob=args.bob, eve_eligible_edges=args.eve_edges), seed=args.seed)
    Path(args.out).write_text(ir.to_json(), encoding="utf-8")
    roles_note = ""
    if assign:
        a = (ir.nodes_with_role("alice") or ["?"])[0]
        b = (ir.nodes_with_role("bob") or ["?"])[0]
        roles_note = f", Alice={a}, Bob={b}, eve_edges={args.eve_edges}"
    print(f"wrote {args.out}: {len(ir.nodes)} nodes, {len(ir.edges)} edges{roles_note}")
    return 0


def cmd_train_eve(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_eve_experiment(config, Path(args.out))
    summary = result.summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"artifacts written to {args.out} (scope: toy)")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs)
    summaries = {}
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summaries[summary_path.parent.name] = json.loads(
            summary_path.read_text(encoding="utf-8"))
    if not summaries:
        print(f"no run summaries found under {runs_dir}", file=sys.stderr)
        return 1
    write_json(summaries, Path(args.out))
    print(f"aggregated {len(summaries)} runs into {args.out}")
    return 0


def _model_by_kind(models: dict, kind: str):
    matches = [m for m in models.values() if m.device_kind == kind]
    if not matches:
        raise SystemExit(f"no resolved {kind!r} model found in physical configs")
    return matches[0]


def cmd_game(args: argparse.Namespace) -> int:
    """Run the strategic routing-vs-attack game on the E91/BBM92 control."""
    ir = TopologyIR.from_json(Path(args.topology).read_text(encoding="utf-8"))
    alice = args.alice or (ir.nodes_with_role("alice") or [None])[0]
    bob = args.bob or (ir.nodes_with_role("bob") or [None])[0]
    if not alice or not bob:
        raise SystemExit("specify --alice and --bob (or assign roles in the topology)")

    models = load_models_from_dir(Path(args.configs), require_resolved=True).models
    config = GameConfig(
        alice=alice, bob=bob,
        source_model=_model_by_kind(models, "source"),
        detector_model=_model_by_kind(models, "detector"),
        fiber_model=_model_by_kind(models, "fiber_channel"),
        run_config=E91RunConfig(
            num_periods=args.periods,
            sifting=SiftingConfig(qber_threshold=args.qber_threshold,
                                  min_sifted_samples=args.min_sifted)),
        num_trials=args.trials, base_seed=args.seed)
    result = evaluate_strategic_game(ir, _POLICIES[args.policy](), config)

    print(f"route: {' -> '.join(result.route)}  (Alice={alice}, Bob={bob}, "
          f"policy={args.policy}, trials={args.trials}, periods={args.periods})")
    header = f"{'strategy':30} {'accept':>6} {'denies':>6} {'qber':>6} {'I(bits)':>8} {'frac':>5}"
    print(header)
    for o in result.outcomes:
        q = f"{o.mean_qber:.3f}" if o.mean_qber is not None else "  -  "
        print(f"{o.strategy_id:30} {o.accept_rate:>6.2f} {str(o.prevents_key):>6} "
              f"{q:>6} {o.mean_information_bits:>8.3f} {o.mean_fraction_correct:>5.2f}")
    best = result.best_information_strategy()
    if best is not None:
        print(f"\nbest info-gain strategy: {best.strategy_id} "
              f"({best.mean_information_bits:.3f} bits/sifted bit, denies_key={best.prevents_key})")
    print(f"denial strategies: {[o.strategy_id for o in result.denial_strategies()]}")
    print("note: intercept-resend leaks information but drives QBER up so the key "
          "aborts (detected); this is a toy/simulator result, not a security proof")
    if args.out:
        write_json(result.to_dict(), Path(args.out))
        print(f"wrote {args.out}")
    return 0


def cmd_xai(args: argparse.Namespace) -> int:
    """DT surrogate + LLM faithfulness benchmark on a trained Q-table."""
    import numpy as np

    from ..xai.surrogate import (
        RouteFeatures,
        build_dataset,
        encode_state,
        extract_decision_path,
        fit_surrogate_dt,
        format_path_text,
    )
    from ..xai.prompt_builder import CounterfactualRow, PromptContext
    from ..xai.benchmark import run_benchmark

    training_path = Path(args.training_summary)
    outcome_path = Path(args.outcome_table)
    if not training_path.exists():
        print(f"training summary not found: {training_path}", file=sys.stderr)
        return 1
    if not outcome_path.exists():
        print(f"outcome table not found: {outcome_path}", file=sys.stderr)
        return 1

    training_data = json.loads(training_path.read_text(encoding="utf-8"))
    outcome_data = json.loads(outcome_path.read_text(encoding="utf-8"))
    entries = training_data["q_table_entries"]["entries"]
    action_names = training_data["action_space"]
    outcome_rows = outcome_data["rows"]

    entries = [e for e in entries if e["state"][0] != "start"]
    r0 = RouteFeatures(bottleneck_score=0.5, hops=1, length_m=10_000.0, overlap_with_other=0)
    r1 = RouteFeatures(bottleneck_score=0.5, hops=1, length_m=10_000.0, overlap_with_other=0)

    dataset = build_dataset(entries, action_names, r0, r1)
    dt = fit_surrogate_dt(dataset, max_depth=args.dt_depth)
    fidelity = float(np.mean(dt.predict(dataset.X) == dataset.y))
    print(f"DT surrogate: fidelity={fidelity:.2%}  depth={args.dt_depth}  samples={len(entries)}")

    prev_action, prev_outcome = args.query_state
    query_X = np.array([encode_state(prev_action, prev_outcome) + [
        r0.bottleneck_score, r1.bottleneck_score, float(r0.hops), float(r1.hops)
    ]])
    path = extract_decision_path(dt, dataset, query_X)
    print(f"Decision path for ({prev_action!r}, {prev_outcome!r}):")
    print(format_path_text(path))

    on_route = [r for r in outcome_rows if r["selected_route"] == "route_r0"]
    chosen = next(
        (r for r in on_route if r["attack_action"] == path.predicted_action), None
    )
    if chosen is None:
        print(f"predicted action {path.predicted_action!r} not in outcome table", file=sys.stderr)
        return 1

    def _row(r: dict, is_chosen: bool) -> CounterfactualRow:
        return CounterfactualRow(
            action=r["attack_action"], route=r["selected_route"],
            outcome=r["public_outcome"], fidelity=r.get("fidelity"),
            qber=r.get("qber"), quality_metric=r.get("quality_metric"),
            is_chosen=is_chosen,
        )

    ctx = PromptContext(
        dt_path_text=format_path_text(path),
        predicted_action=path.predicted_action,
        chosen_row=_row(chosen, True),
        alternative_rows=[_row(r, False) for r in on_route if r["attack_action"] != path.predicted_action],
        r0_bottleneck=r0.bottleneck_score,
        r1_bottleneck=r1.bottleneck_score,
        r0_hops=r0.hops,
        r1_hops=r1.hops,
        dt_fidelity=path.fidelity,
    )

    if args.dry_run:
        from ..xai.prompt_builder import build_prompt
        print("\n--- Prompt (dry run) ---")
        print(build_prompt(ctx))
        return 0

    models = args.models or _DEFAULT_XAI_MODELS
    out_path = Path(args.out) if args.out else None
    print(f"\nBenchmarking {len(models)} model(s)...")
    run_benchmark(ctx, models=models, temperature=args.temperature, output_path=out_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sequence_game.experiments.cli",
        description="Toy quantum-network game experiments (no physical claims)")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-topology", help="generate an ER topology JSON")
    gen.add_argument("--n", type=int, required=True)
    gen.add_argument("--p", type=float, required=True)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--edge-length", type=float, required=True,
                     help="fixed edge length in meters (toy)")
    gen.add_argument("--require-connected", action="store_true")
    gen.add_argument("--assign-roles", action="store_true",
                     help="assign Alice/Bob (random via --seed) and eve-eligible edges")
    gen.add_argument("--alice", help="explicit Alice/source node id")
    gen.add_argument("--bob", help="explicit Bob/sink node id")
    gen.add_argument("--eve-edges", choices=["all", "none"], default="all",
                     help="which edges Eve may attack (default all fibers)")
    gen.add_argument("--out", required=True)
    gen.set_defaults(func=cmd_gen_topology)

    train = sub.add_parser("train-eve", help="run a toy Eve Q-learning experiment")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.set_defaults(func=cmd_train_eve)

    agg = sub.add_parser("aggregate", help="merge run summaries into one JSON")
    agg.add_argument("--runs", required=True)
    agg.add_argument("--out", required=True)
    agg.set_defaults(func=cmd_aggregate)

    game = sub.add_parser("game",
                          help="run the strategic E91/BBM92 routing-vs-attack game")
    game.add_argument("--topology", required=True, help="topology JSON (from gen-topology)")
    game.add_argument("--alice", help="Alice/source node (default: topology 'alice' role)")
    game.add_argument("--bob", help="Bob/sink node (default: topology 'bob' role)")
    game.add_argument("--configs", default="configs/physical",
                      help="directory of resolved physical-model JSONs")
    game.add_argument("--policy", choices=sorted(_POLICIES), default="shortest_hop")
    game.add_argument("--periods", type=int, default=4000,
                      help="emission periods per trial (toy default keeps it quick)")
    game.add_argument("--trials", type=int, default=2)
    game.add_argument("--seed", type=int, default=0)
    game.add_argument("--qber-threshold", type=float, default=0.15)
    game.add_argument("--min-sifted", type=int, default=20)
    game.add_argument("--out", help="optional JSON payoff-table output path")
    game.set_defaults(func=cmd_game)

    xai = sub.add_parser("xai", help="DT surrogate + LLM faithfulness benchmark")
    xai.add_argument("--training-summary", default=_DEFAULT_TRAINING_SUMMARY,
                     help="path to q_eve training summary JSON")
    xai.add_argument("--outcome-table", default=_DEFAULT_OUTCOME_TABLE,
                     help="path to runtime outcome table JSON")
    xai.add_argument("--models", nargs="+", default=None,
                     help="Ollama model tags to benchmark (default: all configured)")
    xai.add_argument("--out", default="results/xai/benchmark.json",
                     help="output path for benchmark results JSON")
    xai.add_argument("--dry-run", action="store_true",
                     help="print DT path and prompt only; skip LLM queries")
    xai.add_argument("--dt-depth", type=int, default=4,
                     help="max decision tree depth (default 4)")
    xai.add_argument("--temperature", type=float, default=0.1,
                     help="LLM sampling temperature (default 0.1)")
    xai.add_argument("--query-state", nargs=2, default=["no_attack", "accepted"],
                     metavar=("PREV_ACTION", "PREV_OUTCOME"),
                     help="Q-table state to explain (default: no_attack accepted)")
    xai.set_defaults(func=cmd_xai)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
