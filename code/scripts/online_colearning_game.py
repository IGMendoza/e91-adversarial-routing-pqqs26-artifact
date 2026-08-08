#!/usr/bin/env python3
"""Run Alice/Eve bandit co-learning on the active compiled histogram backend.

The default experiment loads the first assignment from the normalized
Topology Bench corpus (1--50 km links), resolves the Cui-2026 rolling purified
near-term composite separately for every route, and runs 100,000 simultaneous
partial-feedback turns. Alice learns Bell-qualified saturated service rate;
Eve learns its public complement minus declared action cost. Structural target
overlap and Eve's private instrument records are diagnostics, not rewards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning import (  # noqa: E402
    HistogramCoLearningConfig,
    HistogramPhysicsPool,
    histogram_cache_plan,
    load_topology_bench_case,
    run_histogram_colearning,
)
from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    ALICE_LEARNER_MODES,
    DEFAULT_CORPUS_DB_PATH,
    DEFAULT_REPEATER_SERVICE_PROFILE_PATH,
    EVE_LEARNER_MODES,
    resolve_eve_structured_exploration,
    resolve_structured_exploration,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--corpus-id", default=None)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--assignment-id", default=None)
    parser.add_argument("--assignment-policy", default=None)
    parser.add_argument("--assignment-index", type=int, default=0)
    parser.add_argument("--turns", type=int, default=100_000)
    parser.add_argument("--requested-pairs", type=int, default=1_200)
    parser.add_argument("--final-window", type=int, default=10_000)
    parser.add_argument("--record-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--game-mode",
        choices=[
            "attack_surface_dos",
            "attack_surface_mixed",
            "attack_surface_all",
        ],
        default="attack_surface_all",
    )
    parser.add_argument(
        "--schedule",
        choices=["constant", "anytime"],
        default="anytime",
    )
    parser.add_argument(
        "--alice-learner",
        choices=sorted(ALICE_LEARNER_MODES),
        default="flat_exp3",
        help=(
            "Alice learner. maiti_dag_ftrl and ito_dag_ftrl implement the "
            "modern aggregate-feedback DAG methods; tsallis_inf and "
            "exp3_plus_plus are adaptive independent-arm controls; "
            "linear_route_exp2 is the working finite structured baseline."
        ),
    )
    parser.add_argument(
        "--eve-learner",
        choices=sorted(EVE_LEARNER_MODES),
        default="flat_exp3",
        help=(
            "Eve independent-arm learner. Adaptive controls use the same "
            "selected-action scalar-feedback boundary as Alice."
        ),
    )
    parser.add_argument(
        "--eve-structured-exploration",
        choices=["uniform", "g_optimal"],
        default=None,
        help=(
            "Eve LinearExp2 exploration geometry. Defaults to g_optimal for "
            "linear_action_exp2 and uniform otherwise."
        ),
    )
    parser.add_argument(
        "--tsallis-estimator", choices=["iw", "rv"], default="rv"
    )
    parser.add_argument("--alice-tsallis-eta-scale", type=float, default=1.0)
    parser.add_argument("--eve-tsallis-eta-scale", type=float, default=1.0)
    parser.add_argument("--exp3pp-confidence-alpha", type=float, default=3.0)
    parser.add_argument(
        "--alice-exp3pp-exploration-beta", type=float, default=256.0
    )
    parser.add_argument(
        "--eve-exp3pp-exploration-beta", type=float, default=256.0
    )
    parser.add_argument("--alice-exp3pp-eta-scale", type=float, default=1.0)
    parser.add_argument("--eve-exp3pp-eta-scale", type=float, default=1.0)
    parser.add_argument("--maiti-failure-probability", type=float, default=0.05)
    parser.add_argument("--maiti-eta-scale", type=float, default=1.0)
    parser.add_argument(
        "--maiti-implicit-exploration-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--ito-regularizer",
        choices=["tsallis", "log_barrier"],
        default="tsallis",
    )
    parser.add_argument("--ito-eta-scale", type=float, default=1.0)
    parser.add_argument("--ito-beta", type=float, default=2.0)
    parser.add_argument("--path-ftrl-solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--path-ftrl-solver-max-iterations", type=int, default=60)
    parser.add_argument(
        "--structured-exploration",
        choices=["uniform", "g_optimal"],
        default=None,
        help=(
            "LinearExp2 exploration geometry. Defaults to g_optimal for "
            "linear_route_exp2 and uniform otherwise; no route quality enters "
            "the design."
        ),
    )
    parser.add_argument("--alice-gamma", type=float, default=0.07)
    parser.add_argument("--eve-gamma", type=float, default=0.07)
    parser.add_argument("--alice-eta-c", type=float, default=1.0)
    parser.add_argument("--eve-eta-c", type=float, default=1.0)
    parser.add_argument("--exp3-t0", type=float, default=10_000.0)
    parser.add_argument("--gamma-max", type=float, default=0.20)
    parser.add_argument(
        "--alice-physics-prior-mix",
        type=float,
        default=0.0,
        help=(
            "Explicit sensitivity-only route-capacity prior mixture. The "
            "production default is 0 so Alice starts uniformly."
        ),
    )
    parser.add_argument(
        "--repeater-service-profile",
        type=Path,
        default=DEFAULT_REPEATER_SERVICE_PROFILE_PATH,
    )
    parser.add_argument("--accepted-alive-time-s", type=float, default=None)
    parser.add_argument("--age-points-per-edge", type=int, default=8)
    parser.add_argument("--fock-cutoff", type=int, default=3)
    parser.add_argument("--evaluated-cache-size", type=int, default=256)
    parser.add_argument(
        "--no-known-detector-approximation",
        action="store_true",
        help=(
            "Fail closed instead of applying the explicitly reported 1550 nm "
            "scalar PNR operating point at Cui's 1558 nm output."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    case = load_topology_bench_case(
        args.corpus_db,
        corpus_id=args.corpus_id,
        graph_id=args.graph_id,
        assignment_id=args.assignment_id,
        assignment_policy=args.assignment_policy,
        assignment_index=args.assignment_index,
    )
    structured_exploration = resolve_structured_exploration(
        args.alice_learner,
        args.structured_exploration,
    )
    eve_structured_exploration = resolve_eve_structured_exploration(
        args.eve_learner,
        args.eve_structured_exploration,
    )
    config = HistogramCoLearningConfig(
        turns=args.turns,
        requested_pairs_per_turn=args.requested_pairs,
        final_window=args.final_window,
        record_every=args.record_every,
        seed=args.seed,
        game_mode=args.game_mode,
        alice_learner_mode=args.alice_learner,
        eve_learner_mode=args.eve_learner,
        structured_exploration_mode=structured_exploration,
        eve_structured_exploration_mode=eve_structured_exploration,
        exp3_schedule_mode=args.schedule,
        alice_exp3_gamma=args.alice_gamma,
        eve_exp3_gamma=args.eve_gamma,
        alice_exp3_eta_c=args.alice_eta_c,
        eve_exp3_eta_c=args.eve_eta_c,
        exp3_t0=args.exp3_t0,
        exp3_gamma_max=args.gamma_max,
        tsallis_inf_estimator=args.tsallis_estimator,
        alice_tsallis_inf_eta_scale=args.alice_tsallis_eta_scale,
        eve_tsallis_inf_eta_scale=args.eve_tsallis_eta_scale,
        exp3pp_confidence_alpha=args.exp3pp_confidence_alpha,
        alice_exp3pp_exploration_beta=(
            args.alice_exp3pp_exploration_beta
        ),
        eve_exp3pp_exploration_beta=args.eve_exp3pp_exploration_beta,
        alice_exp3pp_eta_scale=args.alice_exp3pp_eta_scale,
        eve_exp3pp_eta_scale=args.eve_exp3pp_eta_scale,
        maiti_failure_probability=args.maiti_failure_probability,
        maiti_eta_scale=args.maiti_eta_scale,
        maiti_implicit_exploration_scale=(
            args.maiti_implicit_exploration_scale
        ),
        ito_regularizer=args.ito_regularizer,
        ito_eta_scale=args.ito_eta_scale,
        ito_beta=args.ito_beta,
        path_ftrl_solver_tolerance=args.path_ftrl_solver_tolerance,
        path_ftrl_solver_max_iterations=args.path_ftrl_solver_max_iterations,
        alice_physics_prior_mix=args.alice_physics_prior_mix,
        repeater_service_profile_path=args.repeater_service_profile,
        accepted_alive_time_s=args.accepted_alive_time_s,
        age_points_per_edge=args.age_points_per_edge,
        allow_known_1550_to_1558_detector_approximation=(
            not args.no_known_detector_approximation
        ),
        fock_cutoff=args.fock_cutoff,
        evaluated_cache_size=args.evaluated_cache_size,
    )
    cache_plan = histogram_cache_plan(case, config)
    print(
        f"case={case.graph_id}/{case.assignment_id} "
        f"routes={len(case.routes)} turns={config.turns} "
        f"alice_learner={config.alice_learner_mode} "
        f"eve_learner={config.eve_learner_mode} "
        f"backend=compiled_histogram profile={config.repeater_service_profile_path.name}",
        flush=True,
    )
    print("resolving route-specific rolling physics...", flush=True)
    print(
        "cache preflight: "
        f"{cache_plan['global_route_action_pair_count']} route/action pairs -> "
        f"{cache_plan['canonical_unique_move_count_total']} canonical entries; "
        f"max {cache_plan['maximum_canonical_moves_on_one_route']}/route, "
        f"limit {cache_plan['configured_entries_per_route']}/route, "
        "eviction-free="
        f"{cache_plan['sufficient_for_declared_action_surface_without_eviction']}, "
        f"estimated resident cache "
        f"{cache_plan['estimated_full_action_surface_cache_mib']:.1f} MiB",
        flush=True,
    )
    physics = HistogramPhysicsPool(case, config)
    for row in physics.calibrations:
        print(
            f"  {row.route_id}: h={row.hop_count} "
            f"distance={row.total_distance_km:.3f} km "
            f"rate={row.service_rate_hz:.4f}/s "
            f"availability={row.route_availability:.6f} "
            f"CHSH={row.clean_expected_chsh_s}",
            flush=True,
        )
    summary = run_histogram_colearning(
        case,
        config,
        physics=physics,
        progress_callback=lambda message: print(f"  {message}", flush=True),
    )
    payload = summary.to_dict()
    _print_final(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}", flush=True)
    return payload


def _print_final(payload: dict[str, Any]) -> None:
    actions = payload["actions"]
    eve = np.asarray(payload["eve_strategy"], dtype=float)
    alice = np.asarray(payload["alice_strategy"], dtype=float)
    routes = payload["case"]["route_ids"]
    top_eve = sorted(range(len(actions)), key=lambda index: -eve[index])[:8]
    print(
        "FINAL "
        f"Alice utility={payload['mean_alice_utility']:.4f} "
        f"final={payload['final_alice_utility']:.4f} "
        f"Bell-qualified rate={payload['mean_bell_qualified_rate_hz']:.4f}/s "
        f"Eve utility={payload['mean_eve_public_utility']:.4f} "
        f"elapsed={payload['elapsed_seconds']:.2f}s",
        flush=True,
    )
    print(
        "  Alice strategy: "
        + ", ".join(
            f"{route}={alice[index]:.3f}"
            for index, route in enumerate(routes)
        ),
        flush=True,
    )
    print(
        "  Eve top actions: "
        + ", ".join(
            f"{actions[index]['id']}={eve[index]:.3f}"
            for index in top_eve
        ),
        flush=True,
    )
    oracle = (payload.get("empirical_game") or {}).get("oracle")
    learned = (payload.get("empirical_game") or {}).get("learned_profile")
    if oracle and learned:
        print(
            f"  post-hoc empirical oracle Alice utility="
            f"{oracle['alice_public_utility']:.4f}; "
            f"learned-equivalent exploitability="
            f"{learned['zero_sum_equivalent_exploitability']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
