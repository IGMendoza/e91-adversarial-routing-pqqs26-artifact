#!/usr/bin/env python3
"""Produce the resumable 100k histogram co-learning paper artifact bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning import (  # noqa: E402
    HistogramCoLearningConfig,
)
from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    ALICE_LEARNER_MODES,
    DEFAULT_CORPUS_DB_PATH,
    DEFAULT_REPEATER_SERVICE_PROFILE_PATH,
    EVE_LEARNER_MODES,
    resolve_eve_structured_exploration,
    resolve_structured_exploration,
)
from sequence_game.experiments.histogram_colearning.paper import (  # noqa: E402
    DEFAULT_PAPER_CASES,
    DEFAULT_PAPER_SEEDS,
    run_histogram_paper_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/histogram_colearning_paper_100k"),
    )
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--turns", type=int, default=100_000)
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_PAPER_SEEDS)),
        help="Comma-separated independent learner seeds (default: 42,43,44).",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.slug for case in DEFAULT_PAPER_CASES],
        help="Run only this fixed paper case; repeat to select several.",
    )
    parser.add_argument("--requested-pairs", type=int, default=1_200)
    parser.add_argument(
        "--alice-learner",
        choices=sorted(ALICE_LEARNER_MODES),
        default="flat_exp3",
        help=(
            "Alice learner. flat_exp3 is the frozen comparator; "
            "linear_route_exp2 is the working finite structured baseline; "
            "tsallis_inf and exp3_plus_plus are adaptive independent-arm "
            "controls; maiti_dag_ftrl and ito_dag_ftrl are the modern "
            "structured targets; "
            "dag_path_comband and dag_path_exp2 are diagnostic baselines. "
            "DAG modes require an exact DAG route closure."
        ),
    )
    parser.add_argument(
        "--eve-learner",
        choices=sorted(EVE_LEARNER_MODES),
        default="flat_exp3",
        help=(
            "Eve learner using selected-action scalar feedback. "
            "linear_action_exp2 shares information through the audited "
            "family/target/topology action span."
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
            "LinearExp2 exploration geometry. Defaults to g_optimal for the "
            "finite structured learner and uniform otherwise. g_optimal uses "
            "topology identifiability, not route quality."
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
    parser.add_argument("--final-window", type=int, default=10_000)
    parser.add_argument("--record-every", type=int, default=1_000)
    parser.add_argument(
        "--posthoc-samples-per-cell",
        type=int,
        default=256,
        help=(
            "Independent stratified payoff-audit draws per route/action cell; "
            "these never update either learner."
        ),
    )
    parser.add_argument("--age-points-per-edge", type=int, default=8)
    parser.add_argument("--fock-cutoff", type=int, default=3)
    parser.add_argument("--evaluated-cache-size", type=int, default=256)
    parser.add_argument(
        "--repeater-service-profile",
        type=Path,
        default=DEFAULT_REPEATER_SERVICE_PROFILE_PATH,
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Re-run and atomically replace matching run records instead of resuming.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    selected = set(args.case or [])
    cases = tuple(
        case for case in DEFAULT_PAPER_CASES
        if not selected or case.slug in selected
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
        posthoc_samples_per_cell=args.posthoc_samples_per_cell,
        alice_learner_mode=args.alice_learner,
        eve_learner_mode=args.eve_learner,
        structured_exploration_mode=structured_exploration,
        eve_structured_exploration_mode=eve_structured_exploration,
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
        seed=seeds[0] if seeds else 42,
        age_points_per_edge=args.age_points_per_edge,
        fock_cutoff=args.fock_cutoff,
        evaluated_cache_size=args.evaluated_cache_size,
        repeater_service_profile_path=args.repeater_service_profile,
    )
    print(
        f"paper bundle: backend=compiled_histogram_transcript "
        f"alice_learner={config.alice_learner_mode} "
        f"eve_learner={config.eve_learner_mode} "
        f"cases={len(cases)} seeds={seeds} turns/replicate={config.turns:,}",
        flush=True,
    )
    result = run_histogram_paper_bundle(
        args.out_dir,
        base_config=config,
        seeds=seeds,
        corpus_db_path=args.corpus_db,
        cases=cases,
        resume=not args.fresh,
        progress_callback=lambda message: print(f"  {message}", flush=True),
    )
    summary = result["run_summary"]
    print(
        f"complete: {summary['run_count']} runs, "
        f"{summary['total_bandit_turns']:,} total histogram turns; "
        f"artifacts={result['output_dir']}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
