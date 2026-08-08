#!/usr/bin/env python3
"""Screen Maiti schedule scales on live histogram co-learning feedback."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    ALICE_LEARNER_MAITI_DAG_FTRL,
    DEFAULT_CORPUS_DB_PATH,
    HistogramCoLearningConfig,
)
from sequence_game.experiments.histogram_colearning.corpus import (  # noqa: E402
    load_topology_bench_case,
)
from sequence_game.experiments.histogram_colearning.learning import (  # noqa: E402
    run_histogram_colearning,
)
from sequence_game.experiments.histogram_colearning.paper import (  # noqa: E402
    DEFAULT_PAPER_CASES,
)
from sequence_game.experiments.histogram_colearning.physics import (  # noqa: E402
    HistogramPhysicsPool,
    histogram_cache_plan,
)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(row.strip()) for row in value.split(",") if row.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(row.strip()) for row in value.split(",") if row.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    long_case = next(
        case for case in DEFAULT_PAPER_CASES if case.slug == "long_composable"
    )
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--graph-id", default=long_case.graph_id)
    parser.add_argument("--assignment-id", default=long_case.assignment_id)
    parser.add_argument("--turns", type=int, default=1_000)
    parser.add_argument("--deployment-turns", type=int, default=100_000)
    parser.add_argument("--eta-scales", default="1,4")
    parser.add_argument("--ix-scales", default="1")
    parser.add_argument("--loss-baselines", default="1")
    parser.add_argument("--seeds", default="51")
    parser.add_argument("--record-every", type=int, default=1_000)
    parser.add_argument("--final-window", type=int, default=10_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/maiti_histogram_sensitivity/sensitivity.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.deployment_turns < args.turns:
        raise ValueError("deployment-turns must be >= turns")
    eta_scales = _floats(args.eta_scales)
    ix_scales = _floats(args.ix_scales)
    loss_baselines = _floats(args.loss_baselines)
    seeds = _ints(args.seeds)
    if not eta_scales or not ix_scales or not loss_baselines or not seeds:
        raise ValueError(
            "eta-scales, ix-scales, loss-baselines, and seeds must be non-empty"
        )
    case = load_topology_bench_case(
        args.corpus_db,
        graph_id=args.graph_id,
        assignment_id=args.assignment_id,
    )
    base = HistogramCoLearningConfig(
        turns=args.turns,
        final_window=min(args.final_window, args.turns),
        record_every=args.record_every,
        posthoc_samples_per_cell=0,
        alice_learner_mode=ALICE_LEARNER_MAITI_DAG_FTRL,
        maiti_schedule_horizon=args.deployment_turns,
    )
    plan = histogram_cache_plan(case, base)
    print(
        f"cache preflight: {plan['global_route_action_pair_count']} nominal "
        f"cells -> {plan['canonical_unique_move_count_total']} canonical; "
        f"estimated {plan['estimated_full_action_surface_cache_mib']:.1f} MiB",
        flush=True,
    )
    physics = HistogramPhysicsPool(case, base)
    started = time.perf_counter()
    runs = []
    for eta_scale in eta_scales:
        for ix_scale in ix_scales:
            for loss_baseline in loss_baselines:
                for seed in seeds:
                    config = replace(
                        base,
                        seed=seed,
                        maiti_eta_scale=eta_scale,
                        maiti_implicit_exploration_scale=ix_scale,
                        maiti_reward_loss_baseline=loss_baseline,
                    )
                    label = (
                        f"maiti_eta{eta_scale:g}_ix{ix_scale:g}_"
                        f"baseline{loss_baseline:g}"
                    )
                    print(
                        f"{label}/seed-{seed}: {args.turns:,} prefix turns under "
                        f"{args.deployment_turns:,}-turn schedule",
                        flush=True,
                    )
                    summary = run_histogram_colearning(
                        case,
                        config,
                        physics=physics,
                        progress_callback=(
                            lambda message, active=label, active_seed=seed: print(
                                f"{active}/seed-{active_seed}: {message}",
                                flush=True,
                            )
                        ),
                    ).to_dict()
                    current = summary["alice_current_strategy"]
                    runs.append({
                        "label": label,
                        "eta_scale": eta_scale,
                        "ix_scale": ix_scale,
                        "reward_loss_baseline": loss_baseline,
                        "seed": seed,
                        "mean_alice_utility": summary["mean_alice_utility"],
                        "final_alice_utility": summary["final_alice_utility"],
                        "maximum_current_route_probability": max(current),
                        "false_signal_policy_total_variation": summary[
                            "alice_false_signal_telemetry"
                        ]["policy_attribution"]["policy_total_variation"],
                        "summary": summary,
                    })
    result = {
        "schema": "histogram_maiti_schedule_sensitivity.v1",
        "scientific_scope": (
            "global schedule sensitivity on live histogram co-learning; "
            "not a route-quality prior and not held-out production evidence"
        ),
        "case": {
            "graph_id": case.graph_id,
            "assignment_id": case.assignment_id,
            "route_count": len(case.routes),
        },
        "turns": args.turns,
        "deployment_turns": args.deployment_turns,
        "eta_scales": list(eta_scales),
        "ix_scales": list(ix_scales),
        "reward_loss_baselines": list(loss_baselines),
        "seeds": list(seeds),
        "cache_plan": plan,
        "selection_performed": False,
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(f"wrote {args.out}")
    for row in runs:
        print(
            f"{row['label']}/seed-{row['seed']}: mean="
            f"{row['mean_alice_utility']:.4f}, final="
            f"{row['final_alice_utility']:.4f}, max-current-policy="
            f"{row['maximum_current_route_probability']:.3f}"
        )
    return result


if __name__ == "__main__":
    main()
