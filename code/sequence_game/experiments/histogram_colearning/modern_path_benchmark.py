"""Matched histogram-backend benchmark for modern path learners.

All learners see the same declared route set, Eve action surface, public scalar
feedback, and compiled physics pool.  Calibrated hyperparameters are read from
the leakage-free synthetic-loss artifact rather than selected from benchmark
outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import (
    ALICE_LEARNER_FLAT_EXP3,
    ALICE_LEARNER_ITO_DAG_FTRL,
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_MAITI_DAG_FTRL,
    STRUCTURED_EXPLORATION_G_OPTIMAL,
    HistogramCoLearningConfig,
)
from .corpus import HistogramGameCase
from .learning import run_histogram_colearning
from .physics import HistogramPhysicsPool, histogram_cache_plan


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ModernPathBenchmarkSpec:
    label: str
    config_overrides: Mapping[str, Any]
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "config_overrides": dict(self.config_overrides),
            "role": self.role,
        }


def benchmark_specs_from_calibration(
    calibration: Mapping[str, Any],
) -> tuple[ModernPathBenchmarkSpec, ...]:
    """Build matched baseline/modern specs from a v2 calibration artifact."""

    if calibration.get("schema") != "histogram_modern_path_calibration.v2":
        raise ValueError("modern benchmark requires calibration schema v2")
    selected = calibration.get("selected_by_family")
    required = {"maiti", "ito_tsallis", "ito_log_barrier"}
    if not isinstance(selected, Mapping) or set(selected) != required:
        raise ValueError("calibration must select all three modern families")

    def parameters(family: str) -> Mapping[str, Any]:
        row = selected[family]
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid selected calibration row for {family}")
        candidate = row.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError(f"missing selected candidate for {family}")
        values = candidate.get("parameters")
        if not isinstance(values, Mapping):
            raise ValueError(f"missing selected parameters for {family}")
        return values

    maiti = parameters("maiti")
    tsallis = parameters("ito_tsallis")
    log_barrier = parameters("ito_log_barrier")
    return (
        ModernPathBenchmarkSpec(
            "flat_exp3",
            {"alice_learner_mode": ALICE_LEARNER_FLAT_EXP3},
            "frozen finite-action compatibility comparator",
        ),
        ModernPathBenchmarkSpec(
            "linear_exp2_g_optimal",
            {
                "alice_learner_mode": ALICE_LEARNER_LINEAR_ROUTE_EXP2,
                "structured_exploration_mode": STRUCTURED_EXPLORATION_G_OPTIMAL,
            },
            "working finite structured baseline",
        ),
        ModernPathBenchmarkSpec(
            "maiti_dag_ftrl",
            {
                "alice_learner_mode": ALICE_LEARNER_MAITI_DAG_FTRL,
                "maiti_eta_scale": float(maiti["eta_scale"]),
                "maiti_implicit_exploration_scale": float(
                    maiti["implicit_exploration_scale"]
                ),
                "maiti_failure_probability": float(
                    maiti["failure_probability"]
                ),
                "maiti_reward_loss_baseline": float(
                    maiti.get("reward_loss_baseline", 1.0)
                ),
            },
            "modern adaptive-adversary DAG path learner",
        ),
        ModernPathBenchmarkSpec(
            "ito_dag_ftrl_tsallis",
            {
                "alice_learner_mode": ALICE_LEARNER_ITO_DAG_FTRL,
                "ito_regularizer": str(tsallis["regularizer"]),
                "ito_eta_scale": float(tsallis["eta_scale"]),
                "ito_beta": float(tsallis["beta"]),
            },
            "modern best-of-both-worlds DAG path learner (Tsallis)",
        ),
        ModernPathBenchmarkSpec(
            "ito_dag_ftrl_log_barrier",
            {
                "alice_learner_mode": ALICE_LEARNER_ITO_DAG_FTRL,
                "ito_regularizer": str(log_barrier["regularizer"]),
                "ito_eta_scale": float(log_barrier["eta_scale"]),
                "ito_beta": float(log_barrier["beta"]),
            },
            "modern best-of-both-worlds DAG path learner (log barrier)",
        ),
    )


def run_modern_path_benchmark(
    case: HistogramGameCase,
    base_config: HistogramCoLearningConfig,
    calibration: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (42,),
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run matched learners while reusing one compiled histogram physics pool."""

    normalized_seeds = tuple(map(int, seeds))
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("benchmark seeds must be non-empty and unique")
    specs = benchmark_specs_from_calibration(calibration)
    cache_plan = histogram_cache_plan(case, base_config)
    if not cache_plan["sufficient_for_declared_action_surface_without_eviction"]:
        raise MemoryError(
            "configured evaluated histogram cache cannot hold the declared "
            "canonical action surface without eviction"
        )
    started = time.perf_counter()
    _notify(
        progress_callback,
        f"compiling shared physics pool: {cache_plan['route_count']} routes, "
        f"{cache_plan['declared_action_count']} actions, "
        f"{cache_plan['canonical_unique_move_count_total']} canonical entries, "
        f"estimated {cache_plan['estimated_full_action_surface_cache_mib']:.1f} MiB",
    )
    physics = HistogramPhysicsPool(case, base_config)
    runs: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs, start=1):
        for seed in normalized_seeds:
            config = replace(base_config, seed=seed, **spec.config_overrides)
            _notify(
                progress_callback,
                f"learner {spec_index}/{len(specs)} {spec.label}, seed {seed}: "
                f"{config.turns:,} turns",
            )
            summary = run_histogram_colearning(
                case,
                config,
                physics=physics,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda message, label=spec.label, active_seed=seed: _notify(
                        progress_callback,
                        f"{label}/seed-{active_seed}: {message}",
                    )
                ),
            )
            payload = summary.to_dict()
            runs.append({
                "benchmark_spec": spec.to_dict(),
                "seed": seed,
                "summary": payload,
                "comparison_row": _comparison_row(spec, seed, payload),
            })
    return {
        "schema": "histogram_modern_path_benchmark.v1",
        "backend": "compiled_histogram_transcript",
        "deprecated_sequence_backend_executed": False,
        "case": {
            "graph_id": case.graph_id,
            "assignment_id": case.assignment_id,
            "route_count": len(case.routes),
        },
        "base_config": base_config.to_dict(),
        "seeds": list(normalized_seeds),
        "calibration": {
            "schema": calibration["schema"],
            "scientific_scope": calibration["scientific_scope"],
            "selection_rule": calibration["selection_rule"],
            "selected_by_family": {
                family: row["candidate"]
                for family, row in calibration["selected_by_family"].items()
            },
            "benchmark_outcomes_used_for_tuning": False,
        },
        "cache_plan": cache_plan,
        "specs": [spec.to_dict() for spec in specs],
        "comparison_rows": [row["comparison_row"] for row in runs],
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _comparison_row(
    spec: ModernPathBenchmarkSpec,
    seed: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    game = payload["empirical_game"]
    learned = game.get("learned_profile")
    false_signal = payload["alice_false_signal_telemetry"]
    solver = payload["alice_learner"].get("runtime_diagnostics", {}).get(
        "solver"
    )
    return {
        "label": spec.label,
        "role": spec.role,
        "seed": seed,
        "turns": payload["turns"],
        "turns_per_second": payload["turns_per_second"],
        "mean_alice_utility": payload["mean_alice_utility"],
        "final_alice_utility": payload["final_alice_utility"],
        "mean_bell_qualified_rate_hz": payload["mean_bell_qualified_rate_hz"],
        "final_bell_qualified_rate_hz": payload[
            "final_bell_qualified_rate_hz"
        ],
        "alice_false_signal_rate_all_turns": false_signal[
            "false_signal_rate_all_turns"
        ],
        "alice_false_signal_policy_total_variation": false_signal[
            "policy_attribution"
        ]["policy_total_variation"],
        "empirical_matrix_sampling_design": game["sampling_design"],
        "empirical_matrix_minimum_cell_samples": game["minimum_cell_samples"],
        "empirical_matrix_maximum_cell_standard_error": game[
            "maximum_cell_standard_error"
        ],
        "zero_sum_equivalent_exploitability": (
            None
            if learned is None
            else learned["zero_sum_equivalent_exploitability"]
        ),
        "solver_mean_newton_iterations": (
            None if solver is None else solver["mean_newton_iterations"]
        ),
        "solver_maximum_newton_decrement_squared_half": (
            None
            if solver is None
            else solver["maximum_newton_decrement_squared_half"]
        ),
    }


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
