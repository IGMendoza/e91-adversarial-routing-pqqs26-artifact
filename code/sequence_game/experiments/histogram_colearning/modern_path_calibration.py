"""Leakage-free calibration for the modern histogram path learners.

Calibration uses the declared route geometry but synthetic edge-additive loss
sequences.  It never reads histogram service quality, Eve targets, or a payoff
audit.  Candidate selection and held-out evaluation seeds are disjoint.  This
keeps tuning global to an algorithm/topology regime instead of turning it into
a hidden route-quality prior.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import numpy as np

from .corpus import HistogramGameCase
from .modern_path_learners import (
    HistogramItoPathFTRL,
    HistogramMaitiPathFTRL,
)


CALIBRATION_SCENARIOS = (
    "stationary_edge_losses",
    "piecewise_adversarial_edge_losses",
    "sparse_corrupted_edge_losses",
)


@dataclass(frozen=True)
class ModernPathCandidate:
    family: str
    label: str
    parameters: dict[str, Any]
    literature_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "label": self.label,
            "parameters": dict(self.parameters),
            "literature_default": self.literature_default,
        }


def default_modern_path_candidates() -> tuple[ModernPathCandidate, ...]:
    rows = []
    # Log-spaced around the theorem-prescribed unit multipliers.  The wider
    # upper tail is intentional: the first pilot grid was censored at eta=4
    # for both Ito regularizers, so declaring that boundary a calibrated
    # optimum would have been unjustified.
    for eta_scale in (0.5, 1.0, 2.0, 4.0):
        for ix_scale in (0.5, 1.0, 2.0):
            rows.append(ModernPathCandidate(
                family="maiti",
                label=f"maiti_eta{eta_scale:g}_ix{ix_scale:g}",
                parameters={
                    "eta_scale": eta_scale,
                    "implicit_exploration_scale": ix_scale,
                    "failure_probability": 0.05,
                    "reward_loss_baseline": 1.0,
                },
                literature_default=eta_scale == 1.0 and ix_scale == 1.0,
            ))
    for eta_scale in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        rows.append(ModernPathCandidate(
            family="ito_tsallis",
            label=f"ito_tsallis_eta{eta_scale:g}",
            parameters={
                "regularizer": "tsallis",
                "eta_scale": eta_scale,
                # The proof uses beta=2; it is not swept.
                "beta": 2.0,
            },
            literature_default=eta_scale == 1.0,
        ))
        rows.append(ModernPathCandidate(
            family="ito_log_barrier",
            label=f"ito_log_barrier_eta{eta_scale:g}",
            parameters={
                "regularizer": "log_barrier",
                "eta_scale": eta_scale,
                "beta": 2.0,
            },
            literature_default=eta_scale == 1.0,
        ))
    return tuple(rows)


def calibrate_modern_path_learners(
    case: HistogramGameCase,
    *,
    horizon: int = 3_000,
    learner_horizon: int | None = None,
    calibration_seeds: Sequence[int] = (101, 102, 103),
    evaluation_seeds: Sequence[int] = (201, 202, 203),
    candidates: Sequence[ModernPathCandidate] | None = None,
    solver_tolerance: float = 1e-6,
    solver_max_iterations: int = 60,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Select per-family scales and evaluate them on disjoint synthetic seeds."""

    if horizon < 100:
        raise ValueError("calibration horizon must be >= 100")
    schedule_horizon = int(learner_horizon or horizon)
    if schedule_horizon < horizon:
        raise ValueError("learner_horizon must be >= calibration horizon")
    calibration = tuple(map(int, calibration_seeds))
    evaluation = tuple(map(int, evaluation_seeds))
    if (
        not calibration
        or not evaluation
        or set(calibration) & set(evaluation)
        or len(set(calibration)) != len(calibration)
        or len(set(evaluation)) != len(evaluation)
    ):
        raise ValueError("calibration/evaluation seeds must be non-empty and disjoint")
    candidate_rows = tuple(candidates or default_modern_path_candidates())
    if not candidate_rows or len({row.label for row in candidate_rows}) != len(
        candidate_rows
    ):
        raise ValueError("candidate labels must be non-empty and unique")

    started = time.perf_counter()
    calibration_results = []
    for index, candidate in enumerate(candidate_rows, start=1):
        if progress_callback is not None:
            progress_callback(
                f"calibration candidate {index}/{len(candidate_rows)}: "
                f"{candidate.label}"
            )
        calibration_results.append(_evaluate_candidate(
            case,
            candidate,
            horizon=horizon,
            learner_horizon=schedule_horizon,
            seeds=calibration,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        ))
    selected: dict[str, dict[str, Any]] = {}
    for family in sorted({row.family for row in candidate_rows}):
        family_rows = [
            row for row in calibration_results if row["candidate"]["family"] == family
        ]
        raw_winner = min(
            family_rows,
            key=lambda row: (
                row["selection_score"],
                row["worst_scenario_mean_regret_per_turn"],
                row["candidate"]["label"],
            ),
        )
        # A one-standard-error rule avoids over-interpreting Monte Carlo
        # differences.  Among candidates statistically indistinguishable
        # from the raw winner, prefer the least departure from the paper's
        # unit scale.  This is a tuning-conservatism rule, not a route prior:
        # it never inspects route quality or histogram rewards.
        threshold = (
            raw_winner["selection_score"]
            + raw_winner["selection_standard_error"]
        )
        eligible = [
            row for row in family_rows
            if row["selection_score"] <= threshold
        ]
        winner = min(
            eligible,
            key=lambda row: (
                _distance_from_literature_default(row["candidate"]),
                row["selection_score"],
                row["worst_scenario_mean_regret_per_turn"],
                row["candidate"]["label"],
            ),
        )
        candidate = next(
            row for row in candidate_rows
            if row.label == winner["candidate"]["label"]
        )
        if progress_callback is not None:
            progress_callback(
                f"held-out evaluation for {family}: {candidate.label}"
            )
        held_out = _evaluate_candidate(
            case,
            candidate,
            horizon=horizon,
            learner_horizon=schedule_horizon,
            seeds=evaluation,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        )
        selected[family] = {
            "candidate": candidate.to_dict(),
            "raw_empirical_best": raw_winner,
            "one_standard_error_threshold": threshold,
            "one_standard_error_eligible_labels": [
                row["candidate"]["label"] for row in eligible
            ],
            "calibration": winner,
            "held_out_evaluation": held_out,
        }

    return {
        "schema": "histogram_modern_path_calibration.v2",
        "scientific_scope": "algorithmic_calibration_on_synthetic_edge_additive_losses",
        "case": {
            "graph_id": case.graph_id,
            "assignment_id": case.assignment_id,
            "route_count": len(case.routes),
            "route_hops_maximum": max(
                int(route["hop_count"]) for route in case.routes
            ),
        },
        "horizon": horizon,
        "learner_schedule_horizon": schedule_horizon,
        "scenarios": list(CALIBRATION_SCENARIOS),
        "calibration_seeds": list(calibration),
        "evaluation_seeds": list(evaluation),
        "candidate_count": len(candidate_rows),
        "solver": {
            "tolerance": solver_tolerance,
            "maximum_iterations": solver_max_iterations,
            "acceptance_statistic": (
                "Newton decrement squared / 2 is the curvature-normalized "
                "local convergence statistic; projected-gradient residual is "
                "retained as a scale-sensitive diagnostic. No general "
                "objective-gap bound is claimed for the mixed regularizers."
            ),
        },
        "selection_rule": (
            "compute mean fixed-path regret per turn from independent "
            "seed-block means; identify the raw minimum; admit candidates "
            "within one standard error of that raw minimum; among admitted "
            "candidates choose the smallest log2-distance from the paper's "
            "unit scale, then mean regret, worst scenario, and label"
        ),
        "physics_payoffs_used_for_tuning": False,
        "opponent_actions_used_for_tuning": False,
        "route_quality_prior_created": False,
        "calibration_results": calibration_results,
        "selected_by_family": selected,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_candidate(
    case: HistogramGameCase,
    candidate: ModernPathCandidate,
    *,
    horizon: int,
    learner_horizon: int,
    seeds: Sequence[int],
    solver_tolerance: float,
    solver_max_iterations: int,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        losses = _synthetic_route_losses(case, horizon=horizon, seed=seed)
        for scenario, route_losses in losses.items():
            learner = _build_candidate(
                case,
                candidate,
                horizon=horizon,
                learner_horizon=learner_horizon,
                seed=seed,
                solver_tolerance=solver_tolerance,
                solver_max_iterations=solver_max_iterations,
            )
            chosen_loss = 0.0
            for turn in range(horizon):
                route = learner.sample()
                loss = float(route_losses[turn, route])
                chosen_loss += loss
                learner.update(1.0 - loss)
            comparator_losses = np.sum(route_losses, axis=0)
            regret = chosen_loss - float(np.min(comparator_losses))
            diagnostics = learner.full_diagnostics()["solver"]
            rows.append({
                "seed": seed,
                "scenario": scenario,
                "chosen_cumulative_loss": chosen_loss,
                "best_fixed_path_cumulative_loss": float(
                    np.min(comparator_losses)
                ),
                "regret": regret,
                "regret_per_turn": regret / horizon,
                "final_best_path_probability": float(
                    learner.policy()[int(np.argmin(comparator_losses))]
                ),
                "solver_mean_newton_iterations": diagnostics[
                    "mean_newton_iterations"
                ],
                "solver_maximum_residual": diagnostics[
                    "maximum_projected_gradient_residual"
                ],
                "solver_maximum_newton_decrement_squared_half": diagnostics[
                    "maximum_newton_decrement_squared_half"
                ],
            })
    scenario_means = {
        scenario: float(np.mean([
            row["regret_per_turn"] for row in rows
            if row["scenario"] == scenario
        ]))
        for scenario in CALIBRATION_SCENARIOS
    }
    seed_block_means = [
        float(np.mean([
            row["regret_per_turn"] for row in rows
            if row["seed"] == seed
        ]))
        for seed in seeds
    ]
    selection_standard_error = (
        float(np.std(seed_block_means, ddof=1) / np.sqrt(len(seed_block_means)))
        if len(seed_block_means) > 1
        else 0.0
    )
    return {
        "candidate": candidate.to_dict(),
        "runs": rows,
        "selection_score": float(np.mean([
            row["regret_per_turn"] for row in rows
        ])),
        "seed_block_mean_regret_per_turn": seed_block_means,
        "selection_standard_error": selection_standard_error,
        "scenario_mean_regret_per_turn": scenario_means,
        "worst_scenario_mean_regret_per_turn": max(scenario_means.values()),
        "maximum_solver_residual": max(
            row["solver_maximum_residual"] for row in rows
        ),
        "maximum_solver_newton_decrement_squared_half": max(
            row["solver_maximum_newton_decrement_squared_half"]
            for row in rows
        ),
    }


def _distance_from_literature_default(candidate: dict[str, Any]) -> float:
    parameters = candidate["parameters"]
    distance = abs(float(np.log2(float(parameters["eta_scale"]))))
    if candidate["family"] == "maiti":
        distance += abs(float(np.log2(
            float(parameters["implicit_exploration_scale"])
        )))
    return distance


def _build_candidate(
    case: HistogramGameCase,
    candidate: ModernPathCandidate,
    *,
    horizon: int,
    learner_horizon: int,
    seed: int,
    solver_tolerance: float,
    solver_max_iterations: int,
):
    common = {
        "rng": np.random.default_rng([seed, 617]),
        "solver_tolerance": solver_tolerance,
        "solver_max_iterations": solver_max_iterations,
    }
    if candidate.family == "maiti":
        return HistogramMaitiPathFTRL(
            case,
            horizon=learner_horizon,
            failure_probability=float(
                candidate.parameters["failure_probability"]
            ),
            eta_scale=float(candidate.parameters["eta_scale"]),
            implicit_exploration_scale=float(
                candidate.parameters["implicit_exploration_scale"]
            ),
            reward_loss_baseline=float(
                candidate.parameters.get("reward_loss_baseline", 1.0)
            ),
            **common,
        )
    return HistogramItoPathFTRL(
        case,
        # Ito's schedule is anytime; the executed calibration prefix is its
        # natural horizon and does not use Maiti's deployment-horizon control.
        horizon=horizon,
        regularizer=str(candidate.parameters["regularizer"]),
        eta_scale=float(candidate.parameters["eta_scale"]),
        beta=float(candidate.parameters["beta"]),
        **common,
    )


def _synthetic_route_losses(
    case: HistogramGameCase,
    *,
    horizon: int,
    seed: int,
) -> dict[str, np.ndarray]:
    # Construct raw edge-incidence from declared paths.  Edge coefficients are
    # sampled without inspecting length, capacity, device, or reward fields.
    edge_ids = sorted({
        str(edge)
        for route in case.routes
        for edge in route.get("edge_ids") or ()
    })
    edge_index = {edge: index for index, edge in enumerate(edge_ids)}
    incidence = np.zeros((len(case.routes), len(edge_ids)), dtype=float)
    for route_index, route in enumerate(case.routes):
        for edge in route.get("edge_ids") or ():
            incidence[route_index, edge_index[str(edge)]] = 1.0
    rng = np.random.default_rng([seed, 991])
    first = rng.uniform(0.004, 0.028, size=len(edge_ids))
    second = rng.uniform(0.004, 0.028, size=len(edge_ids))
    first_route = incidence @ first
    second_route = incidence @ second
    scale = 0.75 / max(
        float(np.max(first_route)), float(np.max(second_route)), 0.75
    )
    first_route *= scale
    second_route *= scale

    stationary = np.repeat(first_route[np.newaxis, :], horizon, axis=0)
    switching = stationary.copy()
    switching[horizon // 2:] = second_route
    corrupted = stationary.copy()
    corruption_rounds = rng.random(horizon) < 0.12
    for turn in np.flatnonzero(corruption_rounds):
        attacked_edges = rng.random(len(edge_ids)) < 0.15
        edge_corruption = 0.08 * attacked_edges.astype(float)
        corrupted[turn] = np.minimum(
            1.0, first_route + incidence @ edge_corruption
        )
    return {
        "stationary_edge_losses": stationary,
        "piecewise_adversarial_edge_losses": switching,
        "sparse_corrupted_edge_losses": corrupted,
    }
