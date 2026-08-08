"""Paired learner validation against one shared independent payoff audit."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import t as student_t

from sequence_game.game_theory import exploitability, solve_zero_sum_lp


def pooled_independent_payoff_audit(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Pool stratified post-hoc cells without using adaptive learning samples."""

    if not payloads:
        raise ValueError("at least one flat-reference payload is required")
    _validate_compatible_payloads(payloads)
    means = []
    counts = []
    variances = []
    for payload in payloads:
        empirical = payload.get("empirical_game") or {}
        if empirical.get("sampling_design") != "stratified_independent_posthoc":
            raise ValueError("reference payload lacks an independent stratified audit")
        mean = np.asarray(empirical["alice_public_utility_matrix"], dtype=float)
        count = np.asarray(empirical["cell_counts"], dtype=float)
        standard_error = np.asarray(empirical["cell_standard_errors"], dtype=float)
        if mean.shape != count.shape or mean.shape != standard_error.shape:
            raise ValueError("reference audit matrices have inconsistent shapes")
        if np.any(count < 2) or not np.all(np.isfinite(mean)):
            raise ValueError("every pooled reference cell needs at least two samples")
        means.append(mean)
        counts.append(count)
        variances.append(standard_error ** 2 * count)

    total_count = np.sum(counts, axis=0)
    pooled_mean = sum(
        count * mean for count, mean in zip(counts, means)
    ) / total_count
    within_and_between_ss = sum(
        (count - 1.0) * variance + count * (mean - pooled_mean) ** 2
        for count, mean, variance in zip(counts, means, variances)
    )
    pooled_variance = within_and_between_ss / (total_count - 1.0)
    pooled_standard_error = np.sqrt(pooled_variance / total_count)
    actions = payloads[0]["actions"]
    costs = np.asarray([float(action.get("cost", 0.0)) for action in actions])
    equivalent = pooled_mean + costs[np.newaxis, :]
    value, oracle_alice, oracle_eve = solve_zero_sum_lp(equivalent)
    return {
        "sampling_design": "pooled_stratified_independent_posthoc",
        "source_run_count": len(payloads),
        "samples_per_cell": sorted({int(value) for value in total_count.ravel()}),
        "total_samples": int(np.sum(total_count)),
        "payoff": pooled_mean,
        "standard_errors": pooled_standard_error,
        "action_costs": costs,
        "equivalent_game": equivalent,
        "oracle_equivalent_value": float(value),
        "oracle_alice_strategy": oracle_alice,
        "oracle_eve_strategy": oracle_eve,
    }


def paired_learner_validation(
    *,
    flat_payloads: Sequence[dict[str, Any]],
    structured_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare seed-matched learners on the same pooled flat-run audit."""

    reference = pooled_independent_payoff_audit(flat_payloads)
    all_payloads = [*flat_payloads, *structured_payloads]
    _validate_compatible_payloads(all_payloads)
    flat_by_seed = _payloads_by_seed(flat_payloads, label="flat")
    structured_by_seed = _payloads_by_seed(structured_payloads, label="structured")
    seeds = sorted(set(flat_by_seed) & set(structured_by_seed))
    if not seeds:
        raise ValueError("flat and structured payloads have no matched seeds")
    if set(flat_by_seed) != set(structured_by_seed):
        raise ValueError("flat and structured runs must have the same seed set")

    game = reference["equivalent_game"]
    oracle_eve = reference["oracle_eve_strategy"]
    pairs = []
    for seed in seeds:
        flat = flat_by_seed[seed]
        structured = structured_by_seed[seed]
        flat_metrics = _profile_metrics(flat, game)
        structured_metrics = _profile_metrics(structured, game)
        flat_alice = np.asarray(flat["alice_strategy"], dtype=float)
        structured_alice = np.asarray(structured["alice_strategy"], dtype=float)
        flat_eve = np.asarray(flat["eve_strategy"], dtype=float)
        structured_eve = np.asarray(structured["eve_strategy"], dtype=float)
        common_opponent = {
            "flat_eve": {
                "flat_alice_regret": _alice_regret(game, flat_alice, flat_eve),
                "structured_alice_regret": _alice_regret(
                    game, structured_alice, flat_eve
                ),
            },
            "structured_eve": {
                "flat_alice_regret": _alice_regret(
                    game, flat_alice, structured_eve
                ),
                "structured_alice_regret": _alice_regret(
                    game, structured_alice, structured_eve
                ),
            },
            "oracle_eve": {
                "flat_alice_regret": _alice_regret(game, flat_alice, oracle_eve),
                "structured_alice_regret": _alice_regret(
                    game, structured_alice, oracle_eve
                ),
            },
        }
        for row in common_opponent.values():
            row["structured_minus_flat"] = (
                row["structured_alice_regret"] - row["flat_alice_regret"]
            )
        deltas = {
            key: structured_metrics[key] - flat_metrics[key]
            for key in (
                "reference_exploitability",
                "reference_alice_best_response_gap",
                "reference_eve_best_response_gap",
                "mean_alice_utility",
                "final_alice_utility",
                "mean_bell_qualified_rate_hz",
                "final_bell_qualified_rate_hz",
            )
        }
        deltas.update({
            f"alice_regret_against_{name}": row["structured_minus_flat"]
            for name, row in common_opponent.items()
        })
        pairs.append({
            "seed": seed,
            "flat": flat_metrics,
            "structured": structured_metrics,
            "structured_minus_flat": deltas,
            "common_opponent_alice_regret": common_opponent,
        })

    aggregate = {
        key: _paired_summary([
            float(pair["structured_minus_flat"][key]) for pair in pairs
        ])
        for key in pairs[0]["structured_minus_flat"]
    }


    first = flat_payloads[0]
    structured_features = (
        (structured_payloads[0].get("alice_learner") or {}).get("route_features")
        or {}
    )
    route_count = len(first["case"]["route_ids"])
    feature_rank = structured_features.get("rank")
    assessment = _structured_progress_assessment(
        aggregate,
        exact_full_span=bool(
            feature_rank is not None and int(feature_rank) == route_count
        ),
    )
    return {
        "schema": "histogram_paired_learner_validation.v2",
        "case": {
            "graph_id": first["case"]["graph_id"],
            "assignment_id": first["case"]["assignment_id"],
            "route_ids": list(first["case"]["route_ids"]),
            "action_ids": [str(action["id"]) for action in first["actions"]],
            "structured_route_feature_rank": feature_rank,
            "structured_dimension_reduction_active": bool(
                feature_rank is not None and int(feature_rank) < route_count
            ),
            "structured_equivalence_note": (
                "feature rank equals route count, so the orthonormal embedding "
                "spans the complete flat route space and LinearExp2 is not a "
                "lower-dimensional information-sharing model on this case"
                if feature_rank is not None and int(feature_rank) == route_count else
                "topology incidence compresses the route action space"
                if feature_rank is not None else
                "structured route-feature metadata unavailable"
            ),
        },
        "matched_seeds": seeds,
        "reference_audit": {
            "sampling_design": reference["sampling_design"],
            "source_run_count": reference["source_run_count"],
            "samples_per_cell": reference["samples_per_cell"],
            "total_samples": reference["total_samples"],
            "maximum_cell_standard_error": float(
                np.max(reference["standard_errors"])
            ),
            "oracle_equivalent_value": reference["oracle_equivalent_value"],
            "oracle_alice_strategy": reference["oracle_alice_strategy"].tolist(),
            "oracle_eve_strategy": reference["oracle_eve_strategy"].tolist(),
            "used_for_learning": False,
            "source": "flat-run independent post-hoc audits only",
        },
        "pairs": pairs,
        "paired_delta_summary": aggregate,
        "structured_progress_assessment": assessment,
        "information_boundary": (
            "both online learners saw only their declared scalar bandit feedback; "
            "the pooled matrix is an after-learning validation oracle"
        ),
        "claim_boundary": (
            "three matched seeds on one topology; confidence intervals quantify "
            "run variability but do not establish corpus-level generalization"
        ),
    }


def unilateral_learner_validation(
    *,
    baseline_payloads: Sequence[dict[str, Any]],
    alice_structured_payloads: Sequence[dict[str, Any]],
    eve_structured_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Isolate each structured learner through matched offline recombination.

    Every strategy is learned online in its stated simultaneous two-player run.
    The independent baseline audit then recombines learned marginals so that
    structured Alice is evaluated with baseline Eve and structured Eve is
    evaluated with baseline Alice.
    """

    reference = pooled_independent_payoff_audit(baseline_payloads)
    all_payloads = [
        *baseline_payloads,
        *alice_structured_payloads,
        *eve_structured_payloads,
    ]
    _validate_compatible_payloads(all_payloads)
    _validate_unilateral_run_contract(
        baseline_payloads=baseline_payloads,
        alice_structured_payloads=alice_structured_payloads,
        eve_structured_payloads=eve_structured_payloads,
    )
    collections = {
        "baseline": _payloads_by_seed(baseline_payloads, label="baseline"),
        "alice_structured": _payloads_by_seed(
            alice_structured_payloads,
            label="Alice-structured",
        ),
        "eve_structured": _payloads_by_seed(
            eve_structured_payloads,
            label="Eve-structured",
        ),
    }
    seed_sets = {tuple(sorted(rows)) for rows in collections.values()}
    if len(seed_sets) != 1:
        raise ValueError("all unilateral collections must have the same seed set")
    seeds = list(next(iter(seed_sets)))
    game = reference["equivalent_game"]
    oracle_alice = reference["oracle_alice_strategy"]
    oracle_eve = reference["oracle_eve_strategy"]

    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        baseline = collections["baseline"][seed]
        alice_run = collections["alice_structured"][seed]
        eve_run = collections["eve_structured"][seed]
        x0 = np.asarray(baseline["alice_strategy"], dtype=float)
        y0 = np.asarray(baseline["eve_strategy"], dtype=float)
        xa = np.asarray(alice_run["alice_strategy"], dtype=float)
        ye = np.asarray(eve_run["eve_strategy"], dtype=float)

        recombined = {
            "baseline": _reference_profile_metrics(game, x0, y0),
            "alice_changed_only": _reference_profile_metrics(game, xa, y0),
            "eve_changed_only": _reference_profile_metrics(game, x0, ye),
            "both_structured_factorial": _reference_profile_metrics(game, xa, ye),
        }
        common_opponents = {
            "alice": {
                "baseline_eve": _regret_comparison(
                    _alice_regret(game, x0, y0),
                    _alice_regret(game, xa, y0),
                ),
                "structured_eve": _regret_comparison(
                    _alice_regret(game, x0, ye),
                    _alice_regret(game, xa, ye),
                ),
                "oracle_eve": _regret_comparison(
                    _alice_regret(game, x0, oracle_eve),
                    _alice_regret(game, xa, oracle_eve),
                ),
            },
            "eve": {
                "baseline_alice": _regret_comparison(
                    _eve_regret(game, x0, y0),
                    _eve_regret(game, x0, ye),
                ),
                "structured_alice": _regret_comparison(
                    _eve_regret(game, xa, y0),
                    _eve_regret(game, xa, ye),
                ),
                "oracle_alice": _regret_comparison(
                    _eve_regret(game, oracle_alice, y0),
                    _eve_regret(game, oracle_alice, ye),
                ),
            },
        }
        deltas = {
            **{
                f"alice_changed_only.{metric}": (
                    recombined["alice_changed_only"][metric]
                    - recombined["baseline"][metric]
                )
                for metric in recombined["baseline"]
            },
            **{
                f"eve_changed_only.{metric}": (
                    recombined["eve_changed_only"][metric]
                    - recombined["baseline"][metric]
                )
                for metric in recombined["baseline"]
            },
            **{
                f"alice_regret_against_{name}": row["structured_minus_baseline"]
                for name, row in common_opponents["alice"].items()
            },
            **{
                f"eve_regret_against_{name}": row["structured_minus_baseline"]
                for name, row in common_opponents["eve"].items()
            },
        }
        pairs.append({
            "seed": int(seed),
            "online_observed_profiles": {
                "baseline": _profile_metrics(baseline, game),
                "alice_structured": _profile_metrics(alice_run, game),
                "eve_structured": _profile_metrics(eve_run, game),
            },
            "offline_recombined_profiles": recombined,
            "changed_player_common_opponent_regret": common_opponents,
            "unilateral_minus_baseline": deltas,
            "unilateral_strategy_sources": {
                "alice_changed_only": {
                    "alice": "Alice-structured online run",
                    "eve": "baseline online run",
                },
                "eve_changed_only": {
                    "alice": "baseline online run",
                    "eve": "Eve-structured online run",
                },
                "both_structured_factorial": {
                    "alice": "Alice-structured online run",
                    "eve": "Eve-structured online run",
                },
            },
        })

    first = baseline_payloads[0]
    modes = {
        label: _collection_learner_modes(payloads)
        for label, payloads in (
            ("baseline", baseline_payloads),
            ("alice_structured", alice_structured_payloads),
            ("eve_structured", eve_structured_payloads),
        )
    }
    aggregate = {
        key: _paired_summary([
            float(pair["unilateral_minus_baseline"][key]) for pair in pairs
        ])
        for key in pairs[0]["unilateral_minus_baseline"]
    }
    return {
        "schema": "histogram_unilateral_learner_validation.v1",
        "case": {
            "graph_id": first["case"]["graph_id"],
            "assignment_id": first["case"]["assignment_id"],
            "route_ids": list(first["case"]["route_ids"]),
            "action_ids": [str(action["id"]) for action in first["actions"]],
        },
        "learner_modes": modes,
        "matched_seeds": seeds,
        "reference_audit": _reference_audit_metadata(reference),
        "pairs": pairs,
        "paired_delta_summary": aggregate,
        "metric_directions": {
            "reference_exploitability": "lower",
            "reference_alice_best_response_gap": "lower",
            "reference_eve_best_response_gap": "lower",
            "changed_player_common_opponent_regret": "lower",
            "reference_joint_equivalent_utility": (
                "higher for Alice and lower for Eve; not an equilibrium metric alone"
            ),
        },
        "online_information_boundary": (
            "each learner received only its selected-action scalar reward; the "
            "independent payoff audit and all recombinations occur after learning"
        ),
        "recombination_boundary": (
            "factorial profiles evaluate learned marginal strategies on the shared "
            "audit; they are not additional online co-learning runs"
        ),
        "decision_rule": (
            "no numerical acceptance margin or pre-existing route/action ranking "
            "is applied; paired values, uncertainty, and signs are reported"
        ),
    }


def multi_case_paired_learner_validation(
    *,
    flat_payloads_by_case: dict[str, Sequence[dict[str, Any]]],
    structured_payloads_by_case: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Apply the same seed-paired audit across a predeclared topology set."""

    flat_cases = set(flat_payloads_by_case)
    structured_cases = set(structured_payloads_by_case)
    if not flat_cases:
        raise ValueError("at least one validation case is required")
    if flat_cases != structured_cases:
        raise ValueError("flat and structured bundles must have the same case set")
    case_results = {
        case: paired_learner_validation(
            flat_payloads=flat_payloads_by_case[case],
            structured_payloads=structured_payloads_by_case[case],
        )
        for case in sorted(flat_cases)
    }
    seed_sets = {
        tuple(result["matched_seeds"]) for result in case_results.values()
    }
    if len(seed_sets) != 1:
        raise ValueError("every case must use the same matched seed set")
    matched_seeds = list(next(iter(seed_sets)))
    metric_names = list(
        next(iter(case_results.values()))["paired_delta_summary"]
    )
    per_seed_case_average = []
    for seed in matched_seeds:
        rows = {
            case: next(
                pair for pair in result["pairs"] if int(pair["seed"]) == int(seed)
            )
            for case, result in case_results.items()
        }
        averages = {
            metric: float(np.mean([
                row["structured_minus_flat"][metric] for row in rows.values()
            ]))
            for metric in metric_names
        }
        per_seed_case_average.append({
            "seed": int(seed),
            "structured_minus_flat_case_average": averages,
        })
    macro_summary = {
        metric: _paired_summary([
            row["structured_minus_flat_case_average"][metric]
            for row in per_seed_case_average
        ])
        for metric in metric_names
    }
    macro_assessment = _structured_progress_assessment(
        macro_summary,
        exact_full_span=False,
    )
    every_case_supported = all(
        result["structured_progress_assessment"]["development_supported"]
        for result in case_results.values()
    )
    return {
        "schema": "histogram_multi_case_paired_learner_validation.v2",
        "cases": case_results,
        "case_count": len(case_results),
        "matched_seeds": matched_seeds,
        "per_seed_topology_average": per_seed_case_average,
        "topology_averaged_paired_delta_summary": macro_summary,
        "structured_progress_assessment": {
            "development_supported": bool(
                macro_assessment["point_estimate_parity_or_better"]
                and every_case_supported
            ),
            "topology_averaged": macro_assessment,
            "every_individual_case_supported": every_case_supported,
            "supported_cases": sorted(
                case for case, result in case_results.items()
                if result["structured_progress_assessment"][
                    "development_supported"
                ]
            ),
            "criterion": (
                "progress means exact full-span equivalence or no worse paired "
                "point estimates for exploitability, Alice regret against the "
                "shared oracle Eve, and mean Bell-qualified rate; confidence "
                "intervals remain separately visible"
            ),
            "population_generalization_claim": False,
        },
        "information_boundary": (
            "each case uses only pooled independent flat-run post-hoc audits; "
            "no audit matrix was supplied to either online learner"
        ),
        "claim_boundary": (
            "four predeclared Topology Bench cases and three shared seeds; the "
            "topology average is a matched-seed macro summary, not a population "
            "estimate over either Topology Bench corpus"
        ),
    }


def _profile_metrics(payload: dict[str, Any], game: np.ndarray) -> dict[str, float]:
    alice = np.asarray(payload["alice_strategy"], dtype=float)
    eve = np.asarray(payload["eve_strategy"], dtype=float)
    joint = float(alice @ game @ eve)
    return {
        "reference_joint_equivalent_utility": joint,
        "reference_exploitability": float(exploitability(game, alice, eve)),
        "reference_alice_best_response_gap": float(np.max(game @ eve) - joint),
        "reference_eve_best_response_gap": float(joint - np.min(alice @ game)),
        "mean_alice_utility": float(payload["mean_alice_utility"]),
        "final_alice_utility": float(payload["final_alice_utility"]),
        "mean_bell_qualified_rate_hz": float(payload["mean_bell_qualified_rate_hz"]),
        "final_bell_qualified_rate_hz": float(payload["final_bell_qualified_rate_hz"]),
    }


def _reference_profile_metrics(
    game: np.ndarray,
    alice: np.ndarray,
    eve: np.ndarray,
) -> dict[str, float]:
    joint = float(alice @ game @ eve)
    return {
        "reference_joint_equivalent_utility": joint,
        "reference_exploitability": float(exploitability(game, alice, eve)),
        "reference_alice_best_response_gap": float(np.max(game @ eve) - joint),
        "reference_eve_best_response_gap": float(joint - np.min(alice @ game)),
    }


def _reference_audit_metadata(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampling_design": reference["sampling_design"],
        "source_run_count": reference["source_run_count"],
        "samples_per_cell": reference["samples_per_cell"],
        "total_samples": reference["total_samples"],
        "maximum_cell_standard_error": float(
            np.max(reference["standard_errors"])
        ),
        "oracle_equivalent_value": reference["oracle_equivalent_value"],
        "oracle_alice_strategy": reference["oracle_alice_strategy"].tolist(),
        "oracle_eve_strategy": reference["oracle_eve_strategy"].tolist(),
        "used_for_learning": False,
        "source": "baseline-run independent post-hoc audits only",
    }


def _regret_comparison(
    baseline: float,
    structured: float,
) -> dict[str, float]:
    return {
        "baseline": float(baseline),
        "structured": float(structured),
        "structured_minus_baseline": float(structured - baseline),
    }


def _eve_regret(game: np.ndarray, alice: np.ndarray, eve: np.ndarray) -> float:
    return float(alice @ game @ eve - np.min(alice @ game))


_MATCHED_UNILATERAL_CONFIG_FIELDS = (
    "turns",
    "requested_pairs_per_turn",
    "game_mode",
    "exp3_schedule_mode",
    "alice_exp3_gamma",
    "eve_exp3_gamma",
    "alice_exp3_eta_c",
    "eve_exp3_eta_c",
    "exp3_t0",
    "exp3_gamma_max",
    "alice_physics_prior_mix",
    "repeater_service_profile_path",
    "accepted_alive_time_s",
    "age_points_per_edge",
    "allow_known_1550_to_1558_detector_approximation",
    "fock_cutoff",
)


def _validate_unilateral_run_contract(
    *,
    baseline_payloads: Sequence[dict[str, Any]],
    alice_structured_payloads: Sequence[dict[str, Any]],
    eve_structured_payloads: Sequence[dict[str, Any]],
) -> None:
    collections = {
        "baseline": baseline_payloads,
        "Alice-structured": alice_structured_payloads,
        "Eve-structured": eve_structured_payloads,
    }
    for label, payloads in collections.items():
        if not payloads:
            raise ValueError(f"{label} payload collection is empty")
        _collection_learner_modes(payloads)
        for payload in payloads:
            if payload.get("backend") not in {None, "compiled_histogram_transcript"}:
                raise ValueError(f"{label} payload is not from the histogram backend")
            if payload.get("deprecated_sequence_backend_executed") not in {None, False}:
                raise ValueError(f"{label} payload executed the deprecated backend")

    baseline_modes = _collection_learner_modes(baseline_payloads)
    alice_modes = _collection_learner_modes(alice_structured_payloads)
    eve_modes = _collection_learner_modes(eve_structured_payloads)
    if alice_modes["eve"] != baseline_modes["eve"]:
        raise ValueError("Alice-only runs changed Eve's learner mode")
    if alice_modes["alice"] == baseline_modes["alice"]:
        raise ValueError("Alice-only runs did not change Alice's learner mode")
    if eve_modes["alice"] != baseline_modes["alice"]:
        raise ValueError("Eve-only runs changed Alice's learner mode")
    if eve_modes["eve"] == baseline_modes["eve"]:
        raise ValueError("Eve-only runs did not change Eve's learner mode")

    expected = {
        key: (baseline_payloads[0].get("config") or {}).get(key)
        for key in _MATCHED_UNILATERAL_CONFIG_FIELDS
    }
    for label, payloads in collections.items():
        for payload in payloads:
            config = payload.get("config") or {}
            mismatches = {
                key: (expected[key], config.get(key))
                for key in _MATCHED_UNILATERAL_CONFIG_FIELDS
                if config.get(key) != expected[key]
            }
            if mismatches:
                raise ValueError(
                    f"{label} run changed matched physics/learning settings: "
                    f"{mismatches}"
                )


def _collection_learner_modes(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, str]:
    modes = {
        (
            str(
                (payload.get("alice_learner") or {}).get(
                    "mode",
                    (payload.get("config") or {}).get(
                        "alice_learner_mode",
                        "flat_exp3",
                    ),
                )
            ),
            str(
                (payload.get("eve_learner") or {}).get(
                    "mode",
                    (payload.get("config") or {}).get(
                        "eve_learner_mode",
                        "flat_exp3",
                    ),
                )
            ),
        )
        for payload in payloads
    }
    if len(modes) != 1:
        raise ValueError("one learner collection contains inconsistent modes")
    alice, eve = next(iter(modes))
    return {"alice": alice, "eve": eve}


def _structured_progress_assessment(
    aggregate: dict[str, dict[str, Any]],
    *,
    exact_full_span: bool,
) -> dict[str, Any]:
    """Separate useful structured progress from a strict superiority claim.

    No arbitrary non-inferiority margin is introduced. Point estimates and
    uncertainty are both retained, and strict superiority is reported only as
    an optional stronger diagnostic rather than an adoption veto.
    """

    exploitability = aggregate["reference_exploitability"]
    alice_regret = aggregate["alice_regret_against_oracle_eve"]
    mean_rate = aggregate["mean_bell_qualified_rate_hz"]
    point_parity_or_better = bool(
        exploitability["mean"] <= 0.0
        and alice_regret["mean"] <= 0.0
        and mean_rate["mean"] >= 0.0
    )
    strict_superiority = bool(
        exploitability["confidence_interval_95"] is not None
        and exploitability["confidence_interval_95"][1] < 0.0
        and alice_regret["confidence_interval_95"] is not None
        and alice_regret["confidence_interval_95"][1] < 0.0
    )
    return {
        "development_supported": bool(exact_full_span or point_parity_or_better),
        "exact_full_span_equivalence_expected": bool(exact_full_span),
        "point_estimate_parity_or_better": point_parity_or_better,
        "strict_primary_superiority_supported": strict_superiority,
        "noninferiority_margin_declared": False,
        "formal_noninferiority_claim": False,
        "criterion": (
            "exact full-span equivalence, or structured-minus-flat point "
            "estimates <= 0 for exploitability and oracle-Eve Alice regret "
            "and >= 0 for mean Bell-qualified rate"
        ),
        "uncertainty_boundary": (
            "paired confidence intervals are reported independently; failure "
            "to prove strict superiority is not treated as failure of the "
            "structured-learning development path"
        ),
    }


def _alice_regret(game: np.ndarray, alice: np.ndarray, eve: np.ndarray) -> float:
    return float(np.max(game @ eve) - alice @ game @ eve)


def _paired_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    standard_error = sd / math.sqrt(array.size) if array.size > 1 else None
    critical = (
        float(student_t.ppf(0.975, array.size - 1))
        if standard_error is not None else None
    )
    return {
        "values": array.tolist(),
        "mean": mean,
        "sd": sd,
        "standard_error": standard_error,
        "confidence_interval_95": (
            [mean - critical * standard_error, mean + critical * standard_error]
            if critical is not None else None
        ),
        "sample_count": int(array.size),
        "direction": (
            "structured_lower" if mean < 0.0
            else "structured_higher" if mean > 0.0 else "equal"
        ),
    }


def _validate_compatible_payloads(payloads: Sequence[dict[str, Any]]) -> None:
    if not payloads:
        raise ValueError("payload collection is empty")
    first = payloads[0]
    route_ids = list(first["case"]["route_ids"])
    action_ids = [str(action["id"]) for action in first["actions"]]
    action_costs = [float(action.get("cost", 0.0)) for action in first["actions"]]
    expected_shape = (len(route_ids), len(action_ids))
    for payload in payloads[1:]:
        if list(payload["case"]["route_ids"]) != route_ids:
            raise ValueError("learner payloads have different route universes")
        if [str(action["id"]) for action in payload["actions"]] != action_ids:
            raise ValueError("learner payloads have different action universes")
        if [float(action.get("cost", 0.0)) for action in payload["actions"]] != action_costs:
            raise ValueError("learner payloads have different action costs")
        if payload["case"]["graph_id"] != first["case"]["graph_id"]:
            raise ValueError("learner payloads have different graph ids")
        if payload["case"]["assignment_id"] != first["case"]["assignment_id"]:
            raise ValueError("learner payloads have different assignments")
    for payload in payloads:
        if np.asarray(payload["alice_strategy"], dtype=float).shape != (expected_shape[0],):
            raise ValueError("Alice strategy does not match the route universe")
        if np.asarray(payload["eve_strategy"], dtype=float).shape != (expected_shape[1],):
            raise ValueError("Eve strategy does not match the action universe")


def _payloads_by_seed(
    payloads: Sequence[dict[str, Any]], *, label: str
) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        seed = _seed(payload)
        if seed in by_seed:
            raise ValueError(f"duplicate {label} seed {seed}")
        by_seed[seed] = payload
    return by_seed


def _seed(payload: dict[str, Any]) -> int:
    return int((payload.get("paper_run") or {}).get("seed", payload["config"]["seed"]))
