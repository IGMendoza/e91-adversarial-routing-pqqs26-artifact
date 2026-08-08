"""Two-sided learner-family cross-play on one independent histogram audit."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import t as student_t

from sequence_game.game_theory import exploitability

from .learner_validation import (
    _MATCHED_UNILATERAL_CONFIG_FIELDS,
    _collection_learner_modes,
    _payloads_by_seed,
    _reference_audit_metadata,
    _validate_compatible_payloads,
    _validate_unilateral_run_contract,
    pooled_independent_payoff_audit,
)
from .strategic_metrics import EquilibriumSetProjector


CELL_BASELINE = "flat_alice__flat_eve"
CELL_ALICE_STRUCTURED = "structured_alice__flat_eve"
CELL_EVE_STRUCTURED = "flat_alice__structured_eve"
CELL_BOTH_STRUCTURED = "structured_alice__structured_eve"
CELL_ORDER = (
    CELL_BASELINE,
    CELL_ALICE_STRUCTURED,
    CELL_EVE_STRUCTURED,
    CELL_BOTH_STRUCTURED,
)

_ALICE_STRUCTURED_CONFIG_FIELDS = (
    "structured_exploration_mode",
)
_EVE_STRUCTURED_CONFIG_FIELDS = (
    "eve_structured_feature_mode",
    "eve_structured_exploration_mode",
)

REFERENCE_METRICS = (
    "reference_joint_equivalent_utility",
    "reference_exploitability",
    "reference_alice_best_response_gap",
    "reference_eve_best_response_gap",
    "reference_alice_equilibrium_set_total_variation",
    "reference_eve_equilibrium_set_total_variation",
)
ONLINE_METRICS = (
    "mean_alice_utility",
    "final_alice_utility",
    "mean_bell_qualified_rate_hz",
    "final_bell_qualified_rate_hz",
    "mean_eve_bandit_reward",
)
LIVE_SCALAR_METRICS = (*REFERENCE_METRICS, *ONLINE_METRICS)


def two_sided_crossplay_validation(
    *,
    baseline_payloads: Sequence[dict[str, Any]],
    alice_structured_payloads: Sequence[dict[str, Any]],
    eve_structured_payloads: Sequence[dict[str, Any]],
    both_structured_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate all four live cells and every learned marginal cross-play."""

    collections = {
        CELL_BASELINE: baseline_payloads,
        CELL_ALICE_STRUCTURED: alice_structured_payloads,
        CELL_EVE_STRUCTURED: eve_structured_payloads,
        CELL_BOTH_STRUCTURED: both_structured_payloads,
    }
    reference = pooled_independent_payoff_audit(baseline_payloads)
    _validate_two_sided_contract(collections)
    by_seed = {
        cell: _payloads_by_seed(payloads, label=cell)
        for cell, payloads in collections.items()
    }
    seed_sets = {tuple(sorted(payloads)) for payloads in by_seed.values()}
    if len(seed_sets) != 1:
        raise ValueError("all four live cells must have the same seed set")
    seeds = list(next(iter(seed_sets)))
    game = np.asarray(reference["equivalent_game"], dtype=float)
    projector = EquilibriumSetProjector(
        game,
        value=float(reference["oracle_equivalent_value"]),
    )

    live_profiles = []
    for seed in seeds:
        live_profiles.append({
            "seed": int(seed),
            "cells": {
                cell: _payload_profile(by_seed[cell][seed], game, projector)
                for cell in CELL_ORDER
            },
        })
    live_summary = {
        cell: {
            metric: _sample_summary([
                row["cells"][cell][metric] for row in live_profiles
            ])
            for metric in LIVE_SCALAR_METRICS
        }
        for cell in CELL_ORDER
    }
    contrasts = _factorial_contrasts(live_profiles)
    source_crossplay = _full_source_crossplay(by_seed, seeds, game, projector)
    family_crossplay = _learner_family_crossplay(
        by_seed,
        seeds,
        game,
        projector,
    )
    joint_comparison = _joint_vs_offline_factorial(
        by_seed,
        seeds,
        game,
        projector,
    )
    modes = {
        cell: _collection_learner_modes(collections[cell])
        for cell in CELL_ORDER
    }
    first = baseline_payloads[0]
    return {
        "schema": "histogram_two_sided_crossplay_validation.v1",
        "case": {
            "graph_id": first["case"]["graph_id"],
            "assignment_id": first["case"]["assignment_id"],
            "route_ids": list(first["case"]["route_ids"]),
            "action_ids": [str(action["id"]) for action in first["actions"]],
        },
        "cell_order": list(CELL_ORDER),
        "learner_modes": modes,
        "matched_seeds": seeds,
        "reference_audit": _reference_audit_metadata(reference),
        "live_profiles": live_profiles,
        "live_cell_summary": live_summary,
        "paired_factorial_contrasts": contrasts,
        "full_source_crossplay": source_crossplay,
        "learner_family_crossplay": family_crossplay,
        "joint_live_vs_offline_factorial": joint_comparison,
        "metric_directions": {
            "reference_exploitability": "lower",
            "reference_alice_best_response_gap": "lower for Alice learning",
            "reference_eve_best_response_gap": "lower for Eve learning",
            "reference_alice_equilibrium_set_total_variation": "lower",
            "reference_eve_equilibrium_set_total_variation": "lower",
            "reference_joint_equivalent_utility": (
                "higher favors Alice and lower favors Eve"
            ),
            "mean_bell_qualified_rate_hz": "higher favors Alice service",
        },
        "online_information_boundary": (
            "all four live cells learned only from their selected-action scalar "
            "feedback; the pooled audit and every cross-play occur after learning"
        ),
        "cross_seed_inference_boundary": (
            "the nine seed-pair cross-products per source cell are descriptive; "
            "confidence intervals use only the three independent matched live seeds"
        ),
        "decision_rule": (
            "no numerical acceptance margin or pre-existing route/action ranking "
            "is applied"
        ),
    }


def _validate_two_sided_contract(
    collections: dict[str, Sequence[dict[str, Any]]],
) -> None:
    all_payloads = [
        payload
        for cell in CELL_ORDER
        for payload in collections[cell]
    ]
    _validate_compatible_payloads(all_payloads)
    _validate_unilateral_run_contract(
        baseline_payloads=collections[CELL_BASELINE],
        alice_structured_payloads=collections[CELL_ALICE_STRUCTURED],
        eve_structured_payloads=collections[CELL_EVE_STRUCTURED],
    )
    modes = {
        cell: _collection_learner_modes(collections[cell])
        for cell in CELL_ORDER
    }
    expected_joint = {
        "alice": modes[CELL_ALICE_STRUCTURED]["alice"],
        "eve": modes[CELL_EVE_STRUCTURED]["eve"],
    }
    if modes[CELL_BOTH_STRUCTURED] != expected_joint:
        raise ValueError(
            "the both-structured cell must use the structured Alice and Eve modes"
        )
    _validate_joint_role_configuration(
        joint_payloads=collections[CELL_BOTH_STRUCTURED],
        alice_reference=collections[CELL_ALICE_STRUCTURED][0],
        eve_reference=collections[CELL_EVE_STRUCTURED][0],
    )
    baseline_config = collections[CELL_BASELINE][0].get("config") or {}
    expected = {
        key: baseline_config.get(key)
        for key in _MATCHED_UNILATERAL_CONFIG_FIELDS
    }
    for payload in collections[CELL_BOTH_STRUCTURED]:
        if payload.get("backend") not in {None, "compiled_histogram_transcript"}:
            raise ValueError(
                "both-structured payload is not from the histogram backend"
            )
        if payload.get("deprecated_sequence_backend_executed") not in {None, False}:
            raise ValueError("both-structured payload executed the deprecated backend")
        config = payload.get("config") or {}
        mismatches = {
            key: (expected[key], config.get(key))
            for key in _MATCHED_UNILATERAL_CONFIG_FIELDS
            if config.get(key) != expected[key]
        }
        if mismatches:
            raise ValueError(
                "both-structured run changed matched physics/learning settings: "
                f"{mismatches}"
            )


def _validate_joint_role_configuration(
    *,
    joint_payloads: Sequence[dict[str, Any]],
    alice_reference: dict[str, Any],
    eve_reference: dict[str, Any],
) -> None:
    alice_config = alice_reference.get("config") or {}
    eve_config = eve_reference.get("config") or {}
    expected = {
        **{
            key: alice_config.get(key)
            for key in _ALICE_STRUCTURED_CONFIG_FIELDS
        },
        **{
            key: eve_config.get(key)
            for key in _EVE_STRUCTURED_CONFIG_FIELDS
        },
    }
    for payload in joint_payloads:
        config = payload.get("config") or {}
        mismatches = {
            key: (expected_value, config.get(key))
            for key, expected_value in expected.items()
            if config.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                "the both-structured cell changed a calibrated role-specific "
                f"configuration: {mismatches}"
            )


def _payload_profile(
    payload: dict[str, Any],
    game: np.ndarray,
    projector: EquilibriumSetProjector,
) -> dict[str, Any]:
    alice = _probability_vector(payload["alice_strategy"], game.shape[0], "Alice")
    eve = _probability_vector(payload["eve_strategy"], game.shape[1], "Eve")
    return {
        **_reference_profile(game, alice, eve, projector),
        **{metric: float(payload[metric]) for metric in ONLINE_METRICS},
        "alice_strategy": alice.tolist(),
        "eve_strategy": eve.tolist(),
        "alice_strategy_diagnostics": _strategy_diagnostics(alice),
        "eve_strategy_diagnostics": _strategy_diagnostics(eve),
    }


def _reference_profile(
    game: np.ndarray,
    alice: np.ndarray,
    eve: np.ndarray,
    projector: EquilibriumSetProjector,
) -> dict[str, float]:
    joint = float(alice @ game @ eve)
    return {
        "reference_joint_equivalent_utility": joint,
        "reference_exploitability": float(exploitability(game, alice, eve)),
        "reference_alice_best_response_gap": float(np.max(game @ eve) - joint),
        "reference_eve_best_response_gap": float(joint - np.min(alice @ game)),
        "reference_alice_equilibrium_set_total_variation": (
            projector.total_variation("alice", alice)
        ),
        "reference_eve_equilibrium_set_total_variation": (
            projector.total_variation("eve", eve)
        ),
    }


def _factorial_contrasts(
    live_profiles: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    definitions = {
        "alice_effect_with_flat_eve": (
            CELL_ALICE_STRUCTURED,
            CELL_BASELINE,
            "Alice structured minus Alice flat while Eve's assigned learner is flat",
        ),
        "alice_effect_with_structured_eve": (
            CELL_BOTH_STRUCTURED,
            CELL_EVE_STRUCTURED,
            "Alice structured minus Alice flat while Eve's assigned learner "
            "is structured",
        ),
        "eve_effect_with_flat_alice": (
            CELL_EVE_STRUCTURED,
            CELL_BASELINE,
            "Eve structured minus Eve flat while Alice's assigned learner is flat",
        ),
        "eve_effect_with_structured_alice": (
            CELL_BOTH_STRUCTURED,
            CELL_ALICE_STRUCTURED,
            "Eve structured minus Eve flat while Alice's assigned learner "
            "is structured",
        ),
        "both_structured_minus_baseline": (
            CELL_BOTH_STRUCTURED,
            CELL_BASELINE,
            "live both-structured cell minus the live flat/flat cell",
        ),
    }
    output: dict[str, Any] = {}
    for name, (positive, negative, definition) in definitions.items():
        per_seed = [
            {
                "seed": int(row["seed"]),
                **{
                    metric: float(
                        row["cells"][positive][metric]
                        - row["cells"][negative][metric]
                    )
                    for metric in LIVE_SCALAR_METRICS
                },
            }
            for row in live_profiles
        ]
        output[name] = {
            "definition": definition,
            "positive_cell": positive,
            "negative_cell": negative,
            "per_seed": per_seed,
            "summary": {
                metric: _sample_summary([row[metric] for row in per_seed])
                for metric in LIVE_SCALAR_METRICS
            },
        }
    interaction_rows = []
    for row in live_profiles:
        cells = row["cells"]
        interaction_rows.append({
            "seed": int(row["seed"]),
            **{
                metric: float(
                    cells[CELL_BOTH_STRUCTURED][metric]
                    - cells[CELL_ALICE_STRUCTURED][metric]
                    - cells[CELL_EVE_STRUCTURED][metric]
                    + cells[CELL_BASELINE][metric]
                )
                for metric in LIVE_SCALAR_METRICS
            },
        })
    output["learner_assignment_interaction"] = {
        "definition": "both - Alice-only - Eve-only + baseline",
        "per_seed": interaction_rows,
        "summary": {
            metric: _sample_summary([row[metric] for row in interaction_rows])
            for metric in LIVE_SCALAR_METRICS
        },
    }
    return output


def _full_source_crossplay(
    by_seed: dict[str, dict[int, dict[str, Any]]],
    seeds: Sequence[int],
    game: np.ndarray,
    projector: EquilibriumSetProjector,
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    matrices = {
        metric: np.zeros((len(CELL_ORDER), len(CELL_ORDER)), dtype=float)
        for metric in REFERENCE_METRICS
    }
    for alice_index, alice_source in enumerate(CELL_ORDER):
        cells[alice_source] = {}
        for eve_index, eve_source in enumerate(CELL_ORDER):
            rows = []
            for alice_seed in seeds:
                alice = _probability_vector(
                    by_seed[alice_source][alice_seed]["alice_strategy"],
                    game.shape[0],
                    "Alice",
                )
                for eve_seed in seeds:
                    eve = _probability_vector(
                        by_seed[eve_source][eve_seed]["eve_strategy"],
                        game.shape[1],
                        "Eve",
                    )
                    rows.append({
                        "alice_seed": int(alice_seed),
                        "eve_seed": int(eve_seed),
                        "matched_seed": bool(alice_seed == eve_seed),
                        **_reference_profile(game, alice, eve, projector),
                    })
            matched = [row for row in rows if row["matched_seed"]]
            cell = {
                "profile_count": len(rows),
                "matched_seed_profile_count": len(matched),
                "profiles": rows,
                "matched_seed_summary": {
                    metric: _sample_summary([row[metric] for row in matched])
                    for metric in REFERENCE_METRICS
                },
                "all_seed_pair_descriptive": {
                    metric: _descriptive_summary([row[metric] for row in rows])
                    for metric in REFERENCE_METRICS
                },
            }
            cells[alice_source][eve_source] = cell
            for metric in REFERENCE_METRICS:
                matrices[metric][alice_index, eve_index] = cell[
                    "all_seed_pair_descriptive"
                ][metric]["mean"]
    return {
        "alice_source_order": list(CELL_ORDER),
        "eve_source_order": list(CELL_ORDER),
        "cells": cells,
        "mean_matrices": {
            metric: matrix.tolist() for metric, matrix in matrices.items()
        },
        "inference_boundary": (
            "each source/source cell contains all nine seed cross-products; only "
            "the three matched-seed rows receive a confidence interval"
        ),
    }


def _learner_family_crossplay(
    by_seed: dict[str, dict[int, dict[str, Any]]],
    seeds: Sequence[int],
    game: np.ndarray,
    projector: EquilibriumSetProjector,
) -> dict[str, Any]:
    alice_sources = {
        "flat": (CELL_BASELINE, CELL_EVE_STRUCTURED),
        "structured": (CELL_ALICE_STRUCTURED, CELL_BOTH_STRUCTURED),
    }
    eve_sources = {
        "flat": (CELL_BASELINE, CELL_ALICE_STRUCTURED),
        "structured": (CELL_EVE_STRUCTURED, CELL_BOTH_STRUCTURED),
    }
    family_order = ("flat", "structured")
    cells: dict[str, dict[str, Any]] = {}
    matrices = {
        metric: np.zeros((2, 2), dtype=float) for metric in REFERENCE_METRICS
    }
    for alice_index, alice_family in enumerate(family_order):
        cells[alice_family] = {}
        for eve_index, eve_family in enumerate(family_order):
            values = {metric: [] for metric in REFERENCE_METRICS}
            for alice_source in alice_sources[alice_family]:
                for alice_seed in seeds:
                    alice = _probability_vector(
                        by_seed[alice_source][alice_seed]["alice_strategy"],
                        game.shape[0],
                        "Alice",
                    )
                    for eve_source in eve_sources[eve_family]:
                        for eve_seed in seeds:
                            eve = _probability_vector(
                                by_seed[eve_source][eve_seed]["eve_strategy"],
                                game.shape[1],
                                "Eve",
                            )
                            profile = _reference_profile(
                                game,
                                alice,
                                eve,
                                projector,
                            )
                            for metric in REFERENCE_METRICS:
                                values[metric].append(profile[metric])
            summaries = {
                metric: _descriptive_summary(metric_values)
                for metric, metric_values in values.items()
            }
            cells[alice_family][eve_family] = {
                "profile_count": len(next(iter(values.values()))),
                "alice_source_cells": list(alice_sources[alice_family]),
                "eve_source_cells": list(eve_sources[eve_family]),
                "descriptive": summaries,
            }
            for metric in REFERENCE_METRICS:
                matrices[metric][alice_index, eve_index] = summaries[metric]["mean"]
    return {
        "alice_family_order": list(family_order),
        "eve_family_order": list(family_order),
        "cells": cells,
        "mean_matrices": {
            metric: matrix.tolist() for metric, matrix in matrices.items()
        },
        "inference_boundary": (
            "family cells average source-condition and seed cross-products and are "
            "descriptive; the live-cell and paired-contrast sections carry seed "
            "inference"
        ),
    }


def _joint_vs_offline_factorial(
    by_seed: dict[str, dict[int, dict[str, Any]]],
    seeds: Sequence[int],
    game: np.ndarray,
    projector: EquilibriumSetProjector,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        alice_offline = _probability_vector(
            by_seed[CELL_ALICE_STRUCTURED][seed]["alice_strategy"],
            game.shape[0],
            "Alice",
        )
        eve_offline = _probability_vector(
            by_seed[CELL_EVE_STRUCTURED][seed]["eve_strategy"],
            game.shape[1],
            "Eve",
        )
        alice_live = _probability_vector(
            by_seed[CELL_BOTH_STRUCTURED][seed]["alice_strategy"],
            game.shape[0],
            "Alice",
        )
        eve_live = _probability_vector(
            by_seed[CELL_BOTH_STRUCTURED][seed]["eve_strategy"],
            game.shape[1],
            "Eve",
        )
        offline = _reference_profile(
            game,
            alice_offline,
            eve_offline,
            projector,
        )
        live = _reference_profile(
            game,
            alice_live,
            eve_live,
            projector,
        )
        rows.append({
            "seed": int(seed),
            "offline_factorial": offline,
            "live_both_structured": live,
            "live_minus_offline": {
                metric: float(live[metric] - offline[metric])
                for metric in REFERENCE_METRICS
            },
            "strategy_adaptation_total_variation": {
                "alice": float(0.5 * np.sum(np.abs(alice_live - alice_offline))),
                "eve": float(0.5 * np.sum(np.abs(eve_live - eve_offline))),
            },
        })
    return {
        "per_seed": rows,
        "live_minus_offline_summary": {
            metric: _sample_summary([
                row["live_minus_offline"][metric] for row in rows
            ])
            for metric in REFERENCE_METRICS
        },
        "strategy_adaptation_total_variation_summary": {
            player: _sample_summary([
                row["strategy_adaptation_total_variation"][player]
                for row in rows
            ])
            for player in ("alice", "eve")
        },
        "interpretation": (
            "the offline profile combines the Alice-only and Eve-only learned "
            "marginals; the live profile comes from simultaneous structured learning"
        ),
    }


def _probability_vector(
    values: Sequence[float],
    expected_size: int,
    label: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (expected_size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} strategy has the wrong shape or non-finite values")
    if np.any(vector < -1e-12) or not np.isclose(vector.sum(), 1.0, atol=1e-9):
        raise ValueError(f"{label} strategy is not a probability distribution")
    vector = np.maximum(vector, 0.0)
    return vector / vector.sum()


def _strategy_diagnostics(strategy: np.ndarray) -> dict[str, float]:
    positive = strategy[strategy > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "entropy_nats": entropy,
        "effective_action_count": float(math.exp(entropy)),
        "maximum_action_probability": float(np.max(strategy)),
    }


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a nonempty finite vector")
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
    }


def _descriptive_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("descriptive values must be a nonempty finite vector")
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "profile_count": int(array.size),
        "confidence_interval_reported": False,
    }
