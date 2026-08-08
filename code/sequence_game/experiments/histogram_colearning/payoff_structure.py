"""Post-run bilinear structure audits for histogram payoff games."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from sequence_game.game_theory import exploitability, solve_zero_sum_lp

from .corpus import HistogramGameCase
from .structure import (
    EVE_FEATURE_ACTION_IDENTITY,
    EVE_FEATURE_FAMILY,
    EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY,
    EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
    EVE_FEATURE_MODES,
    eve_action_embedding,
    route_action_incidence,
    route_resource_embedding,
    validate_eve_action_contract,
)


def analyze_payoff_structure(
    *,
    payoff: np.ndarray,
    cell_standard_errors: np.ndarray,
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Fit nested two-sided models and measure equilibrium distortion."""

    utility = np.asarray(payoff, dtype=float)
    standard_errors = np.asarray(cell_standard_errors, dtype=float)
    expected_shape = (len(case.routes), len(actions))
    if utility.shape != expected_shape:
        raise ValueError(
            f"payoff shape {utility.shape} does not match routes/actions {expected_shape}"
        )
    if not np.all(np.isfinite(utility)):
        raise ValueError("payoff audit must contain a finite value for every cell")
    if standard_errors.shape != expected_shape:
        raise ValueError("cell_standard_errors has the wrong shape")
    if np.any(np.isfinite(standard_errors) & (standard_errors < 0.0)):
        raise ValueError("cell standard errors must be nonnegative")

    normalized_actions = validate_eve_action_contract(case, actions)
    incidence = route_action_incidence(case, normalized_actions)
    alice_embedding = route_resource_embedding(case)
    eve_embeddings = {
        mode: eve_action_embedding(case, normalized_actions, mode=mode)
        for mode in EVE_FEATURE_MODES
    }
    costs = np.asarray([row["cost"] for row in normalized_actions], dtype=float)
    families = len({
        row["attack_type"]
        for row in normalized_actions
        if row["attack_type"] != "no_attack"
    })

    predictions: dict[str, np.ndarray] = {
        "route_intercepts_only": _route_intercept_model(utility),
        "route_plus_attack_family_incidence": _route_family_model(
            utility, incidence, normalized_actions
        ),
        "route_plus_target_action_incidence": _route_action_model(
            utility, incidence
        ),
        "resource_incidence_linear_per_action": (
            alice_embedding.embedding @ alice_embedding.embedding.T @ utility
        ),
    }
    parameter_counts = {
        "route_intercepts_only": len(case.routes),
        "route_plus_attack_family_incidence": len(case.routes) + families,
        "route_plus_target_action_incidence": len(case.routes) + max(0, len(actions) - 1),
        "resource_incidence_linear_per_action": alice_embedding.rank * len(actions),
    }
    bilinear_modes = {
        "bilinear_family": EVE_FEATURE_FAMILY,
        "bilinear_family_target_topology": EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
        "bilinear_family_by_target_topology": EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY,
        "bilinear_action_identity_ceiling": EVE_FEATURE_ACTION_IDENTITY,
    }
    for model_name, mode in bilinear_modes.items():
        eve_embedding = eve_embeddings[mode]
        predictions[model_name] = _bilinear_projection(
            utility,
            alice_embedding.embedding,
            eve_embedding.embedding,
        )
        parameter_counts[model_name] = alice_embedding.rank * eve_embedding.rank

    diagnostics = {
        name: _model_diagnostics(
            truth=utility,
            prediction=prediction,
            standard_errors=standard_errors,
            action_costs=costs,
            parameter_count=parameter_counts[name],
        )
        for name, prediction in predictions.items()
    }
    route_groups = [
        (str(route["route_id"]), np.asarray([index], dtype=int))
        for index, route in enumerate(case.routes)
    ]
    target_groups = [
        (
            target,
            np.asarray(
                [
                    index
                    for index, action in enumerate(normalized_actions)
                    if action["target"] == target
                ],
                dtype=int,
            ),
        )
        for target in sorted({
            row["target"] for row in normalized_actions if row["target"]
        })
    ]
    for model_name, mode in bilinear_modes.items():
        eve_embedding = eve_embeddings[mode]
        diagnostics[model_name]["bilinear"] = {
            "alice_feature_rank": alice_embedding.rank,
            "eve_feature_mode": mode,
            "eve_feature_rank": eve_embedding.rank,
            "coefficient_matrix_shape": [alice_embedding.rank, eve_embedding.rank],
            "leave_one_route_out": _bilinear_axis_holdout(
                truth=utility,
                standard_errors=standard_errors,
                alice_features=alice_embedding.embedding,
                eve_features=eve_embedding.embedding,
                groups=route_groups,
                axis=0,
            ),
            "leave_one_target_out": _bilinear_axis_holdout(
                truth=utility,
                standard_errors=standard_errors,
                alice_features=alice_embedding.embedding,
                eve_features=eve_embedding.embedding,
                groups=target_groups,
                axis=1,
            ),
        }

    active_support = incidence.sum(axis=0)
    attack_mask = np.asarray([
        action["attack_type"] != "no_attack" for action in normalized_actions
    ])
    attacked_support = active_support[attack_mask]
    return {
        "schema": "histogram_payoff_structure.v2",
        "case": case.to_dict(),
        "matrix_shape": list(utility.shape),
        "route_universe": dict(case.route_universe),
        "route_resource_embedding": alice_embedding.to_dict(),
        "eve_action_embeddings": {
            mode: embedding.to_dict() for mode, embedding in eve_embeddings.items()
        },
        "eve_feature_contract": {
            "action_fields": ["id", "attack_type", "kind", "target", "cost"],
            "target_topology_fields": [
                "target kind",
                "route support",
                "edge length",
                "node or endpoint degree",
                "per-route target incidence",
                "per-route normalized distance from Alice",
            ],
            "selected_alice_route_in_features": False,
            "payoff_observations_in_features": False,
            "action_cost_treatment": (
                "excluded from the public-utility fit and added only as the "
                "declared Eve column cost in equivalent-game diagnostics"
            ),
        },
        "incidence": {
            "active_cell_fraction": float(np.mean(incidence)),
            "attack_action_count": int(np.sum(attack_mask)),
            "actions_shared_by_at_least_two_routes": int(np.sum(attacked_support >= 2)),
            "maximum_route_support_per_action": int(attacked_support.max(initial=0)),
        },
        "models": diagnostics,
        "interpretation_boundary": (
            "post-run structural adequacy audit over an independently sampled "
            "payoff matrix; no fitted value or held-out cell is supplied to either "
            "online learner"
        ),
    }

def _bilinear_projection(
    utility: np.ndarray,
    alice_features: np.ndarray,
    eve_features: np.ndarray,
) -> np.ndarray:
    coefficients = alice_features.T @ utility @ eve_features
    return alice_features @ coefficients @ eve_features.T


def _bilinear_axis_holdout(
    *,
    truth: np.ndarray,
    standard_errors: np.ndarray,
    alice_features: np.ndarray,
    eve_features: np.ndarray,
    groups: Sequence[tuple[str, np.ndarray]],
    axis: int,
) -> dict[str, Any]:
    if axis not in (0, 1):
        raise ValueError("holdout axis must be 0 or 1")
    held_truth: list[np.ndarray] = []
    held_prediction: list[np.ndarray] = []
    held_baseline: list[np.ndarray] = []
    held_standard_errors: list[np.ndarray] = []
    group_rows: list[dict[str, Any]] = []
    full_indices = np.arange(truth.shape[axis], dtype=int)
    for group_label, test_indices in groups:
        test = np.unique(np.asarray(test_indices, dtype=int))
        train = np.setdiff1d(full_indices, test, assume_unique=True)
        if test.size < 1 or train.size < 1:
            raise ValueError(f"invalid holdout group {group_label!r}")
        if axis == 0:
            alice_train = alice_features[train]
            alice_pinv, training_rank = _numerical_pinv(alice_train)
            eve_transpose_pinv, _eve_rank = _numerical_pinv(eve_features.T)
            coefficients = (
                alice_pinv
                @ truth[train, :]
                @ eve_transpose_pinv
            )
            prediction = alice_features[test] @ coefficients @ eve_features.T
            observed = truth[test, :]
            errors = standard_errors[test, :]
            baseline = np.repeat(
                truth[train, :].mean(axis=0, keepdims=True), test.size, axis=0
            )
            identifiable = _row_span_identifiable_energy(
                alice_features[test], alice_train
            )
        else:
            eve_train = eve_features[train]
            alice_pinv, _alice_rank = _numerical_pinv(alice_features)
            eve_transpose_pinv, training_rank = _numerical_pinv(eve_train.T)
            coefficients = (
                alice_pinv
                @ truth[:, train]
                @ eve_transpose_pinv
            )
            prediction = alice_features @ coefficients @ eve_features[test].T
            observed = truth[:, test]
            errors = standard_errors[:, test]
            baseline = np.repeat(
                truth[:, train].mean(axis=1, keepdims=True), test.size, axis=1
            )
            identifiable = _row_span_identifiable_energy(
                eve_features[test], eve_train
            )
        residual = observed - prediction
        group_rows.append({
            "group": group_label,
            "heldout_index_count": int(test.size),
            "heldout_cell_count": int(observed.size),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "baseline_rmse": float(np.sqrt(np.mean((observed - baseline) ** 2))),
            "mean_feature_identifiable_energy": float(np.mean(identifiable)),
            "minimum_feature_identifiable_energy": float(np.min(identifiable)),
            "training_feature_rank": int(training_rank["rank"]),
            "training_feature_dimension": int(training_rank["maximum_rank"]),
            "training_feature_rank_loss": int(
                training_rank["maximum_rank"] - training_rank["rank"]
            ),
            "training_feature_condition_number": training_rank["condition_number"],
            "numerical_rank_tolerance": training_rank["tolerance"],
        })
        held_truth.append(observed.ravel())
        held_prediction.append(prediction.ravel())
        held_baseline.append(baseline.ravel())
        held_standard_errors.append(errors.ravel())
    observed_all = np.concatenate(held_truth)
    predicted_all = np.concatenate(held_prediction)
    baseline_all = np.concatenate(held_baseline)
    errors_all = np.concatenate(held_standard_errors)
    residual_all = observed_all - predicted_all
    model_rmse = float(np.sqrt(np.mean(residual_all ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((observed_all - baseline_all) ** 2)))
    finite_se = np.isfinite(errors_all) & (errors_all > 0.0)
    rms_se = (
        float(np.sqrt(np.mean(errors_all[finite_se] ** 2)))
        if np.any(finite_se) else None
    )
    identifiable_values = np.asarray([
        row["mean_feature_identifiable_energy"] for row in group_rows
    ])
    return {
        "group_count": len(group_rows),
        "heldout_cell_count": int(observed_all.size),
        "rmse": model_rmse,
        "mae": float(np.mean(np.abs(residual_all))),
        "baseline_axis_mean_rmse": baseline_rmse,
        "relative_rmse_reduction_vs_axis_mean": (
            float(1.0 - model_rmse / baseline_rmse) if baseline_rmse > 0.0 else None
        ),
        "rms_cell_standard_error": rms_se,
        "rmse_to_rms_cell_standard_error": (
            float(model_rmse / rms_se) if rms_se not in (None, 0.0) else None
        ),
        "mean_group_feature_identifiable_energy": float(np.mean(identifiable_values)),
        "fully_identifiable_group_fraction": float(
            np.mean(identifiable_values >= 1.0 - 1e-10)
        ),
        "groups": group_rows,
    }


def _row_span_identifiable_energy(
    test_features: np.ndarray,
    train_features: np.ndarray,
) -> np.ndarray:
    train_pinv, _rank = _numerical_pinv(train_features)
    projector = train_pinv @ train_features
    projected = test_features @ projector
    denominator = np.sum(test_features ** 2, axis=1)
    numerator = np.sum(projected ** 2, axis=1)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return np.clip(ratio, 0.0, 1.0)


def _numerical_pinv(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Truncated-SVD inverse with a retained-subspace condition-number cap.

    Axis holdouts can make an otherwise well-conditioned topology feature
    basis numerically rank deficient. Retaining directions at machine-epsilon
    scale produces unbounded extrapolations from bounded payoff matrices. The
    square-root epsilon cutoff caps the retained spectral condition number at
    ``1 / sqrt(machine epsilon)``.
    """

    values = np.asarray(matrix, dtype=float)
    left, singular_values, right = np.linalg.svd(values, full_matrices=False)
    relative_tolerance = float(np.sqrt(np.finfo(float).eps))
    tolerance = (
        relative_tolerance * singular_values[0]
        if singular_values.size else 0.0
    )
    retained = singular_values > tolerance
    rank = int(np.sum(retained))
    inverse = (
        (right[retained].T / singular_values[retained]) @ left[:, retained].T
        if rank
        else np.zeros((values.shape[1], values.shape[0]), dtype=float)
    )
    condition_number = (
        float(singular_values[0] / singular_values[retained][-1])
        if rank else None
    )
    return inverse, {
        "rank": rank,
        "maximum_rank": int(min(values.shape)),
        "tolerance": float(tolerance),
        "relative_tolerance": relative_tolerance,
        "rank_rule": (
            "retain singular values greater than "
            "sqrt(machine_epsilon) * largest_singular_value"
        ),
        "condition_number": condition_number,
    }


def _route_intercept_model(utility: np.ndarray) -> np.ndarray:
    return np.repeat(utility.mean(axis=1, keepdims=True), utility.shape[1], axis=1)


def _route_family_model(
    utility: np.ndarray,
    incidence: np.ndarray,
    actions: Sequence[dict[str, Any]],
) -> np.ndarray:
    routes, action_count = utility.shape
    families = sorted({
        str(action.get("attack_type") or "no_attack")
        for action in actions
        if str(action.get("attack_type") or "no_attack") != "no_attack"
    })
    design = np.zeros((routes * action_count, routes + len(families)), dtype=float)
    family_index = {family: index for index, family in enumerate(families)}
    for route_index in range(routes):
        for action_index, action in enumerate(actions):
            row = route_index * action_count + action_index
            design[row, route_index] = 1.0
            if incidence[route_index, action_index]:
                family = str(action.get("attack_type") or "no_attack")
                design[row, routes + family_index[family]] = 1.0
    coefficients, *_ = np.linalg.lstsq(design, utility.ravel(), rcond=None)
    return (design @ coefficients).reshape(utility.shape)


def _route_action_model(utility: np.ndarray, incidence: np.ndarray) -> np.ndarray:
    routes, actions = utility.shape
    design = np.zeros((routes * actions, routes + max(0, actions - 1)), dtype=float)
    for route_index in range(routes):
        for action_index in range(actions):
            row = route_index * actions + action_index
            design[row, route_index] = 1.0
            if action_index > 0 and incidence[route_index, action_index]:
                design[row, routes + action_index - 1] = 1.0
    coefficients, *_ = np.linalg.lstsq(design, utility.ravel(), rcond=None)
    return (design @ coefficients).reshape(utility.shape)


def _model_diagnostics(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    standard_errors: np.ndarray,
    action_costs: np.ndarray,
    parameter_count: int,
) -> dict[str, Any]:
    residual = truth - prediction
    squared = float(np.sum(residual ** 2))
    centered = float(np.sum((truth - truth.mean()) ** 2))
    finite_se = np.isfinite(standard_errors) & (standard_errors > 0.0)
    rms_se = (
        float(np.sqrt(np.mean(standard_errors[finite_se] ** 2)))
        if np.any(finite_se) else None
    )
    within_two_se = (
        float(np.mean(np.abs(residual[finite_se]) <= 2.0 * standard_errors[finite_se]))
        if np.any(finite_se) else None
    )

    true_game = truth + action_costs[np.newaxis, :]
    predicted_game = prediction + action_costs[np.newaxis, :]
    true_value, true_alice, true_eve = solve_zero_sum_lp(true_game)
    predicted_value, predicted_alice, predicted_eve = solve_zero_sum_lp(predicted_game)
    return {
        "parameter_count": int(parameter_count),
        "cell_count": int(truth.size),
        "parameter_to_cell_ratio": float(parameter_count / truth.size),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "r_squared": float(1.0 - squared / centered) if centered > 0.0 else None,
        "rms_cell_standard_error": rms_se,
        "rmse_to_rms_cell_standard_error": (
            float(np.sqrt(np.mean(residual ** 2)) / rms_se)
            if rms_se not in (None, 0.0) else None
        ),
        "fraction_residuals_within_two_cell_standard_errors": within_two_se,
        "true_equivalent_game_value": float(true_value),
        "predicted_equivalent_game_value": float(predicted_value),
        "absolute_equilibrium_value_error": float(abs(predicted_value - true_value)),
        "predicted_equilibrium_exploitability_in_true_game": float(
            exploitability(true_game, predicted_alice, predicted_eve)
        ),
        "true_equilibrium_exploitability_in_predicted_game": float(
            exploitability(predicted_game, true_alice, true_eve)
        ),
        "strategy_distance_warning": (
            "equilibrium strategy distances are omitted because redundant "
            "route/target actions make LP optima non-unique"
        ),
    }
