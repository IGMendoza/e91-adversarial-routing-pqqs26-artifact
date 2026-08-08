"""Response-aware Stackelberg commitments over physical Eve interventions.

The stationary leader problem in this module is separate from the ordered
Prime-and-Probe controller.  Every leader action remains an existing
histogram-backend Eve intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog


STACKELBERG_COMMITMENT_SCHEMA = "histogram_stackelberg_commitment.v1"
STACKELBERG_ORACLE_SCHEMA = "histogram_stackelberg_oracle.v1"


@dataclass(frozen=True)
class StackelbergCommitmentEvaluation:
    """One commitment evaluated on a declared payoff matrix."""

    alice_expected_rewards: tuple[float, ...]
    alice_best_response_indices: tuple[int, ...]
    eve_expected_utilities_by_route: tuple[float, ...]
    optimistic_eve_value: float
    pessimistic_eve_value: float
    alice_response_margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StackelbergCommitment:
    """One LP solution associated with an intended Alice response route."""

    intended_alice_route_index: int
    probabilities: tuple[float, ...]
    response_certified: bool
    certified_eve_value_lower_bound: float
    planning_evaluation: StackelbergCommitmentEvaluation
    minimum_robust_response_slack: float
    required_response_margin: float = 0.0

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size < 1:
            raise ValueError("commitment probabilities must be a nonempty vector")
        if np.any(probabilities < -1e-12) or not math.isclose(
            float(np.sum(probabilities)),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("commitment probabilities must form a simplex point")
        if not 0 <= self.intended_alice_route_index < len(
            self.planning_evaluation.alice_expected_rewards
        ):
            raise ValueError("intended Alice route is outside the payoff matrix")
        if (
            not math.isfinite(self.required_response_margin)
            or self.required_response_margin < 0.0
        ):
            raise ValueError("required response margin must be finite and nonnegative")

    @property
    def support_size(self) -> int:
        return int(np.sum(np.asarray(self.probabilities) > 1e-12))

    def to_dict(
        self,
        *,
        action_ids: Sequence[str] | None = None,
        route_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        probabilities = np.asarray(self.probabilities, dtype=float)
        payload = {
            "schema": STACKELBERG_COMMITMENT_SCHEMA,
            **asdict(self),
            "support_size": self.support_size,
            "support_indices": [
                int(index) for index in np.flatnonzero(probabilities > 1e-12)
            ],
        }
        if action_ids is not None:
            if len(action_ids) != probabilities.size:
                raise ValueError("action identities do not match commitment")
            payload["support_actions"] = [
                {
                    "action_id": str(action_ids[index]),
                    "probability": float(probabilities[index]),
                }
                for index in np.flatnonzero(probabilities > 1e-12)
            ]
        if route_ids is not None:
            if len(route_ids) != len(
                self.planning_evaluation.alice_expected_rewards
            ):
                raise ValueError("route identities do not match commitment")
            payload["intended_alice_route_id"] = str(
                route_ids[self.intended_alice_route_index]
            )
            payload["planning_alice_best_response_ids"] = [
                str(route_ids[index])
                for index in self.planning_evaluation.alice_best_response_indices
            ]
        return payload


@dataclass(frozen=True)
class StackelbergOracleResult:
    """Complete route-facet search for one leader payoff model."""

    selected: StackelbergCommitment
    candidates: tuple[StackelbergCommitment, ...]
    route_count: int
    action_count: int
    uncertainty_applied: bool
    certified_candidate_count: int

    def to_dict(
        self,
        *,
        action_ids: Sequence[str] | None = None,
        route_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": STACKELBERG_ORACLE_SCHEMA,
            "route_count": self.route_count,
            "action_count": self.action_count,
            "uncertainty_applied": self.uncertainty_applied,
            "certified_candidate_count": self.certified_candidate_count,
            "selected": self.selected.to_dict(
                action_ids=action_ids,
                route_ids=route_ids,
            ),
            "candidates": [
                candidate.to_dict(action_ids=action_ids, route_ids=route_ids)
                for candidate in self.candidates
            ],
        }


def eve_public_payoff_matrix(
    alice_payoff_matrix: np.ndarray,
    action_costs: np.ndarray,
) -> np.ndarray:
    """Return Eve's declared public utility ``1 - Alice reward - action cost``."""

    alice = _validated_alice_matrix(alice_payoff_matrix)
    costs = _validated_action_costs(action_costs, alice.shape[1])
    return 1.0 - alice - costs[np.newaxis, :]


def evaluate_stackelberg_commitment(
    alice_payoff_matrix: np.ndarray,
    action_costs: np.ndarray,
    probabilities: np.ndarray,
    *,
    best_response_tolerance: float = 1e-10,
) -> StackelbergCommitmentEvaluation:
    """Evaluate one leader commitment under both tie-breaking conventions."""

    alice = _validated_alice_matrix(alice_payoff_matrix)
    costs = _validated_action_costs(action_costs, alice.shape[1])
    policy = _validated_policy(probabilities, alice.shape[1])
    if (
        not math.isfinite(best_response_tolerance)
        or best_response_tolerance < 0.0
    ):
        raise ValueError("best_response_tolerance must be finite and nonnegative")
    eve = eve_public_payoff_matrix(alice, costs)
    alice_values = alice @ policy
    maximum = float(np.max(alice_values))
    best_responses = tuple(
        map(
            int,
            np.flatnonzero(
                alice_values >= maximum - float(best_response_tolerance)
            ),
        )
    )
    eve_values = eve @ policy
    tied_eve_values = eve_values[list(best_responses)]
    sorted_alice = np.sort(alice_values)
    response_margin = (
        float(sorted_alice[-1] - sorted_alice[-2])
        if sorted_alice.size > 1
        else math.inf
    )
    return StackelbergCommitmentEvaluation(
        alice_expected_rewards=tuple(map(float, alice_values)),
        alice_best_response_indices=best_responses,
        eve_expected_utilities_by_route=tuple(map(float, eve_values)),
        optimistic_eve_value=float(np.max(tied_eve_values)),
        pessimistic_eve_value=float(np.min(tied_eve_values)),
        alice_response_margin=response_margin,
    )


def solve_stackelberg_oracle(
    alice_payoff_matrix: np.ndarray,
    action_costs: np.ndarray,
    *,
    alice_reward_radii: np.ndarray | None = None,
    minimum_response_margin: float = 0.0,
    best_response_tolerance: float = 1e-10,
) -> StackelbergOracleResult:
    """Solve every Alice-response facet and select Eve's best commitment.

    ``minimum_response_margin`` moves the commitment inside one Alice
    best-response facet: the intended route must exceed every competing route
    by at least that amount.  This is the strict/pessimistic Stackelberg
    construction used to steer a no-regret follower toward a unique response.

    With nonzero reward radii, route ``i`` is certified only when its lower
    expected reward exceeds every competing route's upper expected reward by
    the requested margin.  The LP objective is then a lower bound on Eve's
    public utility under that certified response.  If no response is certified,
    the nominal margin-constrained oracle is returned with
    ``response_certified=False`` rather than replacing the unresolved
    commitment with a no-attack action.
    """

    alice = _validated_alice_matrix(alice_payoff_matrix)
    costs = _validated_action_costs(action_costs, alice.shape[1])
    radii = _validated_radii(alice_reward_radii, alice.shape)
    margin = _validated_response_margin(minimum_response_margin)
    candidates = _solve_route_facets(
        alice,
        costs,
        radii,
        minimum_response_margin=margin,
        best_response_tolerance=best_response_tolerance,
    )
    certified = tuple(
        candidate for candidate in candidates if candidate.response_certified
    )
    selection_pool = certified
    if not selection_pool and np.any(radii > 0.0):
        nominal = _solve_route_facets(
            alice,
            costs,
            np.zeros_like(alice),
            minimum_response_margin=margin,
            best_response_tolerance=best_response_tolerance,
        )
        selection_pool = tuple(
            StackelbergCommitment(
                intended_alice_route_index=row.intended_alice_route_index,
                probabilities=row.probabilities,
                response_certified=False,
                certified_eve_value_lower_bound=row.planning_evaluation.pessimistic_eve_value,
                planning_evaluation=row.planning_evaluation,
                minimum_robust_response_slack=_minimum_robust_slack(
                    alice,
                    radii,
                    row.intended_alice_route_index,
                    np.asarray(row.probabilities),
                ),
                required_response_margin=margin,
            )
            for row in nominal
        )
        candidates = selection_pool
    if not selection_pool:
        raise RuntimeError("no Stackelberg response facet is feasible")
    selected = max(selection_pool, key=_candidate_order_key)
    return StackelbergOracleResult(
        selected=selected,
        candidates=tuple(candidates),
        route_count=alice.shape[0],
        action_count=alice.shape[1],
        uncertainty_applied=bool(np.any(radii > 0.0)),
        certified_candidate_count=len(certified),
    )


def simultaneous_hoeffding_radii(
    observation_counts: np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    """Simultaneous bounded-reward radii for all route/action cells."""

    counts = np.asarray(observation_counts)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("observation counts must be a nonempty matrix")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("observation counts must be nonnegative integers")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    cell_count = counts.size
    logarithm = math.log(2.0 * cell_count / delta)
    radii = np.ones(counts.shape, dtype=float)
    observed = counts > 0
    radii[observed] = np.sqrt(
        logarithm / (2.0 * counts[observed].astype(float))
    )
    return np.minimum(radii, 1.0)


def _solve_route_facets(
    alice: np.ndarray,
    costs: np.ndarray,
    radii: np.ndarray,
    *,
    minimum_response_margin: float,
    best_response_tolerance: float,
) -> tuple[StackelbergCommitment, ...]:
    route_count, action_count = alice.shape
    lower_alice = np.clip(alice - radii, 0.0, 1.0)
    upper_alice = np.clip(alice + radii, 0.0, 1.0)
    candidates = []
    for intended_route in range(route_count):
        competitor_indices = [
            index for index in range(route_count) if index != intended_route
        ]
        differences = np.asarray([
            lower_alice[intended_route] - upper_alice[competitor]
            for competitor in competitor_indices
        ])
        lower_eve = 1.0 - upper_alice[intended_route] - costs
        result = linprog(
            -lower_eve,
            A_ub=-differences if differences.size else None,
            b_ub=(
                -minimum_response_margin
                * np.ones(len(competitor_indices), dtype=float)
                if differences.size else None
            ),
            A_eq=np.ones((1, action_count), dtype=float),
            b_eq=np.ones(1, dtype=float),
            bounds=[(0.0, 1.0)] * action_count,
            method="highs",
        )
        if not result.success:
            continue
        policy = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        policy /= policy.sum()
        evaluation = evaluate_stackelberg_commitment(
            alice,
            costs,
            policy,
            best_response_tolerance=best_response_tolerance,
        )
        if intended_route not in evaluation.alice_best_response_indices:
            continue
        slack = _minimum_robust_slack(
            alice,
            radii,
            intended_route,
            policy,
        )
        candidates.append(StackelbergCommitment(
            intended_alice_route_index=intended_route,
            probabilities=tuple(map(float, policy)),
            response_certified=bool(
                slack >= minimum_response_margin - 1e-9
            ),
            certified_eve_value_lower_bound=float(lower_eve @ policy),
            planning_evaluation=evaluation,
            minimum_robust_response_slack=slack,
            required_response_margin=minimum_response_margin,
        ))
    return tuple(candidates)


def _minimum_robust_slack(
    alice: np.ndarray,
    radii: np.ndarray,
    intended_route: int,
    policy: np.ndarray,
) -> float:
    if alice.shape[0] == 1:
        return math.inf
    lower_target = np.clip(
        alice[intended_route] - radii[intended_route],
        0.0,
        1.0,
    )
    slacks = []
    for competitor in range(alice.shape[0]):
        if competitor == intended_route:
            continue
        upper_competitor = np.clip(
            alice[competitor] + radii[competitor],
            0.0,
            1.0,
        )
        slacks.append(float((lower_target - upper_competitor) @ policy))
    return float(min(slacks))


def _candidate_order_key(
    candidate: StackelbergCommitment,
) -> tuple[float, float, float, float, int]:
    evaluation = candidate.planning_evaluation
    return (
        candidate.certified_eve_value_lower_bound,
        evaluation.pessimistic_eve_value,
        evaluation.optimistic_eve_value,
        evaluation.alice_response_margin,
        -candidate.support_size,
    )


def _validated_alice_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("Alice payoff matrix must be nonempty")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(
        matrix > 1.0
    ):
        raise ValueError("Alice payoff matrix must lie in [0, 1]")
    return matrix


def _validated_action_costs(values: np.ndarray, action_count: int) -> np.ndarray:
    costs = np.asarray(values, dtype=float)
    if costs.shape != (action_count,):
        raise ValueError("action costs do not match the payoff matrix")
    if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("action costs must be finite and nonnegative")
    return costs


def _validated_policy(values: np.ndarray, action_count: int) -> np.ndarray:
    policy = np.asarray(values, dtype=float)
    if policy.shape != (action_count,):
        raise ValueError("commitment does not match the action count")
    if not np.all(np.isfinite(policy)) or np.any(policy < -1e-12):
        raise ValueError("commitment probabilities must be finite and nonnegative")
    total = float(np.sum(policy))
    if total <= 0.0:
        raise ValueError("commitment must have positive mass")
    policy = np.maximum(policy, 0.0)
    policy /= policy.sum()
    return policy


def _validated_radii(
    values: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if values is None:
        return np.zeros(shape, dtype=float)
    radii = np.asarray(values, dtype=float)
    if radii.shape != shape:
        raise ValueError("reward radii do not match the payoff matrix")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("reward radii must be finite and nonnegative")
    return np.minimum(radii, 1.0)


def _validated_response_margin(value: float) -> float:
    margin = float(value)
    if not math.isfinite(margin) or margin < 0.0 or margin > 1.0:
        raise ValueError("minimum response margin must lie in [0, 1]")
    return margin
