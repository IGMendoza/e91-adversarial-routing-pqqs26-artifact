"""Physical Prime/Exploit plan construction for controller-visible routing.

The strong local-access regime observes route-resource reservations passively.
It therefore does not require an intercept/resend or projective operation to
estimate route occupancy. Plans in this module use physical interventions only
for their control objectives: suppressing alternative routes during priming
and attacking the selected target family during exploitation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .prime_probe import route_action_incidence


CONTROLLER_PLAN_SCHEMA = "histogram_stackelberg_controller_plan.v1"
DEFAULT_CONTROLLER_ATTACK_TYPES = (
    "edge_dos",
    "swap_denial",
    "memory_degradation",
)


@dataclass(frozen=True)
class ControllerPrimeExploitPlan:
    """One passive-observation, physical Prime/Exploit plan."""

    plan_id: str
    exploit_action_index: int
    exploit_action_id: str
    exploit_attack_type: str
    exploit_action_cost: float
    target_route_indices: tuple[int, ...]
    target_route_ids: tuple[str, ...]
    alternative_route_indices: tuple[int, ...]
    prime_action_indices: tuple[int, ...]
    prime_action_ids: tuple[str, ...]
    prime_attack_types: tuple[str, ...]
    prime_probabilities: tuple[float, ...]
    prime_physical_loss_floor: float
    exploit_target_service_loss_minimum: float
    exploit_target_eve_advantage_minimum: float
    profitable_target_mass_threshold: float
    selection_score: float

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("controller plan ID must be nonempty")
        if not self.target_route_indices or not self.alternative_route_indices:
            raise ValueError("controller plan requires target and alternative routes")
        if not self.prime_action_indices:
            raise ValueError("controller plan requires at least one prime action")
        probabilities = np.asarray(self.prime_probabilities, dtype=float)
        if probabilities.shape != (len(self.prime_action_indices),):
            raise ValueError("prime probabilities do not match prime actions")
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not math.isclose(
                float(probabilities.sum()),
                1.0,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        ):
            raise ValueError("prime probabilities must form a simplex point")
        if len(self.prime_action_ids) != len(self.prime_action_indices):
            raise ValueError("prime action IDs do not match indices")
        if len(self.prime_attack_types) != len(self.prime_action_indices):
            raise ValueError("prime action families do not match indices")
        if (
            not math.isfinite(self.exploit_action_cost)
            or self.exploit_action_cost < 0.0
        ):
            raise ValueError("exploit action cost must be finite and nonnegative")
        for name, value in (
            ("prime_physical_loss_floor", self.prime_physical_loss_floor),
            (
                "exploit_target_service_loss_minimum",
                self.exploit_target_service_loss_minimum,
            ),
            (
                "exploit_target_eve_advantage_minimum",
                self.exploit_target_eve_advantage_minimum,
            ),
            ("selection_score", self.selection_score),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.profitable_target_mass_threshold)
            or not 0.0 <= self.profitable_target_mass_threshold < 1.0
        ):
            raise ValueError(
                "profitable target-mass threshold must lie in [0, 1)"
            )

    @property
    def target_family_key(self) -> tuple[int, ...]:
        return tuple(self.target_route_indices)

    @property
    def screening_key(self) -> tuple[tuple[int, ...], str]:
        return self.target_family_key, self.exploit_attack_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROLLER_PLAN_SCHEMA,
            **asdict(self),
            "passive_observation": True,
            "active_occupancy_probe_required": False,
            "trigger_semantics": (
                "exploit only when the passive target-family concentration "
                "lower confidence bound exceeds the feedback-derived "
                "profitable target-mass threshold"
            ),
            "profitability_derivation": (
                "non-target exploit utility change is minus the declared "
                "action cost; target utility change uses the minimum feedback-"
                "bank service reduction minus that cost"
            ),
        }


def build_controller_prime_exploit_plans(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    *,
    prime_attack_types: Sequence[str] = DEFAULT_CONTROLLER_ATTACK_TYPES,
    exploit_attack_types: Sequence[str] = DEFAULT_CONTROLLER_ATTACK_TYPES,
    selection_digest_blake2b_128: str | None = None,
) -> tuple[ControllerPrimeExploitPlan, ...]:
    """Enumerate every feasible passive-observation Prime/Exploit plan."""

    validation_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        bank,
        case,
        config,
        selection_digest_blake2b_128=validation_digest,
    )
    prime_families = _validated_attack_types(prime_attack_types)
    exploit_families = _validated_attack_types(exploit_attack_types)
    actions = case.actions(config.game_mode)
    active = route_action_incidence(case, actions)
    reward = bank.feedback.alice_payoff_matrix()
    no_attack_index = bank.action_ids.index("no_attack")
    baseline = reward[:, no_attack_index]
    prime_candidates = tuple(
        index
        for index, action in enumerate(actions)
        if str(action["attack_type"]) in prime_families
    )
    exploit_candidates = tuple(
        index
        for index, action in enumerate(actions)
        if str(action["attack_type"]) in exploit_families
    )
    candidates: list[ControllerPrimeExploitPlan] = []

    for exploit_index in exploit_candidates:
        target = tuple(map(int, np.flatnonzero(active[:, exploit_index])))
        alternatives = tuple(map(int, np.flatnonzero(~active[:, exploit_index])))
        if not target or not alternatives:
            continue
        safe_prime = tuple(
            index
            for index in prime_candidates
            if index != exploit_index
            and not np.any(active[list(target), index])
        )
        if not safe_prime:
            continue
        physical_losses = (
            baseline[list(alternatives), None]
            - reward[np.ix_(alternatives, safe_prime)]
        )
        physical_losses = np.maximum(physical_losses, 0.0)
        useful_columns = np.any(physical_losses > 1e-12, axis=0)
        if not np.any(useful_columns):
            continue
        safe_prime = tuple(
            index
            for index, useful in zip(safe_prime, useful_columns)
            if useful
        )
        physical_losses = physical_losses[:, useful_columns]
        if np.any(np.max(physical_losses, axis=1) <= 1e-12):
            continue
        probabilities, prime_floor = _max_min_physical_loss_distribution(
            physical_losses
        )
        retained = probabilities > 1e-12
        prime_indices = tuple(
            index
            for index, keep in zip(safe_prime, retained)
            if keep
        )
        prime_probabilities = probabilities[retained]
        prime_probabilities /= prime_probabilities.sum()

        cost = float(bank.action_costs[exploit_index])
        target_service_loss = (
            baseline[list(target)] - reward[list(target), exploit_index]
        )
        minimum_service_loss = float(np.min(target_service_loss))
        minimum_eve_advantage = minimum_service_loss - cost
        if minimum_eve_advantage <= 1e-12:
            continue
        profitable_mass = (
            cost / (minimum_eve_advantage + cost)
            if cost > 0.0
            else 0.0
        )
        if not 0.0 <= profitable_mass < 1.0:
            continue
        exploit_action = actions[exploit_index]
        attack_type = str(exploit_action["attack_type"])
        action_id = str(exploit_action["id"])
        selection_score = (
            float(prime_floor)
            * minimum_eve_advantage
            * (1.0 - profitable_mass)
        )
        if selection_score <= 0.0:
            continue
        candidates.append(ControllerPrimeExploitPlan(
            plan_id=f"controller_prime_exploit:{action_id}",
            exploit_action_index=exploit_index,
            exploit_action_id=action_id,
            exploit_attack_type=attack_type,
            exploit_action_cost=cost,
            target_route_indices=target,
            target_route_ids=tuple(bank.route_ids[index] for index in target),
            alternative_route_indices=alternatives,
            prime_action_indices=prime_indices,
            prime_action_ids=tuple(
                str(actions[index]["id"]) for index in prime_indices
            ),
            prime_attack_types=tuple(
                str(actions[index]["attack_type"])
                for index in prime_indices
            ),
            prime_probabilities=tuple(map(float, prime_probabilities)),
            prime_physical_loss_floor=float(prime_floor),
            exploit_target_service_loss_minimum=minimum_service_loss,
            exploit_target_eve_advantage_minimum=minimum_eve_advantage,
            profitable_target_mass_threshold=float(profitable_mass),
            selection_score=selection_score,
        ))

    if not candidates:
        raise ValueError(
            "no physical Prime/Exploit plan satisfies the declared access, "
            "cost, target-safety, and profitability conditions"
        )
    candidates.sort(
        key=lambda plan: (
            plan.selection_score,
            plan.prime_physical_loss_floor,
            plan.exploit_target_eve_advantage_minimum,
            -float(len(plan.target_route_indices)),
            plan.exploit_action_id,
        ),
        reverse=True,
    )
    return tuple(candidates)


def controller_plan_screening_representatives(
    plans: Sequence[ControllerPrimeExploitPlan],
    *,
    maximum_plans: int | None = None,
) -> tuple[ControllerPrimeExploitPlan, ...]:
    """Retain the best plan per target-family and exploit-physics family."""

    values = tuple(plans)
    if not values:
        raise ValueError("at least one controller plan is required")
    if maximum_plans is not None and int(maximum_plans) < 1:
        raise ValueError("maximum_plans must be positive")
    retained: list[ControllerPrimeExploitPlan] = []
    seen: set[tuple[tuple[int, ...], str]] = set()
    for plan in values:
        if plan.screening_key in seen:
            continue
        retained.append(plan)
        seen.add(plan.screening_key)
        if maximum_plans is not None and len(retained) >= int(maximum_plans):
            break
    return tuple(retained)


def controller_target_family_representatives(
    plans: Sequence[ControllerPrimeExploitPlan],
    *,
    maximum_families: int | None = None,
) -> tuple[ControllerPrimeExploitPlan, ...]:
    """Retain the highest-ranked plan for each distinct target route family."""

    values = tuple(plans)
    if not values:
        raise ValueError("at least one controller plan is required")
    if maximum_families is not None and int(maximum_families) < 1:
        raise ValueError("maximum_families must be positive")
    retained: list[ControllerPrimeExploitPlan] = []
    seen: set[tuple[int, ...]] = set()
    for plan in values:
        if plan.target_family_key in seen:
            continue
        retained.append(plan)
        seen.add(plan.target_family_key)
        if (
            maximum_families is not None
            and len(retained) >= int(maximum_families)
        ):
            break
    return tuple(retained)


def controller_plan_family_census(
    plans: Sequence[ControllerPrimeExploitPlan],
) -> dict[str, Any]:
    """Summarize the complete physical plan surface without pruning it."""

    values = tuple(plans)
    by_exploit: dict[str, int] = {}
    by_prime: dict[str, int] = {}
    for plan in values:
        by_exploit[plan.exploit_attack_type] = (
            by_exploit.get(plan.exploit_attack_type, 0) + 1
        )
        for family in set(plan.prime_attack_types):
            by_prime[family] = by_prime.get(family, 0) + 1
    return {
        "plan_count": len(values),
        "distinct_target_route_families": len({
            plan.target_family_key for plan in values
        }),
        "distinct_screening_keys": len({
            plan.screening_key for plan in values
        }),
        "plan_count_by_exploit_attack_type": by_exploit,
        "plan_count_using_prime_attack_type": by_prime,
        "minimum_profitable_target_mass": float(min(
            plan.profitable_target_mass_threshold for plan in values
        )),
        "maximum_profitable_target_mass": float(max(
            plan.profitable_target_mass_threshold for plan in values
        )),
    }


def _validated_attack_types(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(map(str, values)))
    if not result:
        raise ValueError("attack-type set must be nonempty")
    allowed = set(DEFAULT_CONTROLLER_ATTACK_TYPES)
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(
            f"unsupported controller attack types: {sorted(unknown)}"
        )
    return result


def _max_min_physical_loss_distribution(
    physical_losses: np.ndarray,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(physical_losses, dtype=float)
    if (
        matrix.ndim != 2
        or min(matrix.shape) < 1
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
    ):
        raise ValueError(
            "physical-loss matrix must be finite, nonnegative, and nonempty"
        )
    action_count = matrix.shape[1]
    objective = np.zeros(action_count + 1, dtype=float)
    objective[-1] = -1.0
    inequality = np.column_stack((-matrix, np.ones(matrix.shape[0])))
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(matrix.shape[0]),
        A_eq=np.asarray([[*np.ones(action_count), 0.0]]),
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * action_count + [(0.0, None)],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"physical Prime distribution LP failed: {result.message}"
        )
    probabilities = np.maximum(np.asarray(result.x[:-1], dtype=float), 0.0)
    probabilities /= probabilities.sum()
    floor = float(np.min(matrix @ probabilities))
    if floor <= 0.0:
        raise RuntimeError("physical Prime distribution has no positive floor")
    return probabilities, floor
