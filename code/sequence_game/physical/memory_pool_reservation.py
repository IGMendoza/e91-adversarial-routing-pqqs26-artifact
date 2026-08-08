"""Memory-pool reservation model for rolling E91 live-pair inputs.

This module is intentionally standalone. It derives the edge availability and
selected-live-pair age distribution that can be serialized into the histogram
backend's ``rolling_alive_pair`` protocol mode, under the assumptions documented
in docs/E91_MEMORY_POOL_RESERVATION_MODEL.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Sequence


@dataclass(frozen=True)
class EdgeReservationModel:
    """Steady-state reservation model for one elementary edge.

    ``reserved_slots`` is the number of independent memory slots per edge side
    reserved for maintaining accepted elementary entanglement on this edge.
    ``detected_decay_memory_count`` is the number of independently decaying
    memories whose joint survival keeps the slot live; legacy callers retain
    the historical one-memory hazard, while the multiplexed two-ended edge
    calibration sets it to two explicitly.
    """

    success_probability_per_attempt: float
    attempt_period_s: float
    accepted_alive_time_s: float
    detected_decay_lifetime_s: float | None = None
    reserved_slots: int = 1
    selection_policy: str = "youngest_live"
    mean_empty_time_override_s: float | None = None
    initial_memory_age_s: float = 0.0
    detected_decay_memory_count: int = 1

    def __post_init__(self) -> None:
        p = float(self.success_probability_per_attempt)
        if not (0.0 < p <= 1.0):
            raise ValueError("success_probability_per_attempt must be in (0, 1]")
        if float(self.attempt_period_s) <= 0.0:
            raise ValueError("attempt_period_s must be > 0")
        if float(self.accepted_alive_time_s) <= 0.0:
            raise ValueError("accepted_alive_time_s must be > 0")
        if self.detected_decay_lifetime_s is not None and float(self.detected_decay_lifetime_s) <= 0.0:
            raise ValueError("detected_decay_lifetime_s must be > 0 when supplied")
        if int(self.reserved_slots) < 1:
            raise ValueError("reserved_slots must be >= 1")
        if str(self.selection_policy) not in {"youngest_live", "random_live"}:
            raise ValueError("selection_policy must be 'youngest_live' or 'random_live'")
        if self.mean_empty_time_override_s is not None and (
            not math.isfinite(float(self.mean_empty_time_override_s))
            or float(self.mean_empty_time_override_s) <= 0.0
        ):
            raise ValueError("mean_empty_time_override_s must be finite and > 0 when supplied")
        if not math.isfinite(float(self.initial_memory_age_s)) or float(
            self.initial_memory_age_s
        ) < 0.0:
            raise ValueError("initial_memory_age_s must be finite and >= 0")
        if int(self.detected_decay_memory_count) < 1:
            raise ValueError("detected_decay_memory_count must be >= 1")

    def with_slots(self, reserved_slots: int) -> "EdgeReservationModel":
        return replace(self, reserved_slots=int(reserved_slots))


@dataclass(frozen=True)
class RouteReservationPlan:
    reserved_slots: tuple[int, ...]
    edge_availabilities: tuple[float, ...]
    route_availability: float
    target_route_availability: float
    objective: str = "min_total_slots"
    exact: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "reserved_slots": list(self.reserved_slots),
            "edge_availabilities": list(self.edge_availabilities),
            "route_availability": self.route_availability,
            "target_route_availability": self.target_route_availability,
            "objective": self.objective,
            "exact": self.exact,
        }


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _validate_probability_target(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def mean_empty_time_s(edge: EdgeReservationModel) -> float:
    """Mean time a slot spends generating before its next success."""

    if edge.mean_empty_time_override_s is not None:
        return float(edge.mean_empty_time_override_s)
    return float(edge.attempt_period_s) / float(edge.success_probability_per_attempt)


def effective_detected_decay_lifetime_s(edge: EdgeReservationModel) -> float | None:
    """Lifetime of the live edge pair under independent per-memory decay."""

    if edge.detected_decay_lifetime_s is None:
        return None
    return float(edge.detected_decay_lifetime_s) / int(edge.detected_decay_memory_count)


def mean_live_time_s(edge: EdgeReservationModel) -> float:
    """Mean accepted-live duration before stale expiry or detected decay."""

    t_alive = float(edge.accepted_alive_time_s)
    tau = effective_detected_decay_lifetime_s(edge)
    if tau is None:
        return t_alive
    tau_f = float(tau)
    return tau_f * (-math.expm1(-t_alive / tau_f))


def slot_live_probability(edge: EdgeReservationModel) -> float:
    e_live = mean_live_time_s(edge)
    return e_live / (mean_empty_time_s(edge) + e_live)


def edge_live_probability(
    edge: EdgeReservationModel,
    reserved_slots: int | None = None,
) -> float:
    c = int(edge.reserved_slots if reserved_slots is None else reserved_slots)
    if c < 1:
        raise ValueError("reserved_slots must be >= 1")
    a_slot = slot_live_probability(edge)
    return _clamp01(1.0 - (1.0 - a_slot) ** c)


def minimal_slots_for_edge_availability(
    edge: EdgeReservationModel,
    target_availability: float,
) -> int:
    target = _validate_probability_target("target_availability", target_availability)
    if target <= 0.0:
        return 1
    a_slot = slot_live_probability(edge)
    if target >= 1.0:
        if a_slot >= 1.0:
            return 1
        raise ValueError("target_availability=1 cannot be reached with finite slots when a_slot < 1")
    return max(1, math.ceil(math.log1p(-target) / math.log1p(-a_slot)))


def route_live_probability(edges: Sequence[EdgeReservationModel]) -> float:
    probability = 1.0
    for edge in edges:
        probability *= edge_live_probability(edge)
    return _clamp01(probability)


def minimal_slots_for_route_availability(
    edges: Sequence[EdgeReservationModel],
    target_route_availability: float,
    *,
    objective: str = "min_total_slots",
    c_max: int = 64,
) -> RouteReservationPlan:
    """Exact finite discrete search for the minimum-total-slot route plan.

    The dynamic program maximizes sum_i log(A_i(c_i)) for each total slot
    budget. Since c_i is an integer in [1, c_max], this is exact for the
    declared finite search space.  This mathematical optimizer does *not*
    establish that ``c_max`` memories exist in a device.  Production callers
    must obtain that cap from an explicit hardware profile; the repeater-v1
    adapter instead fixes the profile inventory and treats availability as an
    output.
    """

    if objective != "min_total_slots":
        raise ValueError("only objective='min_total_slots' is implemented")
    target = _validate_probability_target("target_route_availability", target_route_availability)
    if c_max < 1:
        raise ValueError("c_max must be >= 1")
    if not edges:
        raise ValueError("at least one edge is required")
    if target <= 0.0:
        slots = tuple(1 for _edge in edges)
        edge_availabilities = tuple(edge_live_probability(edge, 1) for edge in edges)
        return RouteReservationPlan(
            reserved_slots=slots,
            edge_availabilities=edge_availabilities,
            route_availability=math.prod(edge_availabilities),
            target_route_availability=target,
        )
    if target >= 1.0:
        raise ValueError("target_route_availability=1 cannot be guaranteed with finite slots in this model")

    n_edges = len(edges)
    max_total = n_edges * int(c_max)
    log_target = math.log(target)
    log_availability: list[list[float]] = []
    for edge in edges:
        log_availability.append([
            math.log(edge_live_probability(edge, c))
            for c in range(1, int(c_max) + 1)
        ])

    neg_inf = float("-inf")
    dp = [[neg_inf] * (max_total + 1) for _ in range(n_edges + 1)]
    parent: list[list[int | None]] = [[None] * (max_total + 1) for _ in range(n_edges + 1)]
    dp[0][0] = 0.0
    for edge_index in range(n_edges):
        for used in range(edge_index, edge_index * int(c_max) + 1):
            if dp[edge_index][used] == neg_inf:
                continue
            for slots in range(1, int(c_max) + 1):
                total = used + slots
                value = dp[edge_index][used] + log_availability[edge_index][slots - 1]
                if value > dp[edge_index + 1][total] + 1e-15:
                    dp[edge_index + 1][total] = value
                    parent[edge_index + 1][total] = slots

    chosen_total = None
    for total in range(n_edges, max_total + 1):
        if dp[n_edges][total] >= log_target - 1e-15:
            chosen_total = total
            break
    if chosen_total is None:
        best = math.exp(max(dp[n_edges][n_edges:]))
        raise ValueError(
            f"target route availability {target} is unreachable with c_max={c_max}; "
            f"best reachable availability is {best:.17g}"
        )

    slots_out = [0] * n_edges
    cursor = chosen_total
    for edge_index in range(n_edges, 0, -1):
        slots = parent[edge_index][cursor]
        if slots is None:
            raise RuntimeError("internal DP reconstruction failed")
        slots_out[edge_index - 1] = slots
        cursor -= slots
    edge_availabilities = tuple(
        edge_live_probability(edge, slots)
        for edge, slots in zip(edges, slots_out)
    )
    return RouteReservationPlan(
        reserved_slots=tuple(slots_out),
        edge_availabilities=edge_availabilities,
        route_availability=math.prod(edge_availabilities),
        target_route_availability=target,
        objective=objective,
        exact=True,
    )


def live_slot_age_cdf(edge: EdgeReservationModel, t_s: float) -> float:
    t = float(t_s)
    t_alive = float(edge.accepted_alive_time_s)
    if t <= 0.0:
        return 0.0
    if t >= t_alive:
        return 1.0
    tau = effective_detected_decay_lifetime_s(edge)
    if tau is None:
        return t / t_alive
    tau_f = float(tau)
    denom = -math.expm1(-t_alive / tau_f)
    if denom <= 0.0:
        return t / t_alive
    return _clamp01((-math.expm1(-t / tau_f)) / denom)


def live_slot_age_pdf(edge: EdgeReservationModel, t_s: float) -> float:
    t = float(t_s)
    t_alive = float(edge.accepted_alive_time_s)
    if t < 0.0 or t > t_alive:
        return 0.0
    tau = effective_detected_decay_lifetime_s(edge)
    if tau is None:
        return 1.0 / t_alive
    return math.exp(-t / float(tau)) / mean_live_time_s(edge)


def live_slot_age_quantile(edge: EdgeReservationModel, q: float) -> float:
    q = _validate_probability_target("q", q)
    t_alive = float(edge.accepted_alive_time_s)
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return t_alive
    tau = effective_detected_decay_lifetime_s(edge)
    if tau is None:
        return q * t_alive
    tau_f = float(tau)
    denom = -math.expm1(-t_alive / tau_f)
    if denom <= 0.0:
        return q * t_alive
    return -tau_f * math.log1p(-q * denom)


def mean_live_slot_age_s(edge: EdgeReservationModel) -> float:
    t_alive = float(edge.accepted_alive_time_s)
    tau = effective_detected_decay_lifetime_s(edge)
    if tau is None:
        return 0.5 * t_alive
    tau_f = float(tau)
    denom = -math.expm1(-t_alive / tau_f)
    if denom <= 0.0 or t_alive / tau_f < 1e-8:
        return 0.5 * t_alive
    exp_term = math.exp(-t_alive / tau_f)
    return tau_f - (t_alive * exp_term / denom)


def selected_pair_age_cdf(
    edge: EdgeReservationModel,
    t_s: float,
    *,
    reserved_slots: int | None = None,
    selection_policy: str | None = None,
) -> float:
    c = int(edge.reserved_slots if reserved_slots is None else reserved_slots)
    if c < 1:
        raise ValueError("reserved_slots must be >= 1")
    policy = selection_policy or edge.selection_policy
    if policy == "random_live":
        return live_slot_age_cdf(edge, t_s)
    if policy != "youngest_live":
        raise ValueError("selection_policy must be 'youngest_live' or 'random_live'")
    a_slot = slot_live_probability(edge)
    availability = edge_live_probability(edge, c)
    if availability <= 0.0:
        return 0.0
    live_cdf = live_slot_age_cdf(edge, t_s)
    return _clamp01((1.0 - (1.0 - a_slot * live_cdf) ** c) / availability)


def selected_pair_age_quantile(
    edge: EdgeReservationModel,
    q: float,
    *,
    reserved_slots: int | None = None,
    selection_policy: str | None = None,
) -> float:
    q = _validate_probability_target("q", q)
    c = int(edge.reserved_slots if reserved_slots is None else reserved_slots)
    policy = selection_policy or edge.selection_policy
    if policy == "random_live":
        return live_slot_age_quantile(edge, q)
    if policy != "youngest_live":
        raise ValueError("selection_policy must be 'youngest_live' or 'random_live'")
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return float(edge.accepted_alive_time_s)
    a_slot = slot_live_probability(edge)
    availability = edge_live_probability(edge, c)
    live_cdf_value = (1.0 - (1.0 - q * availability) ** (1.0 / c)) / a_slot
    return live_slot_age_quantile(edge, _clamp01(live_cdf_value))


def selected_pair_expected_age_s(
    edge: EdgeReservationModel,
    *,
    reserved_slots: int | None = None,
    selection_policy: str | None = None,
) -> float:
    c = int(edge.reserved_slots if reserved_slots is None else reserved_slots)
    if c < 1:
        raise ValueError("reserved_slots must be >= 1")
    policy = selection_policy or edge.selection_policy
    if policy == "random_live":
        return mean_live_slot_age_s(edge)
    if policy != "youngest_live":
        raise ValueError("selection_policy must be 'youngest_live' or 'random_live'")
    a_slot = slot_live_probability(edge)
    availability = edge_live_probability(edge, c)
    t_alive = float(edge.accepted_alive_time_s)
    if availability <= 0.0:
        return 0.0
    tau = effective_detected_decay_lifetime_s(edge)
    if tau is None or (tau is not None and t_alive / float(tau) < 1e-8):
        integral = t_alive * (1.0 - (1.0 - a_slot) ** (c + 1)) / (a_slot * (c + 1))
    else:
        tau_f = float(tau)
        denom = -math.expm1(-t_alive / tau_f)
        b = a_slot / denom
        alpha = 1.0 - b
        integral = 0.0
        for k in range(c + 1):
            coefficient = math.comb(c, k) * (alpha ** (c - k)) * (b ** k)
            if k == 0:
                integral += coefficient * t_alive
            else:
                integral += coefficient * tau_f * (1.0 - math.exp(-k * t_alive / tau_f)) / k
    numerator = integral - t_alive * ((1.0 - a_slot) ** c)
    return max(0.0, numerator / availability)


def selected_pair_age_quantiles(
    edge: EdgeReservationModel,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, float]:
    return {
        f"q{int(round(100.0 * float(q)))}": selected_pair_age_quantile(edge, float(q))
        for q in quantiles
    }


def discretize_selected_age_distribution(
    edge: EdgeReservationModel,
    *,
    num_points: int | None = None,
    grid: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Return a finite age distribution for ``rolling_alive_pair``.

    With ``num_points``, equal-probability quantile-midpoint support is used.
    With ``grid``, grid entries are interpreted as bin boundaries and weights
    are exact CDF differences; each representative age is the bin's CDF
    midpoint quantile. The resulting finite distribution is an explicit
    discretization of the continuous reservation model.
    """

    if grid is not None and num_points is not None:
        raise ValueError("provide either num_points or grid, not both")
    if grid is not None:
        boundaries = [float(value) for value in grid]
        if len(boundaries) < 2:
            raise ValueError("grid must contain at least two boundaries")
        if abs(boundaries[0]) > 1e-15:
            raise ValueError("grid must start at 0")
        if abs(boundaries[-1] - float(edge.accepted_alive_time_s)) > 1e-12:
            raise ValueError("grid must end at accepted_alive_time_s")
        if any(right < left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("grid boundaries must be nondecreasing")
        ages: list[float] = []
        weights: list[float] = []
        for left, right in zip(boundaries, boundaries[1:]):
            q_left = selected_pair_age_cdf(edge, left)
            q_right = selected_pair_age_cdf(edge, right)
            weight = max(0.0, q_right - q_left)
            if weight <= 0.0:
                continue
            ages.append(selected_pair_age_quantile(edge, 0.5 * (q_left + q_right)))
            weights.append(weight)
        total = math.fsum(weights)
        return ages, [weight / total for weight in weights]
    n = int(num_points or 16)
    if n < 1:
        raise ValueError("num_points must be >= 1")
    weights = [1.0 / n] * n
    ages = [
        selected_pair_age_quantile(edge, (index + 0.5) / n)
        for index in range(n)
    ]
    return ages, weights


def rolling_alive_pair_config_for_homogeneous_edges(
    edges: Sequence[EdgeReservationModel],
    *,
    num_age_points: int = 16,
) -> dict[str, Any]:
    """Serialize exact route availability plus one shared age grid.

    The histogram backend preserves this shared distribution format as a
    backward-compatible alias. Use ``rolling_alive_pair_config_for_edges`` when
    selected-age distributions are heterogeneous.
    """

    if not edges:
        raise ValueError("at least one edge is required")
    first = edges[0]
    signature = _selected_distribution_signature(first)
    if any(_selected_distribution_signature(edge) != signature for edge in edges[1:]):
        raise ValueError(
            "current rolling_alive_pair backend schema accepts one shared age distribution; "
            "heterogeneous edge age distributions require a per-edge age-distribution schema"
        )
    residence_ages, weights = discretize_selected_age_distribution(
        first, num_points=num_age_points
    )
    ages = [float(first.initial_memory_age_s) + age for age in residence_ages]
    return {
        "rolling_edge_availability": [edge_live_probability(edge) for edge in edges],
        "rolling_age_distribution_kind": "discrete",
        "rolling_age_distribution_points_s": ages,
        "rolling_age_distribution_weights": weights,
        "rolling_accepted_alive_time_s": (
            first.accepted_alive_time_s + first.initial_memory_age_s
        ),
        "rolling_selection_policy": first.selection_policy,
        "rolling_age_distribution_source": "memory_pool_reservation_discretized",
        "backend_schema_compatibility": "shared_age_distribution_schema_alias",
        "discretization": {
            "method": "equal_probability_quantile_midpoints",
            "num_points": int(num_age_points),
            "continuous_mean_age_s": (
                first.initial_memory_age_s + selected_pair_expected_age_s(first)
            ),
            "discrete_mean_age_s": math.fsum(age * weight for age, weight in zip(ages, weights)),
        },
    }


def rolling_alive_pair_config_for_edges(
    edges: Sequence[EdgeReservationModel],
    *,
    num_age_points: int = 16,
) -> dict[str, Any]:
    """Serialize heterogeneous per-edge reservation outputs for the backend."""

    if not edges:
        raise ValueError("at least one edge is required")
    edge_points: list[list[float]] = []
    edge_weights: list[list[float]] = []
    edge_summaries: list[dict[str, Any]] = []
    for edge in edges:
        residence_ages, weights = discretize_selected_age_distribution(
            edge, num_points=num_age_points
        )
        ages = [float(edge.initial_memory_age_s) + age for age in residence_ages]
        edge_points.append(ages)
        edge_weights.append(weights)
        edge_summaries.append({
            "reserved_slots": edge.reserved_slots,
            "edge_availability": edge_live_probability(edge),
            "selected_age_mean_s": (
                edge.initial_memory_age_s + selected_pair_expected_age_s(edge)
            ),
            "selected_age_quantiles_s": {
                label: edge.initial_memory_age_s + value
                for label, value in selected_pair_age_quantiles(edge).items()
            },
            "discrete_mean_age_s": math.fsum(age * weight for age, weight in zip(ages, weights)),
        })
        if edge.initial_memory_age_s > 0.0:
            edge_summaries[-1].update({
                "initial_memory_age_s": edge.initial_memory_age_s,
                "selected_live_residence_mean_s": selected_pair_expected_age_s(edge),
                "age_origin": "memory excitation; includes pre-herald storage before pool acceptance",
            })
        if edge.detected_decay_memory_count != 1:
            edge_summaries[-1].update({
                "detected_decay_memory_count": edge.detected_decay_memory_count,
                "effective_live_pair_decay_lifetime_s": (
                    effective_detected_decay_lifetime_s(edge)
                ),
                "detected_decay_semantics": (
                    "independent per-memory detected decay; live-pair survival requires every "
                    "constituent memory to survive"
                ),
            })
    return {
        "rolling_edge_availability": [edge_live_probability(edge) for edge in edges],
        "rolling_age_distribution_kind": "discrete",
        "rolling_age_distribution_points_s": [0.0],
        "rolling_age_distribution_weights": [1.0],
        "rolling_edge_age_distribution_points_s": edge_points,
        "rolling_edge_age_distribution_weights": edge_weights,
        "rolling_accepted_alive_time_s": max(
            edge.accepted_alive_time_s + edge.initial_memory_age_s for edge in edges
        ),
        "rolling_selection_policy": "per_edge_configured_distribution",
        "rolling_age_distribution_source": "memory_pool_reservation_per_edge_discretized",
        "backend_schema_compatibility": "per_edge_age_distribution_schema",
        "discretization": {
            "method": "equal_probability_quantile_midpoints_per_edge",
            "num_points_per_edge": int(num_age_points),
            "support_product": math.prod(len(points) for points in edge_points),
        },
        "edge_summaries": edge_summaries,
    }


def _selected_distribution_signature(edge: EdgeReservationModel) -> tuple[Any, ...]:
    return (
        float(edge.success_probability_per_attempt),
        float(edge.attempt_period_s),
        float(edge.accepted_alive_time_s),
        None if edge.detected_decay_lifetime_s is None else float(edge.detected_decay_lifetime_s),
        int(edge.reserved_slots),
        str(edge.selection_policy),
        None if edge.mean_empty_time_override_s is None else float(edge.mean_empty_time_override_s),
        float(edge.initial_memory_age_s),
        int(edge.detected_decay_memory_count),
    )


def route_reservation_summary(edges: Sequence[EdgeReservationModel]) -> dict[str, Any]:
    def edge_summary(edge: EdgeReservationModel) -> dict[str, Any]:
        payload = {
            "success_probability_per_attempt": edge.success_probability_per_attempt,
            "attempt_period_s": edge.attempt_period_s,
            "accepted_alive_time_s": edge.accepted_alive_time_s,
            "detected_decay_lifetime_s": edge.detected_decay_lifetime_s,
            "reserved_slots": edge.reserved_slots,
            "slot_live_probability": slot_live_probability(edge),
            "edge_live_probability": edge_live_probability(edge),
            "selected_age_mean_s": selected_pair_expected_age_s(edge),
            "selected_age_quantiles_s": selected_pair_age_quantiles(edge),
        }
        if edge.mean_empty_time_override_s is not None:
            payload["mean_empty_time_s"] = mean_empty_time_s(edge)
            payload["mean_empty_time_source"] = "controller_fill_time_override"
        if edge.initial_memory_age_s > 0.0:
            payload["initial_memory_age_s"] = edge.initial_memory_age_s
        if edge.detected_decay_memory_count != 1:
            payload["detected_decay_memory_count"] = edge.detected_decay_memory_count
            payload["effective_live_pair_decay_lifetime_s"] = (
                effective_detected_decay_lifetime_s(edge)
            )
        return payload

    return {
        "edges": [edge_summary(edge) for edge in edges],
        "route_availability": route_live_probability(edges),
    }


def example_route_reservation() -> dict[str, Any]:
    base_edges = [
        EdgeReservationModel(
            success_probability_per_attempt=0.015 + 0.002 * index,
            attempt_period_s=5e-6,
            accepted_alive_time_s=0.004 + 0.0004 * index,
            detected_decay_lifetime_s=0.366,
        )
        for index in range(5)
    ]
    plan = minimal_slots_for_route_availability(base_edges, 0.95, c_max=64)
    planned_edges = [
        edge.with_slots(slots)
        for edge, slots in zip(base_edges, plan.reserved_slots)
    ]
    return {
        "reservation_plan": plan.to_dict(),
        "route_summary": route_reservation_summary(planned_edges),
        "rolling_alive_pair_config": rolling_alive_pair_config_for_edges(
            planned_edges, num_age_points=8
        ),
    }


def main() -> int:
    print(json.dumps(example_route_reservation(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
