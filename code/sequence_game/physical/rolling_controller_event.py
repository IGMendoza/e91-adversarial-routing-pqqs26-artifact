"""Discrete-event reference for the rolling fixed-memory reservation pool.

The histogram backend consumes closed-form steady-state edge availability and
selected-live-pair age distributions from :mod:`memory_pool_reservation`.  This
module realizes the same declared alternating-renewal process with an event
heap.  It is an independent numerical check of those controller inputs at
production hop counts; it is not a second optical/state simulator.

No pair consumption or demand queue is applied in the reference mode.  That is
deliberate: the current Cui composite profile declares
``continuous_demand_queue=false`` and the histogram samples a stationary pool
snapshot per request.  A saturated-service controller would be a different
model and must be introduced explicitly rather than folded into this check.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .memory_pool_reservation import (
    EdgeReservationModel,
    edge_live_probability,
    effective_detected_decay_lifetime_s,
    mean_empty_time_s,
    mean_live_time_s,
    route_live_probability,
    selected_pair_age_quantiles,
    selected_pair_expected_age_s,
)


@dataclass(frozen=True)
class RollingControllerEventConfig:
    """Numerical policy for the event reference (not a physical calibration)."""

    num_samples: int = 100_000
    burn_in_mean_cycles: float = 100.0
    mean_sample_interval_s: float | None = None
    empty_time_distribution: str = "exponential"
    batch_count: int = 20
    seed: int = 20260714

    def __post_init__(self) -> None:
        if int(self.num_samples) < 100:
            raise ValueError("num_samples must be >= 100")
        if float(self.burn_in_mean_cycles) <= 0.0:
            raise ValueError("burn_in_mean_cycles must be > 0")
        if self.mean_sample_interval_s is not None and float(
            self.mean_sample_interval_s
        ) <= 0.0:
            raise ValueError("mean_sample_interval_s must be > 0 when supplied")
        if self.empty_time_distribution not in {"exponential", "geometric_attempts"}:
            raise ValueError(
                "empty_time_distribution must be 'exponential' or 'geometric_attempts'"
            )
        if not 2 <= int(self.batch_count) <= int(self.num_samples):
            raise ValueError("batch_count must lie in [2, num_samples]")


def _mean_cycle_s(edge: EdgeReservationModel) -> float:
    return mean_empty_time_s(edge) + mean_live_time_s(edge)


def _resolved_sample_interval_s(
        edges: Sequence[EdgeReservationModel],
        config: RollingControllerEventConfig,
) -> float:
    if config.mean_sample_interval_s is not None:
        return float(config.mean_sample_interval_s)
    # One quarter of the slowest mean renewal cycle resolves the dynamics while
    # avoiding a prohibitive number of redundant snapshots.
    return 0.25 * max(_mean_cycle_s(edge) for edge in edges)


def rolling_controller_event_resource_estimate(
        edges: Sequence[EdgeReservationModel],
        config: RollingControllerEventConfig,
) -> dict[str, Any]:
    """Conservative memory/event preflight for one sequential validation case."""

    if not edges:
        raise ValueError("at least one edge is required")
    interval = _resolved_sample_interval_s(edges, config)
    burn_in_s = float(config.burn_in_mean_cycles) * max(
        _mean_cycle_s(edge) for edge in edges
    )
    sample_horizon_s = int(config.num_samples) * interval
    total_slots = sum(int(edge.reserved_slots) for edge in edges)
    expected_events = 0.0
    for edge in edges:
        cycles = (burn_in_s + sample_horizon_s) / _mean_cycle_s(edge)
        expected_events += 2.0 * int(edge.reserved_slots) * cycles
    # Main arrays: float64 selected age per edge/sample, route bools, edge bools,
    # plus a factor-two allowance for temporary reductions and quantiles.
    array_bytes = (
        int(config.num_samples) * len(edges) * 8
        + int(config.num_samples) * len(edges)
        + int(config.num_samples)
    )
    estimated_peak_bytes = 2 * array_bytes + total_slots * 512
    return {
        "edge_count": len(edges),
        "total_slots": total_slots,
        "event_heap_upper_bound_entries": total_slots,
        "mean_sample_interval_s": interval,
        "burn_in_s": burn_in_s,
        "sample_horizon_s_mean": sample_horizon_s,
        "expected_processed_events": int(math.ceil(expected_events)),
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "estimated_peak_mib": estimated_peak_bytes / (1024.0 ** 2),
        "route_attack_cache_entries": 0,
        "cache_semantics": "no histogram or route-attack cache in this event reference",
    }


def _batch_statistics(values: np.ndarray, batch_count: int) -> dict[str, Any]:
    chunks = [chunk for chunk in np.array_split(values, int(batch_count)) if chunk.size]
    means = np.asarray([float(np.mean(chunk)) for chunk in chunks])
    estimate = float(np.mean(values))
    standard_error = (
        float(np.std(means, ddof=1) / math.sqrt(means.size))
        if means.size > 1 else None
    )
    interval = (
        [
            max(0.0, estimate - 1.96 * standard_error),
            min(1.0, estimate + 1.96 * standard_error),
        ]
        if standard_error is not None else None
    )
    return {
        "estimate": estimate,
        "batch_standard_error": standard_error,
        "confidence_interval_95_normal_batch_means": interval,
        "batch_count": len(chunks),
    }


def _batch_mean_standard_error(values: np.ndarray, batch_count: int) -> float | None:
    chunks = [chunk for chunk in np.array_split(values, int(batch_count)) if chunk.size]
    means = np.asarray([float(np.mean(chunk)) for chunk in chunks])
    return (
        float(np.std(means, ddof=1) / math.sqrt(means.size))
        if means.size > 1 else None
    )


def simulate_rolling_controller_snapshots(
        edges: Sequence[EdgeReservationModel],
        config: RollingControllerEventConfig = RollingControllerEventConfig(),
) -> dict[str, Any]:
    """Run the event heap and compare sampled stationary outputs to analytics."""

    edges = tuple(edges)
    if not edges:
        raise ValueError("at least one edge is required")
    resource_estimate = rolling_controller_event_resource_estimate(edges, config)
    seed_sequence = np.random.SeedSequence(int(config.seed))
    event_seed, observation_seed, selection_seed = seed_sequence.spawn(3)
    event_rng = np.random.default_rng(event_seed)
    observation_rng = np.random.default_rng(observation_seed)
    selection_rng = np.random.default_rng(selection_seed)

    live_since: list[list[float | None]] = [
        [None] * int(edge.reserved_slots) for edge in edges
    ]
    events: list[tuple[float, int, str, int, int]] = []
    serial = 0

    def empty_duration(edge: EdgeReservationModel) -> float:
        if config.empty_time_distribution == "exponential":
            return max(float(event_rng.exponential(mean_empty_time_s(edge))), 1e-15)
        if edge.mean_empty_time_override_s is not None:
            raise ValueError(
                "geometric_attempts cannot reproduce a mean_empty_time_override; "
                "use the profile-declared exponential renewal"
            )
        attempts = int(event_rng.geometric(edge.success_probability_per_attempt))
        return attempts * float(edge.attempt_period_s)

    def live_duration(edge: EdgeReservationModel) -> float:
        tau = effective_detected_decay_lifetime_s(edge)
        if tau is None:
            return float(edge.accepted_alive_time_s)
        decay = float(event_rng.exponential(tau))
        return min(float(edge.accepted_alive_time_s), max(decay, 1e-15))

    def schedule(time_s: float, kind: str, edge_index: int, slot_index: int) -> None:
        nonlocal serial
        serial += 1
        heapq.heappush(events, (time_s, serial, kind, edge_index, slot_index))

    for edge_index, edge in enumerate(edges):
        for slot_index in range(int(edge.reserved_slots)):
            schedule(empty_duration(edge), "refill", edge_index, slot_index)

    processed_events = 0

    def advance(until_s: float) -> None:
        nonlocal processed_events
        while events and events[0][0] <= until_s:
            event_time, _order, kind, edge_index, slot_index = heapq.heappop(events)
            edge = edges[edge_index]
            processed_events += 1
            if kind == "refill":
                live_since[edge_index][slot_index] = event_time
                schedule(
                    event_time + live_duration(edge),
                    "expire", edge_index, slot_index,
                )
            else:
                live_since[edge_index][slot_index] = None
                schedule(
                    event_time + empty_duration(edge),
                    "refill", edge_index, slot_index,
                )

    burn_in_s = float(resource_estimate["burn_in_s"])
    advance(burn_in_s)
    sample_count = int(config.num_samples)
    edge_live_samples = np.zeros((sample_count, len(edges)), dtype=bool)
    selected_residence_ages = np.full(
        (sample_count, len(edges)), np.nan, dtype=float
    )
    route_live_samples = np.zeros(sample_count, dtype=bool)
    sample_time = burn_in_s
    interval = float(resource_estimate["mean_sample_interval_s"])

    for sample_index in range(sample_count):
        sample_time += max(float(observation_rng.exponential(interval)), 1e-15)
        advance(sample_time)
        route_live = True
        for edge_index, edge in enumerate(edges):
            ages = [
                sample_time - start
                for start in live_since[edge_index]
                if start is not None
            ]
            if not ages:
                route_live = False
                continue
            edge_live_samples[sample_index, edge_index] = True
            if edge.selection_policy == "youngest_live":
                selected = min(ages)
            elif edge.selection_policy == "random_live":
                selected = ages[int(selection_rng.integers(0, len(ages)))]
            else:  # guarded by EdgeReservationModel, retained defensively
                raise ValueError(f"unknown selection policy {edge.selection_policy!r}")
            selected_residence_ages[sample_index, edge_index] = selected
        route_live_samples[sample_index] = route_live

    route_stats = _batch_statistics(route_live_samples.astype(float), config.batch_count)
    analytic_route = route_live_probability(edges)
    route_stats.update({
        "analytic": analytic_route,
        "absolute_error": abs(route_stats["estimate"] - analytic_route),
    })
    edge_rows: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        availability_stats = _batch_statistics(
            edge_live_samples[:, edge_index].astype(float), config.batch_count
        )
        analytic_availability = edge_live_probability(edge)
        availability_stats.update({
            "analytic": analytic_availability,
            "absolute_error": abs(
                availability_stats["estimate"] - analytic_availability
            ),
        })
        observed = selected_residence_ages[:, edge_index]
        observed = observed[np.isfinite(observed)]
        analytic_age_mean = selected_pair_expected_age_s(edge)
        empirical_age_mean = float(np.mean(observed))
        analytic_quantiles = selected_pair_age_quantiles(edge)
        empirical_quantile_values = np.quantile(observed, [0.1, 0.5, 0.9])
        age_standard_error = _batch_mean_standard_error(observed, config.batch_count)
        edge_rows.append({
            "edge_index": edge_index,
            "reserved_slots": int(edge.reserved_slots),
            "availability": availability_stats,
            "selected_residence_age_s": {
                "sample_count": int(observed.size),
                "empirical_mean": empirical_age_mean,
                "analytic_mean": analytic_age_mean,
                "absolute_mean_error": abs(empirical_age_mean - analytic_age_mean),
                "batch_standard_error_of_mean": age_standard_error,
                "empirical_quantiles": dict(zip(
                    ("q10", "q50", "q90"),
                    map(float, empirical_quantile_values),
                )),
                "analytic_quantiles": analytic_quantiles,
                "initial_memory_age_s": float(edge.initial_memory_age_s),
                "physical_age_mean_including_initial_s": (
                    float(edge.initial_memory_age_s) + empirical_age_mean
                ),
            },
        })

    return {
        "schema": "rolling_controller_event_reference.v1",
        "process": {
            "controller": "independent fixed-slot alternating renewal pools",
            "empty_time_distribution": config.empty_time_distribution,
            "live_time_distribution": (
                "min(deterministic_staleness_cutoff, exponential_detected_decay)"
            ),
            "selection_policy_per_edge": [edge.selection_policy for edge in edges],
            "demand_queue": False,
            "pair_consumption": False,
            "snapshot_arrivals": "independent Poisson observation process",
        },
        "config": {
            "num_samples": sample_count,
            "burn_in_mean_cycles": float(config.burn_in_mean_cycles),
            "mean_sample_interval_s": interval,
            "batch_count": int(config.batch_count),
            "seed": int(config.seed),
        },
        "resource_estimate": resource_estimate,
        "events_processed": processed_events,
        "final_event_heap_entries": len(events),
        "route_availability": route_stats,
        "edges": edge_rows,
        "claim_boundary": (
            "independent event realization of the declared stationary rolling-pool "
            "availability/age process; not optical-state propagation, swap execution, "
            "or a continuous-demand service-queue validation"
        ),
    }
