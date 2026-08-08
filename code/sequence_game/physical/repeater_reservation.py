"""Bridge a bounded repeater service profile into rolling-pool inputs.

This adapter is intentionally separate from both the literature equations and
the histogram kernel.  It fixes the memory inventory to the profile's declared
hardware; it never solves for an arbitrary number of slots to manufacture a
target route availability.
"""

from __future__ import annotations

from typing import Any, Sequence

from .memory_pool_reservation import (
    EdgeReservationModel,
    rolling_alive_pair_config_for_edges,
    route_reservation_summary,
)
from .repeater_service import (
    RepeaterServiceError,
    RepeaterServiceProfile,
    elementary_link_estimate,
    purified_elementary_link_estimate,
    quality_restoration_model,
    profile_summary,
    route_bell_quality_estimate,
    route_service_estimate,
)


def reservation_edges_for_service_profile(
    profile: RepeaterServiceProfile,
    edge_lengths_km: Sequence[float],
    *,
    accepted_alive_time_s: float,
    selection_policy: str = "youngest_live",
) -> list[EdgeReservationModel]:
    """One fixed-inventory renewal slot model per elementary route edge."""

    if accepted_alive_time_s <= 0.0:
        raise ValueError("accepted_alive_time_s must be > 0")
    physical_slots = int(profile.hardware["memory_slots_per_adjacent_edge_side"])
    slots = int(
        profile.hardware.get(
            "purified_output_slots_per_adjacent_edge_side", physical_slots
        )
    )
    speed = profile.rate_model.get("fiber_speed_m_per_s")
    decay = profile.memory_model.get("detected_decay_lifetime_s")
    restoration = quality_restoration_model(profile)
    edges: list[EdgeReservationModel] = []
    for length_km in edge_lengths_km:
        raw_estimate = elementary_link_estimate(profile, float(length_km))
        purified = purified_elementary_link_estimate(profile, float(length_km))
        estimate = raw_estimate
        if (
            estimate.attempt_period_s is None
            or estimate.success_probability_per_attempt is None
        ):
            if str(profile.rate_model.get("completion_time_model", "")) != (
                "empirical_exponential_renewal"
            ):
                raise RepeaterServiceError(
                    "an empirical rate without attempt-period/success decomposition cannot be "
                    "mapped to a rolling renewal pool unless the source supplies an empirical "
                    "exponential completion-time model"
                )
        # Once a photon is emitted, its stationary partner waits for propagation
        # to the midpoint and for the herald to return: total L/c.  The local
        # generation overhead precedes that stored interval in the source model.
        initial_age = (
            float(length_km) * 1000.0 / float(speed)
            if speed is not None else 0.0
        )
        if restoration is not None:
            if purified is None:
                raise RepeaterServiceError(
                    "quality-restoration estimate construction failed"
                )
            # The younger of the two completed raw pairs is selected as the
            # retained control.  Its physical memory age at pool acceptance is
            # the herald round trip plus local LOCC and the parity-comparison
            # message; the older pair's additional generation wait is already
            # averaged into the purification input channel.
            initial_age += (
                purified.local_operation_duration_s
                + purified.classical_comparison_duration_s
            )
            attempt_period = purified.mean_cycle_duration_s
            success_probability = purified.purification_success_probability
            mean_empty_time = purified.per_lane_mean_distribution_time_s
        else:
            if (
                estimate.attempt_period_s is None
                or estimate.success_probability_per_attempt is None
            ):
                # EdgeReservationModel consumes the mean empty time directly.
                # The p=1 equivalent cycle only satisfies its serialized
                # attempt fields; it is not claimed to be the experiment's
                # microscopic optical attempt cadence.
                attempt_period = estimate.mean_distribution_time_s
                success_probability = 1.0
            else:
                attempt_period = float(estimate.attempt_period_s)
                success_probability = float(estimate.success_probability_per_attempt)
            mean_empty_time = estimate.mean_distribution_time_s
        edges.append(EdgeReservationModel(
            success_probability_per_attempt=success_probability,
            attempt_period_s=attempt_period,
            accepted_alive_time_s=float(accepted_alive_time_s),
            detected_decay_lifetime_s=(float(decay) if decay is not None else None),
            reserved_slots=slots,
            selection_policy=selection_policy,
            mean_empty_time_override_s=mean_empty_time,
            initial_memory_age_s=initial_age,
            detected_decay_memory_count=2,
        ))
    if not edges:
        raise ValueError("at least one elementary edge is required")
    return edges


def rolling_config_for_service_profile(
    profile: RepeaterServiceProfile,
    edge_lengths_km: Sequence[float],
    *,
    accepted_alive_time_s: float,
    selection_policy: str = "youngest_live",
    num_age_points: int = 8,
) -> dict[str, Any]:
    """Histogram rolling payload plus rate, quality, and resource provenance."""

    edges = reservation_edges_for_service_profile(
        profile,
        edge_lengths_km,
        accepted_alive_time_s=accepted_alive_time_s,
        selection_policy=selection_policy,
    )
    payload = rolling_alive_pair_config_for_edges(
        edges,
        num_age_points=num_age_points,
    )
    physical_slots = int(profile.hardware["memory_slots_per_adjacent_edge_side"])
    slots = int(
        profile.hardware.get(
            "purified_output_slots_per_adjacent_edge_side", physical_slots
        )
    )
    restoration = quality_restoration_model(profile)
    payload.update({
        "rolling_memory_slots_per_edge_side": slots,
        "resource_policy": {
            "kind": "fixed_profile_inventory",
            "memory_slots_per_adjacent_edge_side": physical_slots,
            "rolling_output_pair_slots_per_adjacent_edge_side": slots,
            "memory_qubits_per_node": int(profile.hardware["memory_qubits_per_node"]),
            "communication_qubits_per_node": profile.hardware.get(
                "communication_qubits_per_node"
            ),
            "availability_target_optimizer_used": False,
            "semantics": (
                "availability is an output of the declared physical inventory; "
                "slots are not increased to force a target. With purification, "
                "raw-pair workspace and live purified-output capacity are distinct."
            ),
            "empirical_completion_adapter": (
                "mean-only exponential renewal; the serialized p=1 equivalent cycle is "
                "not a microscopic optical-attempt claim"
                if str(profile.rate_model.get("completion_time_model", ""))
                == "empirical_exponential_renewal"
                else None
            ),
        },
        "repeater_service_profile": profile_summary(profile),
        "route_service": route_service_estimate(profile, edge_lengths_km).to_dict(),
        "bell_quality_without_storage": route_bell_quality_estimate(
            profile, len(edge_lengths_km), edge_lengths_km
        ).to_dict(),
        "reservation_summary": route_reservation_summary(edges),
        "quality_restoration": (
            {
                "model": restoration.to_dict(),
                "per_edge": [
                    purified_elementary_link_estimate(profile, value).to_dict()
                    for value in edge_lengths_km
                ],
                "live_output_slots_per_edge_side": slots,
                "physical_raw_workspace_slots_per_edge_side": physical_slots,
            }
            if restoration is not None else None
        ),
    })
    return payload
