"""Memory storage, physical waiting-time, and route-age models."""

from __future__ import annotations

import copy
import itertools
import math
import time
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from sequence_game.physical.empirical_bell import (
    pauli_probabilities_from_eigenvalues,
    symmetric_local_liu_storage_eigenvalues,
)

from sequence_game.physical.multiplexing_controller import (
    MultiplexingController,
    controller_from_config,
)
from sequence_game.physical.repeater_service import (
    profile_summary as repeater_profile_summary,
    quality_restoration_model,
    route_bell_quality_estimate,
    route_service_estimate,
)

from .devices import *
from .models import *
from .quantum import *
def memory_side_qubits(side: str) -> tuple[int, ...]:
    if side == "left":
        return (1,)
    if side == "right":
        return (2,)
    if side == "both":
        return (1, 2)
    raise ValueError("memory_side must be left, right, or both")


def _finite_positive_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if value_f <= 0.0:
        return None
    return value_f


def memory_lifetime_parameters(front: dict[str, Any]) -> dict[str, Any]:
    memory = front["device_constants"]["memory"]
    storage_model = memory_storage_decoherence_model(front)
    kind = storage_model.get("kind", "none")
    t1 = None
    t2 = None
    t1_source = "not_provided"
    t2_source = "not_provided"
    coherence_time = None
    coherence_source = storage_model.get("source_field", "not_provided")
    if kind == "t1_t2":
        t1 = _finite_positive_or_none(storage_model.get("t1_s"))
        t2 = _finite_positive_or_none(storage_model.get("t2_s"))
        t1_source = f"{storage_model.get('source_field', 'memory_storage_decoherence_model')}.t1_s"
        t2_source = f"{storage_model.get('source_field', 'memory_storage_decoherence_model')}.t2_s"
    elif kind in {"exponential_dephasing", "gaussian_dephasing"}:
        coherence_time = _finite_positive_or_none(storage_model.get("coherence_time_s"))
        t2 = coherence_time if kind == "exponential_dephasing" else None
        t2_source = (
            f"{storage_model.get('source_field', 'memory_storage_decoherence_model')}.coherence_time_s"
            if kind == "exponential_dephasing" and coherence_time is not None else
            "not_used_by_gaussian_storage_model"
        )
    return {
        "storage_model_kind": kind,
        "storage_model_source_field": storage_model.get("source_field", "not_provided"),
        "storage_model_enters_probability": bool(storage_model.get("enters_probability", False)),
        "coherence_time_s": coherence_time,
        "coherence_time_definition": storage_model.get("coherence_time_definition"),
        "coherence_factor_formula": storage_model.get("coherence_factor_formula", "C(t)=1"),
        "legacy_fallback": bool(storage_model.get("legacy_fallback", False)),
        "t1_s": t1,
        "t1_source": t1_source,
        "t2_s": t2,
        "t2_source": t2_source if t2 is not None else "not_provided",
        "coherence_time_source": coherence_source,
        "source_reported_decay_model": memory.get("source_reported_decay_model", "not_specified"),
        "kernel_assumed_storage_decay_model": memory.get(
            "kernel_assumed_storage_decay_model",
            storage_model.get("coherence_factor_formula", "C(t)=1"),
        ),
        "calibration_status": memory.get(
            "memory_storage_calibration_status",
            storage_model.get("calibration_status", "generic approximation unless explicit storage model is supplied"),
        ),
        "amplitude_damping_enabled": t1 is not None and math.isfinite(t1),
        "pure_dephasing_enabled": (
            (kind in {"exponential_dephasing", "gaussian_dephasing"} and coherence_time is not None and math.isfinite(coherence_time))
            or (kind == "t1_t2" and t2 is not None and math.isfinite(t2))
        ),
    }


def memory_storage_channel_probabilities(
    front: dict[str, Any],
    storage_time_ps: float,
    elapsed_storage_time_ps: float = 0.0,
) -> dict[str, Any]:
    storage_time_s = max(0.0, float(storage_time_ps)) * 1e-12
    elapsed_storage_time_s = max(0.0, float(elapsed_storage_time_ps)) * 1e-12
    lifetimes = memory_lifetime_parameters(front)
    kind = lifetimes["storage_model_kind"]
    t1 = lifetimes["t1_s"]
    t2 = lifetimes["t2_s"]
    gamma1 = 0.0
    if kind == "t1_t2" and t1 is not None and math.isfinite(t1):
        gamma1 = clip_probability(1.0 - math.exp(-storage_time_s / t1))
    dephase_probability = 0.0
    t_phi = None
    coherence_factor = 1.0
    pauli_probabilities = None
    if kind == "none":
        pass
    elif kind == "gaussian_dephasing":
        coherence_time = lifetimes["coherence_time_s"]
        if coherence_time is not None and math.isfinite(coherence_time):
            # Gaussian decay C(t)=exp(-(t/T)^2) is defined on the memory's total
            # accumulated storage time and is not interval-divisible, so a
            # storage interval [t0, t0+dt] must contribute the coherence ratio
            # C(t0+dt)/C(t0); sequential intervals then compose to C(t_total).
            total_s = elapsed_storage_time_s + storage_time_s
            c_total = math.exp(-((total_s / coherence_time) ** 2))
            c_prior = math.exp(-((elapsed_storage_time_s / coherence_time) ** 2))
            coherence_factor = c_total / c_prior if c_prior > 0.0 else 0.0
            dephase_probability = clip_probability((1.0 - coherence_factor) / 2.0)
    elif kind == "exponential_dephasing":
        coherence_time = lifetimes["coherence_time_s"]
        if coherence_time is not None and math.isfinite(coherence_time):
            coherence_factor = math.exp(-storage_time_s / coherence_time)
            dephase_probability = clip_probability((1.0 - coherence_factor) / 2.0)
            t_phi = coherence_time
    elif kind == "t1_t2" and t2 is not None and math.isfinite(t2):
        if t1 is not None and math.isfinite(t1):
            inv_phi = (1.0 / t2) - (1.0 / (2.0 * t1))
            if inv_phi < -1e-12:
                raise ValueError(
                    "memory lifetimes are unphysical: T2 must be <= 2*T1 for amplitude damping plus dephasing"
                )
            if inv_phi > 0.0:
                t_phi = 1.0 / inv_phi
        else:
            t_phi = t2
        if t_phi is not None and math.isfinite(t_phi) and t_phi > 0.0:
            coherence_factor = math.exp(-storage_time_s / t_phi)
            dephase_probability = clip_probability((1.0 - coherence_factor) / 2.0)
    elif kind == "liu_2026_anisotropic_pair_fit":
        storage_model = memory_storage_decoherence_model(front)
        total_s = elapsed_storage_time_s + storage_time_s
        total_eigenvalues = symmetric_local_liu_storage_eigenvalues(storage_model, total_s)
        if elapsed_storage_time_s > 0.0:
            prior = symmetric_local_liu_storage_eigenvalues(
                storage_model, elapsed_storage_time_s
            )
            eigenvalues = tuple(
                total / previous if previous > 0.0 else 0.0
                for total, previous in zip(total_eigenvalues, prior)
            )
        else:
            eigenvalues = total_eigenvalues
        pauli_probabilities = pauli_probabilities_from_eigenvalues(*eigenvalues)
        coherence_factor = float(eigenvalues[0])
    else:
        raise ValueError(f"unsupported memory storage decoherence model kind {kind!r}")
    return {
        **lifetimes,
        "storage_time_ps": float(storage_time_ps),
        "storage_time_s": storage_time_s,
        "elapsed_storage_time_ps": float(elapsed_storage_time_ps),
        "elapsed_storage_time_s": elapsed_storage_time_s,
        "gamma_1": gamma1,
        "t_phi_s": t_phi,
        "coherence_factor": coherence_factor,
        "pure_dephasing_probability": dephase_probability,
        "pauli_channel_probabilities": pauli_probabilities,
        "channel_order": "amplitude_damping_then_pure_dephasing",
        "interval_composition": (
            "storage intervals compose on total accumulated storage time; exponential/T1-T2 factors are "
            "interval-divisible, while gaussian_dephasing uses the incremental coherence ratio C(t0+dt)/C(t0)"
        ),
    }


def _apply_storage_channel_params_to_density(
    rho: np.ndarray,
    params: dict[str, Any],
    qubits: tuple[int, ...],
) -> np.ndarray:
    out = clean_density(rho)
    pauli_probabilities = params.get("pauli_channel_probabilities")
    if (
        params["gamma_1"] <= 0.0
        and params["pure_dephasing_probability"] <= 0.0
        and pauli_probabilities is None
    ):
        return out
    for qubit in qubits:
        if pauli_probabilities is not None:
            ops = tuple(
                math.sqrt(float(pauli_probabilities[label])) * PAULI_BY_LABEL[label]
                for label in ("I", "X", "Y", "Z")
                if float(pauli_probabilities[label]) > 0.0
            )
            out = apply_one_qubit_channel(out, qubit, ops)
        if params["gamma_1"] > 0.0:
            out = apply_one_qubit_channel(out, qubit, amplitude_damping_kraus(params["gamma_1"]))
        if params["pure_dephasing_probability"] > 0.0:
            out = apply_one_qubit_channel(
                out,
                qubit,
                dephasing_kraus(params["pure_dephasing_probability"], "Z"),
            )
    return clean_density(out)


def apply_memory_storage_channel_to_density(
    rho: np.ndarray,
    front: dict[str, Any],
    *,
    qubits: tuple[int, ...],
    storage_time_ps: float,
    elapsed_storage_time_ps: float = 0.0,
) -> np.ndarray:
    params = memory_storage_channel_probabilities(
        front,
        storage_time_ps,
        elapsed_storage_time_ps=elapsed_storage_time_ps,
    )
    return _apply_storage_channel_params_to_density(rho, params, qubits)


def apply_memory_storage_channel_to_branches(
    branches: list[Branch],
    front: dict[str, Any],
    *,
    qubits: tuple[int, ...],
    storage_time_ps: float,
    elapsed_storage_time_ps: float = 0.0,
) -> list[Branch]:
    if storage_time_ps <= 0.0:
        return branches
    return normalize_branches([
        Branch(
            probability=branch.probability,
            rho=apply_memory_storage_channel_to_density(
                branch.rho,
                front,
                qubits=qubits,
                storage_time_ps=storage_time_ps,
                elapsed_storage_time_ps=elapsed_storage_time_ps,
            ),
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        )
        for branch in branches
    ])


def apply_memory_storage_channel_to_branches_by_qubit(
    branches: list[Branch],
    front: dict[str, Any],
    *,
    storage_time_by_qubit_ps: dict[int, float],
) -> list[Branch]:
    """Apply total-age storage channels to selected memory qubits.

    This helper is used by the rolling-live-pair protocol, where each edge's
    selected live pair has a configured pre-route age and terminal memories may
    also receive deterministic endpoint-confirmation waits. The times passed
    here are total represented ages from pair creation, so Gaussian dephasing is
    applied once as C(t_total), not as an interval approximation after an
    already-marginalized age mixture.
    """
    if not storage_time_by_qubit_ps or all(float(t) <= 0.0 for t in storage_time_by_qubit_ps.values()):
        return branches
    out: list[Branch] = []
    for branch in branches:
        rho = np.array(branch.rho, dtype=complex, copy=True)
        for qubit, storage_time_ps in sorted(storage_time_by_qubit_ps.items()):
            if float(storage_time_ps) <= 0.0:
                continue
            rho = apply_memory_storage_channel_to_density(
                rho,
                front,
                qubits=(int(qubit),),
                storage_time_ps=float(storage_time_ps),
                elapsed_storage_time_ps=0.0,
            )
        out.append(Branch(
            probability=branch.probability,
            rho=clean_density(rho),
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        ))
    return normalize_branches(out)


def period_from_frequency_ps(frequency_hz: Any) -> float | None:
    value = _finite_positive_or_none(frequency_hz)
    if value is None:
        return None
    return 1e12 / value


def active_multiplexing_controller(
    context: ModelContext,
) -> MultiplexingController | None:
    """Return the opt-in controller; ``None`` preserves legacy semantics."""

    return controller_from_config(context.multiplexing_controller)


def _shared_device_recovery_period_ps(
    front: dict[str, Any],
    context: ModelContext,
) -> tuple[float, dict[str, Any]]:
    memory = front["device_constants"]["memory"]
    fiber = front["device_constants"]["fiber_channel"]
    detector_rate = detector_count_rate_model(front, context)
    candidates = {
        "memory_excitation_period_ps": period_from_frequency_ps(memory.get("frequency_hz")),
        "fiber_channel_scheduling_period_ps": period_from_frequency_ps(fiber.get("frequency_hz")),
        "detector_count_rate_lower_bound_period_ps": period_from_frequency_ps(
            detector_rate["count_rate_hz"]
        ),
    }
    finite = [
        float(value)
        for value in candidates.values()
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0
    ]
    return (max(finite) if finite else 0.0), {
        **candidates,
        "detector_count_rate_hz": detector_rate["count_rate_hz"],
        "detector_count_rate_source": detector_rate["source"],
        "detector_count_rate_profile_field": detector_rate["profile_field"],
        "detector_count_rate_mapping_semantics": detector_rate["mapping_semantics"],
    }


def multiplexed_edge_round(
    *,
    single_mode_success_probability: float,
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    context: ModelContext,
) -> tuple[float, dict[str, Any] | None]:
    """Fold one-mode success into an executable controller round.

    The returned probability is unchanged when no controller is configured.
    This function deliberately changes only the classical round success/rate;
    the successful mode's conditional optical instrument remains the existing
    Fock instrument.
    """

    controller = active_multiplexing_controller(context)
    p_single = clip_probability(single_mode_success_probability)
    if controller is None:
        return p_single, None
    if context.physical_path != "fock_pnr_snspd" or context.source_model != "memory_emission":
        raise NotImplementedError(
            "multiplexing_controller is implemented only for the fock_pnr_snspd "
            "memory_emission optical instrument"
        )
    if int(route.get("hops", len(route.get("edge_ids", ())))) >= 2 and (
        normalized_waiting_protocol(context) != "rolling_alive_pair"
    ):
        raise NotImplementedError(
            "multiplexed multi-edge swap-asap generation requires the conditional within-round "
            "mode-herald offset distribution in the inter-edge waiting kernel; use a calibrated "
            "rolling_alive_pair payload, or leave multiplexing disabled. The controller will not "
            "collapse those offsets to a mean round boundary."
        )
    from .fock import edge_delay_ps

    herald_ps = float(edge_delay_ps(front, route["edge_lengths_m"][edge_id]))
    recovery_ps, recovery_meta = _shared_device_recovery_period_ps(front, context)
    result = controller.evaluate_round(
        single_mode_success_probability=p_single,
        herald_latency_s=herald_ps * 1e-12,
        minimum_shared_device_recovery_s=recovery_ps * 1e-12,
    )
    return result.round_success_probability, {
        "status": "active_exact_independent_time_bin_round_reduction",
        "controller": controller.to_dict(),
        "round": result.to_dict(),
        "shared_device_recovery": {
            **recovery_meta,
            "minimum_shared_device_recovery_period_ps": recovery_ps,
        },
        "conditional_quantum_instrument": (
            "unchanged single-mode detector-conditioned memory instrument; exact because the configured "
            "controller requires mode-invariant instruments and retains the first conclusive mode"
        ),
        "edge_id": edge_id,
    }


def attempt_period_policy(
    route: dict[str, Any],
    front: dict[str, Any],
    context: ModelContext | None = None,
) -> dict[str, Any]:
    from .fock import edge_delay_ps

    context = context or ModelContext()
    recovery_period, recovery_meta = _shared_device_recovery_period_ps(front, context)
    memory_period = recovery_meta["memory_excitation_period_ps"]
    fiber_period = recovery_meta["fiber_channel_scheduling_period_ps"]
    detector_period = recovery_meta["detector_count_rate_lower_bound_period_ps"]
    detector_rate_hz = recovery_meta["detector_count_rate_hz"]
    controller = active_multiplexing_controller(context)
    service_profile = active_repeater_service_profile(context)
    service_restoration_active = bool(
        service_profile is not None
        and quality_restoration_model(service_profile) is not None
    )
    edge_periods: dict[str, Any] = {}
    for edge_id in route["edge_ids"]:
        herald = float(edge_delay_ps(front, route["edge_lengths_m"][edge_id]))
        candidates = {
            "memory_excitation_period_ps": memory_period,
            "fiber_channel_scheduling_period_ps": fiber_period,
            "detector_count_rate_lower_bound_period_ps": detector_period,
            "deterministic_herald_round_trip_ps": herald,
        }
        if controller is None:
            finite_candidates = [
                float(value) for value in candidates.values()
                if value is not None and math.isfinite(float(value)) and float(value) >= 0.0
            ]
            attempt_period = max(finite_candidates) if finite_candidates else herald
            policy = (
                "max(memory excitation period, fiber/channel scheduling period, detector count-rate lower-bound "
                "period, deterministic herald round-trip time)"
            )
            controller_timing = None
        else:
            controller_timing = controller.round_timing(
                herald_latency_s=herald * 1e-12,
                minimum_shared_device_recovery_s=recovery_period * 1e-12,
            )
            attempt_period = float(controller_timing["round_period_s"]) * 1e12
            policy = (
                "explicit independent time-bin controller round: last launch offset + deterministic "
                "midpoint-BSM herald latency + fixed controller overhead; every within/cross-round "
                "mode separation is validated against source/channel/detector recovery"
            )
        edge_period = {
            **candidates,
            "detector_count_rate_hz": detector_rate_hz,
            "detector_count_rate_source": recovery_meta["detector_count_rate_source"],
            "detector_count_rate_profile_field": recovery_meta["detector_count_rate_profile_field"],
            "detector_count_rate_mapping_semantics": recovery_meta[
                "detector_count_rate_mapping_semantics"
            ],
            "attempt_period_ps": attempt_period,
            "policy": policy,
        }
        if controller_timing is not None:
            edge_period["multiplexing_controller_timing"] = controller_timing
        edge_periods[edge_id] = edge_period
    if controller is None:
        return {
            "model": "analytic_iid_geometric_repeat_until_success",
            "scope": "event-free attempt-period approximation; not SeQUeNCe event scheduling",
            "edge_attempt_periods": edge_periods,
            "exclusions": [
                "no multiplexing",
                (
                    "purification is accounted separately by the active repeater-service "
                    "renewal model, not this generic optical attempt-period helper"
                    if service_restoration_active else "no purification"
                ),
                "no adaptive scheduling",
                "no detector dead-time correlations",
                "no resource contention",
                "no event-level asynchronous scheduler dynamics",
            ],
        }
    return {
        "model": "analytic_iid_geometric_repeat_until_success_over_controller_rounds",
        "scope": "event-free attempt-period approximation; not SeQUeNCe event scheduling",
        "edge_attempt_periods": edge_periods,
        "multiplexing_controller": controller.to_dict(),
        "exclusions": [
            (
                "purification is accounted separately by the active repeater-service "
                "renewal model, not this generic time-bin attempt-period helper"
                if service_restoration_active else "no purification"
            ),
            "no adaptive scheduling",
            "no detector dead-time correlations after cadence validation",
            "no resource contention",
            "no event-level asynchronous scheduler dynamics",
            "no repeated-resource excitation or periodic cooling",
        ],
    }


def _equal_period_budgeted_waiting_histories(
    *,
    p0: float,
    p1: float,
    period_ps: float,
    context: ModelContext,
    budget: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    p0 = clip_probability(p0)
    p1 = clip_probability(p1)
    q0 = 1.0 - p0
    q1 = 1.0 - p1
    budget = max(1, int(context.waiting_max_attempts)) if budget is None else max(1, int(budget))
    histories: list[dict[str, float]] = []

    def finite_geometric_sum(ratio: float, count: int) -> float:
        if count <= 0:
            return 0.0
        if abs(1.0 - ratio) <= 1e-15:
            return float(count)
        return (1.0 - ratio ** count) / (1.0 - ratio)

    ratio = q0 * q1
    same_time = p0 * p1 * finite_geometric_sum(ratio, budget)
    histories.append({"left_wait_ps": 0.0, "right_wait_ps": 0.0, "probability": same_time})
    for delta_attempts in range(1, budget):
        overlap_count = budget - delta_attempts
        common = p0 * p1 * finite_geometric_sum(ratio, overlap_count)
        histories.append({
            "left_wait_ps": float(delta_attempts) * period_ps,
            "right_wait_ps": 0.0,
            "probability": common * (q1 ** delta_attempts),
        })
        histories.append({
            "left_wait_ps": 0.0,
            "right_wait_ps": float(delta_attempts) * period_ps,
            "probability": common * (q0 ** delta_attempts),
        })
    retained = max(0.0, min(1.0, sum(row["probability"] for row in histories)))
    ready_by_budget = clip_probability((1.0 - q0 ** budget) * (1.0 - q1 ** budget))
    omitted = max(0.0, min(1.0, 1.0 - ready_by_budget))
    if retained > 0.0:
        for row in histories:
            row["probability"] /= retained
    cap_meta = waiting_cap_diagnostic_metadata(
        configured_route_ready=ready_by_budget,
        capped_route_ready=ready_by_budget,
        cutoffs_reduced_below_budget=False,
        tolerance=float(context.waiting_tail_tolerance),
    )
    return histories, {
        "status": "finite_budget_equal_attempt_period_analytic",
        "attempt_budget_per_link": budget,
        "attempt_difference_cutoff": max(0, budget - 1),
        "history_count": len(histories),
        "retained_history_mass": retained,
        "route_ready_probability_within_budget": ready_by_budget,
        "omitted_tail_mass": omitted,
        "tail_handling": "omitted tail is included as timeout/no_delivery mass, not renormalized into delivered probability",
        "history_probabilities_conditioning": "conditional_on_route_ready_within_attempt_budget",
        "metrics_conditional_on_retained_histories": False,
        "tail_probability_tolerance": float(context.waiting_tail_tolerance),
        "max_attempts_control": int(context.waiting_max_attempts),
        "tail_exceeds_tolerance": omitted > float(context.waiting_tail_tolerance),
        **cap_meta,
    }


def finite_budget_route_ready_probability(
    probabilities: Sequence[float],
    attempt_budgets: Sequence[int],
) -> float:
    ready = 1.0
    for probability, budget in zip(probabilities, attempt_budgets):
        p = clip_probability(float(probability))
        k = max(0, int(budget))
        ready *= 1.0 - (1.0 - p) ** k if k > 0 else 0.0
    return clip_probability(ready)


def waiting_cap_diagnostic_metadata(
    *,
    configured_route_ready: float,
    capped_route_ready: float,
    cutoffs_reduced_below_budget: bool,
    tolerance: float,
) -> dict[str, Any]:
    configured = clip_probability(configured_route_ready)
    capped = clip_probability(capped_route_ready)
    cap_omitted = max(0.0, configured - capped)
    physical_timeout = max(0.0, 1.0 - configured)
    cap_bias = (
        configured / capped
        if capped > 0.0 else
        (math.inf if configured > 0.0 else None)
    )
    compute_cap_limited = cutoffs_reduced_below_budget and cap_omitted > float(tolerance)
    if compute_cap_limited:
        status = "compute_cap_limited"
    elif physical_timeout > float(tolerance):
        status = "physics_deadline_limited"
    else:
        status = "converged_within_configured_attempt_budget"
    return {
        "configured_attempt_budget_route_ready_probability": configured,
        "capped_route_ready_probability": capped,
        "configured_budget_physical_timeout_mass": physical_timeout,
        "compute_cap_omitted_route_ready_mass": cap_omitted,
        "no_delivery_mass_booked_from_waiting": max(0.0, 1.0 - capped),
        "attempt_cutoffs_reduced_below_budget": bool(cutoffs_reduced_below_budget),
        "cap_bias_factor": cap_bias,
        "waiting_precision_status": status,
        "compute_cap_limited": compute_cap_limited,
        "transcript_tv_distance_bound_vs_configured_budget": cap_omitted,
        "error_bound_interpretation": (
            "If retained histories are used while route-ready mass omitted by compute caps is booked "
            "as no_delivery, the unconditional transcript total-variation distance from the configured "
            "finite-attempt-budget waiting model is bounded above by compute_cap_omitted_route_ready_mass "
            "before downstream swap/survival factors. Conditional delivered metrics are not certified "
            "when this bound is large."
        ),
    }


def two_link_waiting_histories(
    *,
    p0: float,
    p1: float,
    period0_ps: float,
    period1_ps: float,
    context: ModelContext,
    budget0: int | None = None,
    budget1: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    p0 = clip_probability(p0)
    p1 = clip_probability(p1)
    budget0 = max(0, int(context.waiting_max_attempts if budget0 is None else budget0))
    budget1 = max(0, int(context.waiting_max_attempts if budget1 is None else budget1))
    if budget0 == 0 or budget1 == 0:
        return [], {
            "status": "no_waiting_histories_zero_attempt_budget",
            "attempt_budgets": [budget0, budget1],
            "retained_history_mass": 0.0,
            "route_ready_probability_within_budget": 0.0,
            "omitted_tail_mass": 1.0,
            "tail_handling": (
                "all mass is timeout/no_delivery because the waiting budget admits zero attempts "
                "on at least one elementary link"
            ),
        }
    if p0 <= 0.0 or p1 <= 0.0:
        return [], {
            "status": "no_waiting_histories_no_elementary_success",
            "retained_history_mass": 0.0,
            "route_ready_probability_within_budget": 0.0,
            "omitted_tail_mass": 1.0,
            "tail_handling": "all mass is timeout/no_delivery because at least one elementary success probability is zero",
        }
    if abs(period0_ps - period1_ps) <= 1e-9 and budget0 == budget1:
        histories, meta = _equal_period_budgeted_waiting_histories(
            p0=p0,
            p1=p1,
            period_ps=float(period0_ps),
            context=context,
            budget=budget0,
        )
    else:
        configured_ready = finite_budget_route_ready_probability([p0, p1], [budget0, budget1])
        cutoff0 = budget0
        cutoff1 = budget1
        while cutoff0 * cutoff1 > int(context.waiting_max_history_pairs) and max(cutoff0, cutoff1) > 1:
            if cutoff0 >= cutoff1:
                cutoff0 -= 1
            else:
                cutoff1 -= 1
        q0 = 1.0 - p0
        q1 = 1.0 - p1
        grouped: dict[tuple[float, float], float] = {}
        for n0 in range(1, cutoff0 + 1):
            prob0 = p0 * (q0 ** (n0 - 1))
            t0 = n0 * float(period0_ps)
            for n1 in range(1, cutoff1 + 1):
                prob = prob0 * p1 * (q1 ** (n1 - 1))
                t1 = n1 * float(period1_ps)
                operation_time = max(t0, t1)
                key = (operation_time - t0, operation_time - t1)
                grouped[key] = grouped.get(key, 0.0) + prob
        retained = sum(grouped.values())
        omitted = max(0.0, min(1.0, 1.0 - retained))
        cap_meta = waiting_cap_diagnostic_metadata(
            configured_route_ready=configured_ready,
            capped_route_ready=retained,
            cutoffs_reduced_below_budget=(cutoff0 < budget0 or cutoff1 < budget1),
            tolerance=float(context.waiting_tail_tolerance),
        )
        cap_tail = cap_meta["compute_cap_omitted_route_ready_mass"]
        histories = [
            {"left_wait_ps": left, "right_wait_ps": right, "probability": prob / retained}
            for (left, right), prob in grouped.items()
            if retained > 0.0 and prob > 0.0
        ]
        meta = {
            "status": "finite_budget_pair_enumeration_unequal_attempt_periods_or_budgets",
            "requested_attempt_budgets": [budget0, budget1],
            "left_attempt_cutoff": cutoff0,
            "right_attempt_cutoff": cutoff1,
            "history_pair_cap": int(context.waiting_max_history_pairs),
            "history_count": len(histories),
            "retained_history_mass": retained,
            "route_ready_probability_within_budget": retained,
            "omitted_tail_mass": omitted,
            "tail_handling": (
                "omitted mass is included as no_delivery and not renormalized into delivered probability; "
                "because pair-history caps reduced the requested attempt budget, this no_delivery mass "
                "contains compute-cap omitted route-ready histories as well as physical timeout mass"
                if cap_tail > 0.0 else
                "omitted tail is included as timeout/no_delivery mass, not renormalized into delivered probability"
            ),
            "history_probabilities_conditioning": "conditional_on_route_ready_within_effective_attempt_cutoffs",
            "metrics_conditional_on_retained_histories": False,
            "tail_probability_tolerance": float(context.waiting_tail_tolerance),
            "max_attempts_control": int(context.waiting_max_attempts),
            "tail_exceeds_tolerance": omitted > float(context.waiting_tail_tolerance),
            **cap_meta,
        }
    expected_left = sum(row["probability"] * row["left_wait_ps"] for row in histories) if histories else None
    expected_right = sum(row["probability"] * row["right_wait_ps"] for row in histories) if histories else None
    max_left = max((row["left_wait_ps"] for row in histories), default=0.0)
    max_right = max((row["right_wait_ps"] for row in histories), default=0.0)
    meta.update({
        "history_probabilities_are_conditional": True,
        "expected_left_extra_wait_ps": expected_left,
        "expected_right_extra_wait_ps": expected_right,
        "expected_left_extra_wait_s": None if expected_left is None else expected_left * 1e-12,
        "expected_right_extra_wait_s": None if expected_right is None else expected_right * 1e-12,
        "max_left_extra_wait_ps_retained": max_left,
        "max_right_extra_wait_ps_retained": max_right,
    })
    return histories, meta


_WAIT_SIGNATURE_MEMO: dict[str, tuple[dict[tuple[float, ...], float], dict[str, Any]]] = {}
_WAIT_SIGNATURE_MEMO_LIMIT = 64


def _multi_link_wait_signature_distribution_fixed(
    *,
    edge_success: list[float],
    edge_periods_ps: list[float],
    attempt_budgets: list[int],
    context: ModelContext,
) -> tuple[dict[tuple[float, ...], float], dict[str, Any]]:
    """Grouped swap-asap waiting distribution for an h>=3 selected chain.

    Model: each elementary link i attempts independently with period Delta_i
    and per-attempt success p_i, so its herald time is T_i = N_i * Delta_i with
    N_i ~ Geometric(p_i) on {1, 2, ...}. Sequential swap-asap scheduling gives
    swap times S_1 = max(T_0, T_1), S_i = max(S_{i-1}, T_i). Every attempt
    vector inside the per-edge cutoffs is enumerated exactly and grouped by its
    wait signature (a_1, b_1, c_2, b_2, ..., c_{h-1}, b_{h-1}) with
    a_1 = S_1 - T_0, b_i = S_i - T_i, c_i = S_i - S_{i-1} -- the exact tuple of
    post-herald storage intervals that enters the quantum composition, so the
    grouping is lossless. Cutoffs are shrunk from the attempt budgets until the
    raw enumeration fits waiting_max_history_pairs and the grouped signature
    count fits waiting_max_wait_signatures; the omitted geometric tail is
    timeout/no_delivery mass, never renormalized into delivered probability.
    Returned signature weights are conditional on route readiness within the
    final cutoffs. Results are memoized (pure function of the arguments), so
    the delivery policy, the composer, and the compiled kernel share one
    enumeration per distinct input.
    """
    hops = len(edge_success)
    if hops < 3:
        raise ValueError("multi_link_wait_signature_distribution requires h >= 3")
    memo_key = stable_hash({
        "edge_success": [float(p) for p in edge_success],
        "edge_periods_ps": [float(p) for p in edge_periods_ps],
        "attempt_budgets": [int(b) for b in attempt_budgets],
        "pair_cap": int(context.waiting_max_history_pairs),
        "signature_cap": int(context.waiting_max_wait_signatures),
        "tail_tolerance": float(context.waiting_tail_tolerance),
        "max_attempts": int(context.waiting_max_attempts),
    })
    cached = _WAIT_SIGNATURE_MEMO.get(memo_key)
    if cached is not None:
        return cached
    probabilities = [clip_probability(p) for p in edge_success]
    if any(p <= 0.0 for p in probabilities):
        return {}, {
            "status": "no_waiting_histories_no_elementary_success",
            "retained_history_mass": 0.0,
            "route_ready_probability_within_budget": 0.0,
            "omitted_tail_mass": 1.0,
            "tail_handling": "all mass is timeout/no_delivery because at least one elementary success probability is zero",
        }
    cutoffs = [max(0, int(budget)) for budget in attempt_budgets]
    if any(cutoff == 0 for cutoff in cutoffs):
        return {}, {
            "status": "no_waiting_histories_zero_attempt_budget",
            "attempt_budgets": list(cutoffs),
            "retained_history_mass": 0.0,
            "route_ready_probability_within_budget": 0.0,
            "omitted_tail_mass": 1.0,
            "tail_handling": (
                "all mass is timeout/no_delivery because the waiting budget admits zero attempts "
                "on at least one elementary link"
            ),
        }
    pair_cap = max(1, int(context.waiting_max_history_pairs))
    signature_cap = max(1, int(context.waiting_max_wait_signatures))
    configured_route_ready = finite_budget_route_ready_probability(probabilities, cutoffs)

    def product(values: list[int]) -> int:
        out = 1
        for value in values:
            out *= value
        return out

    if product(cutoffs) > pair_cap:
        ratio = (pair_cap / product(cutoffs)) ** (1.0 / hops)
        cutoffs = [max(1, int(cutoff * ratio)) for cutoff in cutoffs]
    while product(cutoffs) > pair_cap and max(cutoffs) > 1:
        cutoffs[cutoffs.index(max(cutoffs))] -= 1

    def enumerate_signatures(current_cutoffs: list[int]) -> tuple[dict[tuple[float, ...], float], float]:
        # Vectorized exact enumeration of every attempt vector within the
        # cutoffs, grouped losslessly by the rounded wait signature; identical
        # semantics to the per-vector Python loop it replaces.
        herald_axes = [
            np.arange(1, cutoff + 1, dtype=float) * edge_periods_ps[i]
            for i, cutoff in enumerate(current_cutoffs)
        ]
        weight_axes = [
            probabilities[i] * ((1.0 - probabilities[i]) ** np.arange(0, cutoff, dtype=float))
            for i, cutoff in enumerate(current_cutoffs)
        ]
        herald_grids = np.meshgrid(*herald_axes, indexing="ij")
        weights = weight_axes[0].reshape([-1] + [1] * (hops - 1))
        for i in range(1, hops):
            shape = [1] * hops
            shape[i] = -1
            weights = weights * weight_axes[i].reshape(shape)
        weights = weights.reshape(-1)
        heralds = [grid.reshape(-1) for grid in herald_grids]
        swap_time = np.maximum(heralds[0], heralds[1])
        columns = [np.round(swap_time - heralds[0], 3), np.round(swap_time - heralds[1], 3)]
        for edge_index in range(2, hops):
            next_swap = np.maximum(swap_time, heralds[edge_index])
            columns.append(np.round(next_swap - swap_time, 3))
            columns.append(np.round(next_swap - heralds[edge_index], 3))
            swap_time = next_swap
        signature_matrix = np.stack(columns, axis=1)
        unique_rows, inverse = np.unique(signature_matrix, axis=0, return_inverse=True)
        grouped_weights = np.bincount(inverse, weights=weights, minlength=unique_rows.shape[0])
        grouped = {
            tuple(float(v) for v in row): float(weight)
            for row, weight in zip(unique_rows, grouped_weights)
        }
        return grouped, float(weights.sum())

    grouped, retained = enumerate_signatures(cutoffs)
    shrink_rounds = 0
    while len(grouped) > signature_cap and max(cutoffs) > 1:
        cutoffs = [max(1, int(cutoff * 0.8)) if cutoff > 1 else 1 for cutoff in cutoffs]
        grouped, retained = enumerate_signatures(cutoffs)
        shrink_rounds += 1
    route_ready = 1.0
    for p, cutoff in zip(probabilities, cutoffs):
        route_ready *= 1.0 - (1.0 - p) ** cutoff
    omitted = max(0.0, min(1.0, 1.0 - route_ready))
    reduced_below_budget = any(
        cutoff < int(budget) for cutoff, budget in zip(cutoffs, attempt_budgets)
    )
    cap_meta = waiting_cap_diagnostic_metadata(
        configured_route_ready=configured_route_ready,
        capped_route_ready=route_ready,
        cutoffs_reduced_below_budget=reduced_below_budget,
        tolerance=float(context.waiting_tail_tolerance),
    )
    cap_tail = cap_meta["compute_cap_omitted_route_ready_mass"]
    signatures = (
        {key: weight / retained for key, weight in grouped.items() if weight > 0.0}
        if retained > 0.0 else {}
    )
    meta = {
        "status": "finite_budget_grouped_swap_asap_enumeration",
        "scheduling_policy": "sequential_swap_asap: S_1=max(T_0,T_1); S_i=max(S_{i-1},T_i)",
        "signature_layout": "(a_1=S_1-T_0, b_1=S_1-T_1, c_2=S_2-S_1, b_2=S_2-T_2, ..., c_{h-1}, b_{h-1}) in ps",
        "grouping_exactness": (
            "attempt vectors are grouped by the exact tuple of post-herald storage intervals entering "
            "the quantum composition; grouping is lossless, so accuracy is limited only by the recorded "
            "attempt cutoffs"
        ),
        "requested_attempt_budgets": [int(budget) for budget in attempt_budgets],
        "attempt_cutoffs_after_caps": list(cutoffs),
        "history_pair_cap": pair_cap,
        "wait_signature_cap": signature_cap,
        "signature_cap_shrink_rounds": shrink_rounds,
        "history_count": product(cutoffs),
        "wait_signature_count": len(signatures),
        "retained_history_mass": retained,
        "route_ready_probability_within_budget": clip_probability(route_ready),
        "omitted_tail_mass": omitted,
        "tail_handling": (
            "omitted mass is included as no_delivery and not renormalized into delivered probability; "
            "because h>=3 history/signature caps reduced the requested attempt budget, this no_delivery "
            "mass contains compute-cap omitted route-ready histories as well as physical timeout mass"
            if cap_tail > 0.0 else
            "omitted tail is included as timeout/no_delivery mass, not renormalized into delivered probability"
        ),
        "history_probabilities_conditioning": "conditional_on_route_ready_within_attempt_cutoffs",
        "history_probabilities_are_conditional": True,
        "metrics_conditional_on_retained_histories": False,
        "tail_probability_tolerance": float(context.waiting_tail_tolerance),
        "max_attempts_control": int(context.waiting_max_attempts),
        "tail_exceeds_tolerance": omitted > float(context.waiting_tail_tolerance),
        **cap_meta,
    }
    if len(_WAIT_SIGNATURE_MEMO) >= _WAIT_SIGNATURE_MEMO_LIMIT:
        _WAIT_SIGNATURE_MEMO.pop(next(iter(_WAIT_SIGNATURE_MEMO)))
    _WAIT_SIGNATURE_MEMO[memo_key] = (signatures, meta)
    return signatures, meta


def _estimated_wait_enumeration_memory_mb(hops: int, history_cap: int) -> float:
    # enumerate_signatures materializes h herald grids, one weight array,
    # roughly 2h signature columns, and inverse/grouping work arrays. This is
    # intentionally conservative; it is a guardrail, not a physics parameter.
    bytes_per_history = max(16, 4 * int(hops) + 4) * 8
    return (max(1, int(history_cap)) * bytes_per_history) / (1024.0 * 1024.0)


def _memory_limited_history_cap(context: ModelContext, hops: int) -> int:
    limit_mb = context.waiting_max_memory_mb
    if limit_mb is None or float(limit_mb) <= 0.0:
        return max(1, int(context.waiting_max_history_pairs))
    bytes_per_history = max(16, 4 * int(hops) + 4) * 8
    limit_bytes = float(limit_mb) * 1024.0 * 1024.0
    return max(1, int(limit_bytes // bytes_per_history))


def _adaptive_multi_link_wait_signature_distribution(
    *,
    edge_success: list[float],
    edge_periods_ps: list[float],
    attempt_budgets: list[int],
    context: ModelContext,
) -> tuple[dict[tuple[float, ...], float], dict[str, Any]]:
    started = time.perf_counter()
    base_pairs = max(1, int(context.waiting_max_history_pairs))
    base_signatures = max(1, int(context.waiting_max_wait_signatures))
    max_rounds = max(0, int(context.waiting_max_refinement_rounds))
    memory_pair_cap = _memory_limited_history_cap(context, len(edge_success))
    product_budget = 1
    for budget in attempt_budgets:
        product_budget *= max(1, int(budget))
    best_signatures: dict[tuple[float, ...], float] = {}
    best_meta: dict[str, Any] = {}
    refinements: list[dict[str, Any]] = []
    stop_reason = "max_refinement_rounds_reached"
    for round_index in range(max_rounds + 1):
        elapsed_before = time.perf_counter() - started
        if (
            round_index > 0
            and context.waiting_max_runtime_s is not None
            and elapsed_before >= float(context.waiting_max_runtime_s)
        ):
            stop_reason = "max_runtime_s_reached_before_next_round"
            break
        schedule_multiplier = 1 << round_index
        pair_cap = min(product_budget, memory_pair_cap, base_pairs * schedule_multiplier)
        signature_cap = min(pair_cap, base_signatures * schedule_multiplier)
        round_context = replace(
            context,
            waiting_precision_mode="fixed_caps",
            waiting_max_history_pairs=int(pair_cap),
            waiting_max_wait_signatures=int(signature_cap),
        )
        signatures, meta = _multi_link_wait_signature_distribution_fixed(
            edge_success=edge_success,
            edge_periods_ps=edge_periods_ps,
            attempt_budgets=attempt_budgets,
            context=round_context,
        )
        elapsed = time.perf_counter() - started
        epsilon = float(meta.get("compute_cap_omitted_route_ready_mass", 1.0))
        refinements.append({
            "round": round_index,
            "history_pair_cap": int(pair_cap),
            "wait_signature_cap": int(signature_cap),
            "estimated_enumeration_memory_mb": _estimated_wait_enumeration_memory_mb(
                len(edge_success), int(pair_cap)
            ),
            "route_ready_probability_within_budget": meta.get("route_ready_probability_within_budget"),
            "configured_attempt_budget_route_ready_probability": meta.get(
                "configured_attempt_budget_route_ready_probability"
            ),
            "compute_cap_omitted_route_ready_mass": epsilon,
            "wait_signature_count": meta.get("wait_signature_count"),
            "history_count": meta.get("history_count"),
            "attempt_cutoffs_after_caps": meta.get("attempt_cutoffs_after_caps"),
            "attempt_cutoffs_reduced_below_budget": meta.get("attempt_cutoffs_reduced_below_budget"),
            "wall_seconds_since_start": elapsed,
        })
        best_signatures = signatures
        best_meta = meta
        if epsilon <= float(context.waiting_tv_tolerance):
            stop_reason = "tv_tolerance_reached"
            break
        if pair_cap >= product_budget and not meta.get("attempt_cutoffs_reduced_below_budget", False):
            stop_reason = "full_budget_enumerated"
            break
        if pair_cap >= memory_pair_cap and signature_cap >= pair_cap and memory_pair_cap < product_budget:
            stop_reason = "max_memory_mb_reached"
            break
        if (
            context.waiting_max_runtime_s is not None
            and elapsed >= float(context.waiting_max_runtime_s)
        ):
            stop_reason = "max_runtime_s_reached"
            break
    if not best_meta:
        return {}, {
            "status": "adaptive_bounded_no_evaluation",
            "waiting_precision_mode": "adaptive_bounded",
            "adaptive_refinement_rounds": refinements,
            "adaptive_stop_reason": stop_reason,
            "waiting_precision_status": "waiting_model_requires_user_decision",
        }
    epsilon = float(best_meta.get("compute_cap_omitted_route_ready_mass", 1.0))
    certified = epsilon <= float(context.waiting_tv_tolerance)
    best_meta = {
        **best_meta,
        "status": "adaptive_bounded_grouped_swap_asap_enumeration",
        "base_distribution_status": best_meta.get("status"),
        "waiting_precision_mode": "adaptive_bounded",
        "adaptive_refinement_rounds": refinements,
        "adaptive_round_count": len(refinements),
        "adaptive_stop_reason": stop_reason,
        "adaptive_target_tv_tolerance": float(context.waiting_tv_tolerance),
        "adaptive_target_metric_tolerance": float(context.waiting_metric_tolerance),
        "adaptive_max_refinement_rounds": max_rounds,
        "adaptive_max_runtime_s": context.waiting_max_runtime_s,
        "adaptive_max_memory_mb": context.waiting_max_memory_mb,
        "adaptive_memory_limited_history_pair_cap": memory_pair_cap,
        "adaptive_status": (
            "bounded_support_certified" if certified else "bounded_support_not_certified"
        ),
        "waiting_precision_status": (
            "waiting_model_bounded" if certified else "waiting_model_compute_cap_limited"
        ),
        "exactness_scope": (
            "retained support is enumerated and grouped exactly; omitted full-budget route-ready mass "
            "is bounded by compute_cap_omitted_route_ready_mass and booked as no_delivery"
        ),
    }
    return best_signatures, best_meta


def multi_link_wait_signature_distribution(
    *,
    edge_success: list[float],
    edge_periods_ps: list[float],
    attempt_budgets: list[int],
    context: ModelContext,
) -> tuple[dict[tuple[float, ...], float], dict[str, Any]]:
    if str(context.waiting_precision_mode) == "adaptive_bounded":
        return _adaptive_multi_link_wait_signature_distribution(
            edge_success=edge_success,
            edge_periods_ps=edge_periods_ps,
            attempt_budgets=attempt_budgets,
            context=context,
        )
    if str(context.waiting_precision_mode) != "fixed_caps":
        raise ValueError(
            "unsupported waiting_precision_mode "
            f"{context.waiting_precision_mode!r}; supported modes are fixed_caps, adaptive_bounded"
        )
    return _multi_link_wait_signature_distribution_fixed(
        edge_success=edge_success,
        edge_periods_ps=edge_periods_ps,
        attempt_budgets=attempt_budgets,
        context=context,
    )


def waiting_attempt_budgets(
    context: ModelContext,
    edge_periods_ps: dict[str, float],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Per-edge attempt budgets for the configured waiting budget mode.

    fixed_attempts: every elementary link gets waiting_max_attempts attempts.
    fixed_wall_clock_time: each link gets floor(T_budget / attempt_period)
    attempts, so one histogram draw represents a comparable wall-clock
    route-generation window across different fiber lengths. Budgets are still
    capped by waiting_max_attempts as a compute bound; the cap is recorded.
    """
    mode = str(context.waiting_budget_mode)
    cap = max(1, int(context.waiting_max_attempts))
    if mode == "fixed_attempts":
        budgets = {edge_id: cap for edge_id in edge_periods_ps}
        return budgets, {
            "waiting_budget_mode": mode,
            "attempt_budget_per_edge": dict(budgets),
            "budget_cap_max_attempts": cap,
            "budget_capped_by_max_attempts": False,
        }
    if mode == "fixed_wall_clock_time":
        budget_ps = context.waiting_budget_wall_clock_ps
        if budget_ps is None or float(budget_ps) <= 0.0:
            raise ValueError(
                "waiting_budget_mode=fixed_wall_clock_time requires a positive waiting_budget_wall_clock_ps"
            )
        budget_ps = float(budget_ps)
        budgets: dict[str, int] = {}
        capped = False
        for edge_id, period_ps in edge_periods_ps.items():
            if period_ps <= 0.0:
                budgets[edge_id] = cap
                capped = True
                continue
            allowed = int(budget_ps // float(period_ps))
            if allowed > cap:
                allowed = cap
                capped = True
            budgets[edge_id] = allowed
        return budgets, {
            "waiting_budget_mode": mode,
            "waiting_budget_wall_clock_ps": budget_ps,
            "attempt_budget_per_edge": dict(budgets),
            "attempt_budget_rule": "floor(waiting_budget_wall_clock_ps / edge_attempt_period_ps)",
            "budget_cap_max_attempts": cap,
            "budget_capped_by_max_attempts": capped,
        }
    raise ValueError(
        f"unsupported waiting_budget_mode {mode!r}; supported modes are fixed_attempts, fixed_wall_clock_time"
    )


def normalized_waiting_protocol(context: ModelContext) -> str:
    protocol = str(context.waiting_protocol).strip().lower()
    if protocol not in {"single_buffer_swap_asap", "rolling_alive_pair"}:
        raise ValueError(
            "waiting_protocol must be one of single_buffer_swap_asap, rolling_alive_pair; "
            f"got {context.waiting_protocol!r}"
        )
    return protocol


def normalized_rolling_age_evaluation(context: ModelContext) -> str:
    mode = str(context.rolling_age_evaluation).strip().lower()
    if mode not in {"cartesian", "moment_reduced"}:
        raise ValueError(
            "rolling_age_evaluation must be one of cartesian, moment_reduced; "
            f"got {context.rolling_age_evaluation!r}"
        )
    return mode


def rolling_age_evaluation_metadata(context: ModelContext) -> dict[str, Any]:
    mode = normalized_rolling_age_evaluation(context)
    assumptions = [
        "rolling_alive_pair selected ages are hidden variables and are not transcript coordinates",
        "edge live-pair availability A_i is handled separately from selected-age quality",
        "selected age affects only memory-local storage dephasing after live-pair selection",
        "pre-route stale cutoff and detected decay are already baked into configured A_i and f_i(t)",
        "no post-selection delivery/success branch depends on the hidden selected age",
        "Eve moves and public records do not observe or condition on the hidden selected age",
        "post-selection instruments are linear CP maps on unnormalized branch states",
    ]
    if mode == "cartesian":
        return {
            "mode": mode,
            "status": "exact_cartesian_reference",
            "exactness": (
                "explicit finite product over configured per-edge selected-age support; useful as "
                "an audit/reference path when support_product is tractable"
            ),
            "exactness_assumptions": assumptions,
        }
    return {
        "mode": mode,
        "status": "exact_edge_local_moment_reduction",
        "exactness": (
            "each edge's finite selected-age distribution is averaged into its detector-conditioned "
            "two-memory branch state before the linear route fold; this preserves same-edge age "
            "correlations such as E[C(t)^2] and is exactly equal to Cartesian marginalization under "
            "the listed assumptions"
        ),
        "moments_used": (
            "implemented by direct density/operator averaging over each edge's finite age support; "
            "equivalent to using all required same-edge pure-dephasing moments, not just E[C(t)]"
        ),
        "exactness_assumptions": assumptions,
    }


def _normalize_discrete_rolling_age_distribution(
    points_in: Sequence[float],
    weights_in: Sequence[float],
    *,
    accepted_alive_time_s: float | None,
    prefix: str,
    source: str,
    selection_policy: str,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    points = [float(value) for value in points_in]
    weights = [float(value) for value in weights_in]
    if len(points) != len(weights) or not points:
        raise ValueError(f"{prefix} requires equal non-empty points and weights")
    if any(age < 0.0 or not math.isfinite(age) for age in points):
        raise ValueError(f"{prefix} points must be finite nonnegative seconds")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError(f"{prefix} weights must be finite nonnegative values")
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError(f"{prefix} weights must have positive total mass")
    alive_time_value = accepted_alive_time_s
    if alive_time_value is not None:
        if alive_time_value < 0.0 or not math.isfinite(alive_time_value):
            raise ValueError("rolling_accepted_alive_time_s must be finite and nonnegative when supplied")
        if any(age > alive_time_value + 1e-15 for age in points):
            raise ValueError(f"{prefix} point exceeds rolling_accepted_alive_time_s")
    distribution = [(age, weight / total) for age, weight in zip(points, weights) if weight > 0.0]
    mean = math.fsum(age * weight for age, weight in distribution)
    return distribution, {
        "kind": "discrete",
        "points_s": [age for age, _weight in distribution],
        "weights": [weight for _age, weight in distribution],
        "normalized": abs(total - 1.0) > 1e-12,
        "raw_weight_sum": total,
        "support_size": len(distribution),
        "mean_age_s": mean,
        "max_age_s": max(age for age, _weight in distribution),
        "accepted_alive_time_s": alive_time_value,
        "selection_policy": selection_policy,
        "age_distribution_source": source,
        "exactness": "exact finite sum over configured discrete selected-live-pair age distribution",
    }


def rolling_age_distribution(context: ModelContext) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    kind = str(context.rolling_age_distribution_kind).strip().lower()
    if kind != "discrete":
        raise NotImplementedError(
            "rolling_alive_pair currently supports only rolling_age_distribution_kind='discrete'"
        )
    alive_time = context.rolling_accepted_alive_time_s
    alive_time_value = None if alive_time is None else float(alive_time)
    return _normalize_discrete_rolling_age_distribution(
        context.rolling_age_distribution_points_s,
        context.rolling_age_distribution_weights,
        accepted_alive_time_s=alive_time_value,
        prefix="rolling shared age distribution",
        source=context.rolling_age_distribution_source,
        selection_policy=context.rolling_selection_policy,
    )


def _shared_rolling_age_schema_is_default(context: ModelContext) -> bool:
    return (
        tuple(float(v) for v in context.rolling_age_distribution_points_s) == (0.0,)
        and tuple(float(v) for v in context.rolling_age_distribution_weights) == (1.0,)
    )


def rolling_age_distributions_for_route(
    route: dict[str, Any],
    context: ModelContext,
) -> tuple[list[list[tuple[float, float]]], dict[str, Any]]:
    kind = str(context.rolling_age_distribution_kind).strip().lower()
    if kind != "discrete":
        raise NotImplementedError(
            "rolling_alive_pair currently supports only rolling_age_distribution_kind='discrete'"
        )
    edge_ids = list(route["edge_ids"])
    alive_time = context.rolling_accepted_alive_time_s
    alive_time_value = None if alive_time is None else float(alive_time)
    per_edge_points = context.rolling_edge_age_distribution_points_s
    per_edge_weights = context.rolling_edge_age_distribution_weights
    per_edge_supplied = per_edge_points is not None or per_edge_weights is not None
    if per_edge_supplied:
        if per_edge_points is None or per_edge_weights is None:
            raise ValueError("per-edge rolling age schema requires both points and weights")
        if not _shared_rolling_age_schema_is_default(context):
            raise ValueError(
                "provide either shared rolling_age_distribution_* or per-edge "
                "rolling_edge_age_distribution_*, not both"
            )
        if len(per_edge_points) != len(edge_ids) or len(per_edge_weights) != len(edge_ids):
            raise ValueError("per-edge rolling age distributions must provide one row per route edge")
        distributions: list[list[tuple[float, float]]] = []
        metadata_by_edge: dict[str, Any] = {}
        support_product = 1
        for edge_id, points, weights in zip(edge_ids, per_edge_points, per_edge_weights):
            distribution, meta = _normalize_discrete_rolling_age_distribution(
                points,
                weights,
                accepted_alive_time_s=alive_time_value,
                prefix=f"rolling edge age distribution {edge_id}",
                source=context.rolling_age_distribution_source,
                selection_policy=context.rolling_selection_policy,
            )
            distributions.append(distribution)
            metadata_by_edge[edge_id] = meta
            support_product *= int(meta["support_size"])
        return distributions, {
            "schema": "per_edge",
            "heterogeneous_age_distributions_supported": True,
            "by_edge": metadata_by_edge,
            "support_product": support_product,
            "support_product_limit": int(context.rolling_age_support_product_limit),
            "exactness": "exact finite product over per-edge configured discrete selected-age distributions",
        }
    distribution, meta = rolling_age_distribution(context)
    support_product = int(meta["support_size"]) ** len(edge_ids)
    return [distribution for _edge_id in edge_ids], {
        "schema": "shared",
        **meta,
        "heterogeneous_age_distributions_supported": True,
        "shared": meta,
        "by_edge": {edge_id: meta for edge_id in edge_ids},
        "support_product": support_product,
        "support_product_limit": int(context.rolling_age_support_product_limit),
        "exactness": "exact finite product over shared configured discrete selected-age distribution",
    }


def rolling_edge_availability_for_route(
    route: dict[str, Any],
    context: ModelContext,
) -> tuple[dict[str, float], dict[str, Any]]:
    raw = context.rolling_edge_availability
    edge_ids = list(route["edge_ids"])
    source = "default_all_edges_available"
    if raw is None:
        values = [1.0] * len(edge_ids)
    elif isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]
        if len(values) == 1:
            values = values * len(edge_ids)
        elif len(values) != len(edge_ids):
            raise ValueError(
                "rolling_edge_availability must be a scalar or provide one value per edge"
            )
        source = "configured_per_edge" if len(set(values)) > 1 else "configured_scalar"
    else:
        values = [float(raw)] * len(edge_ids)
        source = "configured_scalar"
    availability: dict[str, float] = {}
    for edge_id, value in zip(edge_ids, values):
        availability[edge_id] = require_probability(f"rolling_edge_availability[{edge_id}]", value)
    route_availability = math.prod(availability.values()) if availability else 1.0
    return availability, {
        "edge_availability": availability,
        "route_availability": clip_probability(route_availability),
        "source": source,
        "semantics": (
            "A_i is the probability that edge i has an accepted live elementary pair at route request time; "
            "it is configured, not inferred from the per-attempt elementary success probability"
        ),
    }


def rolling_endpoint_confirmation_waits(
    route: dict[str, Any],
    front: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    wait_profile = endpoint_classical_wait_profile(route, front)
    swap_count = max(0, int(route.get("hops", 1)) - 1)
    waits = endpoint_extra_waits_for_swap_tails(wait_profile, [0.0] * swap_count)
    return waits[0], waits[1], wait_profile


def rolling_live_pair_policy(
    route: dict[str, Any],
    front: dict[str, Any],
    context: ModelContext,
) -> dict[str, Any]:
    availability, availability_meta = rolling_edge_availability_for_route(route, context)
    _age_distributions, age_meta = rolling_age_distributions_for_route(route, context)
    age_evaluation = rolling_age_evaluation_metadata(context)
    alice_wait, bob_wait, endpoint_profile = rolling_endpoint_confirmation_waits(route, front)
    service_profile = active_repeater_service_profile(context)
    service_model = None
    if service_profile is not None:
        edge_lengths_km = [
            float(route["edge_lengths_m"][edge_id]) / 1000.0
            for edge_id in route["edge_ids"]
        ]
        service = route_service_estimate(service_profile, edge_lengths_km)
        quality = route_bell_quality_estimate(
            service_profile,
            len(edge_lengths_km),
            edge_lengths_km,
        )
        service_model = {
            "profile": repeater_profile_summary(service_profile),
            "route_service": service.to_dict(),
            "bell_quality_without_storage": quality.to_dict(),
            "timing_role": (
                "mean asynchronous controller service time and rate; kept separate from the "
                "random-time rolling-pool availability probability used for this histogram draw"
            ),
            "quality_role": (
                "source-equation Bell-diagonal baseline before explicit selected-age storage "
                "decoherence; the histogram expected metrics include the storage channel"
            ),
        }
    return {
        "waiting_protocol": "rolling_alive_pair",
        "status": "implemented_configured_discrete_age_distribution",
        "supported": True,
        "probability_space": (
            "one route request against continuously maintained live elementary pairs; no_delivery includes "
            "at least one elementary edge lacking an accepted live pair at request time"
        ),
        "mathematical_model": (
            "independent edge live events with configured probabilities A_i; conditional selected-pair "
            "ages are drawn independently from the configured discrete distribution f_i(t), represented "
            "exactly by a finite probability sum"
        ),
        "route_availability": availability_meta["route_availability"],
        "repeater_service_model": service_model,
        "availability": availability_meta,
        "selected_age_distribution": age_meta,
        "rolling_age_evaluation": age_evaluation,
        "heterogeneous_age_distributions_supported": True,
        "age_support_product": age_meta["support_product"],
        "age_support_product_limit": age_meta["support_product_limit"],
        "pre_route_decay_and_stale_semantics": (
            "baked_into_configured_availability_and_age_distribution; pre-route stale cutoff and detected "
            "decay are not applied again by the kernel"
        ),
        "endpoint_classical_confirmation": {
            "alice_extra_wait_ps": alice_wait,
            "bob_extra_wait_ps": bob_wait,
            "policy": endpoint_profile,
            "storage_effect": (
                "terminal selected live-pair memories include the deterministic endpoint confirmation wait "
                "in their total storage age before endpoint measurement"
            ),
        },
        "extra_memory_slots_note": {
            "memory_slots_per_edge_side": int(context.rolling_memory_slots_per_edge_side),
            "not_modeled_yet": (
                "no queueing/resource-contention model or stateful depletion across game turns is included"
            ),
            "future_pool_formula": (
                "A_pool=1-(1-A_slot)^c and, for youngest-live selection, "
                "F_selected(t)=[1-(1-A_slot F(t))^c]/[1-(1-A_slot)^c]"
            ),
        },
        "excluded_timing_dynamics": [
            "event-level memory-pool queueing",
            "stateful depletion across game turns",
            "resource contention between simultaneous routes",
            "renewal-derived live-pair age distribution",
        ],
    }


def rolling_edge_storage_times_ps(
    *,
    edge_index: int,
    hop_count: int,
    age_s: float,
    endpoint_wait_alice_ps: float,
    endpoint_wait_bob_ps: float,
) -> dict[int, float]:
    age_ps = float(age_s) * 1e12
    left_age = age_ps + (float(endpoint_wait_alice_ps) if edge_index == 0 else 0.0)
    right_age = age_ps + (float(endpoint_wait_bob_ps) if edge_index == hop_count - 1 else 0.0)
    return {0: left_age, 1: right_age}


def rolling_live_pair_site_age_distribution(
    route: dict[str, Any],
    front: dict[str, Any],
    context: ModelContext,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact finite age-support distribution over all memory sites.

    The returned ages have columns [e0_left, e0_right, e1_left, e1_right, ...].
    This is intentionally a product over configured per-edge age support, not a
    route-level repeat-until-success history enumeration.
    """
    age_distributions, age_meta = rolling_age_distributions_for_route(route, context)
    support_product = int(age_meta["support_product"])
    limit = int(context.rolling_age_support_product_limit)
    if support_product > limit:
        raise NotImplementedError(
            "rolling_alive_pair per-edge age support product exceeds configured exact limit: "
            f"{support_product} > {limit}"
        )
    alice_wait, bob_wait, _endpoint_profile = rolling_endpoint_confirmation_waits(route, front)
    hops = len(route["edge_ids"])
    rows: list[list[float]] = []
    weights: list[float] = []
    for combo in itertools.product(*(range(len(distribution)) for distribution in age_distributions)):
        row: list[float] = []
        weight = 1.0
        for edge_index, point_index in enumerate(combo):
            age_s, point_weight = age_distributions[edge_index][point_index]
            weight *= point_weight
            times = rolling_edge_storage_times_ps(
                edge_index=edge_index,
                hop_count=hops,
                age_s=age_s,
                endpoint_wait_alice_ps=alice_wait,
                endpoint_wait_bob_ps=bob_wait,
            )
            row.extend([times[0], times[1]])
        rows.append(row)
        weights.append(weight)
    weights_array = np.asarray(weights, dtype=float)
    total = float(weights_array.sum())
    if total <= 0.0:
        raise ValueError("rolling age support has zero probability mass")
    return weights_array / total, np.asarray(rows, dtype=float), {
        "support_rows": len(rows),
        "per_edge_support_sizes": [len(distribution) for distribution in age_distributions],
        "support_product": support_product,
        "support_product_limit": limit,
        "selected_age_distribution": age_meta,
        "exactness": "exact Cartesian product of configured edge-age support",
        "endpoint_waits_included_in_terminal_site_ages": True,
    }


def stochastic_waiting_time_policy(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> dict[str, Any]:
    from .fock import edge_delay_ps
    from .routing import fock_elementary_edge_outcome_probabilities

    attempt_policy = attempt_period_policy(route, front, context)
    if context.physical_path != "fock_pnr_snspd":
        return {"status": "not_applicable_non_fock_path", "attempt_period_policy": attempt_policy}
    if route["hops"] == 1:
        return {
            "status": "not_applicable_single_elementary_link",
            "supported": True,
            "attempt_period_policy": attempt_policy,
            "omitted_tail_mass": 0.0,
            "tail_handling": "no repeat-until-success adjacent-link waiting tail exists for a single elementary link",
        }
    edge_ids = route["edge_ids"]
    edge_success = {}
    controller_rounds: dict[str, Any] = {}
    for edge_id in edge_ids:
        probs, _meta = fock_elementary_edge_outcome_probabilities(edge_id, route, front, attack, context)
        probability, round_meta = multiplexed_edge_round(
            single_mode_success_probability=(
                probs.get("psi_plus", 0.0) + probs.get("psi_minus", 0.0)
            ),
            edge_id=edge_id,
            route=route,
            front=front,
            context=context,
        )
        edge_success[edge_id] = probability
        if round_meta is not None:
            controller_rounds[edge_id] = round_meta
    edge_periods = {
        edge_id: float(attempt_policy["edge_attempt_periods"][edge_id]["attempt_period_ps"])
        for edge_id in edge_ids
    }
    budgets, budget_meta = waiting_attempt_budgets(context, edge_periods)
    wall_clock_budget_by_edge_ps = {
        edge_id: float(budgets[edge_id]) * edge_periods[edge_id]
        for edge_id in edge_ids
    }
    wait_profile = endpoint_classical_wait_profile(route, front)
    herald_list = [
        float(edge_delay_ps(front, route["edge_lengths_m"][edge_id]))
        for edge_id in edge_ids
    ]
    if route["hops"] >= 3:
        signatures, distribution_meta = multi_link_wait_signature_distribution(
            edge_success=[edge_success[edge_id] for edge_id in edge_ids],
            edge_periods_ps=[edge_periods[edge_id] for edge_id in edge_ids],
            attempt_budgets=[budgets[edge_id] for edge_id in edge_ids],
            context=context,
        )
        weighted_ages = [
            (
                weight,
                signature_memory_ages(
                    signature,
                    herald_ages_ps=herald_list,
                    wait_profile=wait_profile,
                )[0],
            )
            for signature, weight in signatures.items()
        ]
        _usable, memory_loss_stats = apply_memory_loss_policies_to_histories(
            weighted_ages, front, context
        )
        history_count = len(signatures)
        status = "implemented_multi_link_grouped_swap_asap_enumeration"
        probability_space = (
            "one finite-budget h>=3 route-generation cycle; no_delivery includes elementary links "
            "not all ready within the (possibly cap-reduced) attempt cutoffs"
        )
        mathematical_model = (
            "N_i ~ Geometric(p_i) on attempts {1,2,...}; T_i=N_i*Delta_t_i; sequential swap-asap "
            "S_1=max(T_0,T_1), S_i=max(S_{i-1},T_i); every attempt vector within the cutoffs is "
            "enumerated exactly and grouped losslessly by its post-herald storage-interval signature"
        )
        storage_state_effect = (
            "wait signatures are composed through the full sequential fold with exact per-memory "
            "(duration, elapsed) storage applications"
            if memory_lifetime_parameters(front)["storage_model_enters_probability"] else
            "no active storage model: waits do not change branch states, so the deterministic "
            "composer is branch-exact and waiting enters only the delivery mass"
        )
    else:
        histories, distribution_meta = two_link_waiting_histories(
            p0=edge_success[edge_ids[0]],
            p1=edge_success[edge_ids[1]],
            period0_ps=edge_periods[edge_ids[0]],
            period1_ps=edge_periods[edge_ids[1]],
            context=context,
            budget0=budgets[edge_ids[0]],
            budget1=budgets[edge_ids[1]],
        )
        endpoint_wait_alice, endpoint_wait_bob = endpoint_extra_waits_for_swap_tails(
            wait_profile, [0.0]
        )
        weighted_ages = [
            (
                float(row["probability"]),
                two_link_history_memory_ages(
                    row,
                    herald_left_ps=herald_list[0],
                    herald_right_ps=herald_list[1],
                    endpoint_wait_alice_ps=endpoint_wait_alice,
                    endpoint_wait_bob_ps=endpoint_wait_bob,
                ),
            )
            for row in histories
        ]
        _usable, memory_loss_stats = apply_memory_loss_policies_to_histories(
            weighted_ages, front, context
        )
        history_count = len(histories)
        status = "implemented_two_link_analytic_iid_geometric"
        probability_space = (
            "one finite-budget h=2 route-generation cycle; no_delivery includes elementary links "
            "not both ready within the waiting budget"
        )
        mathematical_model = (
            "N_i ~ Geometric(p_i) on attempts {1,2,...}; T_i=N_i*Delta_t_i; "
            "the earlier successful elementary memory pair receives extra storage |T_0-T_1| before the swap"
        )
        storage_state_effect = "one-sided waiting mixtures applied exactly before the single swap"
    fixed_wall_clock = context.waiting_budget_mode == "fixed_wall_clock_time"
    payload = {
        "status": status,
        "supported": True,
        "supported_hop_counts": "1, 2 (analytic), and >=3 (grouped swap-asap enumeration within attempt cutoffs)",
        "probability_space": probability_space,
        "storage_state_effect": storage_state_effect,
        "mathematical_model": mathematical_model,
        "delivery_probability_semantics": (
            "delivered branch states are conditioned on route readiness within the finite waiting budget; "
            "the retained readiness probability enters delivery_model, and omitted tail mass is timeout/no_delivery"
        ),
        "game_turn_budget_semantics": {
            "waiting_budget_semantics": "per_pair",
            "parallel_memory_pairs_modeled": False,
            "sequence_reference_alignment": (
                "compiled histogram waiting budget is per selected-route transcript-pair generation cycle; "
                "SeQUeNCe fixed-route trials may use repeater_window_ms and memory_pairs as event-level "
                "trial resources. Those are not silently harmonized here."
            ),
            "budget_type": (
                "fixed_wall_clock_time_budget" if fixed_wall_clock
                else "fixed_attempt_budget_per_elementary_link"
            ),
            "waiting_budget_mode": context.waiting_budget_mode,
            "waiting_precision_mode": context.waiting_precision_mode,
            "waiting_tv_tolerance": context.waiting_tv_tolerance,
            "waiting_metric_tolerance": context.waiting_metric_tolerance,
            "waiting_budget_wall_clock_ps": (
                float(context.waiting_budget_wall_clock_ps) if fixed_wall_clock else None
            ),
            "max_attempts_per_elementary_link": (
                dict(budgets) if fixed_wall_clock else int(context.waiting_max_attempts)
            ),
            "attempt_budget_derivation": budget_meta,
            "fixed_wall_clock_time_budget_supported": True,
            "wall_clock_duration_changes_with_fiber_length": not fixed_wall_clock,
            "wall_clock_budget_by_edge_ps": wall_clock_budget_by_edge_ps,
            "route_generation_budget_wall_clock_ps": (
                float(context.waiting_budget_wall_clock_ps) if fixed_wall_clock
                else (max(wall_clock_budget_by_edge_ps.values()) if wall_clock_budget_by_edge_ps else 0.0)
            ),
            "interpretation": (
                "For fixed T, longer fiber lengths increase each attempt period and therefore reduce the "
                "attempt budget per elementary link; delivered/no-delivery mass reflects a comparable "
                "wall-clock route-generation window across fiber lengths (subject to the recorded "
                "max-attempts compute cap)."
                if fixed_wall_clock else
                "For fixed K, longer fiber lengths increase each attempt period and therefore increase "
                "the wall-clock duration represented by one histogram draw. Use "
                "waiting_budget_mode=fixed_wall_clock_time for a comparable fixed-T game turn."
            ),
        },
        "edge_success_probabilities": edge_success,
        "attempt_period_policy": attempt_policy,
        "waiting_distribution": {
            **distribution_meta,
            "memory_loss_policies": memory_loss_stats,
        },
        "endpoint_classical_wait_profile": wait_profile,
        "retained_history_count": history_count,
        "excluded_timing_dynamics": attempt_policy["exclusions"],
    }
    if controller_rounds:
        payload["multiplexing_controller_rounds"] = controller_rounds
    return payload


def deterministic_storage_timing_policy(route: dict[str, Any], front: dict[str, Any]) -> dict[str, Any]:
    from .fock import edge_delay_ps, fiber_group_speed_m_per_ps

    speed = fiber_group_speed_m_per_ps(front)
    edge_ready = {
        edge_id: edge_delay_ps(front, route["edge_lengths_m"][edge_id])
        for edge_id in route["edge_ids"]
    }
    swaps = []
    left_ready = edge_ready[route["edge_ids"][0]]
    for swap_index, repeater in enumerate(route["nodes"][1:-1]):
        right_edge = route["edge_ids"][swap_index + 1]
        right_ready = edge_ready[right_edge]
        operation_time = max(left_ready, right_ready)
        swaps.append({
            "repeater": repeater,
            "operation_time_ps": operation_time,
            "left_segment_extra_wait_ps": operation_time - left_ready,
            "right_edge_extra_wait_ps": operation_time - right_ready,
            "policy": "deterministic one-shot swap as soon as adjacent elementary heralds are available",
        })
        left_ready = operation_time
    return {
        "edge_herald_ready_time_ps": edge_ready,
        "fiber_group_speed_m_per_ps": speed,
        "fiber_propagation_speed_m_per_ps": speed,
        "fiber_speed_interpretation": (
            "scalar guided-mode propagation/group speed along the fiber path coordinate/fiber arclength; "
            "not vacuum c, not vector velocity, and not a fiber dephasing parameter"
        ),
        "elementary_storage_time_policy": (
            "each elementary memory is stored for one half-link photon propagation plus one half-link classical herald return, "
            "equal to full edge_delay_ps under the shared scalar fiber group-speed model"
        ),
        "swap_operations": swaps,
        "endpoint_measurement_time_ps": left_ready,
        "endpoint_classical_confirmation": endpoint_classical_wait_profile(route, front),
        "excluded_waiting_dynamics": (
            "asynchronous event-scheduler waiting is not modeled; repeat-until-success waiting is represented "
            "separately for Fock/SNSPD routes (analytic for h=2, grouped swap-asap enumeration for h>=3)"
        ),
    }


def endpoint_classical_wait_profile(route: dict[str, Any], front: dict[str, Any]) -> dict[str, Any]:
    from .fock import fiber_group_speed_m_per_ps

    """Classical swap-outcome propagation times from each repeater to the endpoints.

    Endpoint measurement policy: Alice and Bob measure once route readiness is
    confirmed, i.e. once every swap outcome has classically reached them over
    the same fiber path (shared scalar group speed). The endpoint memories
    keep storing (decohering and decaying) during that classical flight.
    t_c(r_i -> Alice) = sum(L_0..L_{i-1})/v and t_c(r_i -> Bob) =
    sum(L_i..L_{h-1})/v. With sequential swap-asap scheduling the Alice-side
    extra wait after the final swap is deterministic (the last swap r_{h-1} is
    both the latest and the farthest from Alice); the Bob-side wait is
    max_i(t_c(r_i -> Bob) - (S_{h-1} - S_i)) and can depend on the waiting
    history for h >= 3.
    """
    speed = fiber_group_speed_m_per_ps(front)
    edge_ids = list(route["edge_ids"])
    hops = len(edge_ids)
    lengths = [float(route["edge_lengths_m"][edge_id]) for edge_id in edge_ids]
    if speed <= 0.0 or hops < 2:
        return {
            "policy": "measure_on_route_ready_confirmation",
            "applicable": False,
            "reason": (
                "single elementary link: the elementary herald round trip already confirms readiness"
                if hops < 2 else "no positive classical propagation speed available"
            ),
            "classical_time_to_alice_ps": [],
            "classical_time_to_bob_ps": [],
            "alice_extra_wait_ps": 0.0,
        }
    to_alice = [sum(lengths[:i]) / speed for i in range(1, hops)]
    to_bob = [sum(lengths[i:]) / speed for i in range(1, hops)]
    return {
        "policy": "measure_on_route_ready_confirmation",
        "applicable": True,
        "classical_speed_m_per_ps": speed,
        "classical_time_to_alice_ps": to_alice,
        "classical_time_to_bob_ps": to_bob,
        "alice_extra_wait_ps": to_alice[-1],
        "alice_extra_wait_rule": "deterministic: last swap r_{h-1} is latest and farthest from Alice",
        "bob_extra_wait_rule": "max_i(t_c(r_i -> Bob) - (S_{h-1} - S_i)); history-dependent for h >= 3",
        "storage_effect": (
            "the endpoint memory qubits receive this extra storage interval (decoherence channel and "
            "detected-decay survival) after the final swap and before endpoint measurement"
        ),
    }


def endpoint_extra_waits_for_swap_tails(
    wait_profile: dict[str, Any],
    swap_tail_after_each_swap_ps: Sequence[float],
) -> tuple[float, float]:
    """(Alice, Bob) extra endpoint storage after the final swap for one history.

    swap_tail_after_each_swap_ps[i-1] = S_{h-1} - S_i for swaps i = 1..h-1
    (zero for the final swap).
    """
    if not wait_profile.get("applicable", False):
        return 0.0, 0.0
    to_alice = wait_profile["classical_time_to_alice_ps"]
    to_bob = wait_profile["classical_time_to_bob_ps"]
    alice_wait = max(
        0.0,
        max(tc - tail for tc, tail in zip(to_alice, swap_tail_after_each_swap_ps)),
    )
    bob_wait = max(
        0.0,
        max(tc - tail for tc, tail in zip(to_bob, swap_tail_after_each_swap_ps)),
    )
    return alice_wait, bob_wait


def memory_survival_model(front: dict[str, Any]) -> dict[str, Any]:
    """Typed detected-decay survival model from memory.detected_decay_lifetime_s.

    Semantics (Cui2025): spontaneous decay of the storage level is detected by
    a mid-circuit measurement, so it is heralded loss that reduces
    delivery mass (failure stage memory_decay_detected), not a state-error
    channel. Each memory survives its total represented storage age t with
    probability exp(-t / lifetime); ages are exact per waiting history.
    """
    memory = front["device_constants"]["memory"]
    lifetime = _finite_positive_or_none(memory.get("detected_decay_lifetime_s"))
    return {
        "enabled": lifetime is not None,
        "lifetime_s": lifetime,
        "source_field": "memory.detected_decay_lifetime_s",
        "failure_stage": "memory_decay_detected",
        "semantics": (
            "per-memory heralded loss: survival = exp(-total represented storage age / lifetime), "
            "detected by mid-circuit measurement and booked as no_delivery, never as state error"
        ),
    }


def survival_probability_for_ages(ages_ps: Sequence[float], lifetime_s: float) -> float:
    total_s = sum(max(0.0, float(age)) for age in ages_ps) * 1e-12
    return math.exp(-total_s / float(lifetime_s))


def apply_memory_loss_policies_to_histories(
    weighted_ages: list[tuple[float, list[float]]],
    front: dict[str, Any],
    context: ModelContext,
) -> tuple[list[float], dict[str, Any]]:
    """Per-history usable weights under stale-pair cutoff and decay survival.

    weighted_ages: (conditional weight, list of per-memory total ages in ps)
    per waiting history, weights conditional on route readiness. Returns a
    per-history usable-weight list (weight * survival, zero when any memory
    age exceeds the cutoff) aligned with the input, plus classical stats. The
    caller renormalizes retained weights for delivered-state conditioning and
    uses the stats for delivery-mass decomposition.
    """
    survival = memory_survival_model(front)
    cutoff_ps = context.memory_cutoff_ps
    cutoff_active = cutoff_ps is not None and float(cutoff_ps) > 0.0
    total = math.fsum(weight for weight, _ages in weighted_ages)
    cut_mass = 0.0
    usable_weights: list[float] = []
    survived_mass = 0.0
    retained_mass = 0.0
    for weight, ages in weighted_ages:
        if cutoff_active and any(age > float(cutoff_ps) for age in ages):
            cut_mass += weight
            usable_weights.append(0.0)
            continue
        retained_mass += weight
        factor = (
            survival_probability_for_ages(ages, survival["lifetime_s"])
            if survival["enabled"] else 1.0
        )
        survived_mass += weight * factor
        usable_weights.append(weight * factor)
    cutoff_fraction = (cut_mass / total) if total > 0.0 else 0.0
    survival_expectation = (survived_mass / retained_mass) if retained_mass > 0.0 else 1.0
    stats = {
        "survival_enabled": survival["enabled"],
        "detected_decay_lifetime_s": survival["lifetime_s"],
        "memory_cutoff_ps": float(cutoff_ps) if cutoff_active else None,
        "cutoff_fraction_of_ready_mass": cutoff_fraction,
        "survival_expectation_given_cutoff_pass": survival_expectation,
        "usable_fraction_of_ready_mass": (
            (1.0 - cutoff_fraction) * survival_expectation
        ),
        "age_accounting": (
            "each memory's age = elementary herald storage + post-herald waits (+ endpoint classical "
            "confirmation wait for the two endpoint memories); exact per waiting history/signature"
        ),
    }
    return usable_weights, stats


def two_link_history_memory_ages(
    history: dict[str, float],
    *,
    herald_left_ps: float,
    herald_right_ps: float,
    endpoint_wait_alice_ps: float,
    endpoint_wait_bob_ps: float,
) -> list[float]:
    left_wait = float(history["left_wait_ps"])
    right_wait = float(history["right_wait_ps"])
    return [
        herald_left_ps + left_wait + endpoint_wait_alice_ps,
        herald_left_ps + left_wait,
        herald_right_ps + right_wait,
        herald_right_ps + right_wait + endpoint_wait_bob_ps,
    ]


def signature_swap_tails_ps(signature: Sequence[float], hops: int) -> list[float]:
    """S_{h-1} - S_i for swaps i = 1..h-1 from a wait signature."""
    c_values = [float(signature[2 * i - 2]) for i in range(2, hops)]
    tails = []
    for swap_index in range(1, hops):
        tails.append(math.fsum(c_values[swap_index - 1:]))
    return tails


def signature_memory_ages(
    signature: Sequence[float],
    *,
    herald_ages_ps: Sequence[float],
    wait_profile: dict[str, Any],
) -> tuple[list[float], float, float]:
    """Per-memory total ages plus (Alice, Bob) endpoint waits for one signature."""
    hops = len(herald_ages_ps)
    a_1 = float(signature[0])
    b_values = [float(signature[2 * i - 1]) for i in range(1, hops)]
    c_values = [float(signature[2 * i - 2]) for i in range(2, hops)]
    tails = signature_swap_tails_ps(signature, hops)
    alice_wait, bob_wait = endpoint_extra_waits_for_swap_tails(wait_profile, tails)
    ages = [herald_ages_ps[0] + a_1 + math.fsum(c_values) + alice_wait]
    ages.append(herald_ages_ps[0] + a_1)
    for j in range(1, hops - 1):
        ages.append(herald_ages_ps[j] + b_values[j - 1])
        ages.append(herald_ages_ps[j] + b_values[j - 1] + c_values[j - 1])
    ages.append(herald_ages_ps[hops - 1] + b_values[hops - 2])
    ages.append(herald_ages_ps[hops - 1] + b_values[hops - 2] + bob_wait)
    return ages, alice_wait, bob_wait




from .performance import BenchmarkResult, RuntimeMeasurement, benchmark_call, measure_call


def edge_delay_ps(front: dict[str, Any], length_m: float) -> int:
    from .fock import edge_delay_ps as calculate

    return calculate(front, length_m)


def half_edge_delay_ps(front: dict[str, Any], length_m: float) -> int:
    from .fock import half_edge_delay_ps as calculate

    return calculate(front, length_m)


def detector_cooldown_period_ps(
    front: dict[str, Any], context: ModelContext | None = None
) -> int:
    from .fock import detector_cooldown_period_ps as calculate

    return calculate(front, context)


def detector_timing_window_at_wavelength(
    front: dict[str, Any], context: ModelContext, wavelength_nm: float
) -> tuple[float, dict[str, Any]]:
    from .fock import detector_timing_window_at_wavelength as calculate

    return calculate(front, context, wavelength_nm)
