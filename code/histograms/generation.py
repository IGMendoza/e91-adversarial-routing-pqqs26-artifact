"""Histogram construction, metrics, audits, and frozen-case evaluation."""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Sequence

import numpy as np

from .devices import *
from .fock import *
from .models import *
from .output import *
from .quantum import *
from .routing import *
from .timing import *
from .topology import *
def setting_pairs(protocol: ProtocolConfig) -> list[tuple[ProtocolSetting, ProtocolSetting, float]]:
    alice = [s for s in protocol.settings if s.party == "alice"]
    bob = [s for s in protocol.settings if s.party == "bob"]
    aw = normalized_weights(alice)
    bw = normalized_weights(bob)
    return [
        (a, b, float(ap * bp))
        for a, ap in zip(alice, aw)
        for b, bp in zip(bob, bw)
    ]


def normalized_weights(settings: list[ProtocolSetting]) -> np.ndarray:
    weights = np.asarray([setting.probability_weight for setting in settings], dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(settings), dtype=float)
    return weights / float(weights.sum())


_JOINT_OUTCOME_EFFECT_CACHE: dict[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    np.ndarray,
] = {}


def _axis_cache_key(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(component) for component in axis)


def joint_outcome_effect_vectors(
    alice_axis: tuple[float, float, float],
    bob_axis: tuple[float, float, float],
) -> np.ndarray:
    key = (_axis_cache_key(alice_axis), _axis_cache_key(bob_axis))
    cached = _JOINT_OUTCOME_EFFECT_CACHE.get(key)
    if cached is not None:
        return cached
    alice_projectors = projectors_for_axis(alice_axis)
    bob_projectors = projectors_for_axis(bob_axis)
    effects = [
        np.kron(alice_projectors[a], bob_projectors[b])
        for a in range(2)
        for b in range(2)
    ]
    trace_vectors = np.asarray(
        [
            effect.reshape(16)
            for effect in effects
        ],
        dtype=complex,
    )
    _JOINT_OUTCOME_EFFECT_CACHE[key] = trace_vectors
    return trace_vectors


def joint_outcome_probabilities(
    rho: np.ndarray,
    alice_axis: tuple[float, float, float],
    bob_axis: tuple[float, float, float],
) -> np.ndarray:
    effect_vectors = joint_outcome_effect_vectors(alice_axis, bob_axis)
    probs = np.real(effect_vectors @ rho.T.reshape(16)).reshape(2, 2)
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total <= 0.0:
        return np.full((2, 2), 0.25)
    return probs / total


def sifting_class(protocol: ProtocolConfig, alice: ProtocolSetting, bob: ProtocolSetting) -> str:
    if alice.use == "key" and bob.use == "key":
        return "key"
    if (alice.label, bob.label) in protocol.chsh_pairs:
        return "chsh"
    return "discarded"


def histogram_rows_from_branches(
    *,
    branches: list[Branch],
    stages: dict[str, Any],
    front: dict[str, Any],
    protocol: ProtocolConfig,
    key_length: int,
    keep_full_bsm: bool,
    keep_detector_click_records: bool,
    context: ModelContext,
) -> list[dict[str, Any]]:
    p_delivery = stages["delivered"]
    readout = memory_readout_model(front)
    rotation = endpoint_analysis_rotation_model(front)
    rotation_probability = (
        clip_probability(float(rotation.get("error_probability", 0.0)))
        if rotation.get("kind") == "rotated_basis_depolarizing" and rotation.get("enters_probability")
        else 0.0
    )
    rotation_ops = depolarizing_kraus(rotation_probability) if rotation_probability > 0.0 else None

    def is_rotated_axis(axis: tuple[float, float, float]) -> bool:
        return tuple(axis) != (0.0, 0.0, 1.0)

    pairs = setting_pairs(protocol)
    rows: list[dict[str, Any]] = []
    for stage in (
        "elementary_generation_or_channel_failure",
        "stale_pair_cutoff",
        "memory_decay_detected",
        "swap_failure",
        "endpoint_detection_failure",
    ):
        probability = stages[stage]
        if probability > 0.0:
            rows.append({
                "delivery_status": "no_delivery",
                "failure_stage": stage,
                "alice_setting_label": "none",
                "bob_setting_label": "none",
                "alice_outcome": -1,
                "bob_outcome": -1,
                "sifting_class": "no_delivery",
                "eve_observation_record": "none",
                "effective_pauli_frame": "I",
                "public_bsm_record": "none",
                "detector_click_record": DETECTOR_CLICK_RECORD_NO_DELIVERY,
                "timing_record": "marginalized",
                "probability": probability,
            })
    for branch in branches:
        rotated_variants: dict[tuple[bool, bool], np.ndarray] = {(False, False): branch.rho}
        if rotation_ops is not None:
            rotated_variants[(True, False)] = apply_one_qubit_channel(branch.rho, 0, rotation_ops)
            rotated_variants[(False, True)] = apply_one_qubit_channel(branch.rho, 1, rotation_ops)
            rotated_variants[(True, True)] = apply_one_qubit_channel(
                rotated_variants[(True, False)], 1, rotation_ops
            )
        for alice_setting, bob_setting, p_setting in pairs:
            variant_key = (
                (rotation_ops is not None and is_rotated_axis(alice_setting.axis)),
                (rotation_ops is not None and is_rotated_axis(bob_setting.axis)),
            )
            measured_rho = rotated_variants.get(variant_key, branch.rho)
            ideal_probs = joint_outcome_probabilities(measured_rho, alice_setting.axis, bob_setting.axis)
            probs = apply_memory_readout_model_to_joint_probabilities(ideal_probs, readout)
            sift = sifting_class(protocol, alice_setting, bob_setting)
            for a_out in (0, 1):
                for b_out in (0, 1):
                    probability = p_delivery * branch.probability * p_setting * float(probs[a_out, b_out])
                    if probability <= 0.0:
                        continue
                    rows.append({
                        "delivery_status": "delivered",
                        "failure_stage": "none",
                        "alice_setting_label": alice_setting.label,
                        "bob_setting_label": bob_setting.label,
                        "alice_outcome": a_out,
                        "bob_outcome": b_out,
                        "sifting_class": sift,
                        "eve_observation_record": branch.eve_observation,
                        "effective_pauli_frame": branch.frame,
                        "public_bsm_record": branch.public_bsm_record if keep_full_bsm else "not_emitted",
                        "detector_click_record": detector_click_record_for_policy(
                            branch.detector_click_record,
                            keep_detector_click_records=keep_detector_click_records,
                        ),
                        "timing_record": "marginalized",
                        "probability": probability,
                    })
    rows = coalesce_rows(rows)
    for row in rows:
        row["expected_count"] = key_length * row["probability"]
    return rows


def generate_histogram_case(
    *,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    key_length: int,
    context: ModelContext | None = None,
    emit_full_bsm_audit: bool = False,
    max_full_bsm_repeaters: int = 2,
    emit_detector_click_audit: bool = False,
    emit_approximation_audit: bool = False,
    max_approximation_audit_repeaters: int = 2,
    seed: int | None = None,
    _suppress_audit: bool = False,
) -> dict[str, Any]:
    context = context or ModelContext()
    protocol = ProtocolConfig(requested_pairs=key_length)
    keep_full_bsm = emit_full_bsm_audit and route["repeaters"] <= max_full_bsm_repeaters
    keep_detector_click_records = bool(emit_detector_click_audit)
    stages = delivery_stage_probabilities(route, front, attack, context)
    branches = route_quantum_branches(
        route,
        front,
        attack,
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
        context=context,
    )
    rows = histogram_rows_from_branches(
        branches=branches,
        stages=stages,
        front=front,
        protocol=protocol,
        key_length=key_length,
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
        context=context,
    )
    metrics = expected_metrics(rows, protocol)
    waiting_certification = waiting_metric_certification(rows, protocol, stages, context)
    validation = validate_histogram(rows)
    audit = (
        {"enabled": False, "status": "suppressed_internal_audit_call"}
        if _suppress_audit else
        bsm_audit_status(
            route=route,
            front=front,
            attack=attack,
            key_length=key_length,
            context=context,
            enabled=emit_full_bsm_audit,
            max_repeaters=max_full_bsm_repeaters,
        )
    )
    if keep_full_bsm:
        audit["full_bsm_in_this_case"] = True
    approximation_audit = approximation_audit_status(
        route=route,
        front=front,
        attack=attack,
        key_length=key_length,
        context=context,
        enabled=emit_approximation_audit,
        max_repeaters=max_approximation_audit_repeaters,
    )
    sample = None
    if seed is not None:
        sample = finite_sample_counts(rows, key_length, seed)
    elementary_model = elementary_link_model_metadata(route, front, attack, context)
    theta_payload = frozen_theta_identity_payload(
        front=front,
        context=context,
        hop_count=route["hops"],
        attack=attack,
        protocol=protocol,
        fiber_length_symbols=fiber_length_symbols_for_route(route),
        emit_full_bsm_audit=emit_full_bsm_audit,
        max_full_bsm_repeaters=max_full_bsm_repeaters,
        emit_detector_click_audit=emit_detector_click_audit,
        emit_approximation_audit=emit_approximation_audit,
        max_approximation_audit_repeaters=max_approximation_audit_repeaters,
    )
    theta_summary = frozen_theta_identity_summary(theta_payload)
    active_memory_process = memory_process_model(front, context)
    active_storage_model = memory_storage_decoherence_model(front)
    active_readout_model = memory_readout_model(front)
    active_service_profile = active_repeater_service_profile(context)
    active_service_edge_channel = repeater_elementary_state_channel(context)
    active_service_swap_channel = repeater_swap_state_channel(context)
    active_service_quality_restoration = (
        quality_restoration_model(active_service_profile).to_dict()
        if active_service_profile is not None
        and quality_restoration_model(active_service_profile) is not None
        else None
    )
    wavelength_report = wavelength_compatibility_report(front, context)
    wavelength_model = {
        **wavelength_report,
        "source": (
            "memory.wavelength selected for memory_emission source model"
            if context.source_model == "memory_emission"
            else "source.wavelength selected for spdc source model"
        ),
        "detector_response_file_loaded": context.detector_response is not None,
        "fiber_response_file_loaded": context.fiber_response is not None,
        "conversion_response_file_loaded": context.conversion_response is not None,
        "scalar_flat_assumptions": elementary_model.get("wavelength_dependency_assumptions", []),
    }
    detector_formulas = elementary_model.get("detector_povm_formulas", {
        "sequence_bk_reference_threshold": "P(click|n)=1-(1-p_dark)*(1-eta)^n inside the SeQUeNCe Barrett-Kok reference abstraction",
    })
    optical_network = elementary_model.get("optical_bell_analyzer", {
        "implemented_network": "sequence_bk_reference uses SingleAtomBSM / Barrett-Kok heralding abstraction, not the Fock/PNR optical analyzer",
        "distinguishable_bell_outcomes": ["phi_plus after feed-forward target state"],
    })
    path_limitations = physical_path_limitations(front, context, hop_count=route["hops"])
    kernel_effects = probability_kernel_effects(front, context, hop_count=route["hops"])
    storage_policy = (
        deterministic_storage_timing_policy(route, front)
        if context.physical_path == "fock_pnr_snspd"
        else None
    )
    waiting_policy = (
        stochastic_waiting_time_policy(route, front, attack, context)
        if context.physical_path == "fock_pnr_snspd"
        and normalized_waiting_protocol(context) == "single_buffer_swap_asap"
        else None
    )
    rolling_policy = (
        rolling_live_pair_policy(route, front, context)
        if context.physical_path == "fock_pnr_snspd"
        and normalized_waiting_protocol(context) == "rolling_alive_pair"
        else None
    )
    resolved_kind = resolved_detector_kind(front, context)
    detector_profile = detector_profile_metadata(front, context)
    detector_rate = detector_count_rate_model(front, context)
    detector_max = detector_max_count(context, detector_profile)
    pnr_meta = pnr_response_model_metadata(detector_profile, resolved_kind, detector_max)
    if context.physical_path == "fock_pnr_snspd" and rolling_policy is not None:
        probability_space = {
            "histogram_draw": "one route request against maintained live elementary pairs",
            "waiting_protocol": "rolling_alive_pair",
            "waiting_model": "configured discrete selected-live-pair age distribution",
            "delivery_mass": "product_i A_i times scalar swap/endpoint success",
            "no_delivery_mass": "includes at least one edge unavailable at request time, plus swap/endpoint failures",
            "delivered_rows": (
                "endpoint transcript distribution conditioned on accepted live elementary pairs and their "
                "configured selected-age distribution, multiplied by delivery_mass"
            ),
            "not_equivalent_to": "single_buffer_swap_asap finite-budget route-from-scratch protocol",
            "pre_route_decay_and_stale_semantics": rolling_policy["pre_route_decay_and_stale_semantics"],
        }
        conditioning_convention = "unconditional_route_request_includes_no_live_pair_no_delivery_bins"
    elif context.physical_path == "fock_pnr_snspd" and route["hops"] >= 2:
        if route["hops"] == 2:
            waiting_model_description = "analytic two-link repeat-until-success waiting"
            runtime_budget = {
                "max_attempts_per_elementary_link": context.waiting_max_attempts,
                "max_history_pairs": context.waiting_max_history_pairs,
                "waiting_precision_mode": context.waiting_precision_mode,
                "waiting_tv_tolerance": context.waiting_tv_tolerance,
                "waiting_metric_tolerance": context.waiting_metric_tolerance,
            }
        else:
            waiting_model_description = (
                "adaptive bounded grouped swap-asap enumeration"
                if context.waiting_precision_mode == "adaptive_bounded" else
                "grouped swap-asap enumeration within attempt/signature caps"
            )
            runtime_budget = {
                "max_attempts_per_elementary_link": context.waiting_max_attempts,
                "max_history_pairs": context.waiting_max_history_pairs,
                "max_wait_signatures": context.waiting_max_wait_signatures,
                "waiting_precision_mode": context.waiting_precision_mode,
                "waiting_tv_tolerance": context.waiting_tv_tolerance,
                "waiting_metric_tolerance": context.waiting_metric_tolerance,
                "max_refinement_rounds": context.waiting_max_refinement_rounds,
                "max_runtime_s": context.waiting_max_runtime_s,
                "max_memory_mb": context.waiting_max_memory_mb,
            }
        probability_space = {
            "histogram_draw": "one finite-budget selected-route generation cycle",
            "waiting_model": waiting_model_description,
            "runtime_budget": runtime_budget,
            "delivery_mass": "probability the route is ready within the finite budget times scalar swap/endpoint success",
            "no_delivery_mass": "includes timeout/omitted waiting tail where route is not ready within budget",
            "delivered_rows": "endpoint transcript distribution conditioned on route readiness within budget and delivered swap branch, multiplied by delivery_mass",
        }
        conditioning_convention = "unconditional_finite_attempt_budget_includes_timeout_no_delivery_bins"
    else:
        probability_space = {
            "histogram_draw": "one synchronized elementary attempt",
            "delivery_mass": "single-attempt delivery probability",
            "no_delivery_mass": "single-attempt elementary/swap/endpoint failure mass",
        }
        conditioning_convention = "unconditional_per_attempt_includes_no_delivery_bins"
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "case_id": case_id(route, attack),
        "physical_path": context.physical_path,
        "physical_path_semantics": (
            "Fock-space optical-mode instrument with SNSPD/PNR detector POVMs"
            if context.physical_path == "fock_pnr_snspd"
            else "SeQUeNCe event-free reference abstraction for RouterNetTopo/Barrett-Kok"
        ),
        "source_model": context.source_model,
        "detector_kind": context.detector_kind,
        "resolved_detector_kind": resolved_kind,
        "detector_profile": detector_profile,
        "detector_profile_role": detector_profile.get("profile_role", "repository_default"),
        "detector_is_demonstrated_or_idealized": (
            "idealized" if resolved_kind == "ideal_pnr" else "demonstrated_profile_or_repository_scalar"
        ),
        "pnr_capability": pnr_meta["pnr_capability"],
        "max_resolvable_photon_number": detector_max,
        "pnr_response_model": pnr_meta["pnr_response_model"],
        "pnr_response_model_is_calibrated_to_profile": pnr_meta[
            "pnr_response_model_is_calibrated_to_profile"
        ],
        "detector_profile_completeness": pnr_meta["detector_profile_completeness"],
        "threshold_coarsening_applied": pnr_meta["threshold_coarsening_applied"],
        "fock_cutoff": context.fock_cutoff,
        "fock_truncation_mass": elementary_model.get("fock_truncation_mass", 0.0),
        "memory_state_construction": elementary_model.get(
            "memory_state_construction",
            "sequence_bk_reference_Barrett_Kok_density_assignment",
        ),
        "memory_process_model": active_memory_process,
        "memory_storage_decoherence_model": active_storage_model,
        "memory_readout_model": active_readout_model,
        "memory_survival_model": memory_survival_model(front),
        "repeater_service_profile": (
            profile_summary(active_service_profile)
            if active_service_profile is not None else None
        ),
        "repeater_service_elementary_state_channel": active_service_edge_channel,
        "repeater_service_quality_restoration": active_service_quality_restoration,
        "repeater_service_swap_state_channel": active_service_swap_channel,
        "endpoint_analysis_rotation_model": endpoint_analysis_rotation_model(front),
        "endpoint_memory_measurement_model": {
            "analysis_rotation_stage": (
                "rotated (non-Z) settings apply the typed endpoint_analysis_rotation_model depolarizing "
                "channel to the measured party's qubit before projection; Z-basis settings are direct "
                "fluorescence readout"
            ),
            "ideal_projector_stage": "rho_AB -> ideal actual-outcome probabilities for the selected E91 axes",
            "readout_stage": "ideal actual outcomes -> reported Alice/Bob transcript outcomes via memory_readout_model",
            "reported_transcript_fields": ["alice_outcome", "bob_outcome"],
            "readout_affects": active_readout_model.get("affects", []),
            "readout_does_not_affect": active_readout_model.get("does_not_affect", []),
            "scalar_fidelity_active": False,
        },
        "memory_fidelity_scalar_status": (
            "metadata/provenance only; not consumed as constructor/interface process noise"
            if not active_memory_process.get("scalar_fidelity_active") else
            "diagnostic override consumed scalar memory.fidelity as p_depol=1-fidelity"
        ),
        "sequence_bell_state_with_fidelity_used": elementary_model.get(
            "sequence_bell_state_with_fidelity_used",
            context.physical_path == "sequence_bk_reference",
        ),
        "model_context": context.to_dict(),
        "physical_model_context_hash": stable_hash(context.to_dict()),
        "frozen_theta_hash": theta_summary["theta_hash"],
        "frozen_theta_identity": theta_summary,
        "wavelength_model": wavelength_model,
        "detector_povm_formulas": detector_formulas,
        "bsm_optical_network_definition": optical_network,
        "distinguishable_bell_outcomes": optical_network.get("distinguishable_bell_outcomes", []),
        "inconclusive_or_failure_bins": {
            "elementary_optical_bsm": "inconclusive contributes to no_delivery through elementary_generation_or_channel_failure",
            "swap_bsm": "swap block or configured swap failure contributes to swap_failure",
        },
        "route": route,
        "linear_chain_contract": linear_chain_contract(route),
        "eve_move_contract": eve_move_contract(attack),
        "length_contract": length_contract(route),
        "fiber_length_variable_names": fiber_length_symbols_for_route(route),
        "device_front_matter_hash": front["front_matter_hash"],
        "hardware_config_source_paths": front["hardware_config_source_paths"],
        "device_parameter_inventory": front.get("device_parameter_inventory", front["mapped_device_parameters"]),
        "used_device_parameters": front["mapped_device_parameters"],
        "unsupported_device_parameters": front["unsupported_device_parameters"],
        "physical_path_limitations": path_limitations,
        "probability_kernel_effects": kernel_effects,
        "fallback_defaults_used": front.get("fallback_defaults_used", []),
        "adapter_conversion_notes": front["adapter_conversion_notes"],
        "repository_process_sources": front["sequence_component_formula_sources"],
        "protocol_config": protocol.to_dict(),
        "eve_move_spec": attack.to_dict(),
        "target_position": attack.target_id,
        "target_active": attack.target_active,
        "transcript_bin_schema": transcript_schema(),
        "transcript_bin_observability": transcript_schema()["observability"],
        "detector_click_record_policy": detector_click_record_policy_metadata(
            emit_detector_click_audit=emit_detector_click_audit
        ),
        "probability_space": probability_space,
        "waiting_budget_semantics": (
            waiting_policy.get("game_turn_budget_semantics")
            if waiting_policy is not None else
            rolling_policy
            if rolling_policy is not None else
            {
                "waiting_budget_semantics": "not_applicable",
                "reason": "no stochastic repeat-until-success waiting model is active",
            }
        ),
        "rolling_live_pair_policy": rolling_policy,
        "histogram_conditioning_convention": conditioning_convention,
        "key_length": key_length,
        "key_bit_definition": "Alice raw key bit for the first configured key setting pair",
        "effective_pauli_frame_encoding": {
            label: {"id": FRAME_ID[label], "bits": list(FRAME_BITS[label])}
            for label in FRAME_LABELS
        },
        "kernel_physics_flags": {
            "emit_full_bsm_audit": bool(emit_full_bsm_audit),
            "max_full_bsm_repeaters": int(max_full_bsm_repeaters),
            "emit_detector_click_audit": bool(emit_detector_click_audit),
            "emit_approximation_audit": bool(emit_approximation_audit),
            "max_approximation_audit_repeaters": int(max_approximation_audit_repeaters),
        },
        "full_bsm_audit_status": audit,
        "approximation_audit_status": approximation_audit,
        "expected_metrics": metrics,
        "waiting_metric_certification": waiting_certification,
        "validation_checks": validation,
        "delivery_model": stages,
        "elementary_link_model": elementary_model,
        "swap_model": {
            "swap_success_probability": front["device_constants"]["swap_bsm"]["success_probability"],
            "swapping_degradation": front["device_constants"]["swap_bsm"].get("swapping_degradation", 1.0),
            "swapping_degradation_mapping": "post-BSM two-qubit depolarizing probability = 1 - swapping_degradation",
            "active_repeater_service_swap_channel": active_service_swap_channel,
            "active_channel_precedence": (
                "repeater service single-qubit Bell-depolarizing Eq. S15 overrides the generic "
                "repository swapping_degradation channel"
                if active_service_swap_channel is not None else
                "repository swap_bsm channel"
            ),
            "phase_error": front["device_constants"]["swap_bsm"].get("phase_error", 0.0),
            "phase_error_mapping": "post-BSM Z-dephasing channel on output-frame qubit with probability phase_error",
            "swap_block": "targeted swap_block forces the selected repeater swap branch into no_delivery",
            "bell_outcome_confusion_model": swap_bell_outcome_confusion_model(front),
            "bell_outcome_confusion_semantics": (
                "typed classical confusion of the announced repeater Bell outcome label: the middle pair is "
                "projected onto the true outcome, but the announced label (which drives the public record, "
                "composed Pauli frame, and applied feed-forward correction) is drawn from the confusion row; "
                "distinct from endpoint memory readout, optical PNR detector confusion, and memory storage "
                "decoherence, and shared by all repeaters on the route"
            ),
        },
        "timing_variables": {
            "mode": "marginalized",
            "status": (
                "partially_modeled_for_fock_snspd_probability_kernel"
                if context.physical_path == "fock_pnr_snspd"
                else "modeled_by_sequence_bk_reference_helpers"
            ),
            "edge_delay_ps": {
                edge: edge_delay_ps(front, route["edge_lengths_m"][edge])
                for edge in route["edge_ids"]
            },
            "half_edge_delay_ps": {
                edge: half_edge_delay_ps(front, route["edge_lengths_m"][edge])
                for edge in route["edge_ids"]
            },
            "fiber_group_speed_m_per_ps": fiber_group_speed_m_per_ps(front),
            "fiber_propagation_speed_m_per_ps": fiber_group_speed_m_per_ps(front),
            "fiber_speed_source_field": front["device_constants"]["fiber_channel"].get(
                "fiber_speed_source_field",
                "fiber_channel.light_speed",
            ),
            "fiber_speed_interpretation": (
                "scalar guided-mode propagation/group speed along the fiber path coordinate/fiber arclength; "
                "not vacuum c, not vector velocity, and not a fiber dephasing parameter"
            ),
            "barrett_kok_trigger_window_ps": front["device_constants"]["detector"]["barrett_kok_trigger_window_ps"],
            "detector_dark_probability_per_trigger_window": detector_dark_window_probability(front),
            "detector_time_resolution_ps": front["device_constants"]["detector"]["time_resolution_ps"],
            "detector_count_rate_hz": detector_rate["count_rate_hz"],
            "detector_count_rate_source": detector_rate["source"],
            "detector_cooldown_period_ps": detector_cooldown_period_ps(front, context),
            "memory_coherence_time_s": front["device_constants"]["memory"]["coherence_time_s"],
            "memory_expiration_mode": (
                "deterministic_storage_time_decoherence_enters_fock_snspd_probability_kernel; "
                "SeQUeNCe-style hard expiration is not modeled; analytic stochastic waiting enters only where supported"
                if context.physical_path == "fock_pnr_snspd"
                else "hard cutoff from Memory._schedule_expiration; BDS decoherence inactive without decoherence_errors"
            ),
            "fock_timing_status": (
                (
                    "detector timing window enters dark-count mean; rolling_alive_pair uses configured "
                    "live-pair availability and selected-age distributions, with memory storage decoherence "
                    "entering branch density matrices through the selected live-pair ages; event-level "
                    "memory-pool queueing remains missing"
                )
                if rolling_policy is not None else
                "detector timing window enters dark-count mean; deterministic fiber-propagation-delay-induced memory storage-time decoherence enters branch density matrices; repeat-until-success waiting enters branch density matrices for h>=2 routes (analytic for h=2, grouped swap-asap enumeration within attempt cutoffs for h>=3); event-level asynchronous timing remains missing"
                if context.physical_path == "fock_pnr_snspd"
                else "not_applicable"
            ),
            "deterministic_storage_timing_policy": storage_policy,
            "stochastic_waiting_time_policy": waiting_policy,
            "rolling_live_pair_policy": rolling_policy,
            "memory_storage_decoherence_model": (
                {
                    "storage_model": active_storage_model,
                    "lifetime_parameters": memory_lifetime_parameters(front),
                    "channel_formula": {
                        "amplitude_damping": "gamma_1(t)=1-exp(-t/T1) when T1 is supplied",
                        "exponential_dephasing": "C(t)=exp(-t/T2); p_phi(t)=(1-C(t))/2",
                        "gaussian_dephasing": "C(t)=exp(-(t/T_gauss)^2); p_phi(t)=(1-C(t))/2",
                        "liu_2026_anisotropic_pair_fit": (
                            "local Pauli eigenvalues are the symmetric square roots of the measured "
                            "pair laws Vxx(t)/Vxx(0)=exp(-t/tau) and Vzz(t)/Vzz(0)=1-r*t/Vzz(0)"
                        ),
                        "combined_T1_T2": "1/T_phi = 1/T2 - 1/(2*T1), with validation T2 <= 2*T1",
                        "interval_composition": (
                            "t is the memory's total accumulated storage time; represented storage "
                            "intervals compose to C(t_total), with gaussian_dephasing applied through "
                            "the incremental coherence ratio C(t0+dt)/C(t0)"
                        ),
                    },
                    "application": "applied to branch stationary memory qubits before the next represented operation",
                    "locality": "memory-local channel; storage coherence parameters are read from the memory block and are not applied to flying photons or fiber modes",
                    "coherence_time_interpretation": active_storage_model.get(
                        "coherence_time_definition",
                        "typed storage model supplied by config",
                    ),
                    "source_reported_decay_model": front["device_constants"]["memory"].get("source_reported_decay_model", "not_specified"),
                    "kernel_assumed_decay_model": front["device_constants"]["memory"].get(
                        "kernel_assumed_storage_decay_model",
                        active_storage_model.get("coherence_factor_formula", "C(t)=1"),
                    ),
                    "storage_calibration_status": front["device_constants"]["memory"].get(
                        "memory_storage_calibration_status",
                        active_storage_model.get("calibration_status", "not_specified"),
                    ),
                    "chsh_loss_interpretation": "CHSH loss at short fiber lengths under tiny T2 is memory-lifetime-limited, not fiber-dephasing-limited",
                    "disabled_condition": "storage model kind none or zero deterministic storage time",
                }
                if context.physical_path == "fock_pnr_snspd"
                else None
            ),
            "implemented_in_probability_kernel": [
                row for row in kernel_effects["implemented_in_probability_kernel"]
                if "timing" in row["effect"] or "dark" in row["effect"]
                or "storage" in row["effect"]
                or "propagation" in row["effect"] or "attempt_rate" in row["effect"]
            ],
            "computed_metadata_only": kernel_effects["computed_metadata_only"],
            "missing": [
                row for row in kernel_effects["missing"]
                if row["effect"] in {"stochastic_waiting_time_decoherence", "detector_dead_time_dynamics"}
            ],
        },
        "probability_representation": "numeric_evaluator",
        "numeric_evaluator": {
            "function": "histogram_numeric_evaluator.evaluate_histogram(case_id, fiber_lengths_m)",
            "sampler": "histogram_numeric_evaluator.sample_histogram(case_id, fiber_lengths_m, shots, seed)",
            "in_process_api": (
                "generator module: load_or_build_front_matter(..., device_front_matter=...) + "
                "kernel_from_case_metadata(front, case_metadata) / evaluate_case_histogram(...) / "
                "sample_transcript_counts(...)"
            ),
            "theta_verification": "rebuilt kernels are checked against the recorded frozen_theta_hash",
            "fiber_lengths_argument": "list[float]",
            "length_contract": length_contract(route),
            "length_formula": (
                "fock_pnr_snspd: half-link optical transmittance T_i=10 ** (-alpha(lambda) * L_i / 20) before detector POVMs; sequence_bk_reference: Barrett-Kok half-edge transmittance uses the same channel law"
            ),
        },
        "optional_finite_sample": sample,
    }
    return {"metadata": metadata, "rows": rows}


def coalesce_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "delivery_status",
        "failure_stage",
        "alice_setting_label",
        "bob_setting_label",
        "alice_outcome",
        "bob_outcome",
        "sifting_class",
        "eve_observation_record",
        "effective_pauli_frame",
        "public_bsm_record",
        "detector_click_record",
        "timing_record",
    ]
    grouped: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped[key] = grouped.get(key, 0.0) + float(row["probability"])
    out = []
    for key, probability in sorted(grouped.items()):
        row = {field: value for field, value in zip(fields, key)}
        row["probability"] = probability
        out.append(row)
    return out


def expected_metrics(rows: list[dict[str, Any]], protocol: ProtocolConfig) -> dict[str, Any]:
    key_rows = [row for row in rows if row["sifting_class"] == "key" and row["delivery_status"] == "delivered"]
    key_prob = sum(row["probability"] for row in key_rows)
    err_prob = sum(row["probability"] for row in key_rows if is_key_error(row, protocol))
    expected_qber = (err_prob / key_prob) if key_prob > 0.0 else None
    if expected_qber is not None and not (-1e-12 <= expected_qber <= 1.0 + 1e-12):
        raise ValueError(
            "computed expected QBER outside [0, 1]: "
            f"numerator={err_prob:.17g}, denominator={key_prob:.17g}, qber={expected_qber:.17g}"
        )
    if expected_qber is not None:
        expected_qber = min(1.0, max(0.0, expected_qber))
    chsh = {}
    for a_label, b_label in protocol.chsh_pairs:
        selected = [
            row for row in rows
            if row["alice_setting_label"] == a_label
            and row["bob_setting_label"] == b_label
            and row["delivery_status"] == "delivered"
        ]
        total = sum(row["probability"] for row in selected)
        if total <= 0.0:
            chsh[f"{a_label}|{b_label}"] = None
            continue
        same = sum(row["probability"] for row in selected if row["alice_outcome"] == row["bob_outcome"])
        chsh[f"{a_label}|{b_label}"] = (2.0 * same - total) / total
    labels = [f"{a}|{b}" for a, b in protocol.chsh_pairs]
    chsh_s = None
    if all(chsh[label] is not None for label in labels):
        raw = chsh[labels[0]] + chsh[labels[1]] + chsh[labels[2]] - chsh[labels[3]]
        chsh_s = -raw
    eve_counts: dict[str, float] = {}
    for row in rows:
        obs = row["eve_observation_record"]
        if obs != "none":
            eve_counts[obs] = eve_counts.get(obs, 0.0) + row["expected_count"]
    guess = eve_guess_probability_from_rows(rows)
    return {
        "delivery_probability": sum(row["probability"] for row in rows if row["delivery_status"] == "delivered"),
        "expected_sifted_key_length": sum(row["expected_count"] for row in key_rows),
        "expected_key_error_count": sum(row["expected_count"] for row in key_rows if is_key_error(row, protocol)),
        "expected_qber": expected_qber,
        "expected_qber_from_histogram": expected_qber,
        "qber_key_probability_mass": key_prob,
        "qber_error_probability_mass": err_prob,
        "qber_numerator_probability": err_prob,
        "qber_denominator_probability": key_prob,
        "qber_histogram_observable": key_prob > 0.0,
        "qber_estimation_power": "adequate" if key_prob > 0.0 else "no_key_mass",
        "qber_diagnostics": {
            "definition": (
                "expected QBER is the infinite-shot key-basis error probability "
                "conditioned on delivered key-basis rows"
            ),
            "numerator_probability": err_prob,
            "denominator_probability": key_prob,
            "observable_from_histogram": key_prob > 0.0,
            "key_error_convention": (
                "anti-correlated key convention: Alice and Bob equal reported bits "
                "are counted as key errors"
            ),
        },
        "expected_chsh_correlations": chsh,
        "expected_chsh_S": chsh_s,
        "expected_eve_observation_counts": dict(sorted(eve_counts.items())),
        "eve_guess_probability": guess["conditional_guess_probability"],
        "eve_guess_probability_numerator": guess["guess_probability_mass"],
        "eve_guess_conditioning_probability": guess["key_probability_mass"],
    }


def _conditional_metric_bound(
    epsilon: float,
    denominator: float,
    metric_tolerance: float,
) -> dict[str, Any]:
    epsilon = max(0.0, float(epsilon))
    denominator = max(0.0, float(denominator))
    if denominator <= 0.0:
        return {
            "status": "not_applicable",
            "denominator": denominator,
            "bound": None,
            "reason": "metric denominator is zero in the retained histogram",
        }
    if epsilon >= denominator:
        return {
            "status": "denominator_too_small",
            "denominator": denominator,
            "bound": None,
            "reason": "omitted route-ready mass is at least as large as this metric denominator",
        }
    bound = 2.0 * epsilon / max(denominator - epsilon, 1e-300)
    return {
        "status": "certified" if bound <= float(metric_tolerance) else "bound_too_large",
        "denominator": denominator,
        "bound": bound,
        "reason": (
            "conditional metric perturbation bounded by 2*epsilon/(denominator-epsilon)"
        ),
    }


def waiting_metric_certification(
    rows: list[dict[str, Any]],
    protocol: ProtocolConfig,
    delivery_model: dict[str, Any],
    context: ModelContext,
) -> dict[str, Any]:
    waiting_model = delivery_model.get("waiting_model") or {}
    distribution = waiting_model.get("waiting_distribution") or {}
    if not distribution:
        return {
            "status": "not_applicable",
            "reason": "no stochastic waiting distribution is active for this case",
        }
    epsilon = float(distribution.get("compute_cap_omitted_route_ready_mass") or 0.0)
    tv_tolerance = float(context.waiting_tv_tolerance)
    metric_tolerance = float(context.waiting_metric_tolerance)
    key_rows = [
        row for row in rows
        if row["sifting_class"] == "key" and row["delivery_status"] == "delivered"
    ]
    key_denominator = sum(row["probability"] for row in key_rows)
    qber = _conditional_metric_bound(epsilon, key_denominator, metric_tolerance)
    eve = _conditional_metric_bound(epsilon, key_denominator, metric_tolerance)
    chsh_pairs: dict[str, dict[str, Any]] = {}
    chsh_bound_sum = 0.0
    chsh_blocking_statuses: list[str] = []
    for a_label, b_label in protocol.chsh_pairs:
        denominator = sum(
            row["probability"] for row in rows
            if row["delivery_status"] == "delivered"
            and row["alice_setting_label"] == a_label
            and row["bob_setting_label"] == b_label
        )
        bound = _conditional_metric_bound(epsilon, denominator, metric_tolerance)
        chsh_pairs[f"{a_label}|{b_label}"] = bound
        if bound["status"] == "certified":
            chsh_bound_sum += float(bound["bound"] or 0.0)
        elif bound["status"] != "not_applicable":
            chsh_blocking_statuses.append(str(bound["status"]))
    if chsh_blocking_statuses:
        chsh_status = (
            "denominator_too_small"
            if "denominator_too_small" in chsh_blocking_statuses else
            "bound_too_large"
        )
    elif not chsh_pairs:
        chsh_status = "not_applicable"
    else:
        chsh_status = "certified" if chsh_bound_sum <= metric_tolerance else "bound_too_large"
    component_statuses = [
        qber["status"],
        eve["status"],
        chsh_status,
    ]
    if epsilon > tv_tolerance:
        overall = "bound_too_large"
    elif "denominator_too_small" in component_statuses:
        overall = "denominator_too_small"
    elif "bound_too_large" in component_statuses:
        overall = "bound_too_large"
    else:
        overall = "certified"
    return {
        "status": overall,
        "waiting_precision_mode": context.waiting_precision_mode,
        "unconditional_tv_bound": epsilon,
        "delivery_probability_bound": epsilon,
        "target_tv_tolerance": tv_tolerance,
        "target_metric_tolerance": metric_tolerance,
        "qber_bound": qber,
        "chsh_bound": {
            "status": chsh_status,
            "setting_pair_bounds": chsh_pairs,
            "S_bound": chsh_bound_sum if chsh_status == "certified" else None,
            "reason": "CHSH S bound is the sum of certified per-setting conditional-correlation bounds",
        },
        "eve_guess_bound": eve,
        "metric_certification_status": overall,
        "interpretation": (
            "retained-support physics is exact; omitted route-ready support is bounded by "
            "unconditional_tv_bound and booked as no_delivery. Conditional metrics are certified "
            "only when their denominators exceed that omitted mass by enough margin."
        ),
    }


def row_probability_distribution(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    fields = [
        "delivery_status",
        "failure_stage",
        "alice_setting_label",
        "bob_setting_label",
        "alice_outcome",
        "bob_outcome",
        "sifting_class",
        "eve_observation_record",
        "effective_pauli_frame",
        "public_bsm_record",
        "detector_click_record",
        "timing_record",
    ]
    return {
        tuple(row[field] for field in fields): float(row["probability"])
        for row in rows
    }


def total_variation_distance(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> float:
    left = row_probability_distribution(left_rows)
    right = row_probability_distribution(right_rows)
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def probability_mapping_total_variation(left: dict[Any, float], right: dict[Any, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def conditional_total_variation_distance(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    delivery_status: str,
) -> float | None:
    def filtered(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float] | None:
        selected = [row for row in rows if row["delivery_status"] == delivery_status]
        total = sum(row["probability"] for row in selected)
        if total <= 0.0:
            return None
        return {
            key: probability / total
            for key, probability in row_probability_distribution(selected).items()
        }

    left = filtered(left_rows)
    right = filtered(right_rows)
    if left is None or right is None:
        return None
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def rows_with_public_bsm_marginalized(rows: list[dict[str, Any]], key_length: int) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        item = dict(row)
        if item["delivery_status"] == "delivered":
            item["public_bsm_record"] = "not_emitted"
        projected.append(item)
    out = coalesce_rows(projected)
    for row in out:
        row["expected_count"] = key_length * row["probability"]
    return out


def rows_with_detector_click_marginalized(rows: list[dict[str, Any]], key_length: int) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        item = dict(row)
        if item["delivery_status"] == "delivered":
            item["detector_click_record"] = DETECTOR_CLICK_RECORD_NOT_EMITTED
        projected.append(item)
    out = coalesce_rows(projected)
    for row in out:
        row["expected_count"] = key_length * row["probability"]
    return out


def detector_click_record_policy_metadata(*, emit_detector_click_audit: bool) -> dict[str, Any]:
    return {
        "default": "marginalized_after_detector_conditioning",
        "emit_detector_click_audit": bool(emit_detector_click_audit),
        "production_row_value": DETECTOR_CLICK_RECORD_NOT_EMITTED,
        "no_delivery_row_value": DETECTOR_CLICK_RECORD_NO_DELIVERY,
        "branch_key_policy": (
            "raw detector records are retained as branch/transcript coordinates"
            if emit_detector_click_audit else
            "raw detector records are replaced by the production sentinel before branch aggregation"
        ),
        "physics_semantics": (
            "raw detector records condition the remote memory state inside the elementary-link instrument "
            "before being marginalized from production transcript coordinates"
        ),
        "audit_mode": (
            "preserves raw detector count/click records for small-route validation"
            if emit_detector_click_audit else
            "disabled; production histograms emit detector_click_record='not_emitted' for delivered rows"
        ),
    }


def optional_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def metric_delta_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    return {
        "delivery_probability_delta": optional_delta(left["delivery_probability"], right["delivery_probability"]),
        "qber_delta": optional_delta(left["expected_qber"], right["expected_qber"]),
        "chsh_S_delta": optional_delta(left["expected_chsh_S"], right["expected_chsh_S"]),
        "eve_guess_probability_delta": optional_delta(left["eve_guess_probability"], right["eve_guess_probability"]),
    }


def bell_basis_density_matrix(rho: np.ndarray) -> np.ndarray:
    basis = np.column_stack([bell_ket(label) for label in BELL_LABELS])
    return basis.conj().T @ rho @ basis


def bell_overlaps(rho: np.ndarray) -> dict[str, float]:
    return {
        label: max(0.0, float(np.real_if_close(np.trace(bell_density(label) @ rho))))
        for label in BELL_LABELS
    }


def bell_basis_off_diagonal_l1(rho: np.ndarray) -> float:
    bell_rho = bell_basis_density_matrix(rho)
    off_diag = bell_rho - np.diag(np.diag(bell_rho))
    return float(np.sum(np.abs(off_diag)))


def pauli_twirled_density(rho: np.ndarray) -> np.ndarray:
    overlaps = bell_overlaps(rho)
    twirled = np.zeros_like(rho, dtype=complex)
    for label, probability in overlaps.items():
        twirled += probability * bell_density(label)
    return clean_density(twirled)


def pauli_twirl_branches(branches: list[Branch]) -> list[Branch]:
    return normalize_branches([
        Branch(
            probability=branch.probability,
            rho=pauli_twirled_density(branch.rho),
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        )
        for branch in branches
    ])


def branch_bell_overlap_summary(branches: list[Branch]) -> dict[str, Any]:
    if not branches:
        return {
            "branch_count": 0,
            "weighted_mean_bell_overlaps": {label: None for label in BELL_LABELS},
            "bell_overlap_spread_by_label": {label: None for label in BELL_LABELS},
            "psi_minus_overlap_spread": None,
            "max_bell_overlap_spread": None,
            "weighted_bell_basis_off_diagonal_l1": None,
            "max_bell_basis_off_diagonal_l1": None,
        }
    total = sum(branch.probability for branch in branches)
    weighted = {label: 0.0 for label in BELL_LABELS}
    values = {label: [] for label in BELL_LABELS}
    max_overlaps = []
    off_diag_values = []
    weighted_off_diag = 0.0
    for branch in branches:
        weight = branch.probability / total if total > 0.0 else 0.0
        overlaps = bell_overlaps(branch.rho)
        off_diag = bell_basis_off_diagonal_l1(branch.rho)
        weighted_off_diag += weight * off_diag
        off_diag_values.append(off_diag)
        max_overlaps.append(max(overlaps.values()))
        for label, overlap in overlaps.items():
            weighted[label] += weight * overlap
            values[label].append(overlap)
    spreads = {
        label: max(label_values) - min(label_values)
        for label, label_values in values.items()
    }
    return {
        "branch_count": len(branches),
        "weighted_mean_bell_overlaps": weighted,
        "bell_overlap_spread_by_label": spreads,
        "psi_minus_overlap_min": min(values["psi_minus"]),
        "psi_minus_overlap_max": max(values["psi_minus"]),
        "psi_minus_overlap_spread": spreads["psi_minus"],
        "max_bell_overlap_min": min(max_overlaps),
        "max_bell_overlap_max": max(max_overlaps),
        "max_bell_overlap_spread": max(max_overlaps) - min(max_overlaps),
        "weighted_bell_basis_off_diagonal_l1": weighted_off_diag,
        "max_bell_basis_off_diagonal_l1": max(off_diag_values),
    }


def histogram_view_from_branches(
    *,
    branches: list[Branch],
    stages: dict[str, Any],
    front: dict[str, Any],
    protocol: ProtocolConfig,
    key_length: int,
    keep_full_bsm: bool,
    context: ModelContext,
    keep_detector_click_records: bool = True,
) -> dict[str, Any]:
    rows = histogram_rows_from_branches(
        branches=branches,
        stages=stages,
        front=front,
        protocol=protocol,
        key_length=key_length,
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
        context=context,
    )
    metrics = expected_metrics(rows, protocol)
    public_records = {
        row["public_bsm_record"]
        for row in rows
        if row["delivery_status"] == "delivered"
    }
    return {
        "rows": rows,
        "metrics": metrics,
        "validation": validate_histogram(rows),
        "row_count": len(rows),
        "public_bsm_record_count": len(public_records),
        "endpoint_marginal_hash": stable_hash(endpoint_marginal(rows)),
        "row_distribution_hash": stable_hash(row_probability_distribution(rows)),
        "bell_overlap_summary": branch_bell_overlap_summary(branches),
    }


def histogram_view_from_rows(
    rows: list[dict[str, Any]],
    *,
    branches: list[Branch],
    protocol: ProtocolConfig,
) -> dict[str, Any]:
    metrics = expected_metrics(rows, protocol)
    public_records = {
        row["public_bsm_record"]
        for row in rows
        if row["delivery_status"] == "delivered"
    }
    return {
        "rows": rows,
        "metrics": metrics,
        "validation": validate_histogram(rows),
        "row_count": len(rows),
        "public_bsm_record_count": len(public_records),
        "endpoint_marginal_hash": stable_hash(endpoint_marginal(rows)),
        "row_distribution_hash": stable_hash(row_probability_distribution(rows)),
        "bell_overlap_summary": branch_bell_overlap_summary(branches),
    }


def compare_histogram_views(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_bell = left["bell_overlap_summary"]
    right_bell = right["bell_overlap_summary"]
    endpoint_tv = probability_mapping_total_variation(
        endpoint_marginal(left["rows"]),
        endpoint_marginal(right["rows"]),
    )
    return {
        "tv_distance": total_variation_distance(left["rows"], right["rows"]),
        "tv_distance_conditioned_on_delivery": conditional_total_variation_distance(
            left["rows"], right["rows"], delivery_status="delivered"
        ),
        "metric_deltas_left_minus_right": metric_delta_summary(left["metrics"], right["metrics"]),
        "bell_overlap_spread_deltas_left_minus_right": {
            "psi_minus_overlap_spread_delta": optional_delta(
                left_bell["psi_minus_overlap_spread"],
                right_bell["psi_minus_overlap_spread"],
            ),
            "max_bell_overlap_spread_delta": optional_delta(
                left_bell["max_bell_overlap_spread"],
                right_bell["max_bell_overlap_spread"],
            ),
            "weighted_bell_basis_off_diagonal_l1_delta": optional_delta(
                left_bell["weighted_bell_basis_off_diagonal_l1"],
                right_bell["weighted_bell_basis_off_diagonal_l1"],
            ),
            "max_bell_basis_off_diagonal_l1_delta": optional_delta(
                left_bell["max_bell_basis_off_diagonal_l1"],
                right_bell["max_bell_basis_off_diagonal_l1"],
            ),
        },
        "endpoint_marginal_tv_distance": endpoint_tv,
        "endpoint_marginal_agrees": endpoint_tv <= 1e-12,
    }


def is_key_error(row: dict[str, Any], protocol: ProtocolConfig) -> bool:
    if protocol.key_outcome_relation == "anti_correlated":
        return int(row["alice_outcome"]) == int(row["bob_outcome"])
    return int(row["alice_outcome"]) != int(row["bob_outcome"])


def eve_guess_probability_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["sifting_class"] != "key" or row["delivery_status"] != "delivered":
            continue
        z_record = (row["eve_observation_record"], row["effective_pauli_frame"])
        grouped.setdefault(z_record, [0.0, 0.0])
        grouped[z_record][int(row["alice_outcome"])] += row["probability"]
    key_mass = sum(sum(values) for values in grouped.values())
    guess_mass = sum(max(values) for values in grouped.values())
    return {
        "key_probability_mass": key_mass,
        "guess_probability_mass": guess_mass,
        "conditional_guess_probability": (guess_mass / key_mass) if key_mass > 0.0 else 0.5,
    }


def validate_histogram(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["probability"] for row in rows)
    min_prob = min((row["probability"] for row in rows), default=0.0)
    return {
        "probability_sum": total,
        "probabilities_sum_to_one": abs(total - 1.0) <= 1e-7,
        "all_probabilities_nonnegative": min_prob >= -TOL,
        "minimum_probability": min_prob,
    }


def approximation_view_metadata(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": view["row_count"],
        "public_bsm_record_count": view["public_bsm_record_count"],
        "endpoint_marginal_hash": view["endpoint_marginal_hash"],
        "row_distribution_hash": view["row_distribution_hash"],
        "validation": view["validation"],
        "metrics": view["metrics"],
        "bell_overlap_summary": view["bell_overlap_summary"],
    }


def approximation_audit_status(
    *,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    key_length: int,
    context: ModelContext | None = None,
    enabled: bool,
    max_repeaters: int,
) -> dict[str, Any]:
    context = context or ModelContext()
    if not enabled:
        return {"enabled": False, "status": "disabled"}
    if context.physical_path != "fock_pnr_snspd":
        return {
            "enabled": True,
            "status": "skipped_unsupported_physical_path",
            "physical_path": context.physical_path,
        }
    if route["repeaters"] > max_repeaters:
        return {
            "enabled": True,
            "status": "skipped_route_too_large",
            "max_approximation_audit_repeaters": max_repeaters,
        }

    protocol = ProtocolConfig(requested_pairs=key_length)
    stages = delivery_stage_probabilities(route, front, attack, context)
    exact_resolved_branches = route_quantum_branches(
        route, front, attack, keep_full_bsm=True, keep_detector_click_records=False, context=context
    )
    exact_marginalized_branches = route_quantum_branches(
        route, front, attack, keep_full_bsm=False, keep_detector_click_records=False, context=context
    )
    twirled_resolved_branches = pauli_twirl_branches(exact_resolved_branches)
    twirled_marginalized_branches = pauli_twirl_branches(exact_marginalized_branches)

    exact_resolved = histogram_view_from_branches(
        branches=exact_resolved_branches,
        stages=stages,
        front=front,
        protocol=protocol,
        key_length=key_length,
        keep_full_bsm=True,
        context=context,
        keep_detector_click_records=False,
    )
    exact_marginalized = histogram_view_from_branches(
        branches=exact_marginalized_branches,
        stages=stages,
        front=front,
        protocol=protocol,
        key_length=key_length,
        keep_full_bsm=False,
        context=context,
        keep_detector_click_records=False,
    )
    twirled_resolved = histogram_view_from_branches(
        branches=twirled_resolved_branches,
        stages=stages,
        front=front,
        protocol=protocol,
        key_length=key_length,
        keep_full_bsm=True,
        context=context,
        keep_detector_click_records=False,
    )
    twirled_marginalized = histogram_view_from_branches(
        branches=twirled_marginalized_branches,
        stages=stages,
        front=front,
        protocol=protocol,
        key_length=key_length,
        keep_full_bsm=False,
        context=context,
        keep_detector_click_records=False,
    )

    exact_resolved_public_marginal = histogram_view_from_rows(
        rows_with_public_bsm_marginalized(exact_resolved["rows"], key_length),
        branches=exact_resolved_branches,
        protocol=protocol,
    )
    twirled_resolved_public_marginal = histogram_view_from_rows(
        rows_with_public_bsm_marginalized(twirled_resolved["rows"], key_length),
        branches=twirled_resolved_branches,
        protocol=protocol,
    )

    comparisons = {
        "exact_vs_pauli_twirled_branch_resolved": compare_histogram_views(
            exact_resolved, twirled_resolved
        ),
        "exact_vs_pauli_twirled_branch_marginalized": compare_histogram_views(
            exact_marginalized, twirled_marginalized
        ),
        "branch_resolved_vs_marginalized_exact_after_public_record_marginalization": compare_histogram_views(
            exact_resolved_public_marginal, exact_marginalized
        ),
        "branch_resolved_vs_marginalized_twirled_after_public_record_marginalization": compare_histogram_views(
            twirled_resolved_public_marginal, twirled_marginalized
        ),
    }
    return {
        "enabled": True,
        "status": "computed",
        "physical_path": context.physical_path,
        "source_model": context.source_model,
        "approximation": {
            "exact_coherent_channel": "route branches retain the density matrices produced by the selected-route Fock/SNSPD instrument",
            "pauli_twirled_approximation": "each delivered branch density matrix is projected to its Bell-basis diagonal before endpoint E91 measurement",
            "branch_resolved_public_records": "full public BSM strings are retained as transcript coordinates",
            "branch_marginalized_public_records": "full public BSM strings are marginalized to the composed effective_pauli_frame coordinate",
        },
        "max_approximation_audit_repeaters": max_repeaters,
        "views": {
            "exact_branch_resolved": approximation_view_metadata(exact_resolved),
            "exact_branch_marginalized": approximation_view_metadata(exact_marginalized),
            "pauli_twirled_branch_resolved": approximation_view_metadata(twirled_resolved),
            "pauli_twirled_branch_marginalized": approximation_view_metadata(twirled_marginalized),
        },
        "comparisons": comparisons,
    }


def bsm_audit_status(
    *,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    key_length: int,
    context: ModelContext | None = None,
    enabled: bool,
    max_repeaters: int,
) -> dict[str, Any]:
    context = context or ModelContext()
    if not enabled:
        return {"enabled": False, "status": "disabled"}
    if route["repeaters"] > max_repeaters:
        return {
            "enabled": True,
            "status": "skipped_route_too_large",
            "max_full_bsm_repeaters": max_repeaters,
        }
    compressed = generate_histogram_case(
        route=route,
        front=front,
        attack=attack,
        key_length=key_length,
        context=context,
        emit_full_bsm_audit=False,
        max_full_bsm_repeaters=max_repeaters,
        seed=None,
        _suppress_audit=True,
    )
    full = generate_histogram_case(
        route=route,
        front=front,
        attack=attack,
        key_length=key_length,
        context=context,
        emit_full_bsm_audit=True,
        max_full_bsm_repeaters=max_repeaters,
        seed=None,
        _suppress_audit=True,
    )
    endpoint = endpoint_marginal(compressed["rows"])
    full_endpoint = endpoint_marginal(full["rows"])
    metrics = compressed["metadata"]["expected_metrics"]
    full_metrics = full["metadata"]["expected_metrics"]
    endpoint_tv = probability_mapping_total_variation(endpoint, full_endpoint)
    endpoint_agrees = endpoint_tv <= 1e-12
    qber_agrees = close_optional(metrics["expected_qber"], full_metrics["expected_qber"])
    chsh_agrees = close_optional(metrics["expected_chsh_S"], full_metrics["expected_chsh_S"])
    posterior_agrees = close_optional(
        metrics["eve_guess_probability"],
        full_metrics["eve_guess_probability"],
    )
    delivery_agrees = close_optional(
        metrics["delivery_probability"],
        full_metrics["delivery_probability"],
    )
    verified = endpoint_agrees and qber_agrees and chsh_agrees and posterior_agrees and delivery_agrees
    return {
        "enabled": True,
        "status": "verified" if verified else "failed",
        "max_full_bsm_repeaters": max_repeaters,
        "endpoint_marginal_hash": stable_hash(endpoint),
        "full_bsm_endpoint_marginal_hash": stable_hash(full_endpoint),
        "endpoint_marginal_tv_distance": endpoint_tv,
        "endpoint_marginal_agrees": endpoint_agrees,
        "qber_agrees": qber_agrees,
        "chsh_agrees": chsh_agrees,
        "eve_posterior_agrees": posterior_agrees,
        "delivery_probability_agrees": delivery_agrees,
        "qber": metrics["expected_qber"],
        "chsh_S": metrics["expected_chsh_S"],
        "eve_guess_probability": metrics["eve_guess_probability"],
        "physical_path": context.physical_path,
        "verification_note": "Default Pauli-frame compression is produced by marginalizing branch records over full BSM strings after composing public correction frames.",
    }


def close_optional(left: float | None, right: float | None, tol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tol


def endpoint_marginal(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for row in rows:
        key = "|".join(str(row[field]) for field in (
            "delivery_status",
            "alice_setting_label",
            "bob_setting_label",
            "alice_outcome",
            "bob_outcome",
            "sifting_class",
            "eve_observation_record",
            "effective_pauli_frame",
        ))
        grouped[key] = grouped.get(key, 0.0) + row["probability"]
    return dict(sorted(grouped.items()))


def finite_sample_counts(rows: list[dict[str, Any]], key_length: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    probabilities = np.asarray([row["probability"] for row in rows], dtype=float)
    probabilities = probabilities / probabilities.sum()
    counts = rng.multinomial(key_length, probabilities)
    return {
        "seed": seed,
        "counts_by_bin_index": {
            str(idx): int(count)
            for idx, count in enumerate(counts) if int(count) > 0
        },
    }


def model_context_from_dict(payload: dict[str, Any]) -> ModelContext:
    """Rebuild a ModelContext from a serialized context.to_dict() payload.

    to_dict() emits derived keys (hashes, *_loaded flags) alongside the
    constructor fields; only the dataclass fields are consumed here.
    """
    from dataclasses import fields as dataclass_fields

    field_names = {field.name for field in dataclass_fields(ModelContext)}
    kwargs = {key: value for key, value in payload.items() if key in field_names}
    return ModelContext(**kwargs)


def eve_move_from_dict(payload: dict[str, Any]) -> EveMove:
    """Rebuild an EveMove from a serialized move.to_dict() payload."""
    return EveMove(**payload)


def kernel_from_case_metadata(front: dict[str, Any], case_metadata: dict[str, Any]) -> FrozenHistogramKernel:
    """Rebuild the frozen-theta kernel family recorded in emitted case metadata.

    ``front`` must be the front matter the case was generated from (load it
    with load_or_build_front_matter(..., device_front_matter=...) from the
    emitted histogram_device_front_matter.json).
    """
    context = model_context_from_dict(case_metadata["model_context"])
    attack = eve_move_from_dict(case_metadata["eve_move_spec"])
    route = case_metadata["route"]
    flags = case_metadata.get("kernel_physics_flags", {})
    symbols = route.get(
        "fiber_length_symbols",
        case_metadata.get("fiber_length_variable_names"),
    )
    return FrozenHistogramKernel(
        front=front,
        context=context,
        hop_count=int(route["hops"]),
        attack=attack,
        key_length=int(case_metadata.get("key_length", 256)),
        fiber_length_symbols=tuple(symbols) if symbols else None,
        emit_full_bsm_audit=bool(flags.get("emit_full_bsm_audit", False)),
        max_full_bsm_repeaters=int(flags.get("max_full_bsm_repeaters", 2)),
        emit_detector_click_audit=bool(flags.get("emit_detector_click_audit", False)),
        emit_approximation_audit=bool(flags.get("emit_approximation_audit", False)),
        max_approximation_audit_repeaters=int(flags.get("max_approximation_audit_repeaters", 2)),
    )


def evaluate_case_histogram(
    front: dict[str, Any],
    case_metadata: dict[str, Any],
    fiber_lengths_m: Sequence[float],
    *,
    strict_theta: bool = True,
) -> TranscriptDistribution:
    """Evaluate the case's frozen kernel at a new fiber-length vector in-process.

    With strict_theta the rebuilt kernel must reproduce the recorded
    frozen_theta_hash, guaranteeing the evaluation belongs to the same frozen
    model family as the emitted case.
    """
    kernel = kernel_from_case_metadata(front, case_metadata)
    recorded = case_metadata.get("frozen_theta_hash")
    if strict_theta and recorded is not None and kernel.theta_hash != recorded:
        raise ValueError(
            "rebuilt kernel theta hash does not match the emitted case "
            f"({kernel.theta_hash} != {recorded}); the supplied front matter or metadata "
            "does not belong to this frozen case family"
        )
    return kernel.evaluate(list(fiber_lengths_m))


def sample_transcript_counts(
    distribution: TranscriptDistribution,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    """Multinomial transcript-record sample from an evaluated distribution."""
    sample = finite_sample_counts(distribution.rows, int(shots), int(seed))
    return {
        **sample,
        "shots": int(shots),
        "rows_by_bin_index": {
            index: distribution.rows[int(index)]
            for index in sample["counts_by_bin_index"]
        },
    }


# ---------------------------------------------------------------------------
# Compiled length-parametric histogram kernel
#
# H_{theta,h,move}(L) evaluated fast: the elementary edge instrument is an
# exact polynomial in s_i = sqrt(eta_path(L_i)) (fitted by probing the trusted
# instrument at eta nodes and verified on held-out nodes), the storage models
# in scope are pure dephasing (coherence factors multiply to C(total age)), and
# every downstream step is linear in the unnormalized branch operators, so the
# composition becomes a signature-batched fold of 4x4 operator tables with
# per-signature coherence masks. The classical layer (waiting signatures,
# survival, cutoff, delivery decomposition) reuses the reference functions
# verbatim, so P(tau) matches FrozenHistogramKernel.evaluate to floating-point
# precision; verify_compiled_kernel checks exactly that.
# ---------------------------------------------------------------------------
