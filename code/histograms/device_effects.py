"""Device-effect coverage, fail-closed checks, and path limitations."""

from __future__ import annotations

from typing import Any

from .device_models import *
from .device_profiles import *
from .models import *
def probability_kernel_effects(
    front: dict[str, Any],
    context: ModelContext,
    hop_count: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from .fock import conversion_noise_rate_per_s, detector_max_count, resolved_detector_kind
    from .timing import memory_survival_model, normalized_waiting_protocol

    if context.physical_path != "fock_pnr_snspd":
        return {
            "implemented_in_probability_kernel": [
                {
                    "effect": "sequence_bk_reference_barrett_kok_abstraction",
                    "status": "implemented_only_in_sequence_reference_helpers",
                    "enters_probability": True,
                }
            ],
            "generic_approximations": [],
            "computed_metadata_only": [],
            "implemented_only_in_sequence_reference_helpers": [],
            "missing": [],
            "intentionally_excluded": [],
        }
    resolved_kind = resolved_detector_kind(front, context)
    profile = detector_profile_metadata(front, context)
    max_count = detector_max_count(context, profile)
    pnr_meta = pnr_response_model_metadata(profile, resolved_kind, max_count)
    memory = front["device_constants"]["memory"]
    process_model = memory_process_model(front, context)
    storage_model = memory_storage_decoherence_model(front)
    survival_model = memory_survival_model(front)
    readout_model = memory_readout_model(front)
    polarization_model = fiber_polarization_noise_model(front)
    indistinguishability_model = photon_indistinguishability_model(front)
    swap_confusion_model = swap_bell_outcome_confusion_model(front)
    service_profile = active_repeater_service_profile(context)
    service_pair_instrument = repeater_elementary_pair_instrument(context)
    service_edge_channel = repeater_elementary_state_channel(context)
    service_swap_channel = repeater_swap_state_channel(context)
    service_restoration = (
        quality_restoration_model(service_profile)
        if service_profile is not None else None
    )
    service_aggregate_link = bool(
        service_profile is not None
        and service_profile.histogram_adapter.get(
            "aggregate_zero_loss_link_probability", False
        )
    )
    endpoint_measurement_aggregated = bool(
        service_pair_instrument is not None
        and service_profile is not None
        and service_profile.histogram_adapter.get(
            "endpoint_measurement_aggregated_in_empirical_state", False
        )
    )
    polarization_active = bool(polarization_model.get("enters_probability", False))
    indistinguishability_active = bool(indistinguishability_model.get("enters_probability", False))
    swap_confusion_active = bool(swap_confusion_model.get("enters_probability", False))
    wavelength_report = wavelength_compatibility_report(front, context)
    has_storage_lifetime_field = bool(storage_model.get("enters_probability", False))
    missing: list[dict[str, Any]] = [] if service_pair_instrument is not None else [
        {
            "effect": "detector_dead_time_dynamics",
            "parameter": "detector_profile.dead_time_or_max_count_rate_hz or detector.count_rate fallback",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "the selected detector count/reset rate bounds the inter-attempt period, but single-window "
                "detector POVMs do not model stateful repeated-window reset/dead-time correlations"
            ),
        },
    ]
    if service_pair_instrument is None and not indistinguishability_active:
        missing.append({
            "effect": "hom_wave_packet_indistinguishability",
            "parameter": "source.bandwidth",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "the optical Bell analyzer assumes perfectly indistinguishable modes; supply a typed "
                "memory.photon_indistinguishability_model for the phenomenological mode-overlap mixture"
            ),
        })
    if service_pair_instrument is None and not polarization_active:
        missing.append({
            "effect": "fiber_polarization_drift",
            "parameter": "fiber_channel.polarization_fidelity",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "polarization_fidelity is provenance only; no polarization noise channel is applied to Fock "
                "modes without a typed fiber_channel.polarization_noise_model"
            ),
        })
    if service_pair_instrument is None:
        missing.append({
            "effect": "physical_repeater_bsm_instrument",
            "parameter": "swap_bsm.success_probability",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "repeater swaps use ideal Bell projection plus scalar success/degradation/phase abstractions"
                + (
                    "; the typed Bell-outcome label confusion model is active but does not replace a physical "
                    "optical repeater-BSM instrument"
                    if swap_confusion_active else
                    "; a typed swap_bsm.bell_outcome_confusion_model may additionally model announced-label errors"
                )
            ),
        })
    if (
        service_pair_instrument is None
        and resolved_kind == "pnr_snspd"
        and not pnr_meta["pnr_response_model_is_calibrated_to_profile"]
    ):
        missing.append({
            "effect": "calibrated_pnr_confusion_matrix",
            "parameter": "detector.pnr_confusion_matrix",
            "status": "missing",
            "enters_probability": False,
            "reason": "PNR-capable detector profile is evaluated with the generic saturated binomial+dark response model",
        })
    if context.source_model == "spdc":
        missing.append({
            "effect": "spdc_multipair_memory_conditioning",
            "parameter": "source.mean_photon_num",
            "status": "missing",
            "enters_probability": False,
            "reason": "SPDC multipair memory-state conditioning is explicitly rejected for this kernel",
        })
    for issue in wavelength_report["issues"]:
        if (
            service_pair_instrument is None
            and issue["status"] == "missing"
            and not context.allow_wavelength_mismatch
        ):
            missing.append({
                "effect": issue["effect"],
                "parameter": issue["parameter"],
                "status": "missing",
                "enters_probability": False,
                "reason": issue["reason"],
            })
    if service_pair_instrument is None and process_model.get("kind") == "none":
        missing.append({
            "effect": "explicit_memory_process_model",
            "parameter": "memory.memory_process_model",
            "status": "missing",
            "enters_probability": False,
            "reason": "scalar memory.fidelity is not a calibrated memory process channel",
        })
    elif process_model.get("diagnostic_override"):
        missing.append({
            "effect": "calibrated_memory_process_model",
            "parameter": "memory.fidelity",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "diagnostic override maps scalar memory.fidelity to depolarizing noise, but scalar "
                "memory.fidelity is not a calibrated memory process channel"
            ),
        })
    if readout_model.get("kind") == "none" and not endpoint_measurement_aggregated:
        missing.append({
            "effect": "endpoint_memory_readout_error_model",
            "parameter": "memory.memory_readout_model",
            "status": "missing",
            "enters_probability": False,
            "reason": "no explicit endpoint memory-readout assignment error model is supplied; endpoint memory measurement is ideal",
        })
    rotation_model = endpoint_analysis_rotation_model(front)
    if rotation_model.get("kind") == "none" and not endpoint_measurement_aggregated:
        missing.append({
            "effect": "endpoint_analysis_rotation_error_model",
            "parameter": "memory.endpoint_analysis_rotation_model",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "no typed analysis-rotation error model is supplied; rotated (non-Z) endpoint "
                "measurement settings use ideal analysis pulses (no node-specific rotation fidelity "
                "is reported in the archived sources)"
            ),
        })
    if not survival_model["enabled"] and service_profile is None:
        missing.append({
            "effect": "detected_memory_decay_survival",
            "parameter": "memory.detected_decay_lifetime_s",
            "status": "missing",
            "enters_probability": False,
            "reason": (
                "no storage-level detected-decay lifetime is supplied, so heralded memory loss during "
                "storage is not represented in delivery mass"
            ),
        })
    if storage_model.get("legacy_fallback"):
        missing.append({
            "effect": "typed_memory_storage_decoherence_model",
            "parameter": "memory.memory_storage_decoherence_model",
            "status": "missing",
            "enters_probability": True,
            "reason": (
                "memory.coherence_time was consumed through a legacy exponential fallback because no typed "
                "memory_storage_decoherence_model was supplied"
            ),
        })
    implemented: list[dict[str, Any]] = []
    if service_profile is not None:
        implemented.append({
            "effect": "asynchronous_repeater_service_profile",
            "parameter": "context.repeater_service_profile",
            "status": "implemented_explicit_claim_tier",
            "enters_probability": True,
            "length_parameterized": True,
            "profile_id": service_profile.profile_id,
            "scope": service_profile.scope,
            "claim_tier": service_profile.claim_tier,
            "resource_inventory": service_profile.hardware,
            "rate_model": service_profile.rate_model.get("nested_time_model"),
        })
    if service_pair_instrument is not None:
        implemented.append({
            "effect": "empirical_remote_memory_pair_instrument",
            "parameter": "repeater_service_profile.state_model",
            "status": "implemented_conditional_accepted_pair",
            "enters_probability": True,
            "length_parameterized": False,
            "instrument": service_pair_instrument,
            "mapping": (
                "the demonstrated optical stack is aggregated into the measured renewal rate "
                "and conditional pair state; the repository two-photon Fock analyzer is bypassed"
            ),
        })
        if endpoint_measurement_aggregated:
            implemented.append({
                "effect": "empirical_endpoint_measurement_aggregation",
                "parameter": "repeater_service_profile.histogram_adapter.endpoint_measurement_aggregated_in_empirical_state",
                "status": "implemented_without_double_counting",
                "enters_probability": True,
                "mapping": (
                    "endpoint rotations, assignment errors, and other state-analysis errors "
                    "are already present in the source-reported effective correlations"
                ),
            })
        known_storage_limitation = service_profile.memory_model.get(
            "known_storage_limitation"
        )
        if known_storage_limitation:
            missing.append({
                "effect": "source_reported_anisotropic_storage_degradation",
                "parameter": "repeater_service_profile.memory_model.known_storage_limitation",
                "status": "missing_component_channel",
                "enters_probability": False,
                "reason": str(known_storage_limitation),
            })
    if service_edge_channel is not None:
        implemented.append({
            "effect": "source_specific_elementary_link_state_channel",
            "parameter": "repeater_service_profile.state_model",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "channel": service_edge_channel,
        })
    if service_restoration is not None:
        implemented.append({
            "effect": "elementary_link_entanglement_purification",
            "parameter": "repeater_service_profile.state_model.quality_restoration",
            "status": "implemented_exact_noisy_two_copy_density_instrument",
            "enters_probability": True,
            "length_parameterized": True,
            "model": service_restoration.to_dict(),
            "resource_inventory": service_profile.hardware,
            "mapping": (
                "detector-conditioned raw pair branches enter a bilateral phase-parity "
                "recurrence before rolling storage and repeater swapping"
            ),
        })
        missing.append({
            "effect": "purification_sacrificial_readout_crosstalk_process",
            "parameter": (
                "repeater_service_profile.state_model.quality_restoration."
                "retained_memory_measurement_crosstalk_model"
            ),
            "status": "missing_component_integration_calibration",
            "enters_probability": False,
            "reason": (
                "the dual-species/multi-zone composite is selected to isolate retained "
                "memory from sacrificial fluorescence, but no process matrix for this exact "
                "integrated node is available; no invented crosstalk probability is applied"
            ),
        })
    if service_swap_channel is not None:
        implemented.append({
            "effect": "source_specific_local_swap_state_channel",
            "parameter": "repeater_service_profile.state_model.local_swap_bell_fidelity",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "channel": service_swap_channel,
        })
    propagation_delay_enters = bool(has_storage_lifetime_field or survival_model["enabled"])
    if propagation_delay_enters:
        implemented.append({
            "effect": "fiber_propagation_delay",
            "parameter": "fiber_channel.light_speed,route edge lengths",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": True,
            "mapping": (
                "round(length_m / fiber_group_speed_m_per_ps); delay sets elementary herald and "
                "classical-confirmation storage ages consumed by storage decoherence and/or detected decay"
            ),
        })
    detector_rate = detector_count_rate_model(front, context)
    attempt_rate_enters = bool(
        detector_rate["count_rate_hz"] is not None
        and normalized_waiting_protocol(context) == "single_buffer_swap_asap"
        and (hop_count is None or hop_count >= 2)
    )
    if attempt_rate_enters:
        implemented.append({
            "effect": "detector_attempt_rate_constraint",
            "parameter": detector_rate["profile_field"],
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "count_rate_hz": detector_rate["count_rate_hz"],
            "source": detector_rate["source"],
            "calibration_status": detector_rate["mapping_semantics"],
            "mapping": (
                "the inter-attempt period is bounded below by 1/count_rate; the selected detector "
                "profile rate takes precedence over the repository scalar"
            ),
        })
    if has_storage_lifetime_field:
        implemented.append({
            "effect": "deterministic_storage_time_decoherence",
            "parameter": storage_model.get("source_field", "memory.memory_storage_decoherence_model"),
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": True,
            "timing_policy": "one-shot elementary herald delay plus deterministic adjacent-link swap wait",
            "channel": storage_model.get("coherence_factor_formula", "typed storage decoherence model"),
            "storage_model_kind": storage_model.get("kind", "none"),
            "locality": "memory-local channel applied to stationary memory qubits only; not applied to flying photons or fiber modes",
            "coherence_time_interpretation": storage_model.get(
                "coherence_time_definition",
                "typed storage model supplied by config",
            ),
            "calibration_status": storage_model.get("calibration_status", "not_specified"),
        })
        waiting_supported = hop_count is None or hop_count >= 2
        if normalized_waiting_protocol(context) == "rolling_alive_pair":
            implemented.append({
                "effect": "rolling_live_pair_age_decoherence",
                "parameter": "rolling_age_distribution_points_s,memory.memory_storage_decoherence_model",
                "status": "implemented",
                "enters_probability": True,
                "length_parameterized": bool(waiting_supported and hop_count != 1),
                "timing_policy": (
                    "configured selected-live-pair ages plus deterministic endpoint confirmation waits; "
                    "not route-from-scratch repeat-until-success waiting"
                ),
                "channel": f"same memory-local {storage_model.get('kind', 'storage')} channel, applied to stationary memory qubits at the selected live-pair total ages",
                "scope": (
                    "exact finite sum over configured discrete live-pair age distribution; availability and "
                    "pre-route stale/decay filtering are configured protocol inputs, not derived event-level "
                    "memory-pool dynamics"
                ),
            })
        else:
            implemented.append({
                "effect": "stochastic_waiting_time_decoherence",
                "parameter": "memory.coherence_time,memory.frequency,fiber_channel.frequency,detector.count_rate,fiber_channel.light_speed",
                "status": (
                    "not_applicable_single_link" if hop_count == 1 else "implemented"
                ),
                "enters_probability": waiting_supported and hop_count != 1,
                "length_parameterized": waiting_supported and hop_count != 1,
                "timing_policy": (
                    "analytic i.i.d. geometric repeat-until-success waiting for h=2; sequential swap-asap "
                    "grouped enumeration within explicit per-edge attempt cutoffs for h>=3; no adjacent-link "
                    "stochastic wait exists for h=1"
                ),
                "channel": f"same memory-local {storage_model.get('kind', 'storage')} channel as deterministic storage, applied per memory with exact (duration, elapsed) pairs from the waiting history/signature",
                "scope": (
                    "event-free finite-attempt-budget waiting marginalization; delivered branch states are "
                    "conditioned on route readiness within the recorded attempt cutoffs and omitted tail is "
                    "timeout/no_delivery, not event-level SeQUeNCe scheduling; for h>=3 the attempt cutoffs "
                    "may be reduced below the configured budget by the recorded history/signature caps"
                ),
            })
    else:
        missing.append({
            "effect": "deterministic_storage_time_decoherence",
            "parameter": "memory.memory_storage_decoherence_model",
            "status": "missing",
            "enters_probability": False,
            "reason": "no active typed storage-decoherence model is available, so storage-time decoherence is disabled rather than silently defaulted",
        })
    implemented.extend([
        {
            "effect": "memory_photon_state_preparation",
            "parameter": "memory.wavelength",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
        },
        {
            "effect": "memory_process_model",
            "parameter": process_model.get("source_field", "memory.memory_process_model"),
            "status": "implemented" if process_model.get("kind") != "none" else "not_applied",
            "enters_probability": bool(process_model.get("enters_probability")),
            "length_parameterized": False,
            "kind": process_model.get("kind", "none"),
            "applied_to": process_model.get("applied_to", "none"),
            "calibration_status": process_model.get("calibration_status", "not_specified"),
            "diagnostic_override": bool(process_model.get("diagnostic_override", False)),
        },
        {
            "effect": "memory_emission_efficiency_as_fock_loss",
            "parameter": (
                "repeater_service_profile.rate_model.per_node_zero_loss_detection_probability"
                if service_aggregate_link else "memory.efficiency"
            ),
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
        },
        {
            "effect": "fiber_attenuation_as_fock_loss",
            "parameter": (
                "repeater_service_profile.rate_model.attenuation_exponent_per_km"
                if service_aggregate_link else "fiber_channel.attenuation"
            ),
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": True,
        },
        {
            "effect": "wavelength_response_lookup",
            "parameter": "memory.wavelength/source.wavelength plus optional response curves",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
        },
        {
            "effect": "linear_optics_elementary_bsm",
            "parameter": "fock_cutoff",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
        },
        {
            "effect": "detector_efficiency_and_dark_window_povm",
            "parameter": "detector.efficiency,dark_count,time_resolution/profile timing window",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
        },
        {
            "effect": "detector_conditioned_remote_memory_density_matrix",
            "parameter": "raw detector/count record",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": True,
        },
        {
            "effect": "endpoint_memory_readout_assignment_error",
            "parameter": readout_model.get("source_field", "memory.memory_readout_model"),
            "status": "implemented" if readout_model.get("kind") != "none" else "not_applied_ideal_readout",
            "enters_probability": bool(readout_model.get("enters_probability", False)),
            "length_parameterized": False,
            "kind": readout_model.get("kind", "none"),
            "applies_to": "reported Alice/Bob endpoint transcript outcomes",
            "does_not_affect": readout_model.get("does_not_affect", ["delivery/no_delivery mass"]),
            "scalar_fidelity_active": False,
        },
    ])
    implemented.append({
        "effect": "endpoint_classical_confirmation_wait",
        "parameter": "fiber_channel.light_speed,route edge lengths",
        "status": "implemented",
        "enters_probability": True,
        "length_parameterized": True,
        "applied_to": (
            "extra storage (decoherence channel and detected-decay survival age) on the two endpoint "
            "memory qubits between the final swap and the classical route-ready confirmation"
        ),
        "policy": "measure_on_route_ready_confirmation; t_c(r_i -> endpoint) over the same fiber path at the scalar group speed",
    })
    if survival_model["enabled"]:
        implemented.append({
            "effect": "detected_memory_decay_survival",
            "parameter": "memory.detected_decay_lifetime_s",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": True,
            "applied_to": (
                "per-memory heralded-loss survival exp(-total represented storage age / lifetime), "
                "reweighting waiting histories and booked as failure stage memory_decay_detected"
            ),
            "lifetime_s": survival_model["lifetime_s"],
            "calibration_status": (
                "fitted |D_5/2> lifetime with mid-circuit decay detection (Cui2025); heralded loss, "
                "never state error"
            ),
        })
    if context.memory_cutoff_ps is not None and float(context.memory_cutoff_ps) > 0.0:
        implemented.append({
            "effect": "stale_pair_cutoff_policy",
            "parameter": "context.memory_cutoff_ps",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": True,
            "applied_to": (
                "waiting histories in which any memory's total represented storage age exceeds the "
                "cutoff are discarded to no_delivery (failure stage stale_pair_cutoff)"
            ),
            "cutoff_ps": float(context.memory_cutoff_ps),
            "calibration_status": "protocol policy knob, not a device measurement",
        })
    if rotation_model.get("enters_probability", False):
        implemented.append({
            "effect": "endpoint_analysis_rotation_error",
            "parameter": rotation_model.get("source_field", "memory.endpoint_analysis_rotation_model"),
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "kind": rotation_model.get("kind"),
            "applied_to": "measured party's memory qubit before projection, rotated (non-Z) settings only",
            "calibration_status": rotation_model.get("calibration_status", "not_specified"),
        })
    conversion_noise_rate = conversion_noise_rate_per_s(context)
    if conversion_noise_rate > 0.0 and resolved_kind != "ideal_pnr":
        implemented.append({
            "effect": "conversion_inband_background_counts",
            "parameter": "conversion_response.conversion_noise_rate_per_s",
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "applied_to": "per-detector background rate added to the dark-count mean inside the single detection window",
            "calibration_status": (
                "detected post-filter in-band conversion background rate supplied by the loaded "
                "conversion response file; split arithmetic (2 channels / 4 detectors) is a stated "
                "mapping, not a measured per-detector value"
            ),
        })
    elif conversion_noise_rate > 0.0 and resolved_kind == "ideal_pnr":
        implemented.append({
            "effect": "conversion_inband_background_counts",
            "parameter": "conversion_response.conversion_noise_rate_per_s",
            "status": "loaded_but_suppressed_by_ideal_detector",
            "enters_probability": False,
            "length_parameterized": False,
            "applied_to": "none in ideal_pnr diagnostic mode",
            "calibration_status": (
                "conversion background rate is present in the loaded conversion response, but ideal_pnr "
                "sets detector dark/background counts to zero by construction"
            ),
        })
    if polarization_active:
        implemented.append({
            "effect": "fiber_polarization_noise_channel",
            "parameter": polarization_model.get("source_field", "fiber_channel.polarization_noise_model"),
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "kind": polarization_model.get("kind"),
            "applied_to": "flying optical polarization qubits per elementary half-link traversal, before the optical BSM",
            "calibration_status": polarization_model.get("calibration_status", "not_specified"),
        })
    if indistinguishability_active:
        implemented.append({
            "effect": "hom_wave_packet_indistinguishability",
            "parameter": indistinguishability_model.get("source_field", "memory.photon_indistinguishability_model"),
            "status": "implemented_phenomenological_mode_overlap_mixture",
            "enters_probability": True,
            "length_parameterized": False,
            "kind": indistinguishability_model.get("kind"),
            "mode_overlap": indistinguishability_model.get("mode_overlap"),
            "applied_to": "two-photon interference at the elementary optical Bell analyzer",
            "calibration_status": indistinguishability_model.get("calibration_status", "not_specified"),
        })
    if swap_confusion_active:
        implemented.append({
            "effect": "repeater_bsm_bell_outcome_confusion",
            "parameter": swap_confusion_model.get("source_field", "swap_bsm.bell_outcome_confusion_model"),
            "status": "implemented",
            "enters_probability": True,
            "length_parameterized": False,
            "kind": swap_confusion_model.get("kind"),
            "applied_to": "announced repeater Bell outcome label, public Pauli frame, and applied feed-forward correction",
            "distinct_from": swap_confusion_model.get("does_not_affect", []),
            "calibration_status": swap_confusion_model.get("calibration_status", "not_specified"),
        })
    generic_approximations = []
    if indistinguishability_active:
        generic_approximations.append({
            "effect": "hom_mode_overlap_mixture",
            "parameter": indistinguishability_model.get("source_field", "memory.photon_indistinguishability_model"),
            "status": "phenomenological_generic_approximation",
            "mapping": (
                "elementary BSM statistics = s * indistinguishable interference instrument "
                "+ (1 - s) * distinguishable classical-routing reference"
            ),
            "calibration_status": (
                "phenomenological mode-overlap/visibility parameter; not a calibrated wave-packet or "
                "spectral-temporal-mode model"
            ),
        })
    if process_model.get("diagnostic_override"):
        generic_approximations.append({
            "effect": "diagnostic_memory_fidelity_depolarizing_channel",
            "parameter": "memory.fidelity",
            "status": "diagnostic_generic_approximation",
            "mapping": "local depolarizing probability = 1 - memory.fidelity",
            "source_semantics": memory.get(
                "fidelity_semantics",
                "configured scalar memory/interface fidelity; not a calibrated depolarizing-channel probability",
            ),
            "calibration_status": (
                "diagnostic override only; scalar memory.fidelity is not a calibrated memory process matrix "
                "or Pauli/depolarizing error probability"
            ),
        })
    generic_approximations.extend([
            {
                "effect": "memory_coherence_time_decay_mapping",
                "parameter": storage_model.get("source_field", "memory.memory_storage_decoherence_model"),
                "status": (
                    "implemented_typed_one_parameter_approximation"
                    if not storage_model.get("legacy_fallback") else
                    "legacy_generic_approximation"
                ),
                "mapping": storage_model.get("coherence_factor_formula", "C(t)=1"),
                "storage_model_kind": storage_model.get("kind", "none"),
                "source_reported_decay_model": memory.get("source_reported_decay_model", "not_specified"),
                "kernel_assumed_decay_model": memory.get(
                    "kernel_assumed_storage_decay_model",
                    storage_model.get("coherence_factor_formula", "C(t)=1"),
                ),
                "coherence_time_definition": storage_model.get("coherence_time_definition"),
                "calibration_status": storage_model.get("calibration_status", "not_specified"),
            },
            {
                "effect": "wavelength_stack_scalar_response_mapping",
                "parameter": "memory.wavelength,fiber_channel.attenuation,detector.wavelength_range_nm,conversion_response",
                "status": "generic_approximation" if wavelength_report["issues"] else "implemented_metadata_consistency_check",
                "mapping": (
                    "source wavelength is mapped through an optional scalar/curve conversion throughput and "
                    "conversion output wavelength; fiber attenuation and detector response are evaluated at "
                    "the propagation/detector wavelength. Conversion bandwidth, temporal-mode distortion, "
                    "and polarization-process effects remain metadata-only unless separately modeled"
                ),
                "wavelength_compatibility_status": wavelength_report["wavelength_compatibility_status"],
            },
            {
                "effect": "pnr_response_model",
                "parameter": "detector.pnr_confusion_matrix",
                "status": (
                    "implemented" if pnr_meta["pnr_response_model_is_calibrated_to_profile"]
                    else "generic_approximation"
                ),
                "model": pnr_meta["pnr_response_model"],
            },
            *([] if service_swap_channel is not None else [{
                "effect": "swap_degradation_and_phase",
                "parameter": "swap_bsm.swapping_degradation,swap_bsm.phase_error",
                "status": "generic_approximation",
                "mapping": "post-swap two-qubit depolarizing plus Z-dephasing",
            }]),
    ])
    return {
        "implemented_in_probability_kernel": implemented,
        "generic_approximations": generic_approximations,
        "computed_metadata_only": [
            *([] if propagation_delay_enters else [{
                "effect": "fiber_propagation_delay",
                "parameter": "fiber_channel.light_speed legacy config field emitted as fiber_group_speed_m_per_ps",
                "status": "metadata_only",
                "enters_probability": False,
                "length_parameterized": True,
                "speed_interpretation": "scalar guided-mode propagation/group speed along fiber arclength; not vacuum c, not vector velocity, and not a fiber dephasing parameter",
            }]),
            *([] if attempt_rate_enters else [{
                "effect": "detector_cooldown_period",
                "parameter": detector_rate["profile_field"],
                "status": "metadata_only",
                "enters_probability": False,
                "length_parameterized": False,
                "count_rate_hz": detector_rate["count_rate_hz"],
                "source": detector_rate["source"],
                "reason": "no adjacent-link repeat-until-success waiting is active for this case",
            }]),
            {
                "effect": "memory_fidelity_scalar_provenance",
                "parameter": "memory.fidelity",
                "status": "metadata_only",
                "enters_probability": False,
                "source_semantics": memory.get(
                    "fidelity_semantics",
                    "configured scalar memory/interface fidelity",
                ),
                "reason": "scalar memory.fidelity is not consumed as process noise without an explicit memory_process_model or diagnostic override",
            },
        ],
        "implemented_only_in_sequence_reference_helpers": [
            {
                "effect": "barrett_kok_round_timing_cooldown_survival",
                "status": "implemented_only_in_sequence_bk_reference_helpers",
                "enters_fock_probability": False,
            }
        ],
        "missing": missing,
        "intentionally_excluded": [
            {
                "effect": "endpoint_detector_hardware",
                "status": "intentionally_excluded",
                "reason": "selected repeater E91 path measures endpoint memories, not endpoint optical detector clicks",
            },
            *([] if service_profile is None or survival_model["enabled"] else [{
                "effect": "detected_memory_decay_survival",
                "status": "intentionally_excluded_by_profile",
                "reason": service_profile.memory_model.get(
                    "decay_semantics",
                    "service profile supplies no detected-decay process",
                ),
            }])
        ],
    }


def fail_closed_unsupported_effects(
    front: dict[str, Any],
    context: ModelContext,
    hop_count: int | None = None,
) -> list[dict[str, Any]]:
    if context.physical_path != "fock_pnr_snspd":
        return []
    effects = probability_kernel_effects(front, context, hop_count=hop_count)
    unsupported = []
    for bucket in ("missing", "computed_metadata_only"):
        for row in effects[bucket]:
            if row["effect"] in {
                "fiber_propagation_delay",
                "detector_cooldown_period",
                "memory_fidelity_scalar_provenance",
            }:
                # These metadata-only rows are represented by stricter missing
                # rows when they are required for the Fock/SNSPD kernel.
                continue
            unsupported.append({
                "parameter": row.get("parameter", row["effect"]),
                "effect": row["effect"],
                "classification": row["status"],
                "reason": row.get("reason", "required physical effect is not in the probability kernel"),
            })
    return unsupported


def validate_detector_profile_for_kind(front: dict[str, Any], context: ModelContext) -> str:
    if context.detector_kind == "ideal_pnr":
        return "ideal_pnr"
    profile = detector_profile_metadata(front, context)
    validate_pnr_confusion_matrix(profile, profile_max_resolvable_photon_number(profile))
    if profile.get("selectable") is False:
        raise ValueError(
            f"detector profile {profile.get('reference_name', '<unnamed>')!r} is marked selectable=false"
        )
    pnr_capable = profile_claims_pnr(profile)
    max_count = profile_max_resolvable_photon_number(profile)
    if context.detector_kind == "pnr_snspd":
        if not pnr_capable or max_count < 2:
            raise ValueError(
                "detector_kind=pnr_snspd requires a loaded detector profile with "
                "pnr_capability=true and max_resolvable_photon_number>=2"
            )
        return "pnr_snspd"
    if context.detector_kind == "threshold_snspd":
        return "threshold_snspd"
    if context.detector_kind == "repository_default":
        return "pnr_snspd" if pnr_capable and max_count >= 2 else "threshold_snspd"
    raise ValueError(f"unsupported detector kind {context.detector_kind!r}")


def physical_path_limitations(
    front: dict[str, Any],
    context: ModelContext,
    hop_count: int | None = None,
) -> list[dict[str, Any]]:
    if context.physical_path != "fock_pnr_snspd":
        return []
    return fail_closed_unsupported_effects(front, context, hop_count=hop_count)
