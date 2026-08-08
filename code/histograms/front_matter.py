"""Physical registry loading, front-matter construction, and inventory mapping."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from sequence_game.physical.loader import load_models_from_dir
from sequence_game.physical.registry import PhysicalModel, PhysicalRegistry

from .device_effects import *
from .device_models import *
from .device_profiles import *
from .device_profiles import _float_or_none
from .models import *
def find_model(registry: PhysicalRegistry, kind: str) -> PhysicalModel | None:
    matches = [model for model in registry.models.values() if model.device_kind == kind]
    return sorted(matches, key=lambda m: m.model_name)[0] if matches else None


def load_or_build_front_matter(
    config_dir: Path,
    *,
    device_front_matter: Path | None = None,
    fail_on_unsupported: bool = False,
) -> dict[str, Any]:
    if device_front_matter is not None:
        with open(device_front_matter, encoding="utf-8") as handle:
            front = json.load(handle)
        front["front_matter_hash"] = stable_hash(front.get("front_matter_without_hash", front))
        return front

    registry = load_models_from_dir(config_dir, require_resolved=True)
    front = build_device_front_matter(registry, config_dir)
    required = [
        row for row in front["unsupported_device_parameters"]
        if row["classification"] == "required_for_current_model"
    ]
    if fail_on_unsupported and required:
        names = [row["parameter"] for row in required]
        raise SystemExit(
            "--fail-on-unsupported-device rejected unsupported required parameters: "
            + ", ".join(names)
        )
    return front


def build_device_front_matter(registry: PhysicalRegistry, config_dir: Path) -> dict[str, Any]:
    models = {
        name: model.to_dict()
        for name, model in sorted(registry.models.items())
    }
    selected = {
        kind: (find_model(registry, kind).model_name if find_model(registry, kind) else None)
        for kind in ("source", "fiber_channel", "detector", "memory", "swap_bsm")
    }
    source_paths = sorted(str(path) for path in config_dir.glob("*.json"))
    mapped, unsupported, conversions, inventory = inventory_device_parameters(registry)
    constants = extract_device_constants(registry)
    front_no_hash = {
        "generator_version": GENERATOR_VERSION,
        "front_matter_role": "frozen non-variable hardware/protocol constants",
        "hardware_config_dir": str(config_dir),
        "hardware_config_source_paths": source_paths,
        "selected_models": selected,
        "models": models,
        "sequence_component_formula_sources": {
            "source_pair_count": "sequence.components.light_source.SPDCSource.emit and _generate_tmsv_state",
            "quantum_channel_transmittance": "sequence.components.optical_channel.QuantumChannel.init",
            "quantum_channel_delay_ps": "sequence.components.optical_channel.QuantumChannel.init",
            "detector_dark_process": "sequence.components.detector.Detector.add_dark_count",
            "detector_threshold_efficiency": "sequence.components.detector.Detector.get",
            "detector_time_rounding_and_cooldown": "sequence.components.detector.Detector.record_detection",
            "router_edge_split": "sequence.topology.router_net_topo.RouterNetTopo._add_qconnections",
            "barrett_kok_protocol": "sequence.entanglement_management.generation.barret_kok.BarretKokA/BarretKokB",
            "single_atom_bsm": "sequence.components.bsm.SingleAtomBSM.get and trigger",
            "barrett_kok_trigger_window": "sequence.entanglement_management.generation.generation_message.valid_trigger_time",
            "barrett_kok_success_state": "sequence.components.bsm._set_state_with_fidelity plus tests/entanglement_management/test_generation.py",
            "memory_emission": "sequence.components.memory.Memory.excite",
            "memory_expiration": "sequence.components.memory.Memory.update_state and _schedule_expiration",
            "memory_bds_decoherence": "sequence.components.memory.Memory.bds_decohere only when decoherence_errors is configured",
            "swapping_circuit": "sequence.entanglement_management.swapping.EntanglementSwappingA.circuit",
            "swapping_success": "sequence.entanglement_management.swapping.EntanglementSwappingA.success_probability",
            "swapping_fidelity_update": "sequence.entanglement_management.swapping.EntanglementSwappingA.updated_fidelity",
            "endpoint_memory_measurement": "sequence.app.e91_endpoint.E91EndpointApp.get_memory",
        },
        "adapter_field_contracts": {
            "source": list(SOURCE_FIELDS),
            "fiber_channel": list(FIBER_FIELDS),
            "detector": list(DETECTOR_FIELDS),
            "memory": list(MEMORY_FIELDS),
            "swap_bsm": list(SWAP_FIELDS),
        },
        "device_constants": constants,
        "device_parameter_inventory": inventory,
        "mapped_device_parameters": mapped,
        "unsupported_device_parameters": unsupported,
        "adapter_conversion_notes": conversions,
        "fallback_defaults_used": constants.pop("fallback_defaults_used", []),
        "event_free_model_scope": {
            "route_construction": "selected-route repeater E91 instrument with explicit elementary edges and repeater-local Bell measurements",
            "timing": (
                "fiber propagation delays, detector cooldown, memory cadence, and memory expiration are inventoried; "
                "the Fock/SNSPD probability kernel uses detector timing as the dark-count integration window and applies "
                "a typed storage-time memory channel from fiber delay. Repeat-until-success waiting-time decoherence "
                "enters P(tau) analytically for h=2 and through sequential swap-asap grouped enumeration within "
                "explicit per-edge attempt cutoffs for h>=3; repeated-window dead-time dynamics are not included "
                "in Fock/SNSPD P(tau)."
            ),
            "source_spdc": "the selected repeater path is RouterNetTopo + Barrett-Kok memory emission; repository SPDCSource/Fock/PNR fields are recorded as source-path metadata and not claimed as active selected-route physics",
            "multipair_fock_pnr": "SPDC/Fock/PNR branches are outside the selected RouterNetTopo single-atom Barrett-Kok path unless a future source-driven route is selected",
            "endpoint_detection": (
                "endpoint E91 measurements are memory measurements through E91EndpointApp; endpoint detector dark counts "
                "are not applied to Alice/Bob outcomes for this selected repeater path. Optional typed memory_readout_model "
                "classically maps ideal endpoint memory outcomes to reported transcript bits."
            ),
        },
    }
    front = dict(front_no_hash)
    front["front_matter_without_hash"] = front_no_hash
    front["front_matter_hash"] = stable_hash(front_no_hash)
    return front


def inventory_device_parameters(
    registry: PhysicalRegistry,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    mapped: dict[str, Any] = {}
    unsupported: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    conversions: dict[str, str] = {}
    used_by_kind = {
        "source": set(),
        "fiber_channel": {"attenuation", "light_speed", "frequency"},
        "detector": {"efficiency", "dark_count", "count_rate", "time_resolution"},
        "memory": {"efficiency", "coherence_time", "frequency", "wavelength"},
        "swap_bsm": {"success_probability"},
    }
    optional_used_by_kind = {
        "memory": {
            "t1",
            "T1",
            "t1_s",
            "T1_s",
            "t2",
            "T2",
            "t2_s",
            "T2_s",
            "process_model",
            "memory_process_model",
            "interface_process_model",
            "pre_emission_process_model",
            "memory_storage_decoherence_model",
            "storage_decoherence_model",
            "storage_model",
            "memory_readout_model",
            "endpoint_readout_model",
            "readout_model",
            "photon_indistinguishability_model",
            "hom_visibility_model",
            "detected_decay_lifetime_s",
            "endpoint_analysis_rotation_model",
            "analysis_rotation_model",
        },
        "fiber_channel": {
            "polarization_noise_model",
            "polarization_drift_model",
        },
        "swap_bsm": {
            "bell_outcome_confusion_model",
            "swap_bell_outcome_model",
        },
    }
    known_not_used = {
        "source.frequency": "source timing cadence belongs to SPDCSource/LightSource and is not used by RouterNetTopo Barrett-Kok memory emission",
        "source.wavelength": "source wavelength belongs to SPDCSource/LightSource and is not used by RouterNetTopo Barrett-Kok memory emission",
        "source.bandwidth": "source bandwidth belongs to SPDCSource/LightSource and is not used by RouterNetTopo Barrett-Kok memory emission",
        "source.mean_photon_num": "SPDC pair-count distribution belongs to SPDCSource and is not used by RouterNetTopo Barrett-Kok memory emission",
        "source.phase_error": "source phase_error belongs to SPDCSource/LightSource and is not used by RouterNetTopo Barrett-Kok memory emission",
        "fiber_channel.polarization_fidelity": (
            "polarization_fidelity is inventoried as provenance only; the Fock/SNSPD kernel applies a photonic "
            "polarization drift/noise channel only through an explicit typed fiber_channel.polarization_noise_model"
        ),
        "memory.fidelity": "scalar fidelity is retained as device provenance; it is not consumed as constructor/process noise unless an explicit memory_process_model or diagnostic override is supplied",
        "memory.wavelength": "Memory.excite attaches this wavelength to emitted photons; no wavelength-dependent channel/detector response exists in the selected configs",
        "memory.ion_photon_entanglement_benchmark": (
            "reported Cui ion-photon Bell-state fidelities are benchmark/provenance metadata; they are not "
            "consumed as a memory_process_model and are not mapped to per-memory-qubit depolarizing noise"
        ),
        "swap_bsm.phase_error": "SingleAtomBSM stores phase_error but its get method does not apply it for single_atom encoding in the inspected runtime",
    }
    for model in registry.models.values():
        kind = model.device_kind
        used = used_by_kind.get(kind, set())
        optional_used = optional_used_by_kind.get(kind, set())
        for name, value in sorted(model.parameters.items()):
            key = f"{kind}.{name}"
            row = {
                "model_name": model.model_name,
                "device_kind": kind,
                "parameter": key,
                "value": value,
                "unit": model.units.get(name, ""),
                "reference_tag": model.reference_tag,
            }
            is_used = name in used or name in optional_used
            if is_used:
                mapped[key] = row
            else:
                pass
            classification = (
                "used_by_selected_route_model"
                if is_used else
                ("metadata_only" if key in known_not_used else "known_but_not_yet_used")
            )
            reason = (
                "field is used by the selected-route repeater E91 instrument"
                if is_used else
                known_not_used.get(
                    key,
                    "parameter is present in the repository device model but is not part of the selected-route instrument",
                )
            )
            inventory[key] = {
                **row,
                "classification": classification,
                "reason": reason,
            }
        for name in sorted(used - set(model.parameters)):
            unsupported.append({
                "model_name": model.model_name,
                "device_kind": kind,
                "parameter": f"{kind}.{name}",
                "value": None,
                "unit": "",
                "reference_tag": model.reference_tag,
                "classification": "required_for_current_model",
                "reason": "field is needed by the selected-route repeater E91 histogram generator",
            })
    for key, reason in sorted(known_not_used.items()):
        if key in inventory and inventory[key]["classification"] != "used_by_selected_route_model":
            unsupported.append({
                **inventory[key],
                "classification": "metadata_only" if "metadata" in reason else "known_but_not_yet_used",
                "reason": reason,
            })
    source = find_model(registry, "source")
    if source and "mean_photon_num" in source.parameters:
        inventory["source.SPDC_multipair_and_Fock_PNR_branches"] = {
            "model_name": source.model_name,
            "device_kind": "source",
            "parameter": "source.SPDC_multipair_and_Fock_PNR_branches",
            "value": source.parameters["mean_photon_num"],
            "unit": source.units.get("mean_photon_num", ""),
            "reference_tag": source.reference_tag,
            "classification": "not_used_by_selected_route_model",
            "reason": "SPDCSource supports Poisson multipair and Fock encoding branches; the selected RouterNetTopo repeater path uses Memory.excite single_atom photons and SingleAtomBSM, not SPDCSource",
        }
    detector = find_model(registry, "detector")
    if detector:
        conversions["detector_dark_probability_per_time_resolution"] = (
            "per-detector dark-window probability 1 - exp(-dark_count * trigger_window_ps * 1e-12), with trigger_window_ps from valid_trigger_time resolution"
        )
        conversions["elementary_attempt_period"] = (
            "max(memory excitation period, fiber-channel scheduling period, selected detector-profile "
            "count/reset period, deterministic herald round-trip time); the repository detector count "
            "rate is used only when the selected profile does not report one"
        )
    fiber = find_model(registry, "fiber_channel")
    if fiber:
        conversions["fiber_transmittance"] = (
            "RouterNetTopo splits each route edge into two half channels; per half-channel T_half_i=10 ** (-attenuation_dB_per_m * (L_i/2) / 10)"
        )
        conversions["fiber_delay"] = (
            "round(length_m / fiber_group_speed_m_per_ps); legacy config field fiber_channel.light_speed "
            "is interpreted as a scalar guided-mode propagation/group speed along fiber arclength, not vacuum c, "
            "not vector velocity, and not a fiber dephasing parameter"
        )
    source = find_model(registry, "source")
    if source:
        conversions["spdc_pair_count_inventory"] = (
            "SPDCSource.emit draws Poisson(mean_photon_num); selected-route repeater generation does not use this source branch"
        )
    swap = find_model(registry, "swap_bsm")
    if swap:
        conversions["swap_success_probability"] = (
            "success_probability is used as the repeater Bell-measurement success probability for each internal route repeater"
        )
    return mapped, unsupported, conversions, inventory


def extract_device_constants(registry: PhysicalRegistry) -> dict[str, Any]:
    source = find_model(registry, "source")
    fiber = find_model(registry, "fiber_channel")
    detector = find_model(registry, "detector")
    memory = find_model(registry, "memory")
    swap = find_model(registry, "swap_bsm")
    if fiber is None or detector is None or memory is None or swap is None:
        missing = [
            kind for kind, model in (
                ("fiber_channel", fiber),
                ("detector", detector),
                ("memory", memory),
                ("swap_bsm", swap),
            )
            if model is None
        ]
        raise SystemExit(f"missing required physical configs: {', '.join(missing)}")
    dark_rate = float(detector.parameters.get("dark_count", 0.0))
    time_resolution_ps = float(detector.parameters.get("time_resolution", 0.0))
    dark_probability = clip_probability(1.0 - math.exp(-dark_rate * time_resolution_ps * 1e-12))
    endpoint_eff = require_probability("detector.efficiency", detector.parameters["efficiency"])
    endpoint_detection = 1.0 - (1.0 - endpoint_eff) * (1.0 - dark_probability)
    dark_only_conditional = (
        ((1.0 - endpoint_eff) * dark_probability / endpoint_detection)
        if endpoint_detection > 0.0 else 0.0
    )
    mu = float(source.parameters.get("mean_photon_num", 0.0)) if source else 0.0
    spdc_zero = math.exp(-max(0.0, mu))
    spdc_one = max(0.0, mu) * spdc_zero
    memory_t1 = (
        memory.parameters.get("t1_s")
        if "t1_s" in memory.parameters else
        memory.parameters.get("T1_s")
        if "T1_s" in memory.parameters else
        memory.parameters.get("t1")
        if "t1" in memory.parameters else
        memory.parameters.get("T1")
    )
    memory_t2 = (
        memory.parameters.get("t2_s")
        if "t2_s" in memory.parameters else
        memory.parameters.get("T2_s")
        if "T2_s" in memory.parameters else
        memory.parameters.get("t2")
        if "t2" in memory.parameters else
        memory.parameters.get("T2")
    )
    memory_coherence_raw = memory.parameters.get("coherence_time")
    memory_coherence_s = float(memory_coherence_raw) if memory_coherence_raw is not None else None
    normalized_memory_process = normalize_memory_process_model_from_config(memory.parameters)
    normalized_storage_decoherence = normalize_memory_storage_decoherence_model_from_config(memory.parameters)
    normalized_memory_readout = normalize_memory_readout_model_from_config(memory.parameters)
    memory_notes = memory.notes or ""
    source_decay_model = (
        "Gaussian decay fit"
        if "Gaussian decay fit" in memory_notes or "Gaussian" in memory_notes
        else "not_specified"
    )
    memory_coherence_missing_note = (
        [] if memory_coherence_raw is not None else [
            {
                "parameter": "memory.coherence_time",
                "value": None,
                "reason": (
                    "memory config has no coherence_time field; deterministic storage-time dephasing is disabled "
                    "unless explicit T1/T2 fields are supplied"
                ),
            }
        ]
    )
    fiber_notes = fiber.notes or ""
    fiber_attenuation_reference_wavelength_nm = _float_or_none(
        fiber.parameters.get("attenuation_reference_wavelength_nm")
    )
    if fiber_attenuation_reference_wavelength_nm is None and "1550" in fiber_notes:
        fiber_attenuation_reference_wavelength_nm = 1550.0
    fiber_attenuation_calibration_status = (
        "scalar telecom/C-band attenuation reference inferred from config notes"
        if fiber_attenuation_reference_wavelength_nm is not None else
        "scalar attenuation with unknown reference wavelength"
    )
    fiber_speed_m_per_ps = float(fiber.parameters["light_speed"])
    return {
        "source": {
            "model_name": source.model_name if source else None,
            "mean_photon_num": mu,
            "spdc_zero_pair_probability": spdc_zero,
            "spdc_single_pair_probability": spdc_one,
            "spdc_multi_pair_probability": max(0.0, 1.0 - spdc_zero - spdc_one),
            "frequency_hz": float(source.parameters.get("frequency", 0.0)) if source else None,
            "wavelength_nm": float(source.parameters.get("wavelength", 0.0)) if source else None,
            "bandwidth_nm": float(source.parameters.get("bandwidth", 0.0)) if source else None,
            "phase_error": float(source.parameters.get("phase_error", 0.0)) if source else None,
            "selected_route_use": "not_used_by_RouterNetTopo_Barrett_Kok_memory_emission",
        },
        "fiber_channel": {
            "model_name": fiber.model_name,
            "attenuation_db_per_m": float(fiber.parameters["attenation"]) if "attenation" in fiber.parameters else float(fiber.parameters["attenuation"]),
            "attenuation_reference_wavelength_nm": fiber_attenuation_reference_wavelength_nm,
            "attenuation_calibration_status": fiber_attenuation_calibration_status,
            "polarization_fidelity": require_probability(
                "fiber_channel.polarization_fidelity",
                fiber.parameters["polarization_fidelity"],
            ),
            "fiber_group_speed_m_per_ps": fiber_speed_m_per_ps,
            "fiber_propagation_speed_m_per_ps": fiber_speed_m_per_ps,
            "light_speed_m_per_ps": fiber_speed_m_per_ps,
            "fiber_speed_source_field": "fiber_channel.light_speed",
            "fiber_speed_interpretation": (
                "scalar guided-mode propagation/group speed along the fiber path coordinate/fiber arclength; "
                "not vacuum c, not vector velocity, and not a fiber dephasing parameter"
            ),
            "frequency_hz": float(fiber.parameters["frequency"]),
            "polarization_noise_model": normalize_fiber_polarization_noise_model_from_config(fiber.parameters),
        },
        "detector": {
            "model_name": detector.model_name,
            "efficiency": endpoint_eff,
            "dark_count_per_s": dark_rate,
            "count_rate_hz": float(detector.parameters["count_rate"]),
            "time_resolution_ps": time_resolution_ps,
            "barrett_kok_trigger_window_ps": time_resolution_ps,
            "dark_probability_per_time_resolution": dark_probability,
            "endpoint_detection_probability_given_arrival": clip_probability(endpoint_detection),
            "dark_only_probability_conditioned_on_detection": clip_probability(dark_only_conditional),
        },
        "memory": {
            "model_name": memory.model_name,
            "fidelity": require_probability("memory.fidelity", memory.parameters["fidelity"]),
            "fidelity_source_field": "memory.parameters.fidelity",
            "fidelity_semantics": memory.units.get("fidelity", "dimensionless scalar fidelity"),
            "fidelity_notes": (
                "The default memory profile treats this scalar as a reported memory-state/storage benchmark datum. "
                "The Fock/SNSPD kernel does not consume this scalar as constructor/interface process noise unless "
                "an explicit memory_process_model or diagnostic override is supplied."
            ),
            "memory_storage_benchmark": {
                "fidelity": require_probability("memory.fidelity", memory.parameters["fidelity"]),
                "storage_time_s": 0.100,
                "basis_or_state_set": "six mutually unbiased basis states",
                "source": memory.reference_tag,
                "model_use": "calibration_target_or_metadata_not_constructor_noise",
                "includes_readout_error": "not_resolved_in_current_stub",
            },
            "ion_photon_entanglement_benchmark": copy.deepcopy(
                memory.parameters.get("ion_photon_entanglement_benchmark")
            ),
            "fidelity_channel_calibration_status": (
                "metadata/provenance only by default; no calibrated process matrix or Pauli/depolarizing error model is supplied"
            ),
            "memory_process_model": normalized_memory_process,
            "memory_storage_decoherence_model": normalized_storage_decoherence,
            "memory_readout_model": normalized_memory_readout,
            "efficiency": require_probability("memory.efficiency", memory.parameters["efficiency"]),
            "frequency_hz": float(memory.parameters["frequency"]),
            "coherence_time_s": memory_coherence_s,
            "coherence_time_source_field": (
                "memory.parameters.coherence_time"
                if memory_coherence_raw is not None else
                "not_provided"
            ),
            "coherence_time_unit": memory.units.get("coherence_time", ""),
            "source_reported_decay_model": source_decay_model,
            "kernel_assumed_storage_decay_model": (
                normalized_storage_decoherence.get(
                    "coherence_factor_formula",
                    "C(t)=1",
                )
            ),
            "memory_process_calibration_status": (
                normalized_memory_process.get(
                    "calibration_status",
                    "no calibrated constructor/interface memory process channel supplied",
                )
            ),
            "memory_storage_calibration_status": (
                normalized_storage_decoherence.get(
                    "calibration_status",
                    "generic approximation unless explicit calibrated T1/T2 fields are supplied",
                )
            ),
            "t1_s": float(memory_t1) if memory_t1 is not None else None,
            "t2_s": float(memory_t2) if memory_t2 is not None else None,
            "detected_decay_lifetime_s": _float_or_none(memory.parameters.get("detected_decay_lifetime_s")),
            "detected_decay_semantics": (
                "spontaneous-decay lifetime of the storage level; decay events are detected by mid-circuit "
                "measurement and convert to heralded loss booked as failure stage memory_decay_detected "
                "(delivery/no-delivery mass), not state error; consumed by the Fock/SNSPD kernel as "
                "per-memory survival exp(-total represented storage age / lifetime) over exact "
                "waiting-history ages"
                if memory.parameters.get("detected_decay_lifetime_s") is not None else None
            ),
            "endpoint_analysis_rotation_model": normalize_endpoint_analysis_rotation_model_from_config(
                memory.parameters
            ),
            "cutoff_ratio": 1.0,
            "decoherence_errors": None,
            "decoherence_model": (
                "Fock/SNSPD kernel applies the typed memory_storage_decoherence_model to stationary memory qubits "
                "during represented storage intervals. The default memory profile uses Gaussian dephasing from the "
                "source-reported coherence fit; this remains a one-parameter storage approximation, not a calibrated "
                "full memory process. "
                "SeQUeNCe Memory.bds_decohere is inactive because the selected physical config has no decoherence_errors field"
            ),
            "wavelength_nm": float(memory.parameters["wavelength"]),
            "photon_indistinguishability_model": normalize_photon_indistinguishability_model_from_config(
                memory.parameters
            ),
        },
        "swap_bsm": {
            "model_name": swap.model_name,
            "success_probability": require_probability(
                "swap_bsm.success_probability",
                swap.parameters["success_probability"],
            ),
            "phase_error": float(swap.parameters.get("phase_error", 0.0)),
            "swapping_degradation": float(swap.parameters.get("swapping_degradation", 1.0)),
            "bell_outcome_confusion_model": normalize_swap_bell_outcome_model_from_config(swap.parameters),
        },
        "fallback_defaults_used": (
            [] if "swapping_degradation" in swap.parameters else [
                {
                    "parameter": "swap_bsm.swapping_degradation",
                    "value": 1.0,
                    "reason": "repository swap_bsm config has no swapping_degradation field; event-free histogram records neutral degradation",
                }
            ]
        ) + memory_coherence_missing_note,
    }
