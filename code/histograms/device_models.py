"""Typed physical-process normalization and memory/swap channel models."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from sequence_game.physical.entanglement_purification import (
    average_gate_fidelity_to_pauli_error,
)
from sequence_game.physical.repeater_service import (
    RepeaterServiceProfile,
    profile_elementary_pair_instrument,
    profile_elementary_state_channel,
    local_bell_measurement_model,
    profile_quality_restoration,
    profile_summary,
    profile_swap_state_channel,
    quality_restoration_model,
    repeater_service_profile_from_context,
)

from .models import *


def active_repeater_service_profile(context: ModelContext) -> RepeaterServiceProfile | None:
    """Validated service profile, or ``None`` for the historical model path."""

    return repeater_service_profile_from_context(context.repeater_service_profile)


def repeater_service_activation_blockers(context: ModelContext) -> list[str]:
    profile = active_repeater_service_profile(context)
    if profile is None:
        return []
    blockers = [str(value) for value in profile.histogram_adapter.get("activation_blockers", [])]
    if str(context.waiting_protocol).strip().lower() != "rolling_alive_pair":
        blockers.append(
            "a repeater service profile requires waiting_protocol='rolling_alive_pair'; "
            "the finite simultaneous-attempt route model would contradict its asynchronous controller"
        )
    expected_slots = int(
        profile.hardware.get(
            "purified_output_slots_per_adjacent_edge_side",
            profile.hardware["memory_slots_per_adjacent_edge_side"],
        )
    )
    if int(context.rolling_memory_slots_per_edge_side) != expected_slots:
        blockers.append(
            "rolling_memory_slots_per_edge_side does not match the profile's physical memory inventory "
            f"({context.rolling_memory_slots_per_edge_side} != {expected_slots})"
        )
    if profile.histogram_adapter.get("apply_elementary_state_channel", False):
        try:
            profile_elementary_state_channel(profile)
        except ValueError as exc:
            blockers.append(str(exc))
    if profile.histogram_adapter.get("elementary_link_instrument_kind") == (
        "empirical_remote_memory_pair"
    ):
        try:
            profile_elementary_pair_instrument(profile)
        except ValueError as exc:
            blockers.append(str(exc))
    if profile.histogram_adapter.get("apply_local_swap_state_channel", False):
        try:
            profile_swap_state_channel(profile)
        except ValueError as exc:
            blockers.append(str(exc))
    if profile.histogram_adapter.get("apply_quality_restoration", False):
        try:
            if quality_restoration_model(profile) is None:
                blockers.append(
                    "histogram adapter requests quality restoration but the profile has no model"
                )
        except ValueError as exc:
            blockers.append(str(exc))
    return blockers


def apply_repeater_service_front_overrides(
    front: dict[str, Any],
    context: ModelContext,
) -> dict[str, Any]:
    """Apply only profile-declared histogram adapters to a front-matter copy.

    The base device files remain untouched.  Every override and the base hash
    are retained in the returned front matter so a composite cannot masquerade
    as its input device stack.
    """

    profile = active_repeater_service_profile(context)
    if profile is None:
        return front
    out = copy.deepcopy(front)
    base_hash = out.get("front_matter_hash")
    adapter = profile.histogram_adapter
    memory = out["device_constants"]["memory"]
    overrides: dict[str, Any] = {}
    if adapter.get("source_wavelength_nm") is not None:
        memory["wavelength_nm"] = float(adapter["source_wavelength_nm"])
        overrides["memory.wavelength_nm"] = {
            "value": memory["wavelength_nm"],
            "reason": "repeater-service optical interface wavelength",
        }
    if adapter.get("apply_memory_storage_override", False):
        configured = copy.deepcopy(profile.memory_model["storage_decoherence_model"])
        kind = str(configured.get("kind", "none"))
        configured.update({
            "source_field": "repeater_service_profile.memory_model.storage_decoherence_model",
            "enters_probability": kind != "none",
            "applied_to": "stationary_memory_qubits_during_storage",
            "legacy_fallback": False,
            "calibration_status": (
                "explicit service-profile memory model with claim tier " + profile.claim_tier
            ),
        })
        if kind == "gaussian_dephasing":
            configured["coherence_factor_formula"] = "C(t)=exp(-(t/T_gauss)^2)"
        elif kind == "exponential_dephasing":
            configured["coherence_factor_formula"] = "C(t)=exp(-t/T2)"
        elif kind == "liu_2026_anisotropic_pair_fit":
            configured["coherence_factor_formula"] = (
                "lambda_x=lambda_y=exp(-t/(2*tau_xx)); "
                "lambda_z=sqrt(max(0,1-r_zz*t/V_zz(0))) per local memory"
            )
        else:
            configured["coherence_factor_formula"] = "C(t)=1"
        memory["memory_storage_decoherence_model"] = configured
        memory["coherence_time_s"] = configured.get("coherence_time_s")
        memory["coherence_time_source_field"] = configured["source_field"]
        memory["memory_storage_calibration_status"] = configured["calibration_status"]
        memory["source_reported_decay_model"] = kind
        memory["kernel_assumed_storage_decay_model"] = configured["coherence_factor_formula"]
        memory["detected_decay_lifetime_s"] = profile.memory_model.get(
            "detected_decay_lifetime_s"
        )
        memory["detected_decay_semantics"] = profile.memory_model.get(
            "decay_semantics", "not supplied by repeater service profile"
        )
        overrides["memory.storage_and_detected_decay"] = {
            "storage_decoherence_model": configured,
            "detected_decay_lifetime_s": memory["detected_decay_lifetime_s"],
        }
    if adapter.get("apply_local_bell_measurement_override", False):
        local_bsm = local_bell_measurement_model(profile)
        if local_bsm is None:
            raise ValueError(
                "local Bell-measurement override requires state_model.local_swap_model"
            )
        swap = out["device_constants"]["swap_bsm"]
        swap["success_probability"] = 1.0
        swap["phase_error"] = 0.0
        swap["swapping_degradation"] = 1.0
        swap["bell_outcome_confusion_model"] = {
            "kind": "bell_outcome_confusion_matrix",
            "enters_probability": True,
            "label_order": list(BELL_LABELS),
            "confusion_matrix": [list(row) for row in local_bsm.confusion_matrix],
            "source_field": "repeater_service_profile.state_model.local_swap_model",
            "calibration_status": "near-term same-isotope optical-qubit composite",
            "component_model": local_bsm.to_dict(),
        }
        overrides["swap_bsm.local_bell_measurement"] = copy.deepcopy(
            swap["bell_outcome_confusion_model"]
        )
    if adapter.get(
        "apply_endpoint_measurement_from_quality_restoration_components", False
    ):
        restoration = quality_restoration_model(profile)
        if restoration is None:
            raise ValueError(
                "endpoint measurement component override requires quality restoration"
            )
        analysis_error = average_gate_fidelity_to_pauli_error(
            restoration.local_single_qubit_average_gate_fidelity,
            dimension=2,
        )
        readout_error = 1.0 - restoration.measurement_readout_fidelity
        memory["endpoint_analysis_rotation_model"] = {
            "kind": "rotated_basis_depolarizing",
            "enters_probability": analysis_error > 0.0,
            "error_probability": analysis_error,
            "source_field": (
                "repeater_service_profile.state_model.quality_restoration."
                "local_single_qubit_gate_model"
            ),
            "source": restoration.component_sources,
            "calibration_status": (
                "near-term composite endpoint analysis pulse using the same declared "
                "trapped-ion single-qubit operation as local purification"
            ),
        }
        assignment = [[1.0 - readout_error, readout_error], [readout_error, 1.0 - readout_error]]
        memory["memory_readout_model"] = {
            "kind": "symmetric_assignment_error",
            "enters_probability": readout_error > 0.0,
            "error_probability": readout_error,
            "alice_confusion_matrix": assignment,
            "bob_confusion_matrix": [list(row) for row in assignment],
            "source_field": (
                "repeater_service_profile.state_model.quality_restoration."
                "measurement_model"
            ),
            "calibration_status": (
                "near-term composite endpoint fluorescence assignment using the same "
                "declared trapped-ion readout operating point as purification"
            ),
        }
        overrides["memory.endpoint_measurement"] = {
            "endpoint_analysis_rotation_model": copy.deepcopy(
                memory["endpoint_analysis_rotation_model"]
            ),
            "memory_readout_model": copy.deepcopy(memory["memory_readout_model"]),
        }
    out["repeater_service_profile"] = {
        **profile_summary(profile),
        "base_front_matter_hash": base_hash,
        "overrides": overrides,
        "histogram_adapter": copy.deepcopy(adapter),
    }
    out.setdefault("adapter_conversion_notes", {})["repeater_service_profile"] = {
        "status": "active_explicit_composite_or_projection",
        "profile_id": profile.profile_id,
        "claim_tier": profile.claim_tier,
        "overrides": overrides,
    }
    out["base_front_matter_hash"] = base_hash
    out["front_matter_hash"] = stable_hash({
        "base_front_matter_hash": base_hash,
        "repeater_service_profile": profile.to_dict(),
        "overrides": overrides,
    })
    return out


def repeater_elementary_state_channel(context: ModelContext) -> dict[str, Any] | None:
    profile = active_repeater_service_profile(context)
    if profile is None or not profile.histogram_adapter.get(
        "apply_elementary_state_channel", False
    ):
        return None
    return profile_elementary_state_channel(profile)


def repeater_elementary_pair_instrument(context: ModelContext) -> dict[str, Any] | None:
    profile = active_repeater_service_profile(context)
    if profile is None:
        return None
    return profile_elementary_pair_instrument(profile)


def repeater_swap_state_channel(context: ModelContext) -> dict[str, Any] | None:
    profile = active_repeater_service_profile(context)
    if profile is None or not profile.histogram_adapter.get(
        "apply_local_swap_state_channel", False
    ):
        return None
    return profile_swap_state_channel(profile)


def repeater_quality_restoration(
    context: ModelContext,
    edge_length_km: float,
) -> dict[str, Any] | None:
    profile = active_repeater_service_profile(context)
    if profile is None or not profile.histogram_adapter.get(
        "apply_quality_restoration", False
    ):
        return None
    return profile_quality_restoration(profile, edge_length_km)

def clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def require_probability(name: str, value: Any) -> float:
    value_f = float(value)
    if not 0.0 <= value_f <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return value_f


def validate_pauli_probabilities(probabilities: dict[str, Any], *, prefix: str) -> dict[str, float]:
    aliases = {
        "I": ("p_I", "p_i", "I", "i"),
        "X": ("p_X", "p_x", "X", "x"),
        "Y": ("p_Y", "p_y", "Y", "y"),
        "Z": ("p_Z", "p_z", "Z", "z"),
    }
    out: dict[str, float] = {}
    for label, names in aliases.items():
        for name in names:
            if name in probabilities:
                out[label] = require_probability(f"{prefix}.{name}", probabilities[name])
                break
    if "I" not in out:
        non_identity = sum(out.get(label, 0.0) for label in ("X", "Y", "Z"))
        if non_identity > 1.0 + 1e-12:
            raise ValueError(f"{prefix} Pauli probabilities exceed 1 without p_I")
        out["I"] = clip_probability(1.0 - non_identity)
    for label in ("X", "Y", "Z"):
        out.setdefault(label, 0.0)
    total = sum(out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{prefix} Pauli probabilities must sum to 1, got {total}")
    return {label: clip_probability(out[label]) for label in ("I", "X", "Y", "Z")}


def normalize_memory_process_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    source_field = "not_provided"
    raw: Any = None
    for key in (
        "memory_process_model",
        "process_model",
        "interface_process_model",
        "pre_emission_process_model",
    ):
        if key in parameters:
            raw = parameters[key]
            source_field = f"memory.parameters.{key}"
            break
    if raw is None:
        return {
            "kind": "none",
            "source_field": source_field,
            "enters_probability": False,
            "applied_to": "none",
            "calibration_status": "no explicit constructor/interface process model supplied",
            "scalar_fidelity_active": False,
            "diagnostic_override": False,
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    if kind in {"", "none", "identity", "ideal"}:
        return {
            "kind": "none",
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": False,
            "applied_to": "none",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit identity/no-constructor-process model",
            ),
            "source": raw.get("source"),
            "scalar_fidelity_active": False,
            "diagnostic_override": False,
        }
    if kind == "local_depolarizing":
        if "depolarizing_probability" not in raw:
            raise ValueError(f"{source_field}.depolarizing_probability is required")
        probability = require_probability(
            f"{source_field}.depolarizing_probability",
            raw["depolarizing_probability"],
        )
        return {
            "kind": "local_depolarizing",
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": True,
            "applied_to": "stationary_memory_qubits_before_emission",
            "depolarizing_probability": probability,
            "channel_semantics": (
                "single_qubit_local_depolarizing; applied independently to each stationary memory qubit, "
                "not an elementary-link Werner fidelity or final-pair depolarizing probability"
            ),
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed process parameter supplied by config",
            ),
            "source": raw.get("source"),
            "scalar_fidelity_active": False,
            "diagnostic_override": False,
        }
    if kind == "pauli_channel":
        probabilities = raw.get("probabilities", raw)
        if not isinstance(probabilities, dict):
            raise ValueError(f"{source_field}.probabilities must be an object for pauli_channel")
        pauli = validate_pauli_probabilities(probabilities, prefix=source_field)
        return {
            "kind": "pauli_channel",
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": True,
            "applied_to": "stationary_memory_qubits_before_emission",
            "pauli_probabilities": pauli,
            "cptp_validation": "probabilities are nonnegative and sum to 1",
            "channel_semantics": (
                "single_qubit_memory_local_pauli_channel; applied independently to each stationary memory qubit, "
                "not an elementary-link Werner fidelity or final-pair Pauli channel"
            ),
            "calibration_status": "explicit typed Pauli process supplied by config",
            "scalar_fidelity_active": False,
            "diagnostic_override": False,
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, "
        "local_depolarizing, pauli_channel"
    )


def _positive_float_from_model(name: str, value: Any) -> float:
    value_f = float(value)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value_f


def normalize_memory_storage_decoherence_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    source_field = "not_provided"
    raw: Any = None
    for key in (
        "memory_storage_decoherence_model",
        "storage_decoherence_model",
        "storage_model",
    ):
        if key in parameters:
            raw = parameters[key]
            source_field = f"memory.parameters.{key}"
            break
    if raw is None:
        t1_raw = (
            parameters.get("t1_s")
            if parameters.get("t1_s") is not None else
            parameters.get("T1_s")
            if parameters.get("T1_s") is not None else
            parameters.get("t1")
            if parameters.get("t1") is not None else
            parameters.get("T1")
        )
        t2_raw = (
            parameters.get("t2_s")
            if parameters.get("t2_s") is not None else
            parameters.get("T2_s")
            if parameters.get("T2_s") is not None else
            parameters.get("t2")
            if parameters.get("t2") is not None else
            parameters.get("T2")
        )
        if t1_raw is not None or t2_raw is not None:
            if t1_raw is None or t2_raw is None:
                raise ValueError("typed t1_t2 storage decoherence needs both T1 and T2")
            t1_s = _positive_float_from_model("memory.T1/t1", t1_raw)
            t2_s = _positive_float_from_model("memory.T2/t2", t2_raw)
            if t2_s > 2.0 * t1_s + 1e-12:
                raise ValueError("memory T1/T2 storage model is unphysical: T2 must be <= 2*T1")
            return {
                "kind": "t1_t2",
                "source_field": "derived_from_memory.parameters.T1_T2",
                "enters_probability": True,
                "applied_to": "stationary_memory_qubits_during_storage",
                "t1_s": t1_s,
                "t2_s": t2_s,
                "coherence_factor_formula": "gamma_1(t)=1-exp(-t/T1); 1/T_phi=1/T2-1/(2*T1); C_phi(t)=exp(-t/T_phi)",
                "calibration_status": "explicit T1/T2 lifetime fields supplied by config",
                "legacy_fallback": False,
            }
        coherence_raw = parameters.get("coherence_time")
        if coherence_raw is not None:
            coherence_s = _positive_float_from_model("memory.coherence_time", coherence_raw)
            return {
                "kind": "exponential_dephasing",
                "source_field": "legacy_fallback:memory.parameters.coherence_time",
                "enters_probability": True,
                "applied_to": "stationary_memory_qubits_during_storage",
                "coherence_time_s": coherence_s,
                "coherence_time_definition": "legacy untyped coherence_time interpreted as exponential pure-dephasing lifetime",
                "coherence_factor_formula": "C(t)=exp(-t/T2)",
                "calibration_status": "legacy fallback generic approximation; structured storage decoherence model not supplied",
                "legacy_fallback": True,
            }
        return {
            "kind": "none",
            "source_field": source_field,
            "enters_probability": False,
            "applied_to": "none",
            "coherence_factor_formula": "C(t)=1",
            "calibration_status": "no storage decoherence model supplied",
            "legacy_fallback": False,
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    if kind in {"", "none", "identity", "ideal"}:
        return {
            "kind": "none",
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": False,
            "applied_to": "none",
            "coherence_factor_formula": "C(t)=1",
            "calibration_status": "explicit no-storage-decoherence model",
            "legacy_fallback": False,
        }
    if kind in {"exponential_dephasing", "gaussian_dephasing"}:
        coherence_raw = (
            raw.get("coherence_time_s")
            if raw.get("coherence_time_s") is not None else
            raw.get("t2_s")
            if raw.get("t2_s") is not None else
            parameters.get("coherence_time")
        )
        if coherence_raw is None:
            raise ValueError(f"{source_field}.{kind} needs coherence_time_s")
        coherence_s = _positive_float_from_model(f"{source_field}.coherence_time_s", coherence_raw)
        formula = "C(t)=exp(-t/T2)" if kind == "exponential_dephasing" else "C(t)=exp(-(t/T_gauss)^2)"
        return {
            "kind": kind,
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": True,
            "applied_to": "stationary_memory_qubits_during_storage",
            "coherence_time_s": coherence_s,
            "coherence_time_definition": raw.get(
                "coherence_time_definition",
                "explicit one-parameter storage coherence time",
            ),
            "coherence_factor_formula": formula,
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed one-parameter storage-dephasing model; not a full calibrated memory process",
            ),
            "source": raw.get("source"),
            "legacy_fallback": False,
        }
    if kind == "liu_2026_anisotropic_pair_fit":
        required = (
            "xx_pair_coherence_time_s",
            "zz_initial_visibility",
            "zz_linear_decay_rate_per_s",
            "fit_domain_max_s",
        )
        missing = [name for name in required if raw.get(name) is None]
        if missing:
            raise ValueError(f"{source_field} is missing {', '.join(missing)}")
        normalized = copy.deepcopy(raw)
        normalized.update({
            "kind": kind,
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": True,
            "applied_to": "stationary_memory_qubits_during_storage",
            "xx_pair_coherence_time_s": _positive_float_from_model(
                f"{source_field}.xx_pair_coherence_time_s",
                raw["xx_pair_coherence_time_s"],
            ),
            "zz_initial_visibility": require_probability(
                f"{source_field}.zz_initial_visibility",
                raw["zz_initial_visibility"],
            ),
            "zz_linear_decay_rate_per_s": float(raw["zz_linear_decay_rate_per_s"]),
            "fit_domain_max_s": _positive_float_from_model(
                f"{source_field}.fit_domain_max_s", raw["fit_domain_max_s"]
            ),
            "coherence_factor_formula": (
                "lambda_x=lambda_y=exp(-t/(2*tau_xx)); "
                "lambda_z=sqrt(max(0,1-r_zz*t/V_zz(0)))"
            ),
            "calibration_status": raw.get(
                "calibration_status", "Liu 2026 measured anisotropic pair-storage fit"
            ),
            "legacy_fallback": False,
        })
        if normalized["zz_linear_decay_rate_per_s"] < 0.0:
            raise ValueError(f"{source_field}.zz_linear_decay_rate_per_s must be nonnegative")
        return normalized
    if kind == "t1_t2":
        t1_s = _positive_float_from_model(
            f"{source_field}.t1_s",
            raw.get("t1_s", raw.get("T1_s", raw.get("t1", raw.get("T1")))),
        )
        t2_s = _positive_float_from_model(
            f"{source_field}.t2_s",
            raw.get("t2_s", raw.get("T2_s", raw.get("t2", raw.get("T2")))),
        )
        if t2_s > 2.0 * t1_s + 1e-12:
            raise ValueError(f"{source_field}.t1_t2 is unphysical: T2 must be <= 2*T1")
        return {
            "kind": "t1_t2",
            "source_field": source_field,
            "raw_model": raw,
            "enters_probability": True,
            "applied_to": "stationary_memory_qubits_during_storage",
            "t1_s": t1_s,
            "t2_s": t2_s,
            "coherence_factor_formula": "gamma_1(t)=1-exp(-t/T1); 1/T_phi=1/T2-1/(2*T1); C_phi(t)=exp(-t/T_phi)",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed T1/T2 storage channel supplied by config",
            ),
            "legacy_fallback": False,
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, exponential_dephasing, gaussian_dephasing, liu_2026_anisotropic_pair_fit, t1_t2"
    )


def validate_assignment_confusion_matrix(raw: Any, *, prefix: str) -> list[list[float]]:
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"{prefix} must be a 2x2 row-stochastic matrix")
    if np.any(matrix < -1e-12):
        raise ValueError(f"{prefix} probabilities must be nonnegative")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, np.ones(2), rtol=1e-9, atol=1e-9):
        raise ValueError(f"{prefix} rows must sum to 1")
    matrix = np.clip(matrix, 0.0, 1.0)
    return [[float(matrix[0, 0]), float(matrix[0, 1])], [float(matrix[1, 0]), float(matrix[1, 1])]]


def normalize_memory_readout_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    source_field = "not_provided"
    raw: Any = None
    for key in (
        "memory_readout_model",
        "endpoint_readout_model",
        "readout_model",
    ):
        if key in parameters:
            raw = parameters[key]
            source_field = f"memory.parameters.{key}"
            break
    if raw is None:
        return {
            "kind": "none",
            "source_field": source_field,
            "enters_probability": False,
            "applies_to": "endpoint_memory_measurement",
            "affects": [],
            "does_not_affect": ["delivery/no_delivery mass"],
            "scalar_fidelity_active": False,
            "calibration_status": "no endpoint memory-readout assignment model supplied",
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    base = {
        "source_field": source_field,
        "raw_model": raw,
        "applies_to": "endpoint_memory_measurement",
        "affects": ["reported Alice/Bob transcript outcomes", "QBER", "CHSH", "Eve posterior from reported key bits"],
        "does_not_affect": [
            "delivery/no_delivery mass",
            "elementary success probability",
            "route-ready probability",
            "optical detector POVM",
            "memory storage channel",
            "memory_process_model",
        ],
        "scalar_fidelity_active": False,
    }
    if kind in {"", "none", "identity", "ideal"}:
        return {
            **base,
            "kind": "none",
            "enters_probability": False,
            "affects": [],
            "calibration_status": "explicit ideal endpoint memory readout",
            "alice_confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "bob_confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
        }
    if kind == "symmetric_assignment_error":
        if "error_probability" not in raw:
            raise ValueError(f"{source_field}.error_probability is required")
        r = require_probability(f"{source_field}.error_probability", raw["error_probability"])
        matrix = [[1.0 - r, r], [r, 1.0 - r]]
        return {
            **base,
            "kind": "symmetric_assignment_error",
            "enters_probability": r > 0.0,
            "error_probability": r,
            "alice_confusion_matrix": matrix,
            "bob_confusion_matrix": matrix,
            "matrix_convention": "rows=actual ideal outcome, columns=reported transcript outcome",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit endpoint memory-readout assignment error parameter supplied by config",
            ),
        }
    if kind == "assignment_confusion_matrix":
        if "alice_confusion_matrix" in raw or "bob_confusion_matrix" in raw:
            alice_raw = raw.get("alice_confusion_matrix", raw.get("confusion_matrix"))
            bob_raw = raw.get("bob_confusion_matrix", raw.get("confusion_matrix"))
            if alice_raw is None or bob_raw is None:
                raise ValueError(
                    f"{source_field} needs both alice_confusion_matrix and bob_confusion_matrix, or one shared confusion_matrix"
                )
            alice = validate_assignment_confusion_matrix(
                alice_raw,
                prefix=f"{source_field}.alice_confusion_matrix",
            )
            bob = validate_assignment_confusion_matrix(
                bob_raw,
                prefix=f"{source_field}.bob_confusion_matrix",
            )
        else:
            if "confusion_matrix" not in raw:
                raise ValueError(f"{source_field}.confusion_matrix is required")
            alice = validate_assignment_confusion_matrix(
                raw["confusion_matrix"],
                prefix=f"{source_field}.confusion_matrix",
            )
            bob = [list(row) for row in alice]
        return {
            **base,
            "kind": "assignment_confusion_matrix",
            "enters_probability": True,
            "alice_confusion_matrix": alice,
            "bob_confusion_matrix": bob,
            "matrix_convention": "rows=actual ideal outcome, columns=reported transcript outcome",
            "cptp_validation": "classical assignment matrices are nonnegative and row-stochastic",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit endpoint memory-readout assignment confusion matrix supplied by config",
            ),
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, symmetric_assignment_error, assignment_confusion_matrix"
    )


def normalize_endpoint_analysis_rotation_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    """Typed endpoint analysis-rotation error model.

    Ion endpoint readout is a Z-basis fluorescence measurement; every non-Z
    E91 setting first needs an analysis rotation pulse. This model applies a
    single-qubit depolarizing channel with the supplied error probability to
    the measured party's memory qubit for rotated (non-Z) settings only,
    before projection. Distinct from memory_readout_model (classical
    assignment confusion after outcomes) and from memory_process_model
    (emission noise).
    """
    source_field = "not_provided"
    raw: Any = None
    for key in ("endpoint_analysis_rotation_model", "analysis_rotation_model"):
        if key in parameters:
            raw = parameters[key]
            source_field = f"memory.parameters.{key}"
            break
    base = {
        "source_field": source_field,
        "applies_to": "endpoint_memory_measurements_in_rotated_(non-Z)_bases",
        "affects": ["rotated-setting outcome probabilities", "CHSH", "X-basis QBER contributions"],
        "does_not_affect": [
            "Z-basis key settings (direct fluorescence readout)",
            "delivery/no_delivery mass",
            "memory_readout_model assignment confusion",
        ],
    }
    if raw is None:
        return {
            **base,
            "kind": "none",
            "enters_probability": False,
            "calibration_status": (
                "no endpoint analysis-rotation error model supplied; rotated-basis analysis pulses are ideal"
            ),
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    if kind in {"", "none", "identity", "ideal"}:
        return {
            **base,
            "kind": "none",
            "raw_model": raw,
            "enters_probability": False,
            "calibration_status": "explicit ideal endpoint analysis rotations",
        }
    if kind == "rotated_basis_depolarizing":
        if "error_probability" not in raw:
            raise ValueError(f"{source_field}.error_probability is required")
        probability = require_probability(f"{source_field}.error_probability", raw["error_probability"])
        return {
            **base,
            "kind": "rotated_basis_depolarizing",
            "raw_model": raw,
            "enters_probability": probability > 0.0,
            "error_probability": probability,
            "channel_semantics": (
                "single-qubit depolarizing with the supplied probability applied to the measured party's "
                "memory qubit before projection, for settings whose measurement axis is not Z"
            ),
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed analysis-rotation error parameter supplied by config",
            ),
            "source": raw.get("source"),
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, rotated_basis_depolarizing"
    )


def endpoint_analysis_rotation_model(front: dict[str, Any]) -> dict[str, Any]:
    memory = front["device_constants"]["memory"]
    return copy.deepcopy(memory.get(
        "endpoint_analysis_rotation_model",
        {"kind": "none", "enters_probability": False},
    ))


def validate_bell_outcome_confusion_matrix(raw: Any, *, prefix: str) -> list[list[float]]:
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{prefix} must be a 4x4 row-stochastic matrix over {BELL_LABELS}")
    if np.any(matrix < -1e-12):
        raise ValueError(f"{prefix} probabilities must be nonnegative")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, np.ones(4), rtol=1e-9, atol=1e-9):
        raise ValueError(f"{prefix} rows must sum to 1")
    matrix = np.clip(matrix, 0.0, 1.0)
    return [[float(value) for value in row] for row in matrix]


def normalize_swap_bell_outcome_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    source_field = "not_provided"
    raw: Any = None
    for key in ("bell_outcome_confusion_model", "swap_bell_outcome_model"):
        if key in parameters:
            raw = parameters[key]
            source_field = f"swap_bsm.parameters.{key}"
            break
    base = {
        "source_field": source_field,
        "applies_to": "internal_repeater_swap_bell_measurement_outcome_label",
        "bell_label_order": list(BELL_LABELS),
        "affects": [
            "reported repeater Bell outcome label",
            "public/effective Pauli frame",
            "applied feed-forward correction and therefore the endpoint pair state",
        ],
        "does_not_affect": [
            "swap success probability / delivery mass",
            "endpoint memory_readout_model",
            "optical PNR detector confusion",
            "memory storage decoherence",
        ],
        "matrix_convention": "rows=actual projected Bell outcome, columns=reported/announced Bell label",
    }
    if raw is None:
        return {
            **base,
            "kind": "none",
            "enters_probability": False,
            "calibration_status": "no repeater Bell-outcome confusion model supplied; reported label equals projected outcome",
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    if kind in {"", "none", "identity", "ideal"}:
        return {
            **base,
            "kind": "none",
            "raw_model": raw,
            "enters_probability": False,
            "calibration_status": "explicit ideal repeater Bell-outcome readout",
        }
    if kind == "symmetric_bell_label_error":
        if "error_probability" not in raw:
            raise ValueError(f"{source_field}.error_probability is required")
        r = require_probability(f"{source_field}.error_probability", raw["error_probability"])
        matrix = [
            [1.0 - r if row == col else r / 3.0 for col in range(4)]
            for row in range(4)
        ]
        return {
            **base,
            "kind": "symmetric_bell_label_error",
            "raw_model": raw,
            "enters_probability": r > 0.0,
            "error_probability": r,
            "confusion_matrix": matrix,
            "calibration_status": raw.get(
                "calibration_status",
                "explicit symmetric repeater Bell-label error parameter supplied by config",
            ),
        }
    if kind == "bell_outcome_confusion_matrix":
        if "confusion_matrix" not in raw:
            raise ValueError(f"{source_field}.confusion_matrix is required")
        matrix = validate_bell_outcome_confusion_matrix(
            raw["confusion_matrix"],
            prefix=f"{source_field}.confusion_matrix",
        )
        identity = all(
            abs(matrix[row][col] - (1.0 if row == col else 0.0)) <= 1e-12
            for row in range(4) for col in range(4)
        )
        return {
            **base,
            "kind": "bell_outcome_confusion_matrix",
            "raw_model": raw,
            "enters_probability": not identity,
            "confusion_matrix": matrix,
            "cptp_validation": "classical Bell-label confusion matrix is nonnegative and row-stochastic",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit repeater Bell-outcome confusion matrix supplied by config",
            ),
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, symmetric_bell_label_error, bell_outcome_confusion_matrix"
    )


def normalize_fiber_polarization_noise_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    source_field = "not_provided"
    raw: Any = None
    for key in ("polarization_noise_model", "polarization_drift_model"):
        if key in parameters:
            raw = parameters[key]
            source_field = f"fiber_channel.parameters.{key}"
            break
    base = {
        "source_field": source_field,
        "applies_to": "flying_optical_polarization_qubits_per_elementary_half_link_traversal",
        "affects": [
            "elementary optical BSM click-pattern probabilities",
            "elementary success/inconclusive split",
            "conditioned remote-memory pair state, QBER, and CHSH",
        ],
        "does_not_affect": [
            "stationary memory storage decoherence",
            "endpoint memory readout",
            "repeater swap Bell-outcome labels",
        ],
        "length_dependence": (
            "none modeled; parameters are per half-link photon traversal regardless of fiber length "
            "until a calibrated per-length polarization drift model is supplied"
        ),
    }
    if raw is None:
        return {
            **base,
            "kind": "none",
            "enters_probability": False,
            "calibration_status": (
                "no typed polarization noise model supplied; scalar fiber_channel.polarization_fidelity "
                "remains provenance metadata only"
            ),
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    if kind in {"", "none", "identity", "ideal"}:
        return {
            **base,
            "kind": "none",
            "raw_model": raw,
            "enters_probability": False,
            "calibration_status": "explicit no-polarization-noise model",
        }
    if kind in {"depolarizing_on_flying_polarization", "single_qubit_depolarizing_on_flying_polarization"}:
        if "depolarizing_probability" not in raw:
            raise ValueError(f"{source_field}.depolarizing_probability is required")
        probability = require_probability(
            f"{source_field}.depolarizing_probability",
            raw["depolarizing_probability"],
        )
        return {
            **base,
            "kind": "depolarizing_on_flying_polarization",
            "raw_model": raw,
            "enters_probability": probability > 0.0,
            "depolarizing_probability": probability,
            "channel_semantics": "single-qubit depolarizing random-unitary mixture on each flying polarization qubit",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed flying-polarization depolarizing parameter supplied by config",
            ),
        }
    if kind == "pauli_channel_on_flying_polarization":
        probabilities = raw.get("probabilities", raw)
        if not isinstance(probabilities, dict):
            raise ValueError(f"{source_field}.probabilities must be an object for pauli_channel_on_flying_polarization")
        pauli = validate_pauli_probabilities(probabilities, prefix=source_field)
        return {
            **base,
            "kind": "pauli_channel_on_flying_polarization",
            "raw_model": raw,
            "enters_probability": pauli["I"] < 1.0,
            "pauli_probabilities": pauli,
            "cptp_validation": "probabilities are nonnegative and sum to 1",
            "channel_semantics": "single-qubit Pauli random-unitary mixture on each flying polarization qubit",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed flying-polarization Pauli channel supplied by config",
            ),
        }
    if kind == "unitary_rotation_on_flying_polarization":
        axis = str(raw.get("rotation_axis", "")).strip().upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError(f"{source_field}.rotation_axis must be X, Y, or Z")
        angle = float(raw.get("rotation_angle_rad", 0.0))
        return {
            **base,
            "kind": "unitary_rotation_on_flying_polarization",
            "raw_model": raw,
            "enters_probability": abs(angle) > 0.0,
            "rotation_axis": axis,
            "rotation_angle_rad": angle,
            "channel_semantics": "deterministic polarization-frame rotation exp(-i*theta/2*sigma_axis) on each flying polarization qubit",
            "calibration_status": raw.get(
                "calibration_status",
                "explicit typed flying-polarization unitary rotation supplied by config",
            ),
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, "
        "depolarizing_on_flying_polarization, pauli_channel_on_flying_polarization, "
        "unitary_rotation_on_flying_polarization"
    )


def normalize_photon_indistinguishability_model_from_config(parameters: dict[str, Any]) -> dict[str, Any]:
    source_field = "not_provided"
    raw: Any = None
    for key in ("photon_indistinguishability_model", "hom_visibility_model"):
        if key in parameters:
            raw = parameters[key]
            source_field = f"memory.parameters.{key}"
            break
    base = {
        "source_field": source_field,
        "applies_to": "two_photon_interference_at_elementary_optical_bsm",
        "model_semantics": (
            "phenomenological convex mixture: P = s * indistinguishable interference instrument "
            "+ (1 - s) * distinguishable-photon classical-routing reference, with s the mode overlap; "
            "not a calibrated wave-packet/spectral-temporal model"
        ),
        "affects": [
            "elementary BSM click-pattern probabilities",
            "conditioned remote-memory pair state",
            "final transcript distribution, QBER, and CHSH",
        ],
        "does_not_affect": [
            "memory storage decoherence",
            "endpoint memory readout",
            "repeater swap Bell-outcome labels",
        ],
    }
    if raw is None:
        return {
            **base,
            "kind": "none",
            "enters_probability": False,
            "mode_overlap": 1.0,
            "calibration_status": "no photon indistinguishability model supplied; perfect mode overlap assumed",
        }
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{source_field} must be an object or string kind")
    kind = str(raw.get("kind", raw.get("model", "none"))).strip().lower()
    if kind in {"", "none", "identity", "ideal"}:
        return {
            **base,
            "kind": "none",
            "raw_model": raw,
            "enters_probability": False,
            "mode_overlap": 1.0,
            "calibration_status": "explicit ideal photon indistinguishability",
        }
    if kind in {"mode_overlap", "visibility"}:
        value_key = "mode_overlap" if kind == "mode_overlap" else "visibility"
        if value_key not in raw:
            raise ValueError(f"{source_field}.{value_key} is required")
        overlap = require_probability(f"{source_field}.{value_key}", raw[value_key])
        return {
            **base,
            "kind": kind,
            "raw_model": raw,
            "enters_probability": overlap < 1.0,
            "mode_overlap": overlap,
            "calibration_status": raw.get(
                "calibration_status",
                "explicit phenomenological HOM mode-overlap/visibility parameter supplied by config",
            ),
        }
    raise ValueError(
        f"{source_field}.kind={kind!r} is unsupported; supported kinds are none, mode_overlap, visibility"
    )


def memory_process_model(front: dict[str, Any], context: ModelContext) -> dict[str, Any]:
    configured = copy.deepcopy(front["device_constants"]["memory"].get("memory_process_model", {"kind": "none"}))
    if configured.get("kind") == "none" and context.allow_generic_memory_fidelity_depolarizing:
        fidelity = clip_probability(front["device_constants"]["memory"]["fidelity"])
        return {
            "kind": "local_depolarizing",
            "source_field": "diagnostic_override:memory.fidelity",
            "enters_probability": True,
            "applied_to": "stationary_memory_qubits_before_emission",
            "depolarizing_probability": 1.0 - fidelity,
            "channel_semantics": (
                "diagnostic single_qubit_local_depolarizing; applied independently to each stationary memory qubit, "
                "not an elementary-link Werner fidelity or final-pair depolarizing probability"
            ),
            "calibration_status": (
                "diagnostic generic approximation; scalar memory.fidelity is not a calibrated "
                "depolarizing-channel probability"
            ),
            "scalar_fidelity_active": True,
            "diagnostic_override": True,
            "warning": (
                "Using memory.fidelity as p_depol=1-fidelity reproduces legacy diagnostic behavior "
                "but is not calibrated memory physics."
            ),
        }
    return configured


def memory_storage_decoherence_model(front: dict[str, Any]) -> dict[str, Any]:
    memory = front["device_constants"]["memory"]
    return copy.deepcopy(memory.get("memory_storage_decoherence_model", {"kind": "none"}))


def memory_readout_model(front: dict[str, Any]) -> dict[str, Any]:
    memory = front["device_constants"]["memory"]
    return copy.deepcopy(memory.get("memory_readout_model", {
        "kind": "none",
        "enters_probability": False,
        "alice_confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "bob_confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
    }))


def fiber_polarization_noise_model(front: dict[str, Any]) -> dict[str, Any]:
    fiber = front["device_constants"]["fiber_channel"]
    return copy.deepcopy(fiber.get("polarization_noise_model", {"kind": "none", "enters_probability": False}))


def photon_indistinguishability_model(front: dict[str, Any]) -> dict[str, Any]:
    memory = front["device_constants"]["memory"]
    return copy.deepcopy(memory.get(
        "photon_indistinguishability_model",
        {"kind": "none", "enters_probability": False, "mode_overlap": 1.0},
    ))


def photon_mode_overlap_value(model: dict[str, Any]) -> float:
    if model.get("kind", "none") == "none":
        return 1.0
    return clip_probability(float(model.get("mode_overlap", 1.0)))


def swap_bell_outcome_confusion_model(front: dict[str, Any]) -> dict[str, Any]:
    swap = front["device_constants"]["swap_bsm"]
    return copy.deepcopy(swap.get("bell_outcome_confusion_model", {"kind": "none", "enters_probability": False}))


def bell_outcome_reported_distribution(
    model: dict[str, Any],
    actual_label: str,
) -> tuple[tuple[str, float], ...]:
    """Distribution of announced repeater Bell labels given the projected outcome."""
    kind = model.get("kind", "none")
    if kind == "none" or not model.get("enters_probability", False):
        return ((actual_label, 1.0),)
    matrix = model["confusion_matrix"]
    row = matrix[BELL_LABELS.index(actual_label)]
    return tuple(
        (label, float(prob))
        for label, prob in zip(BELL_LABELS, row)
        if float(prob) > 0.0
    )


def apply_memory_readout_model_to_joint_probabilities(
    ideal_probs: np.ndarray,
    readout_model: dict[str, Any],
) -> np.ndarray:
    if not readout_model.get("enters_probability", False):
        return ideal_probs
    alice = np.asarray(
        readout_model.get("alice_confusion_matrix", [[1.0, 0.0], [0.0, 1.0]]),
        dtype=float,
    )
    bob = np.asarray(
        readout_model.get("bob_confusion_matrix", [[1.0, 0.0], [0.0, 1.0]]),
        dtype=float,
    )
    reported = alice.T @ ideal_probs @ bob
    reported = np.clip(reported, 0.0, 1.0)
    total = float(reported.sum())
    if total <= 0.0:
        return np.full((2, 2), 0.25)
    return reported / total


def memory_process_channel_ops(model: dict[str, Any]) -> tuple[np.ndarray, ...]:
    from .quantum import depolarizing_kraus

    kind = model.get("kind", "none")
    if kind == "none":
        return (I2,)
    if kind == "local_depolarizing":
        return depolarizing_kraus(float(model["depolarizing_probability"]))
    if kind == "pauli_channel":
        probs = model["pauli_probabilities"]
        return tuple(
            math.sqrt(float(probs[label])) * PAULI_BY_LABEL[label]
            for label in ("I", "X", "Y", "Z")
            if float(probs[label]) > 0.0
        )
    raise ValueError(f"unsupported memory_process_model kind {kind!r}")
