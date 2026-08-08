"""Fock-space instruments, detector responses, and link delivery models."""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import sys
from typing import Any, Sequence

import numpy as np

from sequence_game.physical.entanglement_purification import (
    bell_diagonal_density,
    bilateral_phase_error_recurrence,
)
from sequence_game.physical.repeater_service import quality_restoration_model

from .devices import *
from .devices import _float_or_none
from .models import *
from .models import _json_ready
from .quantum import *


def apply_elementary_quality_restoration_to_branches(
    branches: list[Branch],
    *,
    edge_id: str,
    edge_length_m: float,
    context: ModelContext,
) -> tuple[list[Branch], dict[str, Any] | None]:
    """Apply the profile's exact two-copy LOCC instrument to raw edge branches."""

    restoration = repeater_quality_restoration(
        context,
        float(edge_length_m) / 1000.0,
    )
    if restoration is None:
        return branches, None
    profile = active_repeater_service_profile(context)
    if profile is None:
        raise ValueError("quality restoration requires an active repeater service profile")
    model = quality_restoration_model(profile)
    if model is None:
        raise ValueError("quality restoration requested without a validated recurrence model")
    sacrificial_coherence = float(restoration["sacrificial_wait_coherence"])
    raw_branch_count = len(branches)
    raw_records_marginalized = False
    if branches and all(branch.eve_observation == "none" for branch in branches):
        # Purification acts on the density operator, so applying it to the
        # public-record-marginal mixture is exactly the Cartesian sum over two
        # independent raw branch copies.  The profile is Alice-only in this
        # pass; preserving Eve-conditioned branches remains the fail-safe path
        # whenever a nontrivial private observation is present.
        mixed = np.zeros((4, 4), dtype=complex)
        for branch in branches:
            mixed += float(branch.probability) * branch.rho
        branches = [Branch(
            probability=1.0,
            rho=clean_density(mixed),
            frame="I",
            eve_observation="none",
            public_bsm_record=f"{edge_id}:raw_pair_records_marginalized",
            detector_click_record=DETECTOR_CLICK_RECORD_NOT_EMITTED,
        )]
        raw_records_marginalized = True
    restored: list[Branch] = []
    accepted_mass = 0.0
    for control in branches:
        for sacrificial in branches:
            result = bilateral_phase_error_recurrence(
                control.rho,
                sacrificial.rho,
                model,
                target_label=str(restoration["target_bell_state"]),
                sacrificial_wait_coherence=sacrificial_coherence,
            )
            probability = (
                float(control.probability)
                * float(sacrificial.probability)
                * result.success_probability
            )
            if probability <= 0.0:
                continue
            accepted_mass += probability
            restored.append(Branch(
                probability=probability,
                rho=result.output_density_matrix,
                frame=control.frame,
                eve_observation=join_labels(
                    control.eve_observation,
                    sacrificial.eve_observation,
                ),
                public_bsm_record=join_labels(
                    join_labels(control.public_bsm_record, sacrificial.public_bsm_record),
                    f"{edge_id}:purification_accept",
                ),
                detector_click_record=join_detector_click_records(
                    control.detector_click_record,
                    sacrificial.detector_click_record,
                    keep_detector_click_records=True,
                ),
            ))
    if accepted_mass <= 0.0:
        raise ValueError("quality restoration produced zero conditional accepted mass")
    normalized = normalize_branches(restored)
    return normalized, {
        **restoration,
        "conditional_acceptance_probability_for_detector_conditioned_state": (
            accepted_mass
        ),
        "raw_conditional_branch_count": raw_branch_count,
        "purification_input_branch_count": len(branches),
        "accepted_conditional_branch_count": len(normalized),
        "raw_public_records_marginalized": raw_records_marginalized,
        "marginalization_exactness": (
            "exact by bilinearity of the unnormalized two-copy LOCC instrument "
            "for Alice's public-record-marginal density matrix"
            if raw_records_marginalized else
            "Eve-conditioned branches retained"
        ),
        "state_semantics": (
            "full two-copy detector-conditioned density matrices enter the noisy LOCC "
            "instrument; the profile Bell-diagonal estimate is not substituted here"
        ),
    }

def edge_transmittance(front: dict[str, Any], length_m: float) -> float:
    attenuation = front["device_constants"]["fiber_channel"]["attenuation_db_per_m"]
    return clip_probability(10.0 ** (-(attenuation * float(length_m)) / 10.0))


def half_edge_transmittance(front: dict[str, Any], length_m: float) -> float:
    attenuation = front["device_constants"]["fiber_channel"]["attenuation_db_per_m"]
    return clip_probability(10.0 ** (-(attenuation * float(length_m) / 2.0) / 10.0))


def fiber_group_speed_m_per_ps(front: dict[str, Any]) -> float:
    fiber = front["device_constants"]["fiber_channel"]
    return float(
        fiber.get(
            "fiber_group_speed_m_per_ps",
            fiber.get("fiber_propagation_speed_m_per_ps", fiber.get("light_speed_m_per_ps", 0.0)),
        )
    )


def edge_delay_ps(front: dict[str, Any], length_m: float) -> int:
    speed = fiber_group_speed_m_per_ps(front)
    return round(float(length_m) / speed) if speed > 0.0 else 0


def half_edge_delay_ps(front: dict[str, Any], length_m: float) -> int:
    speed = fiber_group_speed_m_per_ps(front)
    return round((float(length_m) / 2.0) / speed) if speed > 0.0 else 0


def detector_dark_window_probability(front: dict[str, Any]) -> float:
    detector = front["device_constants"]["detector"]
    dark_rate = float(detector["dark_count_per_s"])
    window_ps = float(detector["barrett_kok_trigger_window_ps"])
    return clip_probability(1.0 - math.exp(-dark_rate * window_ps * 1e-12))


def detector_cooldown_period_ps(
    front: dict[str, Any],
    context: ModelContext | None = None,
) -> int:
    if context is None:
        count_rate = float(front["device_constants"]["detector"]["count_rate_hz"])
    else:
        resolved = detector_count_rate_model(front, context)
        count_rate = resolved["count_rate_hz"]
    if count_rate is None:
        return 0
    if count_rate <= 0.0:
        return sys.maxsize
    return int(math.ceil(1e12 / count_rate))


def binomial_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    p = clip_probability(p)
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def poisson_pmf(k: int, mean: float) -> float:
    if k < 0:
        return 0.0
    if mean <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mean) * (mean ** k) / math.factorial(k)


def fock_loss_distribution(n: int, eta: float, cutoff: int) -> dict[int, float]:
    if n < 0:
        raise ValueError("Fock photon number must be nonnegative")
    eta = clip_probability(eta)
    out: dict[int, float] = {}
    for k in range(0, min(n, cutoff) + 1):
        out[k] = binomial_pmf(n, k, eta)
    if n > cutoff:
        out[cutoff] = out.get(cutoff, 0.0) + sum(
            binomial_pmf(n, k, eta) for k in range(cutoff + 1, n + 1)
        )
    return out


def poisson_dark_distribution(mean: float, max_count: int) -> dict[int, float]:
    if max_count < 0:
        raise ValueError("max_count must be nonnegative")
    dist = {k: poisson_pmf(k, mean) for k in range(max_count)}
    # The saturated top bin carries all remaining Poisson mass P(K >= max_count).
    dist[max_count] = max(0.0, 1.0 - sum(dist.values()))
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()} if total > 0 else {0: 1.0}


def resolved_detector_kind(front: dict[str, Any], context: ModelContext) -> str:
    return validate_detector_profile_for_kind(front, context)


def detector_max_count(context: ModelContext, profile: dict[str, Any]) -> int:
    if context.detector_kind == "ideal_pnr":
        return max(1, context.fock_cutoff)
    value = profile.get("max_resolvable_photon_number")
    if value is None:
        return max(1, context.fock_cutoff) if profile_claims_pnr(profile) else 1
    return max(1, int(value))


def detector_efficiency_at_wavelength(
    front: dict[str, Any],
    context: ModelContext,
    wavelength_nm: float,
) -> tuple[float, dict[str, Any]]:
    profile = detector_profile_metadata(front, context)
    if context.detector_kind == "ideal_pnr":
        return 1.0, {
            "quantity": "detector_efficiency",
            "wavelength_nm": wavelength_nm,
            "wavelength_dependency": "ideal_detector_override",
            "value": 1.0,
        }
    profile_eff = profile.get("system_detection_efficiency")
    scalar = float(profile_eff if profile_eff is not None else front["device_constants"]["detector"]["efficiency"])
    return evaluate_response_curve(
        context.detector_response,
        wavelength_nm=wavelength_nm,
        value_key="efficiency",
        scalar_default=scalar,
        quantity="detector_efficiency",
    )


def fiber_attenuation_at_wavelength(
    front: dict[str, Any],
    context: ModelContext,
    wavelength_nm: float,
) -> tuple[float, dict[str, Any]]:
    scalar = float(front["device_constants"]["fiber_channel"]["attenuation_db_per_m"])
    return evaluate_response_curve(
        context.fiber_response,
        wavelength_nm=wavelength_nm,
        value_key="attenuation_db_per_m",
        scalar_default=scalar,
        quantity="fiber_attenuation_db_per_m",
    )


def conversion_throughput_at_wavelength(
    context: ModelContext,
    wavelength_nm: float,
) -> tuple[float, dict[str, Any]]:
    return evaluate_response_curve(
        context.conversion_response,
        wavelength_nm=wavelength_nm,
        value_key="throughput",
        scalar_default=1.0,
        quantity="conversion_or_filter_throughput",
    )


def fock_link_transmittance(
    front: dict[str, Any],
    context: ModelContext,
    length_m: float,
) -> tuple[float, dict[str, Any]]:
    stages = wavelength_stages(front, context)
    conversion_input_wavelength = stages["conversion_input_wavelength_nm"]
    propagation_wavelength = stages["propagation_wavelength_nm"]
    attenuation, attenuation_meta = fiber_attenuation_at_wavelength(
        front,
        context,
        propagation_wavelength,
    )
    conversion, conversion_meta = conversion_throughput_at_wavelength(context, conversion_input_wavelength)
    memory_or_source = (
        float(front["device_constants"]["memory"]["efficiency"])
        if context.source_model == "memory_emission"
        else 1.0
    )
    channel = 10.0 ** (-(attenuation * float(length_m) / 2.0) / 10.0)
    total = clip_probability(memory_or_source * conversion * channel)
    service_profile = active_repeater_service_profile(context)
    aggregate_override = (
        service_profile is not None
        and service_profile.histogram_adapter.get(
            "aggregate_zero_loss_link_probability", False
        )
        and service_profile.rate_model.get("kind")
        == "krutyanskiy_midpoint_two_photon"
    )
    aggregate_meta = None
    if aggregate_override:
        detector_efficiency, detector_meta = detector_efficiency_at_wavelength(
            front,
            context,
            stages["detector_wavelength_nm"],
        )
        if detector_efficiency <= 0.0:
            raise ValueError(
                "repeater service aggregate link probability requires positive detector efficiency"
            )
        p0 = float(
            service_profile.rate_model["per_node_zero_loss_detection_probability"]
        )
        gamma = float(service_profile.rate_model["attenuation_exponent_per_km"])
        profile_half_channel = 10.0 ** (
            -gamma * (float(length_m) / 1000.0) / 2.0
        )
        pre_detector = p0 * profile_half_channel / detector_efficiency
        if pre_detector > 1.0 + 1e-12:
            raise ValueError(
                "repeater service aggregate P_link^0 cannot be represented with the selected "
                f"detector: required pre-detector probability {pre_detector:.6g} exceeds 1"
            )
        total = clip_probability(pre_detector)
        channel = profile_half_channel
        aggregate_meta = {
            "active": True,
            "profile_id": service_profile.profile_id,
            "claim_tier": service_profile.claim_tier,
            "per_node_zero_loss_detection_probability": p0,
            "attenuation_exponent_per_km": gamma,
            "selected_detector_efficiency": detector_efficiency,
            "selected_detector_efficiency_metadata": detector_meta,
            "effective_pre_detector_probability": total,
            "mapping": (
                "eta_path=P_link^0*10^(-gamma*L_km/2)/eta_detector, so the "
                "profile's aggregate generation-and-detection probability is not multiplied "
                "again by repository memory efficiency or conversion throughput"
            ),
            "ignored_component_scalars": {
                "memory_or_source_emission_efficiency": memory_or_source,
                "conversion_throughput": conversion,
                "repository_fiber_half_channel_transmittance": (
                    10.0 ** (-(attenuation * float(length_m) / 2.0) / 10.0)
                ),
            },
        }
    return total, {
        "wavelength_nm": propagation_wavelength,
        "source_wavelength_nm": stages["source_wavelength_nm"],
        "conversion_input_wavelength_nm": conversion_input_wavelength,
        "conversion_output_wavelength_nm": stages["conversion_output_wavelength_nm"],
        "propagation_wavelength_nm": propagation_wavelength,
        "detector_wavelength_nm": stages["detector_wavelength_nm"],
        "wavelength_stages": stages,
        "half_link_length_m": float(length_m) / 2.0,
        "memory_or_source_emission_efficiency": memory_or_source,
        "conversion_throughput": conversion,
        "half_channel_transmittance": channel,
        "total_pre_detector_transmittance": total,
        "repeater_service_aggregate_link_override": aggregate_meta,
        "attenuation_response": attenuation_meta,
        "conversion_response": conversion_meta,
    }


def detector_timing_window_at_wavelength(
    front: dict[str, Any],
    context: ModelContext,
    wavelength_nm: float,
) -> tuple[float, dict[str, Any]]:
    if context.detector_kind == "ideal_pnr":
        return 0.0, {
            "quantity": "detector_timing_window_ps",
            "wavelength_nm": wavelength_nm,
            "wavelength_dependency": "ideal_detector_override",
            "value": 0.0,
        }
    profile = detector_profile_metadata(front, context)
    scalar_value = float(
        profile.get("detection_window_ps")
        if profile.get("detection_window_ps") is not None
        else profile.get("timing_jitter_or_resolution_ps")
        if profile.get("timing_jitter_or_resolution_ps") is not None
        else front["device_constants"]["detector"]["time_resolution_ps"]
    )
    timing_curve = profile.get("timing_jitter_response")
    if timing_curve is not None:
        value, meta = evaluate_response_curve(
            timing_curve,
            wavelength_nm=wavelength_nm,
            value_key="timing_jitter_fwhm_ps",
            scalar_default=scalar_value,
            quantity="detector_timing_window_ps",
        )
        meta["source"] = "detector_profile.timing_jitter_response"
        meta["interpretation"] = "FWHM timing jitter used as the single-window dark-count integration interval"
        return float(value), meta
    source = (
        "detector profile scalar"
        if profile.get("detection_window_ps") is not None or profile.get("timing_jitter_or_resolution_ps") is not None
        else "repository scalar config"
    )
    return scalar_value, {
        "quantity": "detector_timing_window_ps",
        "wavelength_nm": wavelength_nm,
        "wavelength_dependency": "scalar_flat_response_assumption",
        "value": scalar_value,
        "source": source,
        "interpretation": "single-window dark-count integration interval",
    }


def conversion_noise_rate_per_s(context: ModelContext) -> float:
    """In-band conversion background rate per converted optical channel.

    Read from the loaded conversion/filter response file
    (conversion_noise_rate_per_s). Semantics: detected in-band background
    counts per second per converted channel after the converter's own filter
    stack (e.g. anti-Stokes Raman scattering of the QFC pump), measured at the
    detector like dark counts; no additional detector-efficiency factor is
    applied.
    """
    if context.conversion_response is None:
        return 0.0
    value = _float_or_none(context.conversion_response.get("conversion_noise_rate_per_s"))
    return max(0.0, value) if value is not None else 0.0


def detector_dark_count_model(
    front: dict[str, Any],
    context: ModelContext,
    wavelength_nm: float | None = None,
) -> tuple[float, dict[str, Any]]:
    wavelength = active_wavelength_nm(front, context) if wavelength_nm is None else wavelength_nm
    if context.detector_kind == "ideal_pnr":
        return 0.0, {
            "dark_count_rate_per_s": 0.0,
            "dark_count_rate_source": "ideal_detector_override",
            "timing_window_ps": 0.0,
            "timing_window_response": {
                "quantity": "detector_timing_window_ps",
                "wavelength_nm": wavelength,
                "wavelength_dependency": "ideal_detector_override",
                "value": 0.0,
            },
            "dark_count_mean_per_window": 0.0,
            "conversion_background_rate_per_detector_per_s": 0.0,
        }
    profile = detector_profile_metadata(front, context)
    profile_dark = profile.get("dark_count_rate_per_s")
    if profile_dark is None:
        dark_rate = float(front["device_constants"]["detector"]["dark_count_per_s"])
        dark_source = "repository scalar config"
    else:
        dark_rate = float(profile_dark)
        dark_source = "detector profile scalar"
    window_ps, timing_meta = detector_timing_window_at_wavelength(front, context, wavelength)
    # Conversion-induced in-band background: each elementary-link optical
    # channel passes one converter, so two converted channels feed the
    # midpoint analyzer. The background is spectrally broad/unpolarized, so
    # each channel's rate splits evenly over the four analyzer detector modes:
    # per-detector rate = 2 channels * rate / 4 = rate / 2.
    noise_rate = conversion_noise_rate_per_s(context)
    background_per_detector = noise_rate / 2.0
    dark_mean = max(0.0, (dark_rate + background_per_detector) * window_ps * 1e-12)
    return dark_mean, {
        "dark_count_rate_per_s": dark_rate,
        "dark_count_rate_source": dark_source,
        "timing_window_ps": window_ps,
        "timing_window_response": timing_meta,
        "dark_count_mean_per_window": dark_mean,
        "conversion_noise_rate_per_channel_per_s": noise_rate,
        "conversion_background_rate_per_detector_per_s": background_per_detector,
        "conversion_background_semantics": (
            "detected in-band conversion background per converted channel (post-filter), split evenly "
            "over the four analyzer detector modes (2 channels / 4 detectors); integrated over the same "
            "single-window interval as dark counts; zero when no conversion response with "
            "conversion_noise_rate_per_s is loaded"
        ),
    }


def detector_dark_mean_for_window(front: dict[str, Any], context: ModelContext) -> float:
    return detector_dark_count_model(front, context)[0]


def snspd_detector_distribution(
    n_incident: int,
    *,
    eta_det: float,
    dark_mean: float,
    detector_kind: str,
    max_count: int,
    pnr_confusion_matrix: dict[str, Any] | None = None,
) -> dict[int, float]:
    if detector_kind == "ideal_pnr":
        return {min(n_incident, max_count): 1.0}
    p_dark = clip_probability(1.0 - math.exp(-dark_mean))
    if detector_kind == "threshold_snspd":
        p_no_click = (1.0 - p_dark) * ((1.0 - clip_probability(eta_det)) ** n_incident)
        return {0: p_no_click, 1: 1.0 - p_no_click}
    if detector_kind != "pnr_snspd":
        raise ValueError(f"unsupported detector kind {detector_kind!r}")
    if pnr_confusion_matrix is not None:
        row_index = min(n_incident, int(pnr_confusion_matrix["max_input_photon_number"]))
        row = pnr_confusion_matrix["reported_count_probabilities"][row_index]
        return {count: float(prob) for count, prob in enumerate(row)}
    signal = {
        k: binomial_pmf(n_incident, k, eta_det)
        for k in range(0, min(n_incident, max_count) + 1)
    }
    if n_incident > max_count:
        signal[max_count] = signal.get(max_count, 0.0) + sum(
            binomial_pmf(n_incident, k, eta_det)
            for k in range(max_count + 1, n_incident + 1)
        )
    dark = poisson_dark_distribution(dark_mean, max_count)
    out: dict[int, float] = {}
    for s_count, s_prob in signal.items():
        for d_count, d_prob in dark.items():
            count = min(max_count, s_count + d_count)
            out[count] = out.get(count, 0.0) + s_prob * d_prob
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0.0 else {0: 1.0}


def fock_detector_distribution_for_emitted_mode(
    n_source: int,
    *,
    pre_detector_eta: float,
    eta_det: float,
    dark_mean: float,
    detector_kind: str,
    max_count: int,
    cutoff: int,
    pnr_confusion_matrix: dict[str, Any] | None = None,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for n_incident, loss_prob in fock_loss_distribution(n_source, pre_detector_eta, cutoff).items():
        det = snspd_detector_distribution(
            n_incident,
            eta_det=eta_det,
            dark_mean=dark_mean,
            detector_kind=detector_kind,
            max_count=max_count,
            pnr_confusion_matrix=pnr_confusion_matrix,
        )
        for count, prob in det.items():
            out[count] = out.get(count, 0.0) + loss_prob * prob
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0.0 else {0: 1.0}


FOCK_MODE_LABELS = ("d0H", "d0V", "d1H", "d1V")
INPUT_MODE_LABELS = ("aH", "aV", "bH", "bV")
FOCK_EDGE_INSTRUMENT_CACHE: dict[str, dict[str, Any]] = {}
FOCK_EDGE_CACHE_HITS = 0
FOCK_EDGE_CACHE_MISSES = 0


def fock_basis(num_modes: int, cutoff: int) -> tuple[tuple[int, ...], ...]:
    if cutoff < 0:
        raise ValueError("Fock cutoff must be nonnegative")
    out: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...], remaining_modes: int, remaining_total: int) -> None:
        if remaining_modes == 0:
            out.append(prefix)
            return
        for value in range(remaining_total + 1):
            rec(prefix + (value,), remaining_modes - 1, remaining_total - value)

    rec((), num_modes, cutoff)
    return tuple(out)


def memory_photon_state_density(front: dict[str, Any], context: ModelContext, basis: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Build rho_{M_L,M_R,P_L,P_R} for |0,H>+|1,V> memory emission.

    Memory qubits are ordered (M_L, M_R). Photonic input modes are ordered
    (a_H, a_V, b_H, b_V). The convention is:
        |psi>_{M,P} = (|0>_M |H>_P + |1>_M |V>_P) / sqrt(2).
    Explicit memory_process_model channels are applied to each stationary memory
    before the optical analyzer. Scalar memory.fidelity is metadata/provenance
    only unless the diagnostic generic-fidelity override is enabled.
    """
    if context.fock_cutoff < 2:
        raise ValueError("fock_pnr_snspd memory_emission requires --fock-cutoff >= 2")
    basis_index = {occ: idx for idx, occ in enumerate(basis)}
    photon_dim = len(basis)
    state = np.zeros(4 * photon_dim, dtype=complex)
    for left_bit in (0, 1):
        for right_bit in (0, 1):
            occ = [0, 0, 0, 0]
            occ[left_bit] = 1
            occ[2 + right_bit] = 1
            mem_index = 2 * left_bit + right_bit
            state[mem_index * photon_dim + basis_index[tuple(occ)]] = 0.5
    rho = density(state)
    process = memory_process_model(front, context)
    if process.get("enters_probability"):
        ops = memory_process_channel_ops(process)
        rho = apply_hybrid_memory_channel(rho, photon_dim, memory_qubit=0, ops=ops)
        rho = apply_hybrid_memory_channel(rho, photon_dim, memory_qubit=1, ops=ops)
    return clean_density(rho)


def apply_hybrid_memory_channel(
    rho: np.ndarray,
    photon_dim: int,
    *,
    memory_qubit: int,
    ops: Iterable[np.ndarray],
) -> np.ndarray:
    out = np.zeros_like(rho, dtype=complex)
    for op in ops:
        mem_op = np.kron(op, I2) if memory_qubit == 0 else np.kron(I2, op)
        lifted = np.kron(mem_op, np.eye(photon_dim, dtype=complex))
        out += lifted @ rho @ lifted.conj().T
    return clean_density(out)


def photon_polarization_projector_operator(
    basis_states: tuple[tuple[int, ...], ...],
    *,
    side: str,
    basis_label: str,
    outcome: int,
) -> np.ndarray:
    if basis_label not in {"Z", "X"}:
        raise ValueError("photonic edge measure-resend basis must be Z or X")
    mode_h, mode_v = (0, 1) if side == "left" else (2, 3)
    if basis_label == "Z":
        ket = np.array([1.0, 0.0], dtype=complex) if outcome == 0 else np.array([0.0, 1.0], dtype=complex)
    else:
        sign = 1.0 if outcome == 0 else -1.0
        ket = np.array([1.0, sign], dtype=complex) / math.sqrt(2.0)
    projector = np.outer(ket, ket.conj())
    index = {occ: idx for idx, occ in enumerate(basis_states)}
    op = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
    for occ in basis_states:
        side_count = occ[mode_h] + occ[mode_v]
        if side_count != 1:
            continue
        in_pol = 0 if occ[mode_h] == 1 else 1
        base = list(occ)
        base[mode_h] = 0
        base[mode_v] = 0
        for out_pol in (0, 1):
            out_occ = list(base)
            out_occ[mode_h if out_pol == 0 else mode_v] = 1
            out_tuple = tuple(out_occ)
            if out_tuple in index:
                op[index[out_tuple], index[occ]] = projector[out_pol, in_pol]
    return op


def apply_hybrid_photon_operator(rho: np.ndarray, photon_op: np.ndarray) -> np.ndarray:
    photon_dim = photon_op.shape[0]
    lifted = np.kron(np.eye(4, dtype=complex), photon_op)
    return lifted @ rho @ lifted.conj().T


def apply_edge_measure_resend_to_memory_photon_state(
    rho: np.ndarray,
    basis_states: tuple[tuple[int, ...], ...],
    *,
    edge_id: str,
    attack: EveMove,
) -> list[tuple[float, np.ndarray, str]]:
    if not (
        attack.target_active
        and attack.move_kind == "edge_measure_resend"
        and attack.target_id == edge_id
    ):
        return [(1.0, rho, "none")]
    probability = clip_probability(attack.edge_measure_probability)
    info_probability = clip_probability(attack.information_probability)
    basis_choice = attack.edge_measure_basis_label if attack.edge_measure_basis_label != "random" else attack.basis_choice
    basis_choices = ("Z", "X") if basis_choice == "random" else (basis_choice,)
    branches: list[tuple[float, np.ndarray, str]] = []
    if probability < 1.0:
        branches.append((1.0 - probability, rho, "none"))
    basis_weight = probability / len(basis_choices)
    for basis_label in basis_choices:
        for outcome in (0, 1):
            op = photon_polarization_projector_operator(
                basis_states,
                side="right",
                basis_label=basis_label,
                outcome=outcome,
            )
            projected = apply_hybrid_photon_operator(rho, op)
            p_outcome = max(0.0, float(np.real_if_close(np.trace(projected))))
            if p_outcome <= 0.0:
                continue
            normalized = clean_density(projected / p_outcome)
            event_probability = basis_weight * p_outcome
            label = f"edge_measure_resend:edge={edge_id}:basis={basis_label}:outcome={outcome}"
            if info_probability > 0.0:
                branches.append((event_probability * info_probability, normalized, label))
            if info_probability < 1.0:
                branches.append((event_probability * (1.0 - info_probability), normalized, "none"))
    total = sum(weight for weight, _state, _label in branches)
    if total <= 0.0:
        return []
    return [(weight / total, state, label) for weight, state, label in branches]


apply_edge_intercept_to_memory_photon_state = apply_edge_measure_resend_to_memory_photon_state


def loss_amplitude(n: int, k: int, eta: float) -> float:
    return math.sqrt(binomial_pmf(n, k, eta))


def apply_loss_channel_hybrid(
    rho: np.ndarray,
    basis_states: tuple[tuple[int, ...], ...],
    eta_by_mode: tuple[float, ...],
) -> np.ndarray:
    photon_dim = len(basis_states)
    index = {occ: idx for idx, occ in enumerate(basis_states)}
    out = np.zeros_like(rho, dtype=complex)
    transmitted_options: dict[tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...], float]]] = {}
    for occ in basis_states:
        options: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

        def rec(mode: int, kept: list[int], lost: list[int], amp: float) -> None:
            if mode == len(occ):
                kept_tuple = tuple(kept)
                if kept_tuple in index:
                    options.append((kept_tuple, tuple(lost), amp))
                return
            n = occ[mode]
            for k in range(n + 1):
                rec(
                    mode + 1,
                    kept + [k],
                    lost + [n - k],
                    amp * loss_amplitude(n, k, eta_by_mode[mode]),
                )

        rec(0, [], [], 1.0)
        transmitted_options[occ] = options
    for mem_row in range(4):
        for mem_col in range(4):
            for occ_row in basis_states:
                row_index = mem_row * photon_dim + index[occ_row]
                for occ_col in basis_states:
                    value = rho[row_index, mem_col * photon_dim + index[occ_col]]
                    if abs(value) <= 0.0:
                        continue
                    for kept_row, lost_row, amp_row in transmitted_options[occ_row]:
                        for kept_col, lost_col, amp_col in transmitted_options[occ_col]:
                            if lost_row != lost_col:
                                continue
                            out[
                                mem_row * photon_dim + index[kept_row],
                                mem_col * photon_dim + index[kept_col],
                            ] += amp_row * amp_col * value
    return np.asarray(np.real_if_close(out), dtype=complex)


def beam_splitter_unitary_matrix() -> np.ndarray:
    inv = 1.0 / math.sqrt(2.0)
    # Rows are input creation operators (aH,aV,bH,bV); columns are output modes
    # (d0H,d0V,d1H,d1V): a_pol -> (d0_pol+d1_pol)/sqrt(2),
    # b_pol -> (d0_pol-d1_pol)/sqrt(2).
    return np.array([
        [inv, 0.0, inv, 0.0],
        [0.0, inv, 0.0, inv],
        [inv, 0.0, -inv, 0.0],
        [0.0, inv, 0.0, -inv],
    ], dtype=complex)


def transform_fock_occupation(
    occupation: tuple[int, ...],
    basis_states: tuple[tuple[int, ...], ...],
    unitary: np.ndarray,
) -> dict[tuple[int, ...], complex]:
    basis_index = {occ: idx for idx, occ in enumerate(basis_states)}
    state: dict[tuple[int, ...], complex] = {(0, 0, 0, 0): 1.0 + 0.0j}
    input_factor = 1.0
    for input_mode, count in enumerate(occupation):
        input_factor *= math.sqrt(math.factorial(count))
        for _ in range(count):
            updated: dict[tuple[int, ...], complex] = {}
            for occ, amplitude in state.items():
                for output_mode in range(4):
                    coeff = unitary[input_mode, output_mode]
                    if abs(coeff) <= 0.0:
                        continue
                    new_occ = list(occ)
                    new_occ[output_mode] += 1
                    new_tuple = tuple(new_occ)
                    if new_tuple not in basis_index:
                        continue
                    creation_factor = math.sqrt(new_occ[output_mode])
                    updated[new_tuple] = updated.get(new_tuple, 0.0j) + amplitude * coeff * creation_factor
            state = updated
    if input_factor > 0.0:
        state = {occ: amp / input_factor for occ, amp in state.items()}
    return state


def fock_mode_transform_matrix(
    basis_states: tuple[tuple[int, ...], ...],
    mode_unitary: np.ndarray,
) -> np.ndarray:
    index = {occ: idx for idx, occ in enumerate(basis_states)}
    transform = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
    for input_occ in basis_states:
        for output_occ, amplitude in transform_fock_occupation(input_occ, basis_states, mode_unitary).items():
            transform[index[output_occ], index[input_occ]] += amplitude
    return transform


def beam_splitter_transform_matrix(basis_states: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return fock_mode_transform_matrix(basis_states, beam_splitter_unitary_matrix())


_FLYING_POLARIZATION_OPERATOR_CACHE: dict[str, np.ndarray] = {}


def flying_polarization_mode_operator(
    basis_states: tuple[tuple[int, ...], ...],
    side: str,
    matrix2: np.ndarray,
) -> np.ndarray:
    """Fock-space operator for a 2x2 polarization-qubit map on one side's modes.

    The single-photon polarization map V acts on creation operators as
    a_in -> sum_out V[out, in] a_out, so the mode matrix passed to
    transform_fock_occupation (rows=input, columns=output) is V.T on the
    side's (H, V) block and identity on the other side.
    """
    if side not in {"left", "right"}:
        raise ValueError("flying polarization side must be left or right")
    matrix2 = np.asarray(matrix2, dtype=complex)
    key = stable_hash({
        "basis_size": len(basis_states),
        "max_total": max(sum(occ) for occ in basis_states),
        "side": side,
        "matrix": _json_ready(matrix2),
    })
    cached = _FLYING_POLARIZATION_OPERATOR_CACHE.get(key)
    if cached is not None:
        return cached
    mode_matrix = np.eye(4, dtype=complex)
    block = matrix2.T
    if side == "left":
        mode_matrix[0:2, 0:2] = block
    else:
        mode_matrix[2:4, 2:4] = block
    operator = fock_mode_transform_matrix(basis_states, mode_matrix)
    _FLYING_POLARIZATION_OPERATOR_CACHE[key] = operator
    return operator


def polarization_noise_mixture_terms(model: dict[str, Any]) -> list[tuple[float, np.ndarray]]:
    """Random-unitary mixture terms (probability, 2x2 unitary) for the model."""
    kind = model.get("kind", "none")
    if kind == "none" or not model.get("enters_probability", False):
        return [(1.0, I2)]
    if kind == "depolarizing_on_flying_polarization":
        p = clip_probability(float(model["depolarizing_probability"]))
        return [(1.0 - p, I2), (p / 3.0, X), (p / 3.0, Y), (p / 3.0, Z)]
    if kind == "pauli_channel_on_flying_polarization":
        probs = model["pauli_probabilities"]
        return [
            (float(probs[label]), PAULI_BY_LABEL[label])
            for label in ("I", "X", "Y", "Z")
            if float(probs[label]) > 0.0
        ]
    if kind == "unitary_rotation_on_flying_polarization":
        axis = PAULI_BY_LABEL[model["rotation_axis"]]
        theta = float(model["rotation_angle_rad"])
        unitary = math.cos(theta / 2.0) * I2 - 1.0j * math.sin(theta / 2.0) * axis
        return [(1.0, unitary)]
    raise ValueError(f"unsupported polarization noise model kind {kind!r}")


def apply_flying_polarization_noise_hybrid(
    rho: np.ndarray,
    basis_states: tuple[tuple[int, ...], ...],
    model: dict[str, Any],
) -> np.ndarray:
    """Apply the typed polarization noise channel to both flying polarization qubits.

    Each side's channel is a mixture of polarization-mode unitaries lifted
    exactly to the truncated Fock space (photon-number conserving), applied
    independently per elementary half-link traversal.
    """
    if model.get("kind", "none") == "none" or not model.get("enters_probability", False):
        return rho
    terms = polarization_noise_mixture_terms(model)
    out = np.asarray(rho, dtype=complex)
    for side in ("left", "right"):
        accumulated = np.zeros_like(out)
        for probability, matrix2 in terms:
            if probability <= 0.0:
                continue
            operator = flying_polarization_mode_operator(basis_states, side, matrix2)
            accumulated += probability * apply_hybrid_photon_operator(out, operator)
        out = np.asarray(np.real_if_close(accumulated), dtype=complex)
    return out


def apply_beam_splitter_hybrid(
    rho: np.ndarray,
    basis_states: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, float]:
    transform = beam_splitter_transform_matrix(basis_states)
    photon_dim = len(basis_states)
    lifted = np.kron(np.eye(4, dtype=complex), transform)
    out = lifted @ rho @ lifted.conj().T
    retained_trace = max(0.0, float(np.real_if_close(np.trace(out))))
    truncation_mass = max(0.0, 1.0 - retained_trace)
    return np.asarray(np.real_if_close(out), dtype=complex), truncation_mass


def detector_count_ranges(detector_kind: str, max_count: int) -> tuple[range, range, range, range]:
    if detector_kind == "threshold_snspd":
        return (range(2), range(2), range(2), range(2))
    return tuple(range(max_count + 1) for _ in range(4))  # type: ignore[return-value]


def raw_detector_record_label(counts: tuple[int, int, int, int]) -> str:
    return "counts:" + ",".join(f"{label}={count}" for label, count in zip(FOCK_MODE_LABELS, counts))


def detector_record_probability_for_occupation(
    occupation: tuple[int, ...],
    record: tuple[int, int, int, int],
    *,
    eta_det: float,
    dark_mean: float,
    detector_kind: str,
    max_count: int,
    pnr_confusion_matrix: dict[str, Any] | None = None,
) -> float:
    probability = 1.0
    for n_incident, observed in zip(occupation, record):
        dist = snspd_detector_distribution(
            n_incident,
            eta_det=eta_det,
            dark_mean=dark_mean,
            detector_kind=detector_kind,
            max_count=max_count,
            pnr_confusion_matrix=pnr_confusion_matrix,
        )
        probability *= dist.get(observed, 0.0)
    return probability


def condition_memory_on_detector_record(
    rho: np.ndarray,
    basis_states: tuple[tuple[int, ...], ...],
    record: tuple[int, int, int, int],
    *,
    eta_det: float,
    dark_mean: float,
    detector_kind: str,
    max_count: int,
    pnr_confusion_matrix: dict[str, Any] | None = None,
    incident_counts_by_occupation: dict[tuple[int, ...], tuple[int, int, int, int]] | None = None,
) -> tuple[float, np.ndarray]:
    photon_dim = len(basis_states)
    memory = np.zeros((4, 4), dtype=complex)
    for occ_index, occupation in enumerate(basis_states):
        incident = (
            incident_counts_by_occupation[occupation]
            if incident_counts_by_occupation is not None
            else occupation
        )
        effect = detector_record_probability_for_occupation(
            incident,
            record,
            eta_det=eta_det,
            dark_mean=dark_mean,
            detector_kind=detector_kind,
            max_count=max_count,
            pnr_confusion_matrix=pnr_confusion_matrix,
        )
        if effect <= 0.0:
            continue
        for mem_row in range(4):
            for mem_col in range(4):
                memory[mem_row, mem_col] += effect * rho[
                    mem_row * photon_dim + occ_index,
                    mem_col * photon_dim + occ_index,
                ]
    probability = max(0.0, float(np.real_if_close(np.trace(memory))))
    if probability <= 0.0:
        return 0.0, np.eye(4, dtype=complex) / 4.0
    return probability, clean_density(memory / probability)


def distinguishable_incident_counts_map(
    basis_states: tuple[tuple[int, ...], ...],
    route_a: int,
    route_b: int,
) -> dict[tuple[int, ...], tuple[int, int, int, int]]:
    """Detector-mode incident counts for distinguishable photons with fixed ports.

    In the distinguishable-photon HOM reference branch the two photons occupy
    orthogonal temporal modes, so there is no two-photon interference: each
    side's photon routes classically to output port route_a/route_b (weight
    1/4 per assignment) while keeping its polarization amplitudes. Detector
    modes are ordered (d0H, d0V, d1H, d1V); side modes are (aH, aV, bH, bV).
    """
    out: dict[tuple[int, ...], tuple[int, int, int, int]] = {}
    for occ in basis_states:
        counts = [0, 0, 0, 0]
        counts[2 * route_a + 0] += occ[0]
        counts[2 * route_a + 1] += occ[1]
        counts[2 * route_b + 0] += occ[2]
        counts[2 * route_b + 1] += occ[3]
        out[occ] = (counts[0], counts[1], counts[2], counts[3])
    return out


def classify_bsm_detector_counts(counts: tuple[int, int, int, int], detector_kind: str) -> str:
    occupied = tuple(idx for idx, count in enumerate(counts) if count > 0)
    if detector_kind in {"pnr_snspd", "ideal_pnr"} and any(count > 1 for count in counts):
        return "inconclusive"
    if occupied in {(0, 1), (2, 3)}:
        return "psi_plus"
    if occupied in {(0, 3), (1, 2)}:
        return "psi_minus"
    return "inconclusive"


def fock_edge_cache_key(
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> str:
    edge_move = (
        attack.to_dict()
        if attack.target_active and attack.target_kind == "edge" and attack.target_id == edge_id
        else {"move_kind": "none", "target_active": False}
    )
    payload = {
        "physical_path": context.physical_path,
        "source_model": context.source_model,
        "detector_kind": context.detector_kind,
        "detector_profile_hash": stable_hash(context.detector_profile),
        "detector_response_hash": stable_hash(context.detector_response),
        "fiber_response_hash": stable_hash(context.fiber_response),
        "conversion_response_hash": stable_hash(context.conversion_response),
        "repeater_service_profile_hash": stable_hash(context.repeater_service_profile),
        "allow_generic_memory_fidelity_depolarizing": context.allow_generic_memory_fidelity_depolarizing,
        "fock_cutoff": context.fock_cutoff,
        "edge_length_m": route["edge_lengths_m"][edge_id],
        "edge_move": edge_move,
        "device_constants_hash": stable_hash(front["device_constants"]),
    }
    return stable_hash(payload)


def fock_instrument_cache_info() -> dict[str, int]:
    return {
        "size": len(FOCK_EDGE_INSTRUMENT_CACHE),
        "hits": FOCK_EDGE_CACHE_HITS,
        "misses": FOCK_EDGE_CACHE_MISSES,
    }


def clear_fock_instrument_cache() -> None:
    global FOCK_EDGE_CACHE_HITS, FOCK_EDGE_CACHE_MISSES
    FOCK_EDGE_INSTRUMENT_CACHE.clear()
    FOCK_EDGE_CACHE_HITS = 0
    FOCK_EDGE_CACHE_MISSES = 0


def fock_edge_data_with_edge_labels(data: dict[str, Any], edge_id: str) -> dict[str, Any]:
    """Return cached Fock edge data with transcript labels rebound to edge_id."""
    labelled_branches = []
    for branch in data["conditional_success_branches"]:
        public_suffix = (
            branch.public_bsm_record.split(":", 1)[1]
            if ":" in branch.public_bsm_record else branch.public_bsm_record
        )
        detector_suffix = (
            branch.detector_click_record.split(":", 1)[1]
            if ":" in branch.detector_click_record else branch.detector_click_record
        )
        labelled_branches.append(Branch(
            probability=branch.probability,
            rho=branch.rho,
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=f"{edge_id}:{public_suffix}",
            detector_click_record=f"{edge_id}:{detector_suffix}",
        ))
    labelled = dict(data)
    labelled["conditional_success_branches"] = labelled_branches
    return labelled


def _fock_edge_unnormalized_instrument(
    *,
    prepared: list[tuple[float, np.ndarray, str]],
    basis_states: tuple[tuple[int, ...], ...],
    eta_path: float,
    eta_det: float,
    dark_mean: float,
    detector_kind: str,
    max_count: int,
    pnr_confusion: dict[str, Any] | None,
    polarization_model: dict[str, Any],
    mode_overlap: float,
    distinguishable_maps: list[dict[tuple[int, ...], tuple[int, int, int, int]]],
) -> tuple[dict[tuple[str, str, str], tuple[float, np.ndarray]], dict[str, float], dict[str, float], float]:
    """Length-parameterized core of the elementary edge instrument.

    All fiber-length dependence of the elementary link enters through the
    single pre-detector transmittance eta_path; every other input is frozen
    per theta. Exposed separately so the compiled length-parametric kernel can
    probe the exact same instrument at chosen eta nodes.
    """
    grouped: dict[tuple[str, str, str], tuple[float, np.ndarray]] = {}
    raw_record_probabilities: dict[str, float] = {}
    public_probabilities = {"psi_plus": 0.0, "psi_minus": 0.0, "inconclusive": 0.0}
    truncation_mass = 0.0
    photon_dim = len(basis_states)
    # The distinguishable-photon HOM branch routes each side's photons
    # classically to one fixed output port, which is exact only when each side
    # carries at most one photon. Fail closed on multi-photon side support
    # instead of silently mis-routing (polarization noise and loss conserve or
    # reduce per-side photon number, so checking the prepared state suffices).
    multi_photon_indices = (
        tuple(
            idx for idx, occ in enumerate(basis_states)
            if occ[0] + occ[1] > 1 or occ[2] + occ[3] > 1
        )
        if mode_overlap < 1.0 else ()
    )
    for prep_probability, prep_state, eve_record in prepared:
        if multi_photon_indices:
            multi_photon_mass = sum(
                float(np.real(prep_state[mem * photon_dim + idx, mem * photon_dim + idx]))
                for mem in range(4)
                for idx in multi_photon_indices
            )
            if multi_photon_mass > 1e-9:
                raise NotImplementedError(
                    "distinguishable-photon classical routing (mode_overlap < 1) is exact only for at "
                    "most one photon per elementary-link side; the prepared memory-photon state has "
                    f"multi-photon side support with mass {multi_photon_mass:.3e}. TODO(scientific): a "
                    "distinguishable multi-photon routing model is required before enabling such a source."
                )
        noisy_state = apply_flying_polarization_noise_hybrid(prep_state, basis_states, polarization_model)
        loss_state = apply_loss_channel_hybrid(noisy_state, basis_states, (eta_path, eta_path, eta_path, eta_path))
        # HOM/mode-overlap mixture: with weight s the photons interfere on the
        # analyzer (ideal indistinguishable branch); with weight (1-s)/4 per
        # port assignment they are distinguishable and route classically.
        contributions: list[tuple[float, np.ndarray, dict | None]] = []
        if mode_overlap > 0.0:
            analyzed_state, branch_truncation = apply_beam_splitter_hybrid(loss_state, basis_states)
            truncation_mass += prep_probability * mode_overlap * branch_truncation
            contributions.append((mode_overlap, analyzed_state, None))
        if mode_overlap < 1.0:
            distinguishable_weight = (1.0 - mode_overlap) / 4.0
            contributions.extend(
                (distinguishable_weight, loss_state, incident_map)
                for incident_map in distinguishable_maps
            )
        for weight, conditioned_state, incident_map in contributions:
            for c0 in detector_count_ranges(detector_kind, max_count)[0]:
                for c1 in detector_count_ranges(detector_kind, max_count)[1]:
                    for c2 in detector_count_ranges(detector_kind, max_count)[2]:
                        for c3 in detector_count_ranges(detector_kind, max_count)[3]:
                            record = (c0, c1, c2, c3)
                            record_probability, memory_state = condition_memory_on_detector_record(
                                conditioned_state,
                                basis_states,
                                record,
                                eta_det=eta_det,
                                dark_mean=dark_mean,
                                detector_kind=detector_kind,
                                max_count=max_count,
                                pnr_confusion_matrix=pnr_confusion,
                                incident_counts_by_occupation=incident_map,
                            )
                            probability = prep_probability * weight * record_probability
                            if probability <= 0.0:
                                continue
                            public_label = classify_bsm_detector_counts(record, detector_kind)
                            raw_label = raw_detector_record_label(record)
                            public_probabilities[public_label] = public_probabilities.get(public_label, 0.0) + probability
                            raw_record_probabilities[raw_label] = raw_record_probabilities.get(raw_label, 0.0) + probability
                            if public_label == "inconclusive":
                                continue
                            key_tuple = (public_label, raw_label, eve_record)
                            old_probability, old_rho = grouped.get(key_tuple, (0.0, np.zeros((4, 4), dtype=complex)))
                            grouped[key_tuple] = (
                                old_probability + probability,
                                old_rho + probability * memory_state,
                            )
    return grouped, public_probabilities, raw_record_probabilities, truncation_mass


def fock_memory_emission_elementary_link_data(
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> dict[str, Any]:
    global FOCK_EDGE_CACHE_HITS, FOCK_EDGE_CACHE_MISSES
    if context.source_model != "memory_emission":
        raise NotImplementedError("SPDC multipair memory-state conditioning is not implemented in this pass.")
    key = fock_edge_cache_key(edge_id, route, front, attack, context)
    if key in FOCK_EDGE_INSTRUMENT_CACHE:
        FOCK_EDGE_CACHE_HITS += 1
        return fock_edge_data_with_edge_labels(FOCK_EDGE_INSTRUMENT_CACHE[key], edge_id)
    FOCK_EDGE_CACHE_MISSES += 1

    empirical_pair = repeater_elementary_pair_instrument(context)
    if empirical_pair is not None:
        if attack.target_active:
            raise NotImplementedError(
                "the empirical remote-memory-pair instrument is Alice-only; optical or "
                "memory Eve moves require a source-specific instrument and fail closed"
            )
        service_profile = active_repeater_service_profile(context)
        if service_profile is None:
            raise ValueError("empirical pair instrument requires an active service profile")
        service_profile.validate_edge_length(
            float(route["edge_lengths_m"][edge_id]) / 1000.0
        )
        distribution = empirical_pair["pauli_error_distribution"]
        probabilities = tuple(float(distribution[label]) for label in ("I", "X", "Z", "Y"))
        rho = bell_diagonal_density(
            probabilities,
            target_label=str(empirical_pair["target_bell_state"]),
        )
        result = {
            "conditional_success_branches": [Branch(
                probability=1.0,
                rho=rho,
                frame="I",
                eve_observation="none",
                public_bsm_record=f"{edge_id}:empirical_remote_pair_accepted",
                detector_click_record=DETECTOR_CLICK_RECORD_NOT_EMITTED,
            )],
            "success_probability": 1.0,
            "bsm_outcome_probabilities": {
                "psi_plus": 0.0,
                "psi_minus": 1.0,
                "inconclusive": 0.0,
            },
            "raw_detector_record_probabilities": {},
            "metadata": {
                "physical_path": "empirical_remote_memory_pair",
                "source_model": "source-reported accepted memory-memory pair",
                "profile_id": service_profile.profile_id,
                "claim_tier": service_profile.claim_tier,
                "elementary_pair_instrument": empirical_pair,
                "conditional_pair_semantics": (
                    "unit trace is conditional on a maintained accepted pair; measured renewal "
                    "rate and rolling availability, not this branch trace, determine delivery mass"
                ),
                "optical_stack_bypassed": (
                    "source, conversion, phase stabilization, fiber, detector, and heralding "
                    "errors are already aggregated into the demonstrated rate and state"
                ),
                "sequence_bell_state_with_fidelity_used": False,
                "repeater_service_elementary_state_channel": None,
                "repeater_service_quality_restoration": None,
                "instrument_channel_order": (
                    "source-reported remote memory pair -> public herald-frame correction -> "
                    "declared fidelity-preserving isotropic twirl -> rolling storage"
                ),
            },
        }
        FOCK_EDGE_INSTRUMENT_CACHE[key] = result
        return fock_edge_data_with_edge_labels(result, edge_id)

    detector_kind = resolved_detector_kind(front, context)
    profile = detector_profile_metadata(front, context)
    max_count = detector_max_count(context, profile)
    pnr_confusion = (
        validate_pnr_confusion_matrix(profile, max_count)
        if detector_kind == "pnr_snspd"
        else None
    )
    pnr_meta = pnr_response_model_metadata(profile, detector_kind, max_count)
    basis_states = fock_basis(4, context.fock_cutoff)
    length = route["edge_lengths_m"][edge_id]
    eta_path, link_meta = fock_link_transmittance(front, context, length)
    wavelength_stages_meta = link_meta["wavelength_stages"]
    source_wavelength = wavelength_stages_meta["source_wavelength_nm"]
    detector_wavelength = wavelength_stages_meta["detector_wavelength_nm"]
    if attack.target_active and attack.move_kind == "edge_added_loss" and attack.target_id == edge_id:
        eta_path = clip_probability(
            eta_path * clip_probability(attack.loss_multiplier) * (1.0 - clip_probability(attack.drop_probability))
        )
        link_meta["edge_added_loss"] = {
            "loss_multiplier": attack.loss_multiplier,
            "drop_probability": attack.drop_probability,
            "mode": "additional pre-analyzer loss channel on both elementary-link optical modes",
        }
    eta_det, detector_efficiency_meta = detector_efficiency_at_wavelength(front, context, detector_wavelength)
    dark_mean, dark_count_meta = detector_dark_count_model(front, context, detector_wavelength)
    active_memory_process = memory_process_model(front, context)
    rho_initial = memory_photon_state_density(front, context, basis_states)
    prepared = apply_edge_measure_resend_to_memory_photon_state(
        rho_initial,
        basis_states,
        edge_id=edge_id,
        attack=attack,
    )
    polarization_model = fiber_polarization_noise_model(front)
    indistinguishability_model = photon_indistinguishability_model(front)
    mode_overlap = photon_mode_overlap_value(indistinguishability_model)
    distinguishable_maps = (
        [
            distinguishable_incident_counts_map(basis_states, route_a, route_b)
            for route_a in (0, 1)
            for route_b in (0, 1)
        ]
        if mode_overlap < 1.0 else []
    )
    grouped, public_probabilities, raw_record_probabilities, truncation_mass = (
        _fock_edge_unnormalized_instrument(
            prepared=prepared,
            basis_states=basis_states,
            eta_path=eta_path,
            eta_det=eta_det,
            dark_mean=dark_mean,
            detector_kind=detector_kind,
            max_count=max_count,
            pnr_confusion=pnr_confusion,
            polarization_model=polarization_model,
            mode_overlap=mode_overlap,
            distinguishable_maps=distinguishable_maps,
        )
    )
    success_probability = public_probabilities.get("psi_plus", 0.0) + public_probabilities.get("psi_minus", 0.0)
    conditional_success_branches: list[Branch] = []
    service_elementary_channel = repeater_elementary_state_channel(context)
    if success_probability > 0.0:
        for (public_label, raw_label, eve_record), (probability, rho_sum) in grouped.items():
            correction = CORRECTION_BY_BSM[public_label]
            corrected_rho = apply_frame_correction(clean_density(rho_sum / probability), correction)
            if service_elementary_channel is not None:
                corrected_rho = apply_one_qubit_channel(
                    corrected_rho,
                    int(service_elementary_channel["qubit"]),
                    dephasing_kraus(
                        float(service_elementary_channel["phase_flip_probability"]),
                        "Z",
                    ),
                )
            conditional_success_branches.append(Branch(
                probability=probability / success_probability,
                rho=corrected_rho,
                frame=correction,
                eve_observation=eve_record,
                public_bsm_record=f"{edge_id}:{public_label}",
                detector_click_record=f"{edge_id}:{raw_label}",
            ))
    conditional_success_branches, quality_restoration_meta = (
        apply_elementary_quality_restoration_to_branches(
            normalize_branches(conditional_success_branches),
            edge_id=edge_id,
            edge_length_m=float(length),
            context=context,
        )
        if conditional_success_branches else (conditional_success_branches, None)
    )
    scalar_assumptions = []
    for resp in (
        detector_efficiency_meta,
        dark_count_meta.get("timing_window_response", {}),
        link_meta.get("attenuation_response", {}),
        link_meta.get("conversion_response", {}),
    ):
        if resp.get("wavelength_dependency") == "scalar_flat_response_assumption":
            scalar_assumptions.append(resp)
    result = {
        "conditional_success_branches": normalize_branches(conditional_success_branches),
        "success_probability": success_probability,
        "bsm_outcome_probabilities": public_probabilities,
        "raw_detector_record_probabilities": dict(sorted(raw_record_probabilities.items())),
        "metadata": {
            "physical_path": "fock_pnr_snspd",
            "source_model": "memory_emission",
            "detector_kind": detector_kind,
            "detector_profile": profile,
            "detector_profile_role": profile.get("profile_role", "repository_default"),
            "detector_is_demonstrated_or_idealized": (
                "idealized" if detector_kind == "ideal_pnr" else "demonstrated_profile_or_repository_scalar"
            ),
            "fock_cutoff": context.fock_cutoff,
            "fock_truncation_mass": truncation_mass,
            "active_wavelength_nm": source_wavelength,
            "wavelength_stages": wavelength_stages_meta,
            "source_wavelength_nm": source_wavelength,
            "conversion_input_wavelength_nm": wavelength_stages_meta["conversion_input_wavelength_nm"],
            "conversion_output_wavelength_nm": wavelength_stages_meta["conversion_output_wavelength_nm"],
            "propagation_wavelength_nm": wavelength_stages_meta["propagation_wavelength_nm"],
            "detector_wavelength_nm": detector_wavelength,
            "conversion_model_status": wavelength_stages_meta["conversion_model_status"],
            "conversion_model_limitations": wavelength_stages_meta["conversion_model_limitations"],
            "link_response": link_meta,
            "detector_efficiency_response": detector_efficiency_meta,
            "detector_dark_count_model": dark_count_meta,
            "detector_timing_window_ps": dark_count_meta.get("timing_window_ps"),
            "detector_timing_window_response": dark_count_meta.get("timing_window_response"),
            "dark_count_mean_per_window": dark_mean,
            "pnr_capability": pnr_meta["pnr_capability"],
            "max_resolvable_photon_number": max_count,
            "pnr_response_model": pnr_meta["pnr_response_model"],
            "pnr_response_model_is_calibrated_to_profile": pnr_meta[
                "pnr_response_model_is_calibrated_to_profile"
            ],
            "pnr_response_model_metadata": pnr_meta,
            "detector_profile_completeness": pnr_meta["detector_profile_completeness"],
            "threshold_coarsening_applied": pnr_meta["threshold_coarsening_applied"],
            "calibrated_pnr_confusion_matrix_available": pnr_meta[
                "calibrated_pnr_confusion_matrix_available"
            ],
            "input_modes": list(INPUT_MODE_LABELS),
            "output_detector_modes": list(FOCK_MODE_LABELS),
            "memory_photon_state_convention": "(|0>_M|H>_P + |1>_M|V>_P)/sqrt(2)",
            "memory_process_model": active_memory_process,
            "repeater_service_elementary_state_channel": service_elementary_channel,
            "repeater_service_quality_restoration": quality_restoration_meta,
            "fiber_polarization_noise_model": polarization_model,
            "photon_indistinguishability_model": indistinguishability_model,
            "hom_mode_overlap": mode_overlap,
            "hom_model_semantics": (
                "phenomenological mode-overlap mixture: click-pattern statistics and conditioned memory "
                "states are s * indistinguishable-interference instrument + (1 - s) * distinguishable "
                "classical-routing reference; not a calibrated wave-packet model"
            ),
            "instrument_channel_order": (
                "memory emission -> edge measure-resend move -> flying polarization noise -> "
                "polarization-independent loss -> HOM mixture (analyzer interference vs classical routing) -> "
                "detector POVMs -> source-specific raw Bell channel -> optional noisy two-copy "
                "elementary-link purification"
            ),
            "memory_fidelity_scalar_status": (
                "metadata/provenance only; not consumed as constructor/interface process noise"
                if not active_memory_process.get("scalar_fidelity_active") else
                "diagnostic override consumed scalar memory.fidelity as p_depol=1-fidelity"
            ),
            "optical_analyzer_unitary_convention": {
                "input_creation_order": list(INPUT_MODE_LABELS),
                "output_creation_order": list(FOCK_MODE_LABELS),
                "matrix_rows_input_columns_output": _json_ready(beam_splitter_unitary_matrix()),
                "mode_transform": "a_pol -> (d0_pol + d1_pol)/sqrt(2); b_pol -> (d0_pol - d1_pol)/sqrt(2)",
            },
            "raw_detector_record_schema": "counts:d0H=<int>,d0V=<int>,d1H=<int>,d1V=<int>",
            "public_bsm_mapping": {
                "psi_plus": ["counts with occupied detector pairs d0H+d0V or d1H+d1V"],
                "psi_minus": ["counts with occupied detector pairs d0H+d1V or d0V+d1H"],
                "inconclusive": "all other count/click patterns",
            },
            "effective_pauli_frame_mapping": {
                "psi_plus": CORRECTION_BY_BSM["psi_plus"],
                "psi_minus": CORRECTION_BY_BSM["psi_minus"],
                "convention": "public elementary BSM labels are mapped to the Pauli frame that converts the conditioned memory pair to the E91 psi_minus convention",
            },
            "pauli_feed_forward_convention": (
                "public-label Pauli corrections are applied to the memory pair physically at herald "
                "time; this is exactly equivalent to software Pauli-frame tracking under the "
                "pure-dephasing storage families (dephasing channels are invariant under Pauli "
                "conjugation), but the equivalence would not extend to amplitude-damping (t1_t2) "
                "storage combined with X/Y corrections"
            ),
            "distinguishable_bell_outcomes": ["psi_plus", "psi_minus"],
            "inconclusive_records": "raw detector records not mapped to psi_plus or psi_minus",
            "memory_state_construction": "conditioned_from_memory_photon_state_and_detector_povm_then_public_pauli_frame_correction",
            "sequence_bell_state_with_fidelity_used": False,
            "wavelength_dependency_assumptions": scalar_assumptions,
            "raw_detector_record_probabilities": dict(sorted(raw_record_probabilities.items())),
        },
    }
    FOCK_EDGE_INSTRUMENT_CACHE[key] = result
    return fock_edge_data_with_edge_labels(result, edge_id)


def fock_memory_emission_elementary_link_instrument(
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> list[Branch]:
    data = fock_memory_emission_elementary_link_data(edge_id, route, front, attack, context)
    return [branch.copy() for branch in data["conditional_success_branches"]]


DIAGNOSTIC_PATTERN_CLASSIFIER_INCIDENT_PATTERNS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "psi_plus": ((1, 1, 0, 0), (0, 0, 1, 1)),
    "psi_minus": ((1, 0, 0, 1), (0, 1, 1, 0)),
    "phi_plus": ((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)),
    "phi_minus": ((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)),
}


def diagnostic_pattern_classifier_counts(counts: tuple[int, int, int, int], detector_kind: str) -> str:
    occupied = tuple(idx for idx, count in enumerate(counts) if count > 0)
    if detector_kind in {"pnr_snspd", "ideal_pnr"} and any(count > 1 for count in counts):
        return "inconclusive"
    if occupied in {(0, 1), (2, 3)}:
        return "psi_plus"
    if occupied in {(0, 3), (1, 2)}:
        return "psi_minus"
    return "inconclusive"


def diagnostic_pattern_classifier_probabilities_for_input_bell(
    bell_label: str,
    front: dict[str, Any],
    context: ModelContext,
    length_m: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    detector_kind = resolved_detector_kind(front, context)
    profile = detector_profile_metadata(front, context)
    pre_eta, link_meta = fock_link_transmittance(front, context, length_m)
    detector_wavelength = link_meta["wavelength_stages"]["detector_wavelength_nm"]
    eta_det, det_meta = detector_efficiency_at_wavelength(front, context, detector_wavelength)
    dark_mean, dark_count_meta = detector_dark_count_model(front, context, detector_wavelength)
    max_count = detector_max_count(context, profile)
    pnr_confusion = (
        validate_pnr_confusion_matrix(profile, max_count)
        if detector_kind == "pnr_snspd"
        else None
    )
    pnr_meta = pnr_response_model_metadata(profile, detector_kind, max_count)
    cutoff = context.fock_cutoff
    result = {"psi_plus": 0.0, "psi_minus": 0.0, "inconclusive": 0.0}
    patterns = DIAGNOSTIC_PATTERN_CLASSIFIER_INCIDENT_PATTERNS[bell_label]
    for pattern in patterns:
        weight = 1.0 / len(patterns)
        per_mode = [
            fock_detector_distribution_for_emitted_mode(
                n,
                pre_detector_eta=pre_eta,
                eta_det=eta_det,
                dark_mean=dark_mean,
                detector_kind=detector_kind,
                max_count=max_count,
                cutoff=cutoff,
                pnr_confusion_matrix=pnr_confusion,
            )
            for n in pattern
        ]
        for c0, p0 in per_mode[0].items():
            for c1, p1 in per_mode[1].items():
                for c2, p2 in per_mode[2].items():
                    for c3, p3 in per_mode[3].items():
                        counts = (c0, c1, c2, c3)
                        label = diagnostic_pattern_classifier_counts(counts, detector_kind)
                        result[label] = result.get(label, 0.0) + weight * p0 * p1 * p2 * p3
    total = sum(result.values())
    if total > 0.0:
        result = {k: v / total for k, v in result.items()}
    metadata = {
        "detector_kind": detector_kind,
        "detector_profile": profile,
        "detector_efficiency_response": det_meta,
        "link_response": link_meta,
        "detector_dark_count_model": dark_count_meta,
        "dark_count_mean_per_window": dark_mean,
        "pnr_response_model_metadata": pnr_meta,
        "pnr_response_model": pnr_meta["pnr_response_model"],
        "pnr_response_model_is_calibrated_to_profile": pnr_meta[
            "pnr_response_model_is_calibrated_to_profile"
        ],
        "max_resolvable_photon_number": max_count,
        "fock_cutoff": cutoff,
        "input_bell_label": bell_label,
        "incident_pattern_basis": DIAGNOSTIC_PATTERN_CLASSIFIER_INCIDENT_PATTERNS[bell_label],
    }
    return result, metadata


def diagnostic_pattern_classifier_unconditional_probabilities(
    front: dict[str, Any],
    context: ModelContext,
    length_m: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    prior = {label: 0.25 for label in ("phi_plus", "phi_minus", "psi_plus", "psi_minus")}
    out = {"psi_plus": 0.0, "psi_minus": 0.0, "inconclusive": 0.0}
    by_input: dict[str, Any] = {}
    for label, weight in prior.items():
        probs, meta = diagnostic_pattern_classifier_probabilities_for_input_bell(label, front, context, length_m)
        by_input[label] = {"probabilities": probs, "metadata": meta}
        for out_label, prob in probs.items():
            out[out_label] = out.get(out_label, 0.0) + weight * prob
    return out, {
        "photonic_bell_prior": prior,
        "by_input_bell_state": by_input,
        "distinguishable_bell_outcomes": ["psi_plus", "psi_minus"],
        "inconclusive_outcome": "inconclusive",
        "optical_network_definition": {
            "analyzer": "partial linear-optics polarization Bell analyzer",
            "detector_modes": ["out0_H", "out0_V", "out1_H", "out1_V"],
            "click_pattern_map": {
                "psi_plus": [[0, 1], [2, 3]],
                "psi_minus": [[0, 3], [1, 2]],
                "inconclusive": "all other threshold/count patterns",
            },
        },
    }


def threshold_detector_click_probability(front: dict[str, Any], photon_number: int) -> float:
    detector = front["device_constants"]["detector"]
    eta = clip_probability(detector["efficiency"])
    p_dark = detector_dark_window_probability(front)
    return clip_probability(1.0 - (1.0 - p_dark) * ((1.0 - eta) ** photon_number))


def barrett_kok_round_valid_trigger_probability(front: dict[str, Any], length_m: float) -> float:
    memory_eff = clip_probability(front["device_constants"]["memory"]["efficiency"])
    keep = memory_eff * half_edge_transmittance(front, length_m)
    detector_eff = clip_probability(front["device_constants"]["detector"]["efficiency"])
    real_signal = clip_probability(keep * detector_eff)
    dark = detector_dark_window_probability(front)
    signal_detector_records = 1.0 - (1.0 - real_signal) * (1.0 - dark)
    other_detector_records = dark
    return clip_probability(
        signal_detector_records * (1.0 - other_detector_records)
        + (1.0 - signal_detector_records) * other_detector_records
    )


def barrett_kok_memory_survives(front: dict[str, Any], length_m: float) -> bool:
    coherence_value = front["device_constants"]["memory"].get("coherence_time_s")
    if coherence_value is None:
        return True
    coherence_s = float(coherence_value)
    if coherence_s <= 0.0:
        return True
    qc_delay = half_edge_delay_ps(front, length_m)
    cc_delay = qc_delay
    two_round_duration_ps = 2 * (qc_delay + cc_delay + 10)
    return two_round_duration_ps < coherence_s * 1e12


def barrett_kok_round_separation_ps(front: dict[str, Any], length_m: float) -> int:
    qc_delay = half_edge_delay_ps(front, length_m)
    cc_delay = qc_delay
    return qc_delay + cc_delay + 10


def barrett_kok_detector_cooldown_factor(front: dict[str, Any], length_m: float) -> float:
    separation = barrett_kok_round_separation_ps(front, length_m)
    cooldown = detector_cooldown_period_ps(front)
    if separation > cooldown:
        return 1.0
    # SingleAtomBSM chooses one of two detectors for each herald. If the same
    # detector is selected before cooldown expires, Detector.record_detection
    # suppresses the second record; the opposite detector can still fire.
    return 0.5


def elementary_success_probability(
    front: dict[str, Any],
    length_m: float,
    edge_id: str,
    attack: HistogramAttack,
) -> float:
    trans_multiplier = 1.0
    if attack.target_active and attack.move_kind == "edge_added_loss" and attack.target_id == edge_id:
        trans_multiplier = clip_probability(attack.loss_multiplier) * (1.0 - clip_probability(attack.drop_probability))
    if not barrett_kok_memory_survives(front, length_m):
        return 0.0
    adjusted_front = front
    if trans_multiplier != 1.0:
        adjusted_front = json.loads(json.dumps(front))
        attenuation = adjusted_front["device_constants"]["fiber_channel"]["attenuation_db_per_m"]
        if trans_multiplier <= 0.0:
            return 0.0
        # Represent the added-loss station as an extra transmission factor on the
        # targeted elementary link, equivalent to the repository drop/forward rule.
        extra_db = -10.0 * math.log10(trans_multiplier) / max(float(length_m), 1e-12)
        adjusted_front["device_constants"]["fiber_channel"]["attenuation_db_per_m"] = attenuation + extra_db
    p_trigger = barrett_kok_round_valid_trigger_probability(adjusted_front, length_m)
    cooldown_factor = barrett_kok_detector_cooldown_factor(adjusted_front, length_m)
    # SeQUeNCe Barrett-Kok has a 1/2 first-round odd-parity herald condition;
    # after the first valid herald, the second round succeeds when its detector
    # trigger is valid. Unit hardware therefore gives edge success 1/2.
    return clip_probability(0.5 * p_trigger * p_trigger * cooldown_factor)


def delivery_stage_probabilities(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext | None = None,
) -> dict[str, Any]:
    from .routing import fock_elementary_edge_outcome_probabilities
    from .timing import (
        apply_memory_loss_policies_to_histories,
        memory_survival_model,
        multiplexed_edge_round,
        normalized_waiting_protocol,
        rolling_live_pair_policy,
        stochastic_waiting_time_policy,
    )

    context = context or ModelContext()
    edge_success = {}
    single_mode_edge_success: dict[str, float] = {}
    controller_rounds: dict[str, Any] = {}
    for edge_id in route["edge_ids"]:
        if context.physical_path == "fock_pnr_snspd":
            probs, _meta = fock_elementary_edge_outcome_probabilities(edge_id, route, front, attack, context)
            p_single = probs.get("psi_plus", 0.0) + probs.get("psi_minus", 0.0)
            single_mode_edge_success[edge_id] = p_single
            p, round_meta = multiplexed_edge_round(
                single_mode_success_probability=p_single,
                edge_id=edge_id,
                route=route,
                front=front,
                context=context,
            )
            if round_meta is not None:
                controller_rounds[edge_id] = round_meta
        else:
            p = elementary_success_probability(
                front,
                route["edge_lengths_m"][edge_id],
                edge_id,
                attack,
            )
        edge_success[edge_id] = p
    swap_success = front["device_constants"]["swap_bsm"]["success_probability"]
    p_swap_all = 1.0
    for repeater in route["nodes"][1:-1]:
        p = 0.0 if (
            attack.target_active
            and attack.move_kind == "swap_block"
            and attack.target_id == repeater
        ) else swap_success
        p_swap_all *= p
    p_elementary = 1.0
    for p in edge_success.values():
        p_elementary *= p
    single_attempt_elementary_product = p_elementary
    waiting_model: dict[str, Any] | None = None
    rolling_model: dict[str, Any] | None = None
    waiting_protocol = normalized_waiting_protocol(context) if context.physical_path == "fock_pnr_snspd" else "single_buffer_swap_asap"
    if context.physical_path == "fock_pnr_snspd" and waiting_protocol == "rolling_alive_pair":
        rolling_model = rolling_live_pair_policy(route, front, context)
        p_elementary = clip_probability(rolling_model["route_availability"])
    elif context.physical_path == "fock_pnr_snspd" and route["hops"] >= 2:
        waiting_model = stochastic_waiting_time_policy(route, front, attack, context)
        p_elementary = clip_probability(
            waiting_model["waiting_distribution"].get("route_ready_probability_within_budget", 0.0)
        )
    memory_loss_stats: dict[str, Any] | None = None
    if context.physical_path == "fock_pnr_snspd":
        if rolling_model is not None:
            memory_loss_stats = {
                "survival_enabled": memory_survival_model(front)["enabled"],
                "detected_decay_lifetime_s": memory_survival_model(front)["lifetime_s"],
                "memory_cutoff_ps": float(context.memory_cutoff_ps) if context.memory_cutoff_ps else None,
                "cutoff_fraction_of_ready_mass": 0.0,
                "survival_expectation_given_cutoff_pass": 1.0,
                "usable_fraction_of_ready_mass": 1.0,
                "rolling_alive_pair_semantics": (
                    "pre-route stale cutoff and detected-decay survival are treated as baked into "
                    "configured edge availability A_i and selected-age distribution f_i(t); they are "
                    "not applied again to route-ready live pairs"
                ),
                "endpoint_confirmation_wait_storage": rolling_model["endpoint_classical_confirmation"],
            }
        elif waiting_model is not None:
            memory_loss_stats = waiting_model["waiting_distribution"].get("memory_loss_policies")
        else:
            herald = float(edge_delay_ps(front, route["edge_lengths_m"][route["edge_ids"][0]]))
            _usable, memory_loss_stats = apply_memory_loss_policies_to_histories(
                [(1.0, [herald, herald])], front, context
            )
    cutoff_fraction = (
        memory_loss_stats["cutoff_fraction_of_ready_mass"] if memory_loss_stats else 0.0
    )
    survival_expectation = (
        memory_loss_stats["survival_expectation_given_cutoff_pass"] if memory_loss_stats else 1.0
    )
    p_stale = p_elementary * cutoff_fraction
    p_after_cutoff = p_elementary * (1.0 - cutoff_fraction)
    p_decay = p_after_cutoff * (1.0 - survival_expectation)
    p_usable = p_after_cutoff * survival_expectation
    p_endpoint = 1.0
    p_pre_endpoint = p_usable * p_swap_all
    delivered = p_pre_endpoint * p_endpoint
    payload = {
        "elementary_success_product": clip_probability(p_elementary),
        "elementary_success_product_semantics": (
            "configured route live-pair availability product for rolling_alive_pair; not a single-attempt success product"
            if rolling_model is not None else
            "finite-budget route-ready probability for h>=2 stochastic waiting "
            "(analytic for h=2; grouped swap-asap enumeration within attempt cutoffs for h>=3)"
            if waiting_model is not None else
            "single synchronized controller-round elementary success product"
            if controller_rounds else
            "single synchronized-attempt elementary success product"
        ),
        "elementary_single_attempt_success_product": clip_probability(
            single_attempt_elementary_product
        ),
        "route_ready_probability_within_waiting_budget": (
            clip_probability(p_elementary) if waiting_model is not None else None
        ),
        "route_live_pair_availability": (
            clip_probability(p_elementary) if rolling_model is not None else None
        ),
        "no_live_pair_failure_mass": (
            clip_probability(1.0 - p_elementary) if rolling_model is not None else None
        ),
        "memory_loss_policies": memory_loss_stats,
        "stale_pair_cutoff_fraction": clip_probability(cutoff_fraction),
        "memory_survival_expectation": clip_probability(survival_expectation),
        "memory_usable_fraction_given_ready": clip_probability(
            (1.0 - cutoff_fraction) * survival_expectation
        ),
        "stage_ordering": (
            "route_ready -> stale_pair_cutoff -> memory_decay_detected -> swap -> endpoint; "
            "declared sequential decomposition convention"
        ),
        "swap_success_product": clip_probability(p_swap_all),
        "endpoint_detection_product": clip_probability(p_endpoint),
        "delivered": clip_probability(delivered),
        "elementary_generation_or_channel_failure": clip_probability(1.0 - p_elementary),
        "stale_pair_cutoff": clip_probability(p_stale),
        "memory_decay_detected": clip_probability(p_decay),
        "swap_failure": clip_probability(p_usable * (1.0 - p_swap_all)),
        "endpoint_detection_failure": clip_probability(p_pre_endpoint * (1.0 - p_endpoint)),
        "edge_success_probabilities": edge_success,
        "waiting_model": waiting_model,
        "rolling_live_pair_model": rolling_model,
        "waiting_protocol": waiting_protocol,
        "elementary_generation_failure_interpretation": (
            "at_least_one_elementary_edge_lacks_an_accepted_live_pair_at_route_request"
            if rolling_model is not None else
            "timeout_or_at_least_one_elementary_link_not_ready_within_finite_attempt_budget"
            if waiting_model is not None else
            "single-attempt elementary generation/channel failure"
        ),
        "endpoint_measurement_model": (
            "E91EndpointApp memory measurement; no endpoint detector loss/dark-count stage in selected repeater path"
            if context.physical_path == "sequence_bk_reference"
            else "E91 endpoint memory measurement after Fock/PNR optical BSM heralding"
        ),
    }
    if controller_rounds:
        payload["single_mode_edge_success_probabilities"] = single_mode_edge_success
        payload["multiplexing_controller_rounds"] = controller_rounds
        payload["elementary_single_attempt_probability_unit"] = "controller_round"
        payload["conditional_optical_instrument_semantics"] = (
            "mode-invariant single-mode instrument retained; controller changes only round success, "
            "cadence, and collision accounting"
        )
    return payload


def elementary_link_model_metadata(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext | None = None,
) -> dict[str, Any]:
    context = context or ModelContext()
    if context.physical_path == "fock_pnr_snspd":
        return fock_elementary_link_model_metadata(route, front, attack, context)
    edges: dict[str, Any] = {}
    for edge_id in route["edge_ids"]:
        length = route["edge_lengths_m"][edge_id]
        p_trigger = barrett_kok_round_valid_trigger_probability(front, length)
        edges[edge_id] = {
            "endpoints": route["edge_endpoints"][edge_id],
            "length_m": length,
            "half_channel_transmittance": half_edge_transmittance(front, length),
            "full_edge_transmittance": edge_transmittance(front, length),
            "half_edge_delay_ps": half_edge_delay_ps(front, length),
            "barrett_kok_round_separation_ps": barrett_kok_round_separation_ps(front, length),
            "detector_cooldown_period_ps": detector_cooldown_period_ps(front, context),
            "detector_cooldown_factor": barrett_kok_detector_cooldown_factor(front, length),
            "barrett_kok_first_round_odd_parity_probability": 0.5,
            "round_valid_trigger_probability": p_trigger,
            "memory_survives_two_round_generation": barrett_kok_memory_survives(front, length),
            "success_probability": elementary_success_probability(front, length, edge_id, attack),
            "conditional_state_model": {
                "target_state_after_feed_forward": "phi_plus",
                "density_assignment": "sequence.components.bsm._set_state_with_fidelity",
                "fidelity_parameter": "memory.raw_fidelity from sequence.components.memory.Memory.encoding['raw_fidelity']",
                "repository_test_calibration": "sequence-toolbox-SeQUeNCe/tests/entanglement_management/test_generation.py::test_generation_fidelity_ket",
                "e91_endpoint_convention": "psi_minus",
                "endpoint_convention_frame": BK_REFERENCE_ENDPOINT_CONVENTION_FRAME,
                "endpoint_convention_note": (
                    "the Barrett-Kok phi_plus target is expressed in the generator's E91 psi_minus "
                    "endpoint convention through the recorded public Pauli frame; this is deterministic "
                    "frame bookkeeping consistent with the repeater-swap corrections, not extra physics"
                ),
            },
            "detector_model": {
                "threshold_click_formula": "P(click|n)=1-(1-p_dark)*(1-eta)^n",
                "p_dark_window": detector_dark_window_probability(front),
                "signal_click_probability_for_single_atom_photon": (
                    clip_probability(front["device_constants"]["memory"]["efficiency"])
                    * half_edge_transmittance(front, length)
                    * clip_probability(front["device_constants"]["detector"]["efficiency"])
                ),
                "exact_one_detector_record_required": True,
            },
        }
    return {
        "route_edges": edges,
        "spdc_fock_pnr_selected_route_status": "not_used_by_selected_RouterNetTopo_Barrett_Kok_memory_emission",
        "endpoint_measurement_status": "memory measurement via E91EndpointApp, not endpoint detector hardware",
        "edge_move_basis_choice": attack.basis_choice if attack.move_kind == "edge_measure_resend" else "not_applicable",
    }


def fock_elementary_link_model_metadata(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> dict[str, Any]:
    from .routing import fock_elementary_edge_outcome_probabilities
    from .timing import multiplexed_edge_round

    edges: dict[str, Any] = {}
    stages = wavelength_stages(front, context)
    profile = detector_profile_metadata(front, context)
    detector_kind = resolved_detector_kind(front, context)
    max_count = detector_max_count(context, profile)
    pnr_meta = pnr_response_model_metadata(profile, detector_kind, max_count)
    active_memory_process = memory_process_model(front, context)
    for edge_id in route["edge_ids"]:
        probs, meta = fock_elementary_edge_outcome_probabilities(edge_id, route, front, attack, context)
        single_mode_success = probs.get("psi_plus", 0.0) + probs.get("psi_minus", 0.0)
        round_success, round_meta = multiplexed_edge_round(
            single_mode_success_probability=single_mode_success,
            edge_id=edge_id,
            route=route,
            front=front,
            context=context,
        )
        edge_payload = {
            "endpoints": route["edge_endpoints"][edge_id],
            "length_m": route["edge_lengths_m"][edge_id],
            "wavelength_nm": meta.get("propagation_wavelength_nm", stages["propagation_wavelength_nm"]),
            "wavelength_stages": meta.get("wavelength_stages", stages),
            "bsm_outcome_probabilities": probs,
            "success_probability": round_success,
            "optical_bsm_metadata": meta,
            "conditional_memory_update": "remote memories are conditioned by tracing photons after applying the raw detector-record POVM",
            "raw_detector_record_probabilities": meta.get("raw_detector_record_probabilities", {}),
        }
        if round_meta is not None:
            edge_payload["single_mode_success_probability"] = single_mode_success
            edge_payload["multiplexing_controller_round"] = round_meta
        edges[edge_id] = edge_payload
    scalar_assumptions = []
    sample_edge = next(iter(edges.values())) if edges else {}
    sample_meta = sample_edge.get("optical_bsm_metadata", {})
    scalar_assumptions.extend(sample_meta.get("wavelength_dependency_assumptions", []))
    return {
        "physical_path": "fock_pnr_snspd",
        "source_model": context.source_model,
        "detector_kind": detector_kind,
        "resolved_detector_kind": detector_kind,
        "detector_profile": profile,
        "detector_profile_role": profile.get("profile_role", "repository_default"),
        "detector_is_demonstrated_or_idealized": (
            "idealized" if detector_kind == "ideal_pnr" else "demonstrated_profile_or_repository_scalar"
        ),
        "pnr_capability": pnr_meta["pnr_capability"],
        "max_resolvable_photon_number": max_count,
        "pnr_response_model": pnr_meta["pnr_response_model"],
        "pnr_response_model_is_calibrated_to_profile": pnr_meta[
            "pnr_response_model_is_calibrated_to_profile"
        ],
        "pnr_response_model_metadata": pnr_meta,
        "detector_profile_completeness": pnr_meta["detector_profile_completeness"],
        "threshold_coarsening_applied": pnr_meta["threshold_coarsening_applied"],
        "fock_cutoff": context.fock_cutoff,
        "fock_truncation_mass": max(
            (edge["optical_bsm_metadata"].get("fock_truncation_mass", 0.0) for edge in edges.values()),
            default=0.0,
        ),
        "active_wavelength_nm": stages["source_wavelength_nm"],
        "wavelength_stages": stages,
        "source_wavelength_nm": stages["source_wavelength_nm"],
        "propagation_wavelength_nm": stages["propagation_wavelength_nm"],
        "detector_wavelength_nm": stages["detector_wavelength_nm"],
        "route_edges": edges,
        "wavelength_dependency_assumptions": scalar_assumptions,
        "detector_povm_formulas": {
            "threshold_snspd": "P(no click | n) = (1 - p_dark) * (1 - eta_det)^n; P(click | n)=1-P(no click|n)",
            "pnr_snspd": "P(k detected | n incident) is Binomial(n, eta_det) convolved with Poisson dark counts and saturated at max_count",
            "profile_calibrated_pnr_confusion_matrix": "If pnr_confusion_matrix is supplied by the detector profile, P(reported count | incident photon number) is read from that row-stochastic matrix instead of the generic saturated model",
            "ideal_pnr": "P(k detected | n)=delta_{k,n} truncated at max_count",
        },
        "optical_bell_analyzer": {
            "implemented_network": "50:50 beamsplitter with polarization-resolved detector modes",
            "unitary_convention": sample_meta.get("optical_analyzer_unitary_convention", {}),
            "distinguishable_bell_outcomes": ["psi_plus", "psi_minus"],
            "inconclusive_bell_outcomes": ["phi_plus", "phi_minus", "loss/dark/ambiguous count patterns"],
            "raw_detector_record_schema": sample_meta.get("raw_detector_record_schema"),
            "public_bsm_mapping": sample_meta.get("public_bsm_mapping", {}),
            "effective_pauli_frame_mapping": sample_meta.get("effective_pauli_frame_mapping", {}),
        },
        "memory_state_construction": "conditioned_from_memory_photon_state_and_detector_povm_then_public_pauli_frame_correction",
        "memory_process_model": active_memory_process,
        "memory_fidelity_scalar_status": (
            "metadata/provenance only; not consumed as constructor/interface process noise"
            if not active_memory_process.get("scalar_fidelity_active") else
            "diagnostic override consumed scalar memory.fidelity as p_depol=1-fidelity"
        ),
        "sequence_bell_state_with_fidelity_used": False,
        "approximations": [
            "finite Fock cutoff; memory_emission requires cutoff >= 2 for the two emitted photons",
            "scalar-flat wavelength response when no response curve file is supplied",
            "memory/interface constructor noise is applied only through explicit memory_process_model or diagnostic override",
            (
                "distinguishable-photon HOM branch routes each side's photons classically to a fixed "
                "output port; exact for at most one photon per side (memory_emission), enforced by a "
                "fail-closed multi-photon support guard"
            ),
        ],
    }
