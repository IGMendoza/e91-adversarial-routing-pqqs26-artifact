"""Length-parametric compiled histogram kernels and exactness checks."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sequence_game.physical.entanglement_purification import bell_diagonal_density

from .devices import *
from .fock import *
from .fock import _fock_edge_unnormalized_instrument
from .generation import *
from .models import *
from .models import _json_ready
from .output import *
from .quantum import *
from .routing import *
from .timing import *
from .topology import *
COMPILED_KERNEL_VERSION = "e91_compiled_route_kernel.v3"
_COMPILED_EDGE_POLY_CACHE: dict[str, dict[str, Any]] = {}
_COMPILED_SUPPORTED_STORAGE_KINDS = {
    "none",
    "exponential_dephasing",
    "gaussian_dephasing",
    "liu_2026_anisotropic_pair_fit",
}


def _edge_instrument_probe_at_eta(
    front: dict[str, Any],
    context: ModelContext,
    attack: HistogramAttack,
    edge_id: str,
    eta_path: float,
) -> tuple[dict[tuple[str, str, str], tuple[float, np.ndarray]], dict[str, float]]:
    """Probe the exact elementary instrument at an explicit pre-detector eta."""
    empirical_pair = repeater_elementary_pair_instrument(context)
    if empirical_pair is not None:
        if attack.target_active:
            raise NotImplementedError(
                "compiled empirical remote-memory pairs support only Alice's no-move baseline"
            )
        distribution = empirical_pair["pauli_error_distribution"]
        probabilities = tuple(float(distribution[label]) for label in ("I", "X", "Z", "Y"))
        rho = bell_diagonal_density(
            probabilities,
            target_label=str(empirical_pair["target_bell_state"]),
        )
        return (
            {("psi_minus", "empirical_remote_pair_accepted", "none"): (1.0, rho)},
            {"psi_plus": 0.0, "psi_minus": 1.0, "inconclusive": 0.0},
        )
    detector_kind = resolved_detector_kind(front, context)
    profile = detector_profile_metadata(front, context)
    max_count = detector_max_count(context, profile)
    pnr_confusion = (
        validate_pnr_confusion_matrix(profile, max_count)
        if detector_kind == "pnr_snspd"
        else None
    )
    basis_states = fock_basis(4, context.fock_cutoff)
    stages = wavelength_stages(front, context)
    detector_wavelength = stages["detector_wavelength_nm"]
    eta_det, _det_meta = detector_efficiency_at_wavelength(front, context, detector_wavelength)
    dark_mean, _dark_meta = detector_dark_count_model(front, context, detector_wavelength)
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
    grouped, public_probabilities, _raw, _trunc = _fock_edge_unnormalized_instrument(
        prepared=prepared,
        basis_states=basis_states,
        eta_path=clip_probability(eta_path),
        eta_det=eta_det,
        dark_mean=dark_mean,
        detector_kind=detector_kind,
        max_count=max_count,
        pnr_confusion=pnr_confusion,
        polarization_model=polarization_model,
        mode_overlap=mode_overlap,
        distinguishable_maps=distinguishable_maps,
    )
    return grouped, public_probabilities


def compile_edge_branch_polynomials(
    front: dict[str, Any],
    context: ModelContext,
    attack: HistogramAttack,
    edge_id: str,
    *,
    max_degree: int = 12,
    fit_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Exact polynomial family for the edge instrument in s = sqrt(eta_path).

    Probes the trusted instrument at s nodes and fits, per branch label, the
    unnormalized frame-corrected operator (probability folded into the trace)
    and, per public Bell outcome, the outcome probability. The fit degree is
    increased until held-out nodes reproduce the instrument to fit_tolerance;
    with two emitted photons the entries are degree <= 4 in s, so this
    terminates immediately in practice. Exactness is by polynomial identity,
    not approximation: a failed fit raises instead of degrading silently.
    """
    if context.source_model != "memory_emission":
        raise NotImplementedError("compiled kernel supports memory_emission only")
    holdout_s = (0.083, 0.471, 0.917)
    degree = 4
    while True:
        nodes = np.linspace(0.0, 0.95, degree + 1)
        node_grouped = [
            _edge_instrument_probe_at_eta(front, context, attack, edge_id, float(s) ** 2)
            for s in nodes
        ]
        labels = sorted({key for grouped, _pub in node_grouped for key in grouped})
        vander = np.vander(nodes, degree + 1, increasing=True)
        inv = np.linalg.inv(vander)
        operator_coeffs = np.zeros((len(labels), degree + 1, 4, 4), dtype=complex)
        for label_index, label in enumerate(labels):
            public_label = label[0]
            correction = CORRECTION_BY_BSM[public_label]
            unitary = np.kron(I2, PAULI_BY_LABEL[correction])
            samples = np.zeros((degree + 1, 4, 4), dtype=complex)
            for node_index, (grouped, _pub) in enumerate(node_grouped):
                _prob, rho_sum = grouped.get(label, (0.0, np.zeros((4, 4), dtype=complex)))
                samples[node_index] = unitary @ rho_sum @ unitary.conj().T
            operator_coeffs[label_index] = np.einsum("dn,nij->dij", inv, samples)
        outcome_labels = ("psi_plus", "psi_minus", "inconclusive")
        outcome_samples = np.array([
            [pub.get(outcome, 0.0) for outcome in outcome_labels]
            for _grouped, pub in node_grouped
        ])
        outcome_coeffs = inv @ outcome_samples
        max_residual = 0.0
        for s_hold in holdout_s:
            grouped_h, pub_h = _edge_instrument_probe_at_eta(front, context, attack, edge_id, s_hold ** 2)
            powers = np.array([s_hold ** d for d in range(degree + 1)])
            for label_index, label in enumerate(labels):
                public_label = label[0]
                correction = CORRECTION_BY_BSM[public_label]
                unitary = np.kron(I2, PAULI_BY_LABEL[correction])
                _prob, rho_sum = grouped_h.get(label, (0.0, np.zeros((4, 4), dtype=complex)))
                expected = unitary @ rho_sum @ unitary.conj().T
                fitted = np.einsum("d,dij->ij", powers, operator_coeffs[label_index])
                max_residual = max(max_residual, float(np.abs(fitted - expected).max()))
            fitted_outcomes = powers @ outcome_coeffs
            for outcome_index, outcome in enumerate(outcome_labels):
                max_residual = max(
                    max_residual,
                    abs(float(fitted_outcomes[outcome_index]) - pub_h.get(outcome, 0.0)),
                )
        if max_residual <= fit_tolerance:
            break
        degree += 2
        if degree > max_degree:
            raise ValueError(
                f"edge instrument polynomial fit did not converge (residual {max_residual:.3e}); "
                "the instrument is not polynomial in sqrt(eta) under the current configuration"
            )
    return {
        "labels": [list(label) for label in labels],
        "operator_coeffs": operator_coeffs,
        "outcome_labels": list(outcome_labels),
        "outcome_coeffs": outcome_coeffs,
        "degree": degree,
        "fit_max_residual": max_residual,
        "fit_variable": "s = sqrt(eta_path); eta_path includes memory/conversion prefactor, fiber loss, and any targeted edge_added_loss multiplier",
    }


def _compiled_edge_poly_cache_key(
    front: dict[str, Any],
    context: ModelContext,
    attack: HistogramAttack,
    edge_id: str,
    targeted: bool,
) -> str:
    edge_move = attack.to_dict() if targeted else {"move_kind": "none"}
    return stable_hash({
        "device_constants": front["device_constants"],
        "context": _compiled_edge_poly_context_key(context),
        "edge_move": edge_move,
        "edge_id": edge_id if targeted else "untargeted",
        "compiled_version": COMPILED_KERNEL_VERSION,
    })


def _compiled_edge_poly_context_key(context: ModelContext) -> dict[str, Any]:
    """Context fields that can affect the fitted optical edge instrument.

    Waiting-budget controls and the waiting protocol change route-level
    storage mixtures and delivery mass, not the detector-conditioned
    elementary-link polynomial in eta_path. Keeping them out of this cache key
    lets kernels with different finite-budget or live-pair policies share the
    same fitted edge instrument without changing physics.
    """
    return {
        "physical_path": context.physical_path,
        "source_model": context.source_model,
        "detector_kind": context.detector_kind,
        "fock_cutoff": context.fock_cutoff,
        "detector_profile_path": context.detector_profile_path,
        "detector_profile": context.detector_profile,
        "detector_response_loaded": context.detector_response is not None,
        "detector_response": context.detector_response,
        "detector_response_hash": stable_hash(context.detector_response),
        "conversion_response_loaded": context.conversion_response is not None,
        "conversion_response": context.conversion_response,
        "conversion_response_hash": stable_hash(context.conversion_response),
        "allow_wavelength_mismatch": context.allow_wavelength_mismatch,
        "allow_generic_memory_fidelity_depolarizing": context.allow_generic_memory_fidelity_depolarizing,
    }


def _storage_coherence_factors(front: dict[str, Any], ages_ps: np.ndarray) -> np.ndarray:
    """C(total age) per memory site; exact for the pure-dephasing storage family."""
    lifetimes = memory_lifetime_parameters(front)
    kind = lifetimes["storage_model_kind"]
    if kind == "none":
        return np.ones_like(ages_ps, dtype=float)
    coherence_time = lifetimes["coherence_time_s"]
    if coherence_time is None or not math.isfinite(coherence_time):
        return np.ones_like(ages_ps, dtype=float)
    ages_s = np.maximum(ages_ps, 0.0) * 1e-12
    if kind == "gaussian_dephasing":
        return np.exp(-((ages_s / coherence_time) ** 2))
    if kind == "exponential_dephasing":
        return np.exp(-ages_s / coherence_time)
    raise NotImplementedError(
        f"compiled kernel supports storage kinds {_COMPILED_SUPPORTED_STORAGE_KINDS}, got {kind!r}"
    )


_Q0_COHERENCE_MASK = np.array([[i // 2 != j // 2 for j in range(4)] for i in range(4)])
_Q1_COHERENCE_MASK = np.array([[i % 2 != j % 2 for j in range(4)] for i in range(4)])


def _apply_coherence_mask(
    ops: np.ndarray,
    qubit: int,
    factors: np.ndarray,
    same_factors: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-dephasing channel on one qubit of batched 4x4 operators.

    ops: [..., B, 4, 4]; factors: [B]. Coherences between computational
    branches of the target qubit are scaled by the factor (equal to 1 - 2p for
    dephasing probability p), which is the exact channel action. When
    same_factors is supplied, population (same-bit) entries are scaled by it
    instead of 1 -- this exposes the affine basis (population part, coherence
    part) used by the moment-expansion evaluation mode.
    """
    mask = _Q0_COHERENCE_MASK if qubit == 0 else _Q1_COHERENCE_MASK
    same = 1.0 if same_factors is None else same_factors[..., None, None]
    scale = np.where(mask, factors[..., None, None], same)
    return ops * scale


_PAULI_LIFTED_4 = {
    (qubit, label): np.kron(PAULI_BY_LABEL[label], I2) if qubit == 0 else np.kron(I2, PAULI_BY_LABEL[label])
    for qubit in (0, 1)
    for label in ("X", "Y", "Z")
}


def _apply_depolarizing_on_qubit(ops: np.ndarray, qubit: int, probability: float) -> np.ndarray:
    if probability <= 0.0:
        return ops
    out = (1.0 - probability) * ops
    for label in ("X", "Y", "Z"):
        pauli = _PAULI_LIFTED_4[(qubit, label)]
        out = out + (probability / 3.0) * np.einsum("ij,...jk,kl->...il", pauli, ops, pauli.conj().T)
    return out


def _apply_profile_storage_on_qubit(
    ops: np.ndarray,
    front: dict[str, Any],
    qubit: int,
    age_ps: float,
) -> np.ndarray:
    """Apply a trace-preserving profile storage channel to unnormalized ops."""

    params = memory_storage_channel_probabilities(front, float(age_ps))
    probabilities = params.get("pauli_channel_probabilities")
    if probabilities is None:
        factor = np.asarray([float(params["coherence_factor"])], dtype=float)
        return _apply_coherence_mask(ops, qubit, factor)
    out = np.zeros_like(ops, dtype=complex)
    identity = np.eye(4, dtype=complex)
    for label in ("I", "X", "Y", "Z"):
        probability = float(probabilities[label])
        if probability <= 0.0:
            continue
        lifted = identity if label == "I" else _PAULI_LIFTED_4[(qubit, label)]
        out += probability * np.einsum(
            "ij,...jk,kl->...il", lifted, ops, lifted.conj().T
        )
    return out


def _swap_contract(
    left_ops: np.ndarray,
    right_ops: np.ndarray,
    bell_vector: np.ndarray,
) -> np.ndarray:
    """Post-swap unnormalized operators for one Bell outcome.

    left_ops [nL, B, 4, 4] over qubits (alice_side, mid_left); right_ops
    [nR, B, 4, 4] over (mid_right, far_side); bell_vector [2, 2] with
    |bell> = sum_{mn} bell[m, n] |m n>. Returns [nL, nR, B, 4, 4] over
    (alice_side, far_side), equal to Tr_mid[(I x |bell><bell| x I) rhoL x rhoR]
    as produced by bsm_project: the conjugated Bell vector contracts the
    row-side mid indices (m, n) and the unconjugated vector the column-side
    (M, N). (Numerically identical either way for the four standard Bell kets
    because they are real; the ordering matters if a complex Bell vector is
    ever introduced.)
    """
    n_left, batch = left_ops.shape[0], left_ops.shape[1]
    n_right = right_ops.shape[0]
    left = left_ops.reshape(n_left, batch, 2, 2, 2, 2)
    right = right_ops.reshape(n_right, batch, 2, 2, 2, 2)
    out = np.einsum(
        "lbamcM,mn,rbndNe,MN->lrbadce",
        left,
        bell_vector.conj(),
        right,
        bell_vector,
        optimize=True,
    )
    return out.reshape(n_left, n_right, batch, 4, 4)


_BELL_VECTOR_2x2 = {
    label: bell_ket(label).reshape(2, 2)
    for label in BELL_LABELS
}


def _apply_swap_output_channels(
    ops: np.ndarray,
    reported: str,
    front: dict[str, Any],
    context: ModelContext | None = None,
) -> np.ndarray:
    """Frame correction by the reported label, then degradation and phase error."""
    correction = CORRECTION_BY_BSM[reported]
    unitary = np.kron(I2, PAULI_BY_LABEL[correction])
    out = np.einsum("ij,...jk,kl->...il", unitary, ops, unitary.conj().T)
    service_channel = (
        repeater_swap_state_channel(context)
        if context is not None else None
    )
    if service_channel is not None:
        p_depol = clip_probability(float(service_channel["depolarizing_probability"]))
        out = _apply_depolarizing_on_qubit(
            out,
            int(service_channel["qubit"]),
            p_depol,
        )
    else:
        degradation = float(front["device_constants"]["swap_bsm"].get("swapping_degradation", 1.0))
        p_depol = clip_probability(1.0 - degradation)
    if service_channel is None and p_depol > 0.0:
        # Uniform 15-Pauli two-qubit depolarizing: (1 - 16p/15) rho + (4p/15) tr(rho) I.
        traces = np.trace(out, axis1=-2, axis2=-1)
        out = (1.0 - 16.0 * p_depol / 15.0) * out + (
            (4.0 * p_depol / 15.0) * traces[..., None, None] * np.eye(4, dtype=complex)
        )
    phase_error = clip_probability(float(front["device_constants"]["swap_bsm"].get("phase_error", 0.0)))
    if phase_error > 0.0:
        factor = 1.0 - 2.0 * phase_error
        mask = np.where(_Q1_COHERENCE_MASK, factor, 1.0)
        out = out * mask
    return out


def _expand_repeater_measure(
    labels: list[tuple[str, str, str, str]],
    ops: np.ndarray,
    *,
    qubit: int,
    basis: str,
    label_prefix: str,
    information_probability: float,
) -> tuple[list[tuple[str, str, str, str]], np.ndarray]:
    """Projective expansion of one side's table for a repeater memory measure."""
    projectors = projectors_for_axis(basis_axis(basis))
    info = clip_probability(information_probability)
    new_labels: list[tuple[str, str, str, str]] = []
    new_ops: list[np.ndarray] = []
    for outcome, projector in enumerate(projectors):
        lifted = np.kron(projector, I2) if qubit == 0 else np.kron(I2, projector)
        projected = np.einsum("ij,nbjk,kl->nbil", lifted, ops, lifted.conj().T)
        outcome_label = f"{label_prefix}:outcome={outcome}"
        for index, (frame, eve, public, click) in enumerate(labels):
            if info > 0.0:
                labelled = outcome_label if eve == "none" else eve + "|" + outcome_label
                new_labels.append((frame, labelled, public, click))
                new_ops.append(info * projected[index])
            if info < 1.0:
                new_labels.append((frame, eve, public, click))
                new_ops.append((1.0 - info) * projected[index])
    return new_labels, np.stack(new_ops, axis=0)


def _aggregate_label_table(
    labels: list[tuple[str, str, str, str]],
    ops: np.ndarray,
) -> tuple[list[tuple[str, str, str, str]], np.ndarray]:
    grouped: dict[tuple[str, str, str, str], np.ndarray] = {}
    for label, op in zip(labels, ops):
        if label in grouped:
            grouped[label] = grouped[label] + op
        else:
            grouped[label] = op.copy()
    out_labels = list(grouped.keys())
    return out_labels, np.stack([grouped[label] for label in out_labels], axis=0)


class _InjectedEdgeOutcomes:
    """Temporarily place polynomial-evaluated edge outcome data in the
    elementary-instrument cache so the classical delivery/waiting layer reads
    the compiled values without rebuilding Fock instruments. Entries are
    restored on exit so reference evaluations in the same process are
    untouched."""

    def __init__(
        self,
        front: dict[str, Any],
        context: ModelContext,
        attack: HistogramAttack,
        route: dict[str, Any],
        outcome_probabilities: dict[str, dict[str, float]],
    ) -> None:
        self.front = front
        self.context = context
        self.attack = attack
        self.route = route
        self.outcomes = outcome_probabilities
        self._saved: dict[str, Any] = {}
        self._keys: list[str] = []

    def __enter__(self) -> "_InjectedEdgeOutcomes":
        # Untargeted edges of equal length share one cache key, so save each
        # unique key exactly once before any overwrite; otherwise the second
        # edge would record our own injection as the "prior" entry and leak it
        # into the cache on exit, corrupting later reference evaluations.
        for edge_id in self.route["edge_ids"]:
            key = fock_edge_cache_key(edge_id, self.route, self.front, self.attack, self.context)
            if key in self._keys:
                continue
            if key in FOCK_EDGE_INSTRUMENT_CACHE:
                self._saved[key] = FOCK_EDGE_INSTRUMENT_CACHE[key]
            probs = self.outcomes[edge_id]
            FOCK_EDGE_INSTRUMENT_CACHE[key] = {
                "conditional_success_branches": [],
                "success_probability": probs.get("psi_plus", 0.0) + probs.get("psi_minus", 0.0),
                "bsm_outcome_probabilities": dict(probs),
                "raw_detector_record_probabilities": {},
                "metadata": {"compiled_kernel_injected_outcomes": True},
            }
            self._keys.append(key)
        return self

    def __exit__(self, *_exc: Any) -> None:
        for key in self._keys:
            if key in self._saved:
                FOCK_EDGE_INSTRUMENT_CACHE[key] = self._saved[key]
            else:
                FOCK_EDGE_INSTRUMENT_CACHE.pop(key, None)


class CompiledRouteKernel:
    """Compiled H_{theta,h,move}: evaluate the transcript histogram at any
    fiber-length vector without rebuilding Fock instruments or looping the
    fold in Python. Exact relative to FrozenHistogramKernel.evaluate for the
    supported scope (fock_pnr_snspd, memory_emission, pure-dephasing storage,
    default public-record compression, no audit flags)."""

    def __init__(
        self,
        *,
        front: dict[str, Any],
        context: ModelContext,
        hop_count: int,
        attack: EveMove,
        key_length: int,
        theta_hash: str,
        plain_edge_poly: dict[str, Any],
        targeted_edge_poly: dict[str, Any] | None,
        fiber_length_symbols: tuple[str, ...],
    ) -> None:
        self.front = front
        self.context = context
        self.hop_count = hop_count
        self.attack = attack
        self.key_length = key_length
        self.theta_hash = theta_hash
        self.plain_edge_poly = plain_edge_poly
        self.targeted_edge_poly = targeted_edge_poly
        self.fiber_length_symbols = fiber_length_symbols
        self._edge_table_cache: dict[
            tuple[str, float, float, bool],
            tuple[list[tuple[str, str, str, str]], np.ndarray],
        ] = {}

    # -- compilation -------------------------------------------------------

    @classmethod
    def compile(
        cls,
        *,
        front: dict[str, Any],
        context: ModelContext,
        hop_count: int,
        attack: EveMove,
        key_length: int = 256,
        fiber_length_symbols: tuple[str, ...] | None = None,
    ) -> "CompiledRouteKernel":
        if context.physical_path != "fock_pnr_snspd":
            raise NotImplementedError("compiled kernel supports the fock_pnr_snspd path only")
        lifetimes = memory_lifetime_parameters(front)
        if lifetimes["storage_model_kind"] not in _COMPILED_SUPPORTED_STORAGE_KINDS:
            raise NotImplementedError(
                "compiled kernel does not support storage kind; supported kinds are "
                f"{sorted(_COMPILED_SUPPORTED_STORAGE_KINDS)}, got {lifetimes['storage_model_kind']!r}"
            )
        if (
            lifetimes["storage_model_kind"] == "liu_2026_anisotropic_pair_fit"
            and normalized_rolling_age_evaluation(context) != "moment_reduced"
        ):
            raise NotImplementedError(
                "compiled Liu anisotropic storage currently requires rolling_age_evaluation='moment_reduced'"
            )
        reference = FrozenHistogramKernel(
            front=front,
            context=context,
            hop_count=hop_count,
            attack=attack,
            key_length=key_length,
            fiber_length_symbols=fiber_length_symbols,
        )
        plain_key = _compiled_edge_poly_cache_key(front, context, attack, "", False)
        if plain_key not in _COMPILED_EDGE_POLY_CACHE:
            _COMPILED_EDGE_POLY_CACHE[plain_key] = compile_edge_branch_polynomials(
                front, context, attack, "__untargeted__"
            )
        plain_poly = _COMPILED_EDGE_POLY_CACHE[plain_key]
        targeted_poly = None
        if attack.target_active and attack.target_kind == "edge" and attack.move_kind == "edge_measure_resend":
            targeted_key = _compiled_edge_poly_cache_key(front, context, attack, attack.target_id, True)
            if targeted_key not in _COMPILED_EDGE_POLY_CACHE:
                _COMPILED_EDGE_POLY_CACHE[targeted_key] = compile_edge_branch_polynomials(
                    front, context, attack, attack.target_id
                )
            targeted_poly = _COMPILED_EDGE_POLY_CACHE[targeted_key]
        return cls(
            front=copy.deepcopy(front),
            context=copy.deepcopy(context),
            hop_count=int(hop_count),
            attack=copy.deepcopy(attack),
            key_length=int(key_length),
            theta_hash=reference.theta_hash,
            plain_edge_poly=plain_poly,
            targeted_edge_poly=targeted_poly,
            fiber_length_symbols=tuple(reference.fiber_length_symbols or ()),
        )

    # -- per-length evaluation helpers --------------------------------------

    def _edge_eta(self, length_m: float, edge_id: str) -> float:
        if repeater_elementary_pair_instrument(self.context) is not None:
            return 1.0
        eta, _meta = fock_link_transmittance(self.front, self.context, length_m)
        attack = self.attack
        if attack.target_active and attack.move_kind == "edge_added_loss" and attack.target_id == edge_id:
            eta = clip_probability(
                eta * clip_probability(attack.loss_multiplier) * (1.0 - clip_probability(attack.drop_probability))
            )
        return eta

    def _edge_table(
        self,
        edge_id: str,
        eta: float,
        length_m: float,
    ) -> tuple[list[tuple[str, str, str, str]], np.ndarray]:
        targeted = bool(
            self.targeted_edge_poly is not None
            and self.attack.target_kind == "edge"
            and self.attack.target_id == edge_id
        )
        record_edge_id = edge_id if targeted else "__compiled_plain_edge__"
        cache_edge_id = edge_id if targeted else "__compiled_plain_edge__"
        cache_key = (cache_edge_id, float(eta), float(length_m), targeted)

        def materialize(
            source: list[tuple[str, str, str, str]],
        ) -> list[tuple[str, str, str, str]]:
            if targeted:
                return list(source)
            return [
                (
                    frame,
                    eve,
                    public.replace(record_edge_id, edge_id),
                    clicks.replace(record_edge_id, edge_id),
                )
                for frame, eve, public, clicks in source
            ]

        cached = self._edge_table_cache.get(cache_key)
        if cached is not None:
            return materialize(cached[0]), np.array(cached[1], dtype=complex, copy=True)
        poly = (
            self.targeted_edge_poly
            if self.targeted_edge_poly is not None
            and self.attack.target_kind == "edge"
            and self.attack.target_id == edge_id
            else self.plain_edge_poly
        )
        s = math.sqrt(clip_probability(eta))
        powers = np.array([s ** d for d in range(int(poly["degree"]) + 1)])
        raw_ops = np.einsum("d,ndij->nij", powers, poly["operator_coeffs"])
        grouped: dict[tuple[str, str, str, str], np.ndarray] = {}
        for index, (public_label, _raw_label, eve_record) in enumerate(tuple(row) for row in poly["labels"]):
            frame = CORRECTION_BY_BSM[public_label]
            label = (
                frame,
                eve_record,
                f"{record_edge_id}:{public_label}",
                DETECTOR_CLICK_RECORD_NOT_EMITTED,
            )
            if label not in grouped:
                grouped[label] = np.zeros((4, 4), dtype=complex)
            grouped[label] = grouped[label] + raw_ops[index]
        labels = list(grouped.keys())
        ops = np.stack([grouped[label] for label in labels], axis=0)
        batched = ops[:, None, :, :]
        service_channel = repeater_elementary_state_channel(self.context)
        if service_channel is not None:
            probability = float(service_channel["phase_flip_probability"])
            factor = np.asarray([1.0 - 2.0 * probability], dtype=float)
            batched = _apply_coherence_mask(
                batched,
                int(service_channel["qubit"]),
                factor,
            )
        if repeater_quality_restoration(
            self.context,
            float(length_m) / 1000.0,
        ) is not None:
            traces = np.real(np.trace(batched[:, 0], axis1=-2, axis2=-1))
            total = float(np.sum(traces))
            if total <= 0.0:
                raise ValueError("compiled raw edge table has zero successful mass")
            raw_branches = [
                Branch(
                    probability=float(weight) / total,
                    rho=clean_density(operator / float(weight)),
                    frame=label[0],
                    eve_observation=label[1],
                    public_bsm_record=label[2],
                    detector_click_record=label[3],
                )
                for label, operator, weight in zip(labels, batched[:, 0], traces)
                if float(weight) > 0.0
            ]
            restored, _restoration_meta = apply_elementary_quality_restoration_to_branches(
                raw_branches,
                edge_id=record_edge_id,
                edge_length_m=float(length_m),
                context=self.context,
            )
            labels = [
                (
                    branch.frame,
                    branch.eve_observation,
                    branch.public_bsm_record,
                    branch.detector_click_record,
                )
                for branch in restored
            ]
            batched = np.stack(
                [float(branch.probability) * branch.rho for branch in restored],
                axis=0,
            )[:, None, :, :]
        if len(self._edge_table_cache) >= 128:
            self._edge_table_cache.pop(next(iter(self._edge_table_cache)))
        self._edge_table_cache[cache_key] = (
            list(labels),
            np.array(batched, dtype=complex, copy=True),
        )
        return materialize(labels), batched

    def _rolling_moment_reduced_edge_table(
        self,
        *,
        route: dict[str, Any],
        edge_index: int,
        edge_id: str,
        eta: float,
    ) -> tuple[list[tuple[str, str, str, str]], np.ndarray]:
        labels, base_ops = self._edge_table(
            edge_id,
            eta,
            float(route["edge_lengths_m"][edge_id]),
        )
        age_distributions, _age_meta = rolling_age_distributions_for_route(route, self.context)
        alice_wait, bob_wait, _endpoint_profile = rolling_endpoint_confirmation_waits(route, self.front)
        averaged = np.zeros_like(base_ops, dtype=complex)
        for age_s, age_weight in age_distributions[edge_index]:
            times = rolling_edge_storage_times_ps(
                edge_index=edge_index,
                hop_count=len(route["edge_ids"]),
                age_s=age_s,
                endpoint_wait_alice_ps=alice_wait,
                endpoint_wait_bob_ps=bob_wait,
            )
            aged = _apply_profile_storage_on_qubit(
                base_ops, self.front, 0, float(times[0])
            )
            aged = _apply_profile_storage_on_qubit(
                aged, self.front, 1, float(times[1])
            )
            averaged = averaged + float(age_weight) * aged
        return labels, averaged

    def _edge_outcomes(self, eta: float, edge_id: str) -> dict[str, float]:
        poly = (
            self.targeted_edge_poly
            if self.targeted_edge_poly is not None
            and self.attack.target_kind == "edge"
            and self.attack.target_id == edge_id
            else self.plain_edge_poly
        )
        s = math.sqrt(clip_probability(eta))
        powers = np.array([s ** d for d in range(int(poly["degree"]) + 1)])
        values = powers @ poly["outcome_coeffs"]
        return {
            outcome: max(0.0, float(value))
            for outcome, value in zip(poly["outcome_labels"], values)
        }

    def _signature_batch(
        self,
        route: dict[str, Any],
        edge_success: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """(usable conditional weights [B], per-site ages [B, 2h]) or None when empty."""
        front, context = self.front, self.context
        edge_ids = list(route["edge_ids"])
        hops = len(edge_ids)
        if normalized_waiting_protocol(context) == "rolling_alive_pair":
            weights, ages, _meta = rolling_live_pair_site_age_distribution(route, front, context)
            return weights, ages
        herald_list = [float(edge_delay_ps(front, route["edge_lengths_m"][e])) for e in edge_ids]
        wait_profile = endpoint_classical_wait_profile(route, front)
        if hops == 1:
            ages = np.array([[herald_list[0], herald_list[0]]])
            weights, _stats = apply_memory_loss_policies_to_histories(
                [(1.0, list(ages[0]))], front, context
            )
            if weights[0] <= 0.0:
                return None
            return np.array([1.0]), ages
        attempt_policy = attempt_period_policy(route, front, context)
        edge_periods = {
            e: float(attempt_policy["edge_attempt_periods"][e]["attempt_period_ps"]) for e in edge_ids
        }
        budgets, _meta = waiting_attempt_budgets(context, edge_periods)
        if hops == 2:
            histories, _dist = two_link_waiting_histories(
                p0=edge_success[edge_ids[0]],
                p1=edge_success[edge_ids[1]],
                period0_ps=edge_periods[edge_ids[0]],
                period1_ps=edge_periods[edge_ids[1]],
                context=context,
                budget0=budgets[edge_ids[0]],
                budget1=budgets[edge_ids[1]],
            )
            if not histories:
                return None
            ecw_a, ecw_b = endpoint_extra_waits_for_swap_tails(wait_profile, [0.0])
            ages = np.array([
                two_link_history_memory_ages(
                    row,
                    herald_left_ps=herald_list[0],
                    herald_right_ps=herald_list[1],
                    endpoint_wait_alice_ps=ecw_a,
                    endpoint_wait_bob_ps=ecw_b,
                )
                for row in histories
            ])
            weighted = [(float(row["probability"]), list(age_row)) for row, age_row in zip(histories, ages)]
            usable, _stats = apply_memory_loss_policies_to_histories(weighted, front, context)
            usable = np.asarray(usable)
        else:
            signatures, _meta2 = multi_link_wait_signature_distribution(
                edge_success=[edge_success[e] for e in edge_ids],
                edge_periods_ps=[edge_periods[e] for e in edge_ids],
                attempt_budgets=[budgets[e] for e in edge_ids],
                context=context,
            )
            if not signatures:
                return None
            ages_list = []
            for signature in signatures:
                age_row, _wa, _wb = signature_memory_ages(
                    signature,
                    herald_ages_ps=herald_list,
                    wait_profile=wait_profile,
                )
                ages_list.append(age_row)
            ages = np.array(ages_list)
            weighted = [
                (weight, list(age_row))
                for weight, age_row in zip(signatures.values(), ages)
            ]
            usable, _stats = apply_memory_loss_policies_to_histories(weighted, front, context)
            usable = np.asarray(usable)
        mass = float(usable.sum())
        if mass <= 0.0:
            return None
        keep = usable > 0.0
        return usable[keep] / mass, ages[keep]

    def _apply_repeater_move(
        self,
        step_index: int,
        repeater: str,
        left_labels: list[tuple[str, str, str, str]],
        left_ops: np.ndarray,
        right_labels: list[tuple[str, str, str, str]],
        right_ops: np.ndarray,
    ) -> tuple[
        list[tuple[float, list, np.ndarray, list, np.ndarray]],
    ]:
        """Returns weighted (left table, right table) components after the move."""
        attack = self.attack
        if not (
            attack.target_active
            and attack.target_kind in {"node", "memory"}
            and attack.target_id == repeater
        ):
            return [(1.0, left_labels, left_ops, right_labels, right_ops)]
        side = attack.target_memory_side if attack.target_memory_side in {"left", "right", "both"} else attack.memory_side
        target_left = side in {"left", "both"}
        target_right = side in {"right", "both"}
        if attack.move_kind == "memory_degradation":
            depol = clip_probability(attack.memory_depolarizing_probability)
            dephase = clip_probability(attack.memory_dephasing_probability)
            new_left, new_right = left_ops, right_ops
            if target_left:
                new_left = _apply_depolarizing_on_qubit(new_left, 1, depol)
                if dephase > 0.0:
                    factor = np.full(new_left.shape[1], 1.0 - 2.0 * dephase)
                    new_left = _apply_coherence_mask(new_left, 1, factor)
            if target_right:
                new_right = _apply_depolarizing_on_qubit(new_right, 0, depol)
                if dephase > 0.0:
                    factor = np.full(new_right.shape[1], 1.0 - 2.0 * dephase)
                    new_right = _apply_coherence_mask(new_right, 0, factor)
            return [(1.0, left_labels, new_left, right_labels, new_right)]
        if attack.move_kind == "repeater_memory_dephasing_probe":
            theta = float(attack.dephasing_theta)
            suffix = f"repeater_memory_dephasing_probe:node={repeater}:theta={theta:.12g}"
            factor_value = math.cos(theta)
            new_left, new_right = left_ops, right_ops
            new_left_labels, new_right_labels = left_labels, right_labels
            if target_left:
                factor = np.full(new_left.shape[1], factor_value)
                new_left = _apply_coherence_mask(new_left, 1, factor)
                new_left_labels = [
                    (frame, join_labels(eve, suffix), public, click)
                    for frame, eve, public, click in left_labels
                ]
            if target_right:
                factor = np.full(new_right.shape[1], factor_value)
                new_right = _apply_coherence_mask(new_right, 0, factor)
                if not target_left:
                    new_right_labels = [
                        (frame, join_labels(eve, suffix), public, click)
                        for frame, eve, public, click in right_labels
                    ]
            return [(1.0, new_left_labels, new_left, new_right_labels, new_right)]
        if attack.move_kind in {"repeater_memory_measure_Z", "repeater_memory_measure_X"}:
            basis = "Z" if attack.move_kind.endswith("_Z") else "X"
            probability = clip_probability(attack.measurement_probability)
            info = clip_probability(attack.information_probability)
            components: list[tuple[float, list, np.ndarray, list, np.ndarray]] = []
            if probability < 1.0:
                components.append((1.0 - probability, left_labels, left_ops, right_labels, right_ops))
            measured_left_labels, measured_left = left_labels, left_ops
            measured_right_labels, measured_right = right_labels, right_ops
            if target_left:
                measured_left_labels, measured_left = _expand_repeater_measure(
                    left_labels,
                    left_ops,
                    qubit=1,
                    basis=basis,
                    label_prefix=f"{attack.move_kind}:node={repeater}:side=left:basis={basis}",
                    information_probability=info,
                )
            if target_right:
                measured_right_labels, measured_right = _expand_repeater_measure(
                    right_labels,
                    right_ops,
                    qubit=0,
                    basis=basis,
                    label_prefix=f"{attack.move_kind}:node={repeater}:side=right:basis={basis}",
                    information_probability=info,
                )
            if probability > 0.0:
                components.append(
                    (probability, measured_left_labels, measured_left, measured_right_labels, measured_right)
                )
            return components
        return [(1.0, left_labels, left_ops, right_labels, right_ops)]

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, fiber_lengths_m: Sequence[float]) -> TranscriptDistribution:
        lengths = [float(length) for length in fiber_lengths_m]
        if len(lengths) != self.hop_count:
            raise ValueError(f"expected {self.hop_count} fiber lengths, got {len(lengths)}")
        front, context, attack = self.front, self.context, self.attack
        route = make_route(self.hop_count, lengths)
        route["fiber_length_symbols"] = list(self.fiber_length_symbols)
        edge_ids = list(route["edge_ids"])
        etas = {edge_id: self._edge_eta(route["edge_lengths_m"][edge_id], edge_id) for edge_id in edge_ids}
        outcomes = {edge_id: self._edge_outcomes(etas[edge_id], edge_id) for edge_id in edge_ids}
        with _InjectedEdgeOutcomes(front, context, attack, route, outcomes):
            stages = delivery_stage_probabilities(route, front, attack, context)
        edge_success = {}
        for edge_id in edge_ids:
            edge_success[edge_id], _round_meta = multiplexed_edge_round(
                single_mode_success_probability=(
                    outcomes[edge_id].get("psi_plus", 0.0)
                    + outcomes[edge_id].get("psi_minus", 0.0)
                ),
                edge_id=edge_id,
                route=route,
                front=front,
                context=context,
            )
        protocol = ProtocolConfig(requested_pairs=self.key_length)
        rolling_moment_reduced = (
            normalized_waiting_protocol(context) == "rolling_alive_pair"
            and normalized_rolling_age_evaluation(context) == "moment_reduced"
        )
        batch = None if rolling_moment_reduced else self._signature_batch(route, edge_success)
        branches: list[Branch] = []
        if rolling_moment_reduced or batch is not None:
            if rolling_moment_reduced:
                usable_weights = np.asarray([1.0], dtype=float)
                site_diff = np.ones((1, 2 * len(edge_ids)), dtype=float)
                site_same = np.ones_like(site_diff)
            else:
                assert batch is not None
                usable_weights, ages = batch
                factors = _storage_coherence_factors(front, ages)
                # Exact batch reduction 1: the fold depends on a waiting history
                # only through its per-site coherence-factor vector (survival and
                # cutoff are already folded into the weights), so histories with
                # float-identical factor rows merge losslessly.
                unique_factors, inverse = np.unique(factors, axis=0, return_inverse=True)
                if unique_factors.shape[0] < len(usable_weights):
                    merged = np.zeros(unique_factors.shape[0])
                    np.add.at(merged, inverse, usable_weights)
                    usable_weights = merged
                    factors = unique_factors
                # Exact batch reduction 2 (moment expansion): each site's
                # dephasing mask is affine in its coherence factor, so the fold is
                # multilinear in the per-site factors. When 2^sites is smaller
                # than the history batch, evaluate the fold once per site subset
                # (population part vs coherence part) and weight by the exact
                # joint moments E[prod_{q in T} c_q] of the waiting mixture.
                n_sites = factors.shape[1]
                if (1 << n_sites) < len(usable_weights):
                    subsets = np.array(
                        [[(mask >> q) & 1 for q in range(n_sites)] for mask in range(1 << n_sites)],
                        dtype=float,
                    )
                    selected = np.where(subsets[:, None, :] == 1.0, factors[None, :, :], 1.0)
                    moments = np.prod(selected, axis=2) @ usable_weights
                    usable_weights = moments
                    site_diff = subsets
                    site_same = 1.0 - subsets
                else:
                    site_diff = factors
                    site_same = np.ones_like(factors)
            repeaters = list(route["nodes"][1:-1])
            confusion = swap_bell_outcome_confusion_model(front)
            batch_size = len(usable_weights)
            if rolling_moment_reduced:
                labels, ops = self._rolling_moment_reduced_edge_table(
                    route=route,
                    edge_index=0,
                    edge_id=edge_ids[0],
                    eta=etas[edge_ids[0]],
                )
            else:
                labels, ops = self._edge_table(
                    edge_ids[0],
                    etas[edge_ids[0]],
                    float(route["edge_lengths_m"][edge_ids[0]]),
                )
            ops = np.broadcast_to(ops, (ops.shape[0], batch_size, 4, 4)).copy()
            for step, repeater in enumerate(repeaters, start=1):
                if rolling_moment_reduced:
                    right_labels, right_ops = self._rolling_moment_reduced_edge_table(
                        route=route,
                        edge_index=step,
                        edge_id=edge_ids[step],
                        eta=etas[edge_ids[step]],
                    )
                else:
                    right_labels, right_ops = self._edge_table(
                        edge_ids[step],
                        etas[edge_ids[step]],
                        float(route["edge_lengths_m"][edge_ids[step]]),
                    )
                right_ops = np.broadcast_to(right_ops, (right_ops.shape[0], batch_size, 4, 4)).copy()
                # single-application-at-consumption dephasing: left qubit1 is
                # m_{step-1}R (ages column 2*step-1), right qubit0 is m_stepL
                # (ages column 2*step).
                if not rolling_moment_reduced:
                    ops = _apply_coherence_mask(
                        ops, 1, site_diff[:, 2 * step - 1], site_same[:, 2 * step - 1]
                    )
                    right_ops = _apply_coherence_mask(
                        right_ops, 0, site_diff[:, 2 * step], site_same[:, 2 * step]
                    )
                components = self._apply_repeater_move(
                    step, repeater, labels, ops, right_labels, right_ops
                )
                accumulated_ops: dict[tuple[str, str, str, str], np.ndarray] = {}
                for weight, c_left_labels, c_left_ops, c_right_labels, c_right_ops in components:
                    n_left, n_right = len(c_left_labels), len(c_right_labels)
                    composed_labels = [
                        [
                            (
                                compose_frame(l_frame, r_frame),
                                join_labels(l_eve, r_eve),
                                join_detector_click_records(
                                    l_click,
                                    r_click,
                                    keep_detector_click_records=False,
                                ),
                            )
                            for r_frame, r_eve, _r_public, r_click in c_right_labels
                        ]
                        for l_frame, l_eve, _l_public, l_click in c_left_labels
                    ]
                    # chunk the batch axis so the [nL, nR, chunk, 4, 4]
                    # contraction stays within a fixed memory budget
                    chunk = max(1, (1 << 22) // max(1, n_left * n_right * 16))
                    for start in range(0, batch_size, chunk):
                        stop = min(batch_size, start + chunk)
                        left_chunk = c_left_ops[:, start:stop]
                        right_chunk = c_right_ops[:, start:stop]
                        for outcome in BELL_LABELS:
                            contracted = _swap_contract(
                                left_chunk, right_chunk, _BELL_VECTOR_2x2[outcome]
                            )
                            for reported, report_probability in bell_outcome_reported_distribution(
                                confusion, outcome
                            ):
                                corrected = _apply_swap_output_channels(
                                    contracted,
                                    reported,
                                    front,
                                    context,
                                )
                                correction = CORRECTION_BY_BSM[reported]
                                scale = weight * report_probability
                                for li in range(n_left):
                                    for ri in range(n_right):
                                        pair_frame, pair_eve, pair_click = composed_labels[li][ri]
                                        label = (
                                            compose_frame(pair_frame, correction),
                                            pair_eve,
                                            "frame_compressed",
                                            pair_click,
                                        )
                                        slot = accumulated_ops.get(label)
                                        if slot is None:
                                            slot = np.zeros((batch_size, 4, 4), dtype=complex)
                                            accumulated_ops[label] = slot
                                        slot[start:stop] += scale * corrected[li, ri]
                labels = list(accumulated_ops.keys())
                ops = np.stack([accumulated_ops[label] for label in labels], axis=0)
            if not rolling_moment_reduced:
                ops = _apply_coherence_mask(ops, 0, site_diff[:, 0], site_same[:, 0])
                ops = _apply_coherence_mask(ops, 1, site_diff[:, -1], site_same[:, -1])
            final = np.einsum("b,nbij->nij", usable_weights, ops)
            total = float(np.real(np.trace(final, axis1=-2, axis2=-1).sum()))
            if total > 0.0:
                for label, op in zip(labels, final):
                    weight = float(np.real(np.trace(op)))
                    if weight <= 0.0:
                        continue
                    branches.append(Branch(
                        probability=weight / total,
                        rho=clean_density(op / weight),
                        frame=label[0],
                        eve_observation=label[1],
                        public_bsm_record=label[2],
                        detector_click_record=label[3],
                    ))
        branches = aggregate_branches(
            normalize_branches(branches),
            keep_full_bsm=False,
            keep_detector_click_records=False,
        )
        rows = histogram_rows_from_branches(
            branches=branches,
            stages=stages,
            front=front,
            protocol=protocol,
            key_length=self.key_length,
            keep_full_bsm=False,
            keep_detector_click_records=False,
            context=context,
        )
        metrics = expected_metrics(rows, protocol)
        waiting_certification = waiting_metric_certification(rows, protocol, stages, context)
        validation = validate_histogram(rows)
        metadata = {
            "generator_version": GENERATOR_VERSION,
            "compiled_kernel_version": COMPILED_KERNEL_VERSION,
            "case_id": case_id(route, attack),
            "frozen_theta_hash": self.theta_hash,
            "evaluation_mode": "compiled_length_parametric_kernel",
            "route": route,
            "eve_move_spec": attack.to_dict(),
            "key_length": self.key_length,
            "edge_eta": {edge_id: etas[edge_id] for edge_id in edge_ids},
            "edge_polynomial_degrees": {
                "plain": self.plain_edge_poly["degree"],
                "targeted": (
                    self.targeted_edge_poly["degree"] if self.targeted_edge_poly is not None else None
                ),
            },
            "detector_click_record_policy": detector_click_record_policy_metadata(
                emit_detector_click_audit=False
            ),
            "delivery_model": stages,
            "waiting_budget_semantics": (
                ((stages.get("waiting_model") or {}).get("game_turn_budget_semantics"))
                or stages.get("rolling_live_pair_model")
                or {
                    "waiting_budget_semantics": "not_applicable",
                    "reason": "no stochastic repeat-until-success waiting model is active",
                }
            ),
            "rolling_live_pair_policy": stages.get("rolling_live_pair_model"),
            "waiting_metric_certification": waiting_certification,
            "expected_metrics": metrics,
            "validation_checks": validation,
            "linear_chain_contract": linear_chain_contract(route),
            "eve_move_contract": eve_move_contract(attack),
            "length_contract": {
                **length_contract(route),
                "length_parametric_status": "compiled_in_process_kernel",
                "substitution_method": (
                    "CompiledRouteKernel.evaluate(fiber_lengths_m): edge instruments from exact "
                    "polynomials in sqrt(eta(L_i)), pure-dephasing storage via coherence factors of the "
                    "exact per-memory ages, classical waiting/survival/cutoff layers re-enumerated per L"
                ),
            },
        }
        return TranscriptDistribution(metadata=metadata, rows=rows)

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path | str) -> None:
        payload = {
            "compiled_kernel_version": COMPILED_KERNEL_VERSION,
            "generator_version": GENERATOR_VERSION,
            "theta_hash": self.theta_hash,
            "front": _json_ready(self.front),
            "context": self.context.to_dict(),
            "hop_count": self.hop_count,
            "attack": self.attack.to_dict(),
            "key_length": self.key_length,
            "fiber_length_symbols": list(self.fiber_length_symbols),
            "plain_labels": self.plain_edge_poly["labels"],
            "plain_degree": self.plain_edge_poly["degree"],
            "plain_outcome_labels": self.plain_edge_poly["outcome_labels"],
            "plain_fit_max_residual": self.plain_edge_poly["fit_max_residual"],
            "targeted_labels": (
                self.targeted_edge_poly["labels"] if self.targeted_edge_poly is not None else None
            ),
            "targeted_degree": (
                self.targeted_edge_poly["degree"] if self.targeted_edge_poly is not None else None
            ),
            "targeted_outcome_labels": (
                self.targeted_edge_poly["outcome_labels"] if self.targeted_edge_poly is not None else None
            ),
            "targeted_fit_max_residual": (
                self.targeted_edge_poly.get("fit_max_residual")
                if self.targeted_edge_poly is not None else None
            ),
        }
        arrays: dict[str, np.ndarray] = {
            "plain_operator_coeffs": self.plain_edge_poly["operator_coeffs"],
            "plain_outcome_coeffs": self.plain_edge_poly["outcome_coeffs"],
        }
        if self.targeted_edge_poly is not None:
            arrays["targeted_operator_coeffs"] = self.targeted_edge_poly["operator_coeffs"]
            arrays["targeted_outcome_coeffs"] = self.targeted_edge_poly["outcome_coeffs"]
        np.savez_compressed(
            Path(path),
            spec=np.array(json.dumps(_json_ready(payload))),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path | str) -> "CompiledRouteKernel":
        data = np.load(Path(path), allow_pickle=False)
        payload = json.loads(str(data["spec"]))
        if payload["compiled_kernel_version"] != COMPILED_KERNEL_VERSION:
            raise ValueError(
                f"compiled kernel version mismatch: artifact {payload['compiled_kernel_version']!r}, "
                f"runtime {COMPILED_KERNEL_VERSION!r}"
            )
        plain_poly = {
            "labels": payload["plain_labels"],
            "operator_coeffs": np.asarray(data["plain_operator_coeffs"]),
            "outcome_labels": payload["plain_outcome_labels"],
            "outcome_coeffs": np.asarray(data["plain_outcome_coeffs"]),
            "degree": payload["plain_degree"],
            "fit_max_residual": payload["plain_fit_max_residual"],
        }
        targeted_poly = None
        if payload["targeted_labels"] is not None:
            targeted_poly = {
                "labels": payload["targeted_labels"],
                "operator_coeffs": np.asarray(data["targeted_operator_coeffs"]),
                "outcome_labels": payload["targeted_outcome_labels"],
                "outcome_coeffs": np.asarray(data["targeted_outcome_coeffs"]),
                "degree": payload["targeted_degree"],
                "fit_max_residual": payload.get("targeted_fit_max_residual"),
            }
        kernel = cls(
            front=payload["front"],
            context=model_context_from_dict(payload["context"]),
            hop_count=int(payload["hop_count"]),
            attack=eve_move_from_dict(payload["attack"]),
            key_length=int(payload["key_length"]),
            theta_hash=payload["theta_hash"],
            plain_edge_poly=plain_poly,
            targeted_edge_poly=targeted_poly,
            fiber_length_symbols=tuple(payload["fiber_length_symbols"]),
        )
        reference = FrozenHistogramKernel(
            front=kernel.front,
            context=kernel.context,
            hop_count=kernel.hop_count,
            attack=kernel.attack,
            key_length=kernel.key_length,
            fiber_length_symbols=kernel.fiber_length_symbols or None,
        )
        if reference.theta_hash != kernel.theta_hash:
            raise ValueError(
                "compiled kernel artifact does not reproduce its recorded frozen theta hash; "
                "the artifact does not belong to this runtime's model family"
            )
        return kernel


def verify_compiled_kernel(
    compiled: CompiledRouteKernel,
    fiber_length_vectors: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Exactness check: compiled vs reference row distributions at given L."""
    reference = FrozenHistogramKernel(
        front=compiled.front,
        context=compiled.context,
        hop_count=compiled.hop_count,
        attack=compiled.attack,
        key_length=compiled.key_length,
        fiber_length_symbols=compiled.fiber_length_symbols or None,
    )
    checks = []
    for lengths in fiber_length_vectors:
        compiled_result = compiled.evaluate(lengths)
        reference_result = reference.evaluate(lengths)
        tv = total_variation_distance(compiled_result.rows, reference_result.rows)
        compiled_sum = float(sum(row["probability"] for row in compiled_result.rows))
        checks.append({
            "fiber_lengths_m": [float(v) for v in lengths],
            "tv_distance_vs_reference": tv,
            "compiled_probability_sum": compiled_sum,
            "passed": tv <= tolerance and abs(compiled_sum - 1.0) <= 1e-7,
        })
    return {
        "theta_hash": compiled.theta_hash,
        "tolerance": tolerance,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
