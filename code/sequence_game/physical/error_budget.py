"""Error-budget mapping recipes for the swap_bsm literature config.

Implements, as importable and tested code, the mapping recipes documented in
``configs/physical/swap_bsm_literature_stub.json`` under
``error_budget_mapping_recipes``. This module supplies NO parameter values:
the swap error budget stays TODO(scientific) until node-specific measured
values (a two-qubit gate fidelity or per-ion readout error probabilities) are
supplied by the user from an explicit reference. The helpers here only convert
such user-supplied values into the typed scalars/config fragments consumed by
the histogram generator's normalizers.

Conventions implemented (matching the config recipe text):

- Swap degradation channel: uniform two-qubit Pauli mixture with total
  non-identity probability ``p = 1 - swapping_degradation``. For this channel
  the process/entanglement fidelity is ``F_pro = 1 - p`` and the average gate
  fidelity is ``F_avg = 1 - (4/5) p`` (via ``F_avg = (d * F_pro + 1) / (d + 1)``
  with ``d = 4``).
- Bell-label readout confusion: a gate-based local Bell measurement reads two
  classical bits (parity, phase). A flip of the parity bit (probability
  ``e1``) maps phi+<->psi+ and phi-<->psi-; a flip of the phase bit
  (probability ``e2``) maps phi+<->phi- and psi+<->psi-; both flips map
  phi+<->psi- and phi-<->psi+. Bell label order is
  ``(phi_plus, phi_minus, psi_plus, psi_minus)``.
"""

from __future__ import annotations

from typing import Any

BELL_LABELS = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")

FIDELITY_KINDS = ("process", "average")

_PARITY_BIT_FLIP = {
    "phi_plus": "psi_plus",
    "psi_plus": "phi_plus",
    "phi_minus": "psi_minus",
    "psi_minus": "phi_minus",
}
_PHASE_BIT_FLIP = {
    "phi_plus": "phi_minus",
    "phi_minus": "phi_plus",
    "psi_plus": "psi_minus",
    "psi_minus": "psi_plus",
}


def _require_probability(name: str, value: Any) -> float:
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")
    return out


def uniform_pauli_probability_from_two_qubit_gate_fidelity(
    fidelity: float,
    *,
    fidelity_kind: str,
) -> float:
    """Total non-identity probability p of the uniform two-qubit Pauli mixture.

    ``fidelity_kind='process'``: p = 1 - F_pro.
    ``fidelity_kind='average'``: p = (5/4) * (1 - F_avg); F_avg below 0.2 has
    no uniform-Pauli-mixture representation (p would exceed 1) and fails.
    """
    if fidelity_kind not in FIDELITY_KINDS:
        raise ValueError(
            f"fidelity_kind must be one of {FIDELITY_KINDS}, got {fidelity_kind!r}"
        )
    fidelity = _require_probability("two-qubit gate fidelity", fidelity)
    if fidelity_kind == "process":
        probability = 1.0 - fidelity
    else:
        probability = 1.25 * (1.0 - fidelity)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{fidelity_kind} fidelity {fidelity!r} maps to uniform-Pauli probability "
            f"{probability!r} outside [0, 1]; no uniform two-qubit Pauli mixture represents it"
        )
    return probability


def swapping_degradation_from_two_qubit_gate_fidelity(
    fidelity: float,
    *,
    fidelity_kind: str,
) -> float:
    """``swapping_degradation`` scalar (= 1 - p) for the histogram kernel."""
    return 1.0 - uniform_pauli_probability_from_two_qubit_gate_fidelity(
        fidelity,
        fidelity_kind=fidelity_kind,
    )


def swap_degradation_config_fragment(
    fidelity: float,
    *,
    fidelity_kind: str,
    source: str,
) -> dict[str, Any]:
    """Provenance-carrying fragment for ``swap_bsm.parameters``.

    The scalar under ``swapping_degradation`` is what the generator consumes;
    ``provenance`` documents the mapping and belongs in the config notes.
    Requires a non-empty ``source`` reference: this module supplies recipes,
    not values.
    """
    if not str(source).strip():
        raise ValueError("source reference is required; recipes must not be applied without provenance")
    degradation = swapping_degradation_from_two_qubit_gate_fidelity(
        fidelity,
        fidelity_kind=fidelity_kind,
    )
    return {
        "swapping_degradation": degradation,
        "provenance": {
            "recipe": "swapping_degradation_from_two_qubit_gate_fidelity",
            "input_fidelity": float(fidelity),
            "input_fidelity_kind": fidelity_kind,
            "channel": "uniform two-qubit Pauli mixture, p = 1 - swapping_degradation",
            "source": str(source),
            "scope": "literature",
        },
    }


def bell_outcome_confusion_matrix_from_readout_errors(
    parity_bit_error: float,
    phase_bit_error: float,
) -> list[list[float]]:
    """Row-stochastic 4x4 confusion matrix over BELL_LABELS from per-bit readout errors."""
    e1 = _require_probability("parity_bit_error", parity_bit_error)
    e2 = _require_probability("phase_bit_error", phase_bit_error)
    index = {label: position for position, label in enumerate(BELL_LABELS)}
    matrix = [[0.0, 0.0, 0.0, 0.0] for _ in BELL_LABELS]
    for row, true_label in enumerate(BELL_LABELS):
        matrix[row][index[true_label]] += (1.0 - e1) * (1.0 - e2)
        matrix[row][index[_PARITY_BIT_FLIP[true_label]]] += e1 * (1.0 - e2)
        matrix[row][index[_PHASE_BIT_FLIP[true_label]]] += (1.0 - e1) * e2
        matrix[row][index[_PHASE_BIT_FLIP[_PARITY_BIT_FLIP[true_label]]]] += e1 * e2
    return matrix


def bell_outcome_confusion_model_config_fragment(
    parity_bit_error: float,
    phase_bit_error: float,
    *,
    source: str,
) -> dict[str, Any]:
    """Typed ``bell_outcome_confusion_model`` fragment for ``swap_bsm.parameters``.

    Compatible with the histogram generator's
    ``normalize_swap_bell_outcome_model_from_config`` (kind
    ``bell_outcome_confusion_matrix``). Requires a non-empty ``source``
    reference: this module supplies recipes, not values.
    """
    if not str(source).strip():
        raise ValueError("source reference is required; recipes must not be applied without provenance")
    matrix = bell_outcome_confusion_matrix_from_readout_errors(parity_bit_error, phase_bit_error)
    return {
        "kind": "bell_outcome_confusion_matrix",
        "confusion_matrix": matrix,
        "bell_label_order": list(BELL_LABELS),
        "readout_error_probabilities": {
            "parity_bit": float(parity_bit_error),
            "phase_bit": float(phase_bit_error),
        },
        "calibration_status": (
            "derived from user-supplied per-ion readout error probabilities via the structured "
            "two-bit (parity, phase) Bell-label readout recipe"
        ),
        "source": str(source),
        "scope": "literature",
    }
