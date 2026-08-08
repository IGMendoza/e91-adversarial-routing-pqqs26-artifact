"""Single-qubit operations on one arm of an entangled polarization pair.

SeQUeNCe's legacy polarization path (``FreeQuantumState``, used by ``SPDCSource``)
has no entanglement-aware way to apply a local single-qubit operation: its
``random_noise`` overwrites the whole shared state with a single-qubit vector,
corrupting the pair. This module supplies the missing primitive -- apply a 2x2
unitary to exactly one qubit of the shared pure state vector (``U`` at the
photon's index, identity elsewhere) -- and two uses of it:

- ``apply_polarization_depolarization``: with probability ``1 - fidelity`` apply a
  Haar-random SU(2) to the arm. This is the entanglement-correct realization of
  SeQUeNCe's polarization-fidelity model (a per-photon random-unitary channel);
  averaged over photons it decorrelates the affected arm, giving matched-basis
  QBER ~ (1 - fidelity) / 2. The channel *form* (random unitary) is a standard
  depolarizing model, chosen to reproduce the measurable effect of SeQUeNCe's
  scalar ``polarization_fidelity``; it is a toy/standard model, not a paper-fit.

- ``apply_measurement_rotation``: apply the rotation ``R(-theta)`` so that a
  subsequent Z-basis measurement is equivalent to measuring at polarization angle
  ``theta`` (used by the CHSH/E91 variant for arbitrary measurement angles).

Both operate on the shared ``FreeQuantumState`` in place and keep it a valid
(normalized, pure) joint state.
"""

from __future__ import annotations

from math import cos, sin
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sequence.components.photon import Photon

_I2 = np.eye(2, dtype=complex)


def _apply_single_arm_unitary(photon: "Photon", u: np.ndarray) -> None:
    """Apply 2x2 unitary ``u`` to this photon's qubit of its (possibly entangled)
    shared FreeQuantumState, leaving the partner qubit(s) unchanged."""
    qs = photon.quantum_state
    states = getattr(qs, "entangled_states", None)
    if states is None:  # quantum-manager photon: not handled here
        return
    idx = states.index(qs)
    op = np.array([[1.0 + 0j]])
    for i in range(len(states)):
        op = np.kron(op, u if i == idx else _I2)
    new_state = tuple(op @ np.asarray(qs.state, dtype=complex))
    for s in states:
        if s is not None:
            s.state = new_state


def random_su2(rng: np.random.Generator) -> np.ndarray:
    """Haar-random 2x2 unitary (QR of a complex Gaussian, phase-fixed)."""
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(z)
    diag = np.diagonal(r)
    return q * (diag / np.abs(diag))


def rotation(theta: float) -> np.ndarray:
    """Real polarization rotation by ``theta`` (radians)."""
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta), cos(theta)]], dtype=complex)


def apply_polarization_depolarization(photon: "Photon", fidelity: float,
                                      rng: np.random.Generator) -> bool:
    """With probability ``1 - fidelity`` apply a Haar-random SU(2) to the photon's
    arm (matches SeQUeNCe's ``random() > polarization_fidelity`` error convention).
    Returns True iff noise was applied."""
    if fidelity >= 1.0:
        return False
    if rng.random() > fidelity:
        _apply_single_arm_unitary(photon, random_su2(rng))
        return True
    return False


def apply_measurement_rotation(photon: "Photon", theta: float) -> None:
    """Rotate the photon's arm by ``-theta`` so a later Z measurement equals a
    measurement at polarization angle ``theta``."""
    _apply_single_arm_unitary(photon, rotation(-theta))
