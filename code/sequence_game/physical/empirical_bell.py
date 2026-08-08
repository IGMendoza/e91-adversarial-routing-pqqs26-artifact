"""Small, source-facing Bell-diagonal calibration primitives.

These helpers turn measured, sign-corrected Bell correlations into the Pauli
error representation used by the histogram backend.  They deliberately do
not infer unmeasured correlations unless the profile states the completion
assumption explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class EmpiricalBellError(ValueError):
    """An empirical Bell-state or storage calibration is not physical."""


def _correlation(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise EmpiricalBellError(f"{name} must be finite and in [-1, 1]")
    return result


def psi_minus_pauli_distribution_from_visibilities(
    *,
    xx: float,
    yy: float,
    zz: float,
) -> dict[str, float]:
    """Return I/X/Z/Y weights for positive, Psi-minus-frame visibilities.

    ``xx``, ``yy``, and ``zz`` are positive sign-corrected magnitudes, so the
    physical Pauli correlations are ``(-xx, -yy, -zz)``.  Applying I/X/Z/Y to
    one half of ``|Psi->`` labels the four Bell components in the repository's
    established order.
    """

    vx = _correlation("xx visibility", xx)
    vy = _correlation("yy visibility", yy)
    vz = _correlation("zz visibility", zz)
    weights = {
        "I": (1.0 + vx + vy + vz) / 4.0,
        "X": (1.0 + vx - vy - vz) / 4.0,
        "Z": (1.0 - vx - vy + vz) / 4.0,
        "Y": (1.0 - vx + vy - vz) / 4.0,
    }
    if min(weights.values()) < -1e-12:
        raise EmpiricalBellError(
            "the supplied Bell correlations do not define a positive "
            f"Bell-diagonal state: {weights}"
        )
    clipped = {label: max(0.0, float(value)) for label, value in weights.items()}
    total = sum(clipped.values())
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise EmpiricalBellError(f"Bell-diagonal weights sum to {total}, not 1")
    return clipped


def psi_minus_distribution_from_model(model: Mapping[str, Any]) -> dict[str, float]:
    if str(model.get("target_bell_state", "")) != "psi_minus":
        raise EmpiricalBellError("empirical Bell correlations currently require psi_minus")
    return psi_minus_pauli_distribution_from_visibilities(
        xx=float(model["xx_visibility"]),
        yy=float(model["yy_visibility"]),
        zz=float(model["zz_visibility"]),
    )


def symmetric_local_liu_storage_eigenvalues(
    model: Mapping[str, Any],
    total_storage_time_s: float,
) -> tuple[float, float, float]:
    """Local Pauli eigenvalues whose equal-age product matches Liu's pair fit.

    Liu reports pair-level ``Vxx(t)=Vxx(0) exp(-t/tau)`` and
    ``Vzz(t)=Vzz(0)-r*t``.  The square-root factorization is the unique
    symmetric local factorization.  It is exact when both memories have the
    same age and remains a declared empirical completion for unequal ages.
    """

    age = max(0.0, float(total_storage_time_s))
    tau = float(model["xx_pair_coherence_time_s"])
    vzz0 = float(model["zz_initial_visibility"])
    rate = float(model["zz_linear_decay_rate_per_s"])
    if not math.isfinite(tau) or tau <= 0.0:
        raise EmpiricalBellError("xx_pair_coherence_time_s must be finite and > 0")
    if not math.isfinite(vzz0) or vzz0 <= 0.0 or vzz0 > 1.0:
        raise EmpiricalBellError("zz_initial_visibility must be in (0, 1]")
    if not math.isfinite(rate) or rate < 0.0:
        raise EmpiricalBellError("zz_linear_decay_rate_per_s must be finite and >= 0")
    fit_max = float(model.get("fit_domain_max_s", math.inf))
    if age > fit_max + 1e-15 and str(model.get("extrapolation_policy", "")) == "fail_closed":
        raise EmpiricalBellError(
            f"storage age {age:g} s exceeds the measured fit domain {fit_max:g} s"
        )
    lambda_xy = math.exp(-age / (2.0 * tau))
    pair_zz_ratio = max(0.0, 1.0 - rate * age / vzz0)
    lambda_z = math.sqrt(pair_zz_ratio)
    return lambda_xy, lambda_xy, lambda_z


def pauli_probabilities_from_eigenvalues(
    lambda_x: float,
    lambda_y: float,
    lambda_z: float,
) -> dict[str, float]:
    """Convert a unital qubit channel's Pauli eigenvalues to probabilities."""

    lx = _correlation("lambda_x", lambda_x)
    ly = _correlation("lambda_y", lambda_y)
    lz = _correlation("lambda_z", lambda_z)
    probabilities = {
        "I": (1.0 + lx + ly + lz) / 4.0,
        "X": (1.0 + lx - ly - lz) / 4.0,
        "Y": (1.0 - lx + ly - lz) / 4.0,
        "Z": (1.0 - lx - ly + lz) / 4.0,
    }
    if min(probabilities.values()) < -1e-12:
        raise EmpiricalBellError(
            "storage eigenvalues do not define a completely positive Pauli "
            f"channel: {probabilities}"
        )
    return {label: max(0.0, float(value)) for label, value in probabilities.items()}


def optimal_symmetric_xz_chsh(xx: float, zz: float) -> dict[str, float]:
    """Optimal Bob angle and CHSH value for fixed Alice X/Z axes.

    For an XZ-diagonal correlation tensor, Bob uses unit axes proportional to
    ``zz*Z +/- xx*X``.  This is also the Horodecki optimum when X and Z are the
    two largest singular directions.
    """

    vx = abs(_correlation("xx visibility", xx))
    vz = abs(_correlation("zz visibility", zz))
    norm = math.hypot(vx, vz)
    if norm <= 0.0:
        raise EmpiricalBellError("at least one XZ visibility must be nonzero")
    angle = math.atan2(vx, vz)
    return {
        "bob_angle_from_z_rad": angle,
        "bob_angle_from_z_deg": math.degrees(angle),
        "bob_x_component_magnitude": vx / norm,
        "bob_z_component": vz / norm,
        "optimized_chsh_s": 2.0 * norm,
    }
