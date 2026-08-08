"""Ekert-91 / CHSH trial: 3-angle entanglement-based QKD with a Bell-inequality test.

Alice measures at angles {0, 22.5, 45}°, Bob at {22.5, 45, 67.5}°. Matching angles
(both 22.5° or both 45°) give perfectly correlated key bits; the cross settings
{0, 45}° × {22.5, 67.5}° form the CHSH statistic

    S = E(0,22.5) - E(0,67.5) + E(45,22.5) + E(45,67.5),

which reaches ~2√2 ≈ 2.83 for the Bell state (violating the classical bound
|S| ≤ 2). An eavesdropper that disturbs the state (e.g. intercept-resend) lowers
|S| toward the classical regime and raises the matching-angle key QBER — the Bell
test is the security check. Angles and the CHSH form are textbook; the
correlations are computed from SeQUeNCe's measured outcomes (a toy/simulator
result, not a loophole-free experiment or a security proof).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..physical.registry import PhysicalModel
from ..sequence_build.chsh_build import build_chsh_link
from .chsh_core import (
    ALICE_ANGLES,
    BOB_ANGLES,
    CHSH_PAIRS,
    CLASSICAL_BOUND,
    KEY_PAIRS,
    compute_chsh_statistics,
)

_BELL_PHI_PLUS = [complex(np.sqrt(0.5)), complex(np.sqrt(0.5))]


@dataclass(frozen=True)
class CHSHResult:
    num_periods: int
    coincidences: int
    chsh_s: float
    violates_bell: bool          # |S| > 2
    key_length: int
    key_qber: Optional[float]    # mismatch on matching-angle bits
    eve_present: bool
    correlations: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_periods": self.num_periods, "coincidences": self.coincidences,
            "chsh_s": self.chsh_s, "violates_bell": self.violates_bell,
            "key_length": self.key_length, "key_qber": self.key_qber,
            "eve_present": self.eve_present, "correlations": dict(self.correlations),
        }


def run_chsh_trial(source_model: PhysicalModel, detector_model: PhysicalModel,
                   fiber_model: PhysicalModel, length_m: float, *, num_periods: int,
                   seed: int, eve_intercept: bool = False) -> CHSHResult:
    if num_periods < 1:
        raise ValueError("num_periods must be >= 1")
    freq = float(source_model.parameters["frequency"])
    period = round(1e12 / freq)
    delay_est = round(length_m / float(fiber_model.parameters["light_speed"]))
    stop_time = int((num_periods + 2) * period + delay_est + period)

    rng = np.random.default_rng([seed, 0xC45])
    a_idx = rng.integers(0, len(ALICE_ANGLES), size=num_periods)
    b_idx = rng.integers(0, len(BOB_ANGLES), size=num_periods)
    a_angles = [ALICE_ANGLES[i] for i in a_idx]
    b_angles = [BOB_ANGLES[i] for i in b_idx]

    built = build_chsh_link(
        source_model, detector_model, fiber_model, length_m,
        alice_angles_rad=a_angles, bob_angles_rad=b_angles,
        stop_time_ps=stop_time, seed=seed, eve_intercept=eve_intercept)
    built.source.emit([_BELL_PHI_PLUS] * num_periods)
    built.timeline.run()

    a_rec = built.alice_meas.records
    b_rec = built.bob_meas.records
    coincide = sorted(set(a_rec) & set(b_rec))

    stats = compute_chsh_statistics(
        a_idx=[int(a_idx[p]) for p in coincide],
        b_idx=[int(b_idx[p]) for p in coincide],
        a_out=[a_rec[p] for p in coincide],
        b_out=[b_rec[p] for p in coincide],
    )
    return CHSHResult(
        num_periods=num_periods, coincidences=stats.coincidences, chsh_s=stats.chsh_s,
        violates_bell=stats.violates_bell, key_length=stats.key_length,
        key_qber=stats.key_qber,
        eve_present=eve_intercept, correlations=stats.correlations)
