"""Shared Ekert-91 / CHSH-S core: angle spec, statistic, and the E91 decision.

Platform-agnostic. Both key-generation paths feed measured ``(setting_index,
outcome)`` samples here:

* the bare-fiber edge path measures **photon polarization** at the angles in
  :data:`ALICE_ANGLES` / :data:`BOB_ANGLES` directly;
* the repeater path measures the delivered **memory qubit** on the Bloch sphere
  at :func:`bloch_angle` (= twice the polarization angle, realised by an ``Ry``
  rotation before a Z measurement).

Both realisations share one correlation structure (which setting-index pairs are
key vs. CHSH, and the S form), so the statistic and the security decision live
here once.

Angles and the CHSH form are textbook Ekert-91. The correlations are computed
from SeQUeNCe's simulated measured outcomes: this is a simulator result and a
Bell-violation *monitor*, not a loophole-free experiment or a security proof.

Security decision (hard Bell criterion, user-selected):

* ``|S| <= 2`` (classical bound not violated) -> ``chsh_abort``: the correlations
  admit a local-hidden-variable model, so there is no device-independent
  security certificate and the key is rejected as insecure.
* else ``key_qber > qber_threshold`` -> ``qber_abort``: Bell-secure but too
  noisy for a usable key.
* else ``accepted``.

The Bell (security) check precedes the QBER (quality) check so an insecure key
is always labelled ``chsh_abort`` even when it is also noisy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import pi
from typing import Any, Optional, Sequence

#: Alice/Bob polarization measurement angles (radians). Indices are the settings.
ALICE_ANGLES = (0.0, pi / 8, pi / 4)         # 0, 22.5, 45 degrees
BOB_ANGLES = (pi / 8, pi / 4, 3 * pi / 8)    # 22.5, 45, 67.5 degrees
#: Matching-angle (Alice idx, Bob idx) pairs that produce key bits (angles equal).
KEY_PAIRS = ((1, 0), (2, 1))                  # 22.5==22.5, 45==45
#: CHSH cross settings: Alice {0 (idx0), 45 (idx2)} x Bob {22.5 (idx0), 67.5 (idx2)}.
CHSH_PAIRS = ((0, 0), (0, 2), (2, 0), (2, 2))
#: Classical (local-hidden-variable) bound on |S|. Ideal Bell state gives 2*sqrt(2).
CLASSICAL_BOUND = 2.0

# E91 security-decision outcome labels (shared with the game outcome set).
OUTCOME_ACCEPTED = "accepted"
OUTCOME_CHSH_ABORT = "chsh_abort"
OUTCOME_QBER_ABORT = "qber_abort"
OUTCOME_NO_KEY = "delivery_failure"


def bloch_angle(polarization_angle_rad: float) -> float:
    """Bloch-sphere measurement angle for a memory qubit.

    A polarization analyser at angle theta corresponds to measuring a spin/memory
    qubit along a Bloch vector at ``2*theta`` in the X-Z plane, so a Phi+ pair
    gives the same correlation ``E = cos(2*(theta_a - theta_b))`` on both
    platforms. Realise it as ``Ry(-2*theta)`` before a Z measurement.
    """

    return 2.0 * polarization_angle_rad


@dataclass(frozen=True)
class CHSHStatistics:
    """CHSH-S statistic and key summary computed from coinciding measurements."""

    coincidences: int
    chsh_s: float
    violates_bell: bool          # |S| > CLASSICAL_BOUND
    key_length: int
    key_qber: Optional[float]    # mismatch fraction on matching-angle bits
    correlations: dict[str, float]
    min_chsh_cell_count: int     # fewest samples among the four CHSH settings

    def to_dict(self) -> dict[str, Any]:
        return {
            "coincidences": self.coincidences,
            "chsh_s": self.chsh_s,
            "violates_bell": self.violates_bell,
            "key_length": self.key_length,
            "key_qber": self.key_qber,
            "correlations": dict(self.correlations),
            "min_chsh_cell_count": self.min_chsh_cell_count,
        }


def compute_chsh_statistics(
        a_idx: Sequence[int],
        b_idx: Sequence[int],
        a_out: Sequence[int],
        b_out: Sequence[int],
) -> CHSHStatistics:
    """Compute the CHSH-S statistic and matching-angle key QBER.

    Args:
        a_idx, b_idx: per-coincidence Alice/Bob setting indices (into ALICE/BOB_ANGLES).
        a_out, b_out: per-coincidence binary measurement outcomes (0/1).

    All four sequences are aligned and cover only coinciding rounds (both sides
    detected the same pair). Correlation ``E = <+1 if outcomes agree else -1>``.
    """

    if not (len(a_idx) == len(b_idx) == len(a_out) == len(b_out)):
        raise ValueError("a_idx, b_idx, a_out, b_out must be the same length")

    pair_sum: dict[tuple[int, int], int] = defaultdict(int)
    pair_n: dict[tuple[int, int], int] = defaultdict(int)
    key_len = 0
    key_err = 0
    for ai, bi, ao, bo in zip(a_idx, b_idx, a_out, b_out):
        ai, bi = int(ai), int(bi)
        agree = int(ao) == int(bo)
        pair_sum[(ai, bi)] += 1 if agree else -1
        pair_n[(ai, bi)] += 1
        if (ai, bi) in KEY_PAIRS:
            key_len += 1
            if not agree:
                key_err += 1

    def corr(ai: int, bi: int) -> float:
        return pair_sum[(ai, bi)] / pair_n[(ai, bi)] if pair_n[(ai, bi)] else 0.0

    s = corr(0, 0) - corr(0, 2) + corr(2, 0) + corr(2, 2)
    correlations = {
        f"E({ALICE_ANGLES[ai]:.3f},{BOB_ANGLES[bi]:.3f})": corr(ai, bi)
        for ai, bi in CHSH_PAIRS
    }
    min_cell = min((pair_n[p] for p in CHSH_PAIRS), default=0)
    return CHSHStatistics(
        coincidences=len(a_idx),
        chsh_s=s,
        violates_bell=abs(s) > CLASSICAL_BOUND,
        key_length=key_len,
        key_qber=(key_err / key_len if key_len else None),
        correlations=correlations,
        min_chsh_cell_count=min_cell,
    )


#: Minimum samples per CHSH setting for the S estimate to be trusted. Below this
#: the four correlation estimates are noise and |S| can spuriously exceed the
#: Tsirelson bound 2*sqrt(2); such a block cannot certify (or refute) a Bell
#: violation, so it is treated as an inconclusive run, not a security decision.
DEFAULT_MIN_CHSH_CELL_COUNT = 30


def e91_outcome(
        stats: CHSHStatistics,
        *,
        qber_threshold: float,
        min_key_pairs: int,
        min_coincidences_per_setting: int = DEFAULT_MIN_CHSH_CELL_COUNT,
) -> str:
    """Device-independent E91 accept/abort decision (hard Bell criterion).

    Returns one of ``accepted``, ``chsh_abort``, ``qber_abort``,
    ``delivery_failure``. The Bell (security) check precedes the QBER (quality)
    check: an insecure key is labelled ``chsh_abort`` even when also too noisy.

    A block whose sifted key or per-setting CHSH samples are below the minima is
    inconclusive (the S estimate is not physically meaningful and may exceed the
    Tsirelson bound from noise) and is reported as ``delivery_failure`` -- no
    key, no spurious security verdict.
    """

    if (stats.key_length < min_key_pairs
            or stats.key_qber is None
            or stats.min_chsh_cell_count < min_coincidences_per_setting):
        # Insufficient statistics to run a valid Bell test / extract a key.
        return OUTCOME_NO_KEY
    if not stats.violates_bell:
        # |S| <= 2: no Bell violation -> no device-independent security certificate.
        return OUTCOME_CHSH_ABORT
    if stats.key_qber > qber_threshold:
        return OUTCOME_QBER_ABORT
    return OUTCOME_ACCEPTED
