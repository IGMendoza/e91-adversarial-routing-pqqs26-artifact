"""Eve's attack-scoped information gain about the sifted raw key.

Implements the user-supplied metric: the empirical conditional mutual information
``I(K; E | P)`` in bits, plus a ``fraction_correct`` diagnostic, where

- ``K`` = Alice's sifted raw-key bit (before privacy amplification),
- ``P`` = the public transcript used for sifting (here: the disclosed matched
  basis per kept index; the abort decision is also public but not informative per
  bit),
- ``E`` = Eve's actual attack record at that index, i.e. her (basis, outcome).

This is an estimator over the simulator's recorded data, not an invented physical
quantity, and not a secrecy proof. K and E stay strictly private (read from the
transcript and the Eve station); only this aggregate is ever surfaced.

Sanity (standard intercept-resend in a random basis, matched-basis bits):
QBER ~ 0.25, ``fraction_correct`` ~ 0.75, ``I(K;E|P)`` ~ 0.5 bits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid hard cross-layer imports; these are duck-typed at runtime
    from ..protocol.transcript import TrialTranscript
    from ..sequence_build.e91_builder import BuiltE91Line


@dataclass(frozen=True)
class InfoGainResult:
    num_sifted: int               # sifted (matched-basis) key bits
    num_eve_records: int          # sifted bits for which Eve has an aligned record
    mutual_information_bits: float  # empirical I(K; E | P)
    fraction_correct: float       # Eve's best-guess accuracy of K given (E, P)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_sifted": self.num_sifted,
            "num_eve_records": self.num_eve_records,
            "mutual_information_bits": self.mutual_information_bits,
            "fraction_correct": self.fraction_correct,
        }


def _entropy_bits(counts) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * log2(p)
    return h


def eve_period_records(built: "BuiltE91Line") -> dict[int, tuple[str, int]]:
    """Map period index -> (Eve basis label, Eve outcome) from the intercept
    station. Empty if there is no intercept station (e.g. no attack or DoS only)."""
    records: dict[int, tuple[str, int]] = {}
    if not built.eve_nodes:
        return records
    frequency = float(built.source.frequency)
    for eve_name in built.eve_nodes:
        station = built.forwarders.get(eve_name)
        observations = getattr(station, "observations", None)
        if not observations:
            continue
        # propagation delay Alice -> this Eve node (so measurement time -> period).
        delay = 0
        for hop in built.hops:
            delay += built.qchannels[(hop.src, hop.dst)].delay
            if hop.dst == eve_name:
                break
        for obs in observations:
            period = round((obs["time"] - delay) * frequency * 1e-12)
            records.setdefault(period, (obs["basis"], int(obs["outcome"])))
    return records


def compute_information_gain(transcript: "TrialTranscript",
                             built: "BuiltE91Line") -> InfoGainResult:
    """Compute I(K;E|P) and fraction_correct over the sifted key of one trial."""
    sifted_idx = transcript.sifted_indices
    periods = transcript.extra.get("coincidence_periods", ())
    a_out = transcript.alice_outcomes
    a_bases = transcript.alice_bases
    eve_records = eve_period_records(built)

    # Aligned samples (K, P, E) over sifted (matched-basis) positions Eve saw.
    samples: list[tuple[int, str, tuple[str, int]]] = []
    for j in sifted_idx:
        k = int(a_out[j])
        period = periods[j] if j < len(periods) else None
        e = eve_records.get(period) if period is not None else None
        if e is not None:
            samples.append((k, a_bases[j], e))

    num_sifted = len(sifted_idx)
    if not samples:
        # No Eve record (no attack, or DoS): zero information. With no observation
        # to condition on, Eve's best guess is a coin flip -> 0.5 (chance), not the
        # empirical key majority (which she cannot know).
        return InfoGainResult(num_sifted, 0, 0.0, 0.5)

    total = len(samples)

    # I(K;E|P) = sum_p Pr(P=p) [ H(K|P=p) - H(K|E,P=p) ]
    by_p: dict[str, list[tuple[int, tuple[str, int]]]] = defaultdict(list)
    for k, p, e in samples:
        by_p[p].append((k, e))
    mi = 0.0
    for p, group in by_p.items():
        w = len(group) / total
        h_k = _entropy_bits(Counter(k for k, _ in group))
        by_e: dict[tuple[str, int], Counter] = defaultdict(Counter)
        for k, e in group:
            by_e[e][k] += 1
        h_k_given_e = sum((sum(c.values()) / len(group)) * _entropy_bits(c)
                          for c in by_e.values())
        mi += w * (h_k - h_k_given_e)
    mi = max(0.0, mi)  # guard tiny negative float noise

    # fraction_correct: Eve's MAP guess of K given (P, E).
    cells: dict[tuple[str, tuple[str, int]], Counter] = defaultdict(Counter)
    for k, p, e in samples:
        cells[(p, e)][k] += 1
    correct = sum(max(c.values()) for c in cells.values())
    fraction_correct = correct / total

    return InfoGainResult(num_sifted, total, mi, fraction_correct)
