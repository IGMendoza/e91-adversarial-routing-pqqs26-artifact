"""Legacy SeQUeNCe-backed bare-fiber E91/BBM92 trial runner.

This module is retained for regression coverage and historical BBM92/bare-fiber
experiments. New repeater-network E91 work should use
``protocol.repeater_trial``.

One *trial* emits ``num_periods`` polarization Bell pairs from Alice's SPDC
source. Alice measures her local arm and Bob measures the arm that traverses the
route's fibers; both choose a random Z/X basis per period. Coincidences (periods
where both arms register a valid bit) are kept, matched-basis indices are sifted,
and QBER/accept are computed by the existing ``protocol.postprocessing`` over the
same ``TrialTranscript`` structure the toy backend uses.

Honest scope: SeQUeNCe's polarization encoding has two MUB bases (Z/X), so this
realizes BBM92-style entanglement QKD (the QKD form usually conflated with E91),
not the 3-angle CHSH/Bell-test form of E91. The *physics* (entangled emission,
fiber loss/noise, detector efficiency/dark counts, measurement collapse) is
SeQUeNCe's; this module only orchestrates emission/measurement timing and
classical sifting (no invented physics).

Basis/timing convention (matches SeQUeNCe's BB84, made robust for source
frequencies that do not divide 1e12 ps): bases are set per period via
``QSDetectorPolarization.set_basis_list`` (index list) with a half-period start
offset so the beam-splitter's ``int()`` bucketing is robust to sub-period
rounding; bits are reconstructed with ``round()`` (mirroring ``Node.get_bits``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..claims import REPEATER_RUNTIME
from ..sequence_build.e91_builder import (
    BuiltE91Line,
    E91Hop,
    StationFactory,
    build_e91_line,
)
from ..physical.registry import PhysicalModel
from .postprocessing import PostprocessingError, SiftingConfig, apply_postprocessing
from .transcript import TrialTranscript

#: Per-photon state passed to SPDCSource.emit; its polarization branch maps
#: [s, s] -> joint (s,0,0,s), i.e. the Phi+ Bell state for s = 1/sqrt(2).
_BELL_PHI_PLUS = [complex(np.sqrt(0.5)), complex(np.sqrt(0.5))]


@dataclass(frozen=True)
class E91RunConfig:
    """How many pairs to emit and how to post-process them."""

    num_periods: int
    sifting: SiftingConfig
    scope: str = REPEATER_RUNTIME

    def __post_init__(self) -> None:
        if self.num_periods < 1:
            raise PostprocessingError("num_periods must be >= 1")
        if len(self.sifting.basis_labels) < 2:
            raise PostprocessingError("need at least 2 basis labels (Z/X)")

    def to_dict(self) -> dict[str, Any]:
        return {"num_periods": self.num_periods, "sifting": self.sifting.to_dict(),
                "scope": self.scope}


def _reconstruct_bits(trigger_times: list[list[int]], recon_start_ps: int,
                      frequency_hz: float, num_periods: int) -> list[int]:
    """Map polarization detector trigger times to per-period bits (0/1, or -1 for
    no/ambiguous detection), mirroring ``Node.get_bits``."""
    bits = [-1] * num_periods
    for t in trigger_times[0]:  # |0> detector
        idx = round((t - recon_start_ps) * frequency_hz * 1e-12)
        if 0 <= idx < num_periods:
            bits[idx] = 0
    for t in trigger_times[1]:  # |1> detector
        idx = round((t - recon_start_ps) * frequency_hz * 1e-12)
        if 0 <= idx < num_periods:
            bits[idx] = -1 if bits[idx] == 0 else 1  # both detectors -> ambiguous
    return bits


def _total_delay_ps(hops: list[E91Hop]) -> int:
    total = 0
    for h in hops:
        speed = float(h.fiber_model.parameters["light_speed"])
        total += round(h.length_m / speed) if speed > 0 else 0
    return total


def run_e91_trial(hops: list[E91Hop], *, alice: str, bob: str,
                  source_model: PhysicalModel, detector_model: PhysicalModel,
                  run_config: E91RunConfig, trial_id: str, seed: int,
                  station_factory: Optional[StationFactory] = None,
                  eve_nodes: tuple[str, ...] = ()) -> tuple[TrialTranscript, BuiltE91Line]:
    """Run one BBM92/E91 trial over ``hops`` and return its transcript plus the
    built network (so Eve observation records can be read from the stations)."""
    frequency = float(source_model.parameters["frequency"])
    period = round(1e12 / frequency)
    n = run_config.num_periods
    stop_time = int((n + 2) * period + _total_delay_ps(hops) + period)

    built = build_e91_line(
        hops, alice=alice, bob=bob, source_model=source_model,
        detector_model=detector_model, stop_time_ps=stop_time, seed=seed,
        station_factory=station_factory, eve_nodes=eve_nodes)

    # Basis choice uses an independent stream (2-int seed) so it cannot correlate
    # with any node's single-int simulation seed.
    basis_rng = np.random.default_rng([seed, 0xE91])
    alice_idx = basis_rng.integers(0, 2, size=n)
    bob_idx = basis_rng.integers(0, 2, size=n)

    bob_delay = built.bob_arm_delay_ps
    half = period // 2
    built.alice_detector.set_basis_list([int(b) for b in alice_idx], -half, frequency)
    built.bob_detector.set_basis_list([int(b) for b in bob_idx], bob_delay - half, frequency)

    built.source.emit([_BELL_PHI_PLUS] * n)
    built.timeline.run()

    alice_bits = _reconstruct_bits(built.alice_detector.get_photon_times(), 0, frequency, n)
    bob_bits = _reconstruct_bits(built.bob_detector.get_photon_times(), bob_delay, frequency, n)

    labels = run_config.sifting.basis_labels
    a_bases: list[str] = []
    b_bases: list[str] = []
    a_out: list[int] = []
    b_out: list[int] = []
    periods: list[int] = []
    for i in range(n):
        if alice_bits[i] != -1 and bob_bits[i] != -1:  # coincidence
            a_bases.append(labels[int(alice_idx[i])])
            b_bases.append(labels[int(bob_idx[i])])
            a_out.append(alice_bits[i])
            b_out.append(bob_bits[i])
            periods.append(i)

    transcript = TrialTranscript(trial_id=trial_id, route_id=None,
                                 route_path=built.node_path, scope=run_config.scope)
    transcript.generation_attempts = n
    transcript.generation_successes = len(periods)
    transcript.alice_bases = tuple(a_bases)
    transcript.bob_bases = tuple(b_bases)
    transcript.alice_outcomes = tuple(a_out)
    transcript.bob_outcomes = tuple(b_out)
    transcript.latency_ps = bob_delay
    transcript.extra["coincidence_periods"] = tuple(periods)
    transcript.extra["num_periods"] = n
    apply_postprocessing(transcript, run_config.sifting)
    return transcript, built
