"""Network + detector for the 3-angle CHSH / Ekert-91 variant.

Unlike the BBM92 control (2 fixed Z/X bases via ``QSDetectorPolarization``), the
Ekert-91 / CHSH protocol measures each arm at one of several *angles*. SeQUeNCe's
``Photon.measure`` accepts an arbitrary orthonormal basis, so ``AngleMeasurementDetector``
measures at polarization angle ``theta`` directly (basis ``{(cosθ,sinθ),(-sinθ,cosθ)}``),
applying the detector model's efficiency. The entangled physics and fiber
transport are SeQUeNCe's; the angle sets and CHSH statistic (computed in
``protocol.chsh_trial``) are textbook, not invented.

A direct Alice–Bob link is built (optionally with an Eve intercept-resend station
on the fiber); multi-hop relays are a straightforward extension of the same
pattern but are not needed to exhibit the Bell-inequality test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Optional, Sequence

from sequence.components.photon import Photon
from sequence.kernel.entity import Entity
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from ..physical.registry import PhysicalModel
from .device_adapters import create_fiber, create_source
from .eve_stations import InterceptResendStation
from .relay import PhotonForwarder


class AngleMeasurementDetector(Entity):
    """Measure each arriving photon at a per-period polarization angle, applying
    detector efficiency. Records ``period -> outcome (0/1)`` for detected photons
    (no dark-count model: CHSH here is a correlation demonstration)."""

    def __init__(self, name: str, timeline: "Timeline", node: "Node", frequency_hz: float,
                 angles_rad: Sequence[float], efficiency: float):
        super().__init__(name, timeline)
        self.node = node
        self.frequency_hz = frequency_hz
        self.angles_rad = list(angles_rad)
        self.efficiency = efficiency
        self.start_offset_ps = 0  # set to the arm's propagation delay after init
        self.records: dict[int, int] = {}
        self._hit: set[int] = set()  # periods that received any photon (for ambiguity)

    def init(self) -> None:
        pass

    def get(self, photon: "Photon", **kwargs) -> None:
        period = round((self.timeline.now() - self.start_offset_ps) * self.frequency_hz * 1e-12)
        if period < 0 or period >= len(self.angles_rad):
            return
        if period in self._hit:
            # multi-photon period (multi-pair emission): ambiguous -> discard.
            self.records.pop(period, None)
            return
        self._hit.add(period)
        rng = self.node.get_generator()
        if rng.random() >= self.efficiency:
            return  # photon not detected
        theta = self.angles_rad[period]
        basis = ((complex(cos(theta)), complex(sin(theta))),
                 (complex(-sin(theta)), complex(cos(theta))))
        self.records[period] = int(Photon.measure(basis, photon, rng))


@dataclass
class BuiltCHSHLink:
    timeline: Timeline
    source: object
    alice_meas: AngleMeasurementDetector
    bob_meas: AngleMeasurementDetector
    bob_arm_delay_ps: int
    eve_station: Optional[InterceptResendStation]


def _lossless_model(fiber_model: PhysicalModel) -> PhysicalModel:
    return PhysicalModel("chsh_eve_link", "fiber_channel", "toy", "Eve resend link",
                         {"attenuation": 0.0, "polarization_fidelity": 1.0,
                          "light_speed": float(fiber_model.parameters["light_speed"]),
                          "frequency": float(fiber_model.parameters["frequency"])})


def build_chsh_link(source_model: PhysicalModel, detector_model: PhysicalModel,
                    fiber_model: PhysicalModel, length_m: float, *,
                    alice_angles_rad: Sequence[float], bob_angles_rad: Sequence[float],
                    stop_time_ps: int, seed: int,
                    eve_intercept: bool = False) -> BuiltCHSHLink:
    timeline = Timeline(stop_time_ps)
    freq = float(source_model.parameters["frequency"])
    efficiency = float(detector_model.parameters["efficiency"])
    pol_fid = float(fiber_model.parameters["polarization_fidelity"])

    alice = Node("alice", timeline)
    alice.set_seed(seed)
    bob = Node("bob", timeline)
    bob.set_seed(seed + 1)

    source = create_source(source_model, "alice.spdc", timeline)
    alice.add_component(source)
    alice_meas = AngleMeasurementDetector("alice.meas", timeline, alice, freq,
                                          alice_angles_rad, efficiency)
    alice.add_component(alice_meas)

    eve_station: Optional[InterceptResendStation] = None
    qchannels = []
    if eve_intercept:
        eve = Node("eve", timeline)
        eve.set_seed(seed + 2)
        bob_arm = PhotonForwarder("alice.bobarm", timeline, alice, "eve",
                                  polarization_fidelity=pol_fid)
        alice.add_component(bob_arm)
        qc1 = create_fiber(fiber_model, "qc.alice->eve", timeline, length_m,
                           polarization_fidelity=1.0)
        qc1.set_ends(alice, "eve")
        eve_station = InterceptResendStation("eve.ir", timeline, eve, "bob",
                                             basis_choice="random")
        eve.add_component(eve_station)
        eve.set_first_component(eve_station.name)
        qc2 = create_fiber(_lossless_model(fiber_model), "qc.eve->bob", timeline, 0.0,
                           polarization_fidelity=1.0)
        qc2.set_ends(eve, "bob")
        qchannels = [qc1, qc2]
    else:
        bob_arm = PhotonForwarder("alice.bobarm", timeline, alice, "bob",
                                  polarization_fidelity=pol_fid)
        alice.add_component(bob_arm)
        qc1 = create_fiber(fiber_model, "qc.alice->bob", timeline, length_m,
                           polarization_fidelity=1.0)
        qc1.set_ends(alice, "bob")
        qchannels = [qc1]

    source.add_receiver(alice_meas)  # Alice's local arm
    source.add_receiver(bob_arm)     # Bob's arm onto the fiber

    bob_meas = AngleMeasurementDetector("bob.meas", timeline, bob, freq,
                                        bob_angles_rad, efficiency)
    bob.add_component(bob_meas)
    bob.set_first_component(bob_meas.name)

    timeline.init()
    bob_delay = sum(qc.delay for qc in qchannels)
    bob_meas.start_offset_ps = bob_delay
    return BuiltCHSHLink(timeline=timeline, source=source, alice_meas=alice_meas,
                         bob_meas=bob_meas, bob_arm_delay_ps=bob_delay,
                         eve_station=eve_station)
