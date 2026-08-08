"""Eve's edge attacks against the SeQUeNCe E91/BBM92 control.

This module is the *strategy/wiring* layer: it has no ``sequence`` import. It maps
an ``EdgeAttackSpec`` (which fiber edge, what kind of attack) onto (a) an expanded
hop list that inserts an Eve node at the far end of the attacked fiber, and (b) a
``station_factory`` that places the matching hardware station (from
``sequence_build.eve_stations``) on that Eve node.

Attack model (consistent with the project framing -- Alice/Bob nodes forbidden,
fiber edges fair game):

- ``none``: no attack.
- ``added_loss``: availability/DoS; drop photons on the attacked fiber.
- ``intercept_resend``: measure-and-resend on the attacked fiber; raises QBER and
  yields Eve a classical record for the information-gain metric.

The Eve node sits on the *fiber*, never inside Alice's or Bob's node. The attacked
edge ``u-v`` (length L) becomes ``u ->[fiber L]-> eve ->[zero-length lossless]-> v``
so the photon still experiences the full physical loss/delay of the attacked span
before Eve acts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..physical.registry import PhysicalModel
from ..sequence_build.e91_builder import E91Hop, StationFactory
from ..sequence_build.eve_stations import AddedLossStation, InterceptResendStation
from ..topology.ir import TopologyIR

ATTACK_KINDS = ("none", "added_loss", "intercept_resend")
#: Active attack kinds Eve may choose in the strategic game ("none" is baseline).
SUPPORTED_GAME_KINDS = ("intercept_resend", "added_loss")


class SequenceAttackError(ValueError):
    """Invalid edge-attack specification or target."""


@dataclass(frozen=True)
class EdgeAttackSpec:
    """Which fiber edge Eve attacks and how."""

    kind: str
    target_u: str = ""
    target_v: str = ""
    drop_probability: float = 1.0      # added_loss: 1.0 = full denial
    basis_choice: str = "random"       # intercept_resend: "random" | "Z" | "X"
    cost: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ATTACK_KINDS:
            raise SequenceAttackError(f"unknown attack kind {self.kind!r}; {ATTACK_KINDS}")
        if self.kind != "none" and not (self.target_u and self.target_v):
            raise SequenceAttackError(f"{self.kind} attack needs a target edge (u, v)")
        if self.target_u and self.target_u == self.target_v:
            raise SequenceAttackError("target edge cannot be a self-loop")
        if not 0.0 <= self.drop_probability <= 1.0:
            raise SequenceAttackError("drop_probability must be in [0, 1]")
        if self.basis_choice not in ("random", "Z", "X"):
            raise SequenceAttackError("basis_choice must be 'random'|'Z'|'X'")
        if self.cost < 0:
            raise SequenceAttackError("cost must be >= 0")

    @property
    def is_active(self) -> bool:
        return self.kind != "none"

    @property
    def target_edge(self) -> frozenset:
        return frozenset((self.target_u, self.target_v))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target_u": self.target_u, "target_v": self.target_v,
                "drop_probability": self.drop_probability, "basis_choice": self.basis_choice,
                "cost": self.cost}


def edge_on_path(node_path: list[str], u: str, v: str) -> bool:
    """True if ``u-v`` is a consecutive (undirected) hop of ``node_path``."""
    pair = frozenset((u, v))
    return any(frozenset((a, b)) == pair for a, b in zip(node_path, node_path[1:]))


def spec_active_on_path(node_path: list[str], spec: EdgeAttackSpec) -> bool:
    """True if the attack is active and its target edge lies on this route.

    The strategic game lets Eve pick any eve-eligible edge; one off Alice's chosen
    route simply has no effect (a wasted attack), which callers model by skipping
    the Eve insertion when this returns False."""
    return spec.is_active and edge_on_path(node_path, spec.target_u, spec.target_v)


def _lossless_link_model(fiber_model: PhysicalModel) -> PhysicalModel:
    return PhysicalModel(
        "eve_resend_link", "fiber_channel", "toy",
        "Eve resend link: zero-length lossless hop from Eve's station to the next node",
        {"attenuation": 0.0, "polarization_fidelity": 1.0,
         "light_speed": float(fiber_model.parameters["light_speed"]),
         "frequency": float(fiber_model.parameters["frequency"])})


def build_attacked_hops(ir: TopologyIR, node_path: list[str], fiber_model: PhysicalModel,
                        spec: EdgeAttackSpec, *, eve_name: str = "eve"
                        ) -> tuple[list[E91Hop], tuple[str, ...]]:
    """Expand a route into hops, inserting an Eve node on the attacked edge.

    If the attack is inactive or its target edge is not on ``node_path``, returns
    the plain route hops with no Eve node."""
    if len(node_path) < 2:
        raise SequenceAttackError("node_path must have at least 2 nodes")
    if eve_name in node_path:
        raise SequenceAttackError(f"eve_name {eve_name!r} collides with a route node")

    apply_here = spec_active_on_path(node_path, spec)
    hops: list[E91Hop] = []
    eve_nodes: tuple[str, ...] = ()
    for u, v in zip(node_path, node_path[1:]):
        edge = ir.edge_between(u, v)
        if edge is None:
            raise SequenceAttackError(f"no edge between {u!r} and {v!r}")
        if apply_here and frozenset((u, v)) == spec.target_edge:
            hops.append(E91Hop(u, eve_name, edge.length_m, fiber_model))
            hops.append(E91Hop(eve_name, v, 0.0, _lossless_link_model(fiber_model)))
            eve_nodes = (eve_name,)
        else:
            hops.append(E91Hop(u, v, edge.length_m, fiber_model))
    return hops, eve_nodes


def make_station_factory(spec: EdgeAttackSpec, *, eve_name: str = "eve"
                         ) -> Optional[StationFactory]:
    """Station factory placing Eve's hardware on the inserted Eve node, or None
    when there is no active attack."""
    if not spec.is_active:
        return None

    def factory(node_name: str, node, dest_name: str, timeline):
        if node_name != eve_name:
            return None
        if spec.kind == "added_loss":
            return AddedLossStation(f"{eve_name}.dos", timeline, node, dest_name,
                                    drop_probability=spec.drop_probability)
        if spec.kind == "intercept_resend":
            return InterceptResendStation(f"{eve_name}.ir", timeline, node, dest_name,
                                          basis_choice=spec.basis_choice)
        return None

    return factory
