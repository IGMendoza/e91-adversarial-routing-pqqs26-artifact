"""Named graph fixtures for graph-game control tests.

These fixtures are deterministic topology-IR builders.  They are not hardware
adapters and do not make physical or security claims.
"""

from __future__ import annotations

from typing import Iterable

from ..claims import CONTROL_GAME, IMPLEMENTED_CONTROL_ONLY
from .er_generator import ERTopologyConfig, generate_er_topology
from .ir import EdgeRecord, NodeRecord, TopologyError, TopologyIR, TopologyMetadata

ER_CONNECTED = "ER_CONNECTED"
TWO_CORRIDOR = "TWO_CORRIDOR"
DIAMOND = "DIAMOND"
LADDER = "LADDER"
LINE_CONTROL = "LINE_CONTROL"
BOTTLENECK_CONTROL = "BOTTLENECK_CONTROL"

GRAPH_FAMILIES = frozenset({
    ER_CONNECTED,
    TWO_CORRIDOR,
    DIAMOND,
    LADDER,
    LINE_CONTROL,
    BOTTLENECK_CONTROL,
})

DEGENERATE_CONTROL_FAMILIES = frozenset({LINE_CONTROL, BOTTLENECK_CONTROL})

DEFAULT_ALICE = "a"
DEFAULT_BOB = "b"


def build_graph_fixture(family: str, *, seed: int = 1) -> TopologyIR:
    """Build a deterministic graph fixture by family name."""

    if family == ER_CONNECTED:
        ir = generate_er_topology(
            ERTopologyConfig(
                n=8,
                p=0.45,
                fixed_edge_length_m=1.0,
                require_connected=True,
                max_retries=500,
            ),
            seed=seed,
        )
        return with_graph_family_metadata(ir, family, seed=seed)
    if family == TWO_CORRIDOR:
        return _manual_fixture(
            family,
            nodes=("a", "u0", "u1", "v0", "v1", "b"),
            edges=(
                ("a", "u0", 1.0),
                ("u0", "u1", 1.0),
                ("u1", "b", 1.0),
                ("a", "v0", 1.0),
                ("v0", "v1", 1.0),
                ("v1", "b", 1.0),
            ),
            seed=seed,
        )
    if family == DIAMOND:
        return _manual_fixture(
            family,
            nodes=("a", "x", "y", "b"),
            edges=(
                ("a", "x", 1.0),
                ("x", "b", 1.0),
                ("a", "y", 1.0),
                ("y", "b", 1.0),
                ("x", "y", 2.0),
            ),
            seed=seed,
        )
    if family == LADDER:
        return _manual_fixture(
            family,
            nodes=("a", "t1", "t2", "l1", "l2", "b"),
            edges=(
                ("a", "t1", 1.0),
                ("t1", "t2", 1.0),
                ("t2", "b", 1.0),
                ("a", "l1", 1.0),
                ("l1", "l2", 1.0),
                ("l2", "b", 1.0),
                ("t1", "l1", 1.0),
                ("t2", "l2", 1.0),
            ),
            seed=seed,
        )
    if family == LINE_CONTROL:
        return _manual_fixture(
            family,
            nodes=("a", "r0", "r1", "b"),
            edges=(
                ("a", "r0", 1.0),
                ("r0", "r1", 1.0),
                ("r1", "b", 1.0),
            ),
            seed=seed,
        )
    if family == BOTTLENECK_CONTROL:
        return _manual_fixture(
            family,
            nodes=("a", "left", "right", "cut", "b"),
            edges=(
                ("a", "left", 1.0),
                ("left", "cut", 1.0),
                ("a", "right", 1.0),
                ("right", "cut", 1.0),
                ("cut", "b", 1.0),
            ),
            seed=seed,
        )
    raise TopologyError(f"unknown graph fixture family {family!r}")


def with_graph_family_metadata(ir: TopologyIR, family: str, *, seed: int | None = None) -> TopologyIR:
    if family not in GRAPH_FAMILIES:
        raise TopologyError(f"unknown graph family {family!r}")
    params = dict(ir.metadata.params)
    params.update({
        "graph_family": family,
        "scope_label": CONTROL_GAME,
        "claim_status": IMPLEMENTED_CONTROL_ONLY,
        "degenerate_control": family in DEGENERATE_CONTROL_FAMILIES,
        "benchmark_candidate": family not in DEGENERATE_CONTROL_FAMILIES,
    })
    ir.metadata = TopologyMetadata(ir.metadata.generator, params, seed if seed is not None else ir.metadata.seed)
    ir.validate()
    return ir


def graph_family(ir: TopologyIR) -> str:
    family = ir.metadata.params.get("graph_family")
    if family not in GRAPH_FAMILIES:
        raise TopologyError("topology metadata is missing a known graph_family")
    return str(family)


def is_degenerate_control(ir: TopologyIR) -> bool:
    return bool(ir.metadata.params.get("degenerate_control", False))


def _manual_fixture(family: str,
                    *,
                    nodes: Iterable[str],
                    edges: Iterable[tuple[str, str, float]],
                    seed: int) -> TopologyIR:
    node_records = {
        node_id: NodeRecord(node_id, _roles_for_node(node_id))
        for node_id in nodes
    }
    edge_records = [
        EdgeRecord(f"e{i}", u, v, length, eve_eligible=True)
        for i, (u, v, length) in enumerate(edges)
    ]
    ir = TopologyIR(
        nodes=node_records,
        edges=edge_records,
        metadata=TopologyMetadata("named_graph_fixture", {"graph_family": family}, seed),
    )
    ir.validate()
    return with_graph_family_metadata(ir, family, seed=seed)


def _roles_for_node(node_id: str) -> frozenset[str]:
    if node_id == DEFAULT_ALICE:
        return frozenset({"alice"})
    if node_id == DEFAULT_BOB:
        return frozenset({"bob"})
    return frozenset({"eve_eligible", "swap_candidate"})
