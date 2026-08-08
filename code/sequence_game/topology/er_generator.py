"""Reproducible Erdős–Rényi G(n, p) topology generation onto the topology IR.

Purely graph-structural. Edge lengths are explicit configuration (fixed value
or seeded uniform sampling within a user-given range); they carry no physical
claim until a scope-labelled channel model consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ir import EdgeRecord, NodeRecord, TopologyError, TopologyIR, TopologyMetadata


@dataclass(frozen=True)
class ERTopologyConfig:
    n: int
    p: float
    fixed_edge_length_m: Optional[float] = None
    edge_length_range_m: Optional[tuple[float, float]] = None
    require_connected: bool = False
    max_retries: int = 100

    def __post_init__(self) -> None:
        if self.n < 2:
            raise TopologyError("n must be >= 2")
        if not 0.0 <= self.p <= 1.0:
            raise TopologyError(f"p must be in [0, 1], got {self.p}")
        has_fixed = self.fixed_edge_length_m is not None
        has_range = self.edge_length_range_m is not None
        if has_fixed == has_range:
            raise TopologyError(
                "exactly one of fixed_edge_length_m / edge_length_range_m must be set")
        if has_fixed and self.fixed_edge_length_m < 0:
            raise TopologyError("fixed_edge_length_m must be >= 0")
        if has_range:
            lo, hi = self.edge_length_range_m
            if not 0 <= lo <= hi:
                raise TopologyError(f"invalid edge_length_range_m {self.edge_length_range_m}")
        if self.max_retries < 1:
            raise TopologyError("max_retries must be >= 1")

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "p": self.p,
            "fixed_edge_length_m": self.fixed_edge_length_m,
            "edge_length_range_m": list(self.edge_length_range_m)
            if self.edge_length_range_m else None,
            "require_connected": self.require_connected,
            "max_retries": self.max_retries,
        }


def _sample_graph(config: ERTopologyConfig, rng: np.random.Generator,
                  seed: Optional[int], attempt: int) -> TopologyIR:
    node_ids = [f"n{i}" for i in range(config.n)]
    nodes = {nid: NodeRecord(nid) for nid in node_ids}
    edges: list[EdgeRecord] = []
    for i in range(config.n):
        for j in range(i + 1, config.n):
            if rng.random() < config.p:
                if config.fixed_edge_length_m is not None:
                    length = float(config.fixed_edge_length_m)
                else:
                    lo, hi = config.edge_length_range_m
                    length = float(rng.uniform(lo, hi))
                edges.append(EdgeRecord(f"e{len(edges)}", node_ids[i], node_ids[j], length))
    params = config.to_dict()
    params["attempt"] = attempt
    ir = TopologyIR(nodes=nodes, edges=edges,
                    metadata=TopologyMetadata("erdos_renyi_gnp", params, seed))
    ir.validate()
    return ir


def generate_er_topology(config: ERTopologyConfig, seed: int) -> TopologyIR:
    """Generate a G(n, p) topology; deterministic for a given (config, seed)."""
    rng = np.random.default_rng(seed)
    last = None
    for attempt in range(config.max_retries):
        last = _sample_graph(config, rng, seed, attempt)
        if not config.require_connected or last.is_connected():
            return last
    raise TopologyError(
        f"no connected G(n={config.n}, p={config.p}) graph found in "
        f"{config.max_retries} attempts (seed={seed})")
