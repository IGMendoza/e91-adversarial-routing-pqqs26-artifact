"""Routing policy interface for Alice route selection.

Policies see only the topology IR and the optional public context they are
given; they must never receive Eve-private state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from ..topology.ir import TopologyIR
from .route import Route


class RoutingPolicy(ABC):
    """Selects a route from source to target on a topology IR."""

    name: str = "abstract"

    @abstractmethod
    def select_route(self, topology: TopologyIR, source: str, target: str, *,
                     rng: Optional[np.random.Generator] = None,
                     context: Optional[dict[str, Any]] = None) -> Route:
        """Return a Route or raise NoRouteError. Stochastic policies require
        an explicit rng."""

    def update(self, route: Route, outcome: dict[str, Any]) -> None:
        """Hook for adaptive policies; default is a no-op."""

    def metadata(self) -> dict[str, Any]:
        return {"policy": self.name}
