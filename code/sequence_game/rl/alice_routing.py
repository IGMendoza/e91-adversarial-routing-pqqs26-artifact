"""Optional tabular Q-learning routing policy for Alice.

Strictly separated from Eve's learner: Alice's state is built only from
information available to Alice (her previous route choice and the public
outcome of the previous trial, passed in via ``context``). There is no access
path to Eve's Q-table, actions, or targets through this class.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..claims import CONTROL_GAME, IMPLEMENTED_CONTROL_ONLY
from ..routing.policy import RoutingPolicy
from ..routing.route import NoRouteError, Route, enumerate_simple_paths, make_route
from ..topology.ir import TopologyIR
from .q_learning import QLearningConfig, QTable, epsilon_greedy

#: context keys Alice is allowed to consume; anything else is ignored,
#: so Eve-private data cannot influence the state even if passed by mistake.
ALICE_CONTEXT_KEYS = ("previous_outcome", "previous_route_id")


class QLearningRoutingPolicy(RoutingPolicy):
    """Epsilon-greedy choice among the simple candidate routes between the
    fixed endpoints; updated from public trial outcomes via ``update``."""

    name = "q_learning_routing"

    def __init__(self, topology: TopologyIR, source: str, target: str, *,
                 alpha: float, gamma: float, epsilon: float):
        paths = enumerate_simple_paths(topology, source, target)
        if not paths:
            raise NoRouteError(f"no candidate routes from {source!r} to {target!r}")
        # deterministic candidate ordering = action index mapping
        self.candidate_routes: list[Route] = [make_route(topology, p) for p in paths]
        self.source, self.target = source, target
        self.epsilon = epsilon
        self.q_config = QLearningConfig(alpha=alpha, gamma=gamma)
        self.q_table = QTable(num_actions=len(self.candidate_routes))
        self._route_index = {r.route_id: i for i, r in enumerate(self.candidate_routes)}
        self._last_state: Optional[tuple[str, ...]] = None
        self._last_action: Optional[int] = None

    @property
    def action_route_ids(self) -> tuple[str, ...]:
        return tuple(route.route_id for route in self.candidate_routes)

    @staticmethod
    def state_from_context(context: Optional[dict[str, Any]]) -> tuple[str, ...]:
        context = context or {}
        return tuple(str(context.get(k, "unknown")) for k in ALICE_CONTEXT_KEYS)

    def select_route(self, topology: TopologyIR, source: str, target: str, *,
                     rng: Optional[np.random.Generator] = None,
                     context: Optional[dict[str, Any]] = None) -> Route:
        if (source, target) != (self.source, self.target):
            raise NoRouteError(
                f"policy trained for {self.source!r}->{self.target!r}, "
                f"asked for {source!r}->{target!r}")
        if rng is None:
            raise ValueError(f"{self.name} requires an explicit rng")
        state = self.state_from_context(context)
        action = epsilon_greedy(self.q_table, state, self.epsilon, rng)
        self._last_state, self._last_action = state, action
        return self.candidate_routes[action]

    def update(self, route: Route, outcome: dict[str, Any]) -> None:
        """Q-update from the public outcome of the trial that used ``route``.

        ``outcome`` must contain a numeric ``reward`` and may contain the
        public context for the next state.
        """
        if self._last_state is None or self._last_action is None:
            return
        if route.route_id != self.candidate_routes[self._last_action].route_id:
            raise NoRouteError("update() route does not match the last selected route")
        reward = float(outcome["reward"])
        next_state = self.state_from_context(outcome.get("next_context"))
        self.q_table.update(self._last_state, self._last_action, reward,
                            next_state, self.q_config)

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "scope_label": CONTROL_GAME,
            "claim_status": IMPLEMENTED_CONTROL_ONLY,
            "num_candidate_routes": len(self.candidate_routes),
            "action_route_ids": list(self.action_route_ids),
            "state_keys": list(ALICE_CONTEXT_KEYS),
            "reward_scope": "success_failure_quality_only",
            "epsilon": self.epsilon,
            **self.q_config.to_dict(),
        }
