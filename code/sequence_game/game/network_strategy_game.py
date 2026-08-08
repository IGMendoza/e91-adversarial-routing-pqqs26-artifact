"""General route-vs-attack control game for generated repeater graphs.

This module advances the graph-level research premise up to the XAI boundary:
ER/generated topology with Alice/Bob roles, stable route candidates, Eve actions
as ``[target, attack_type]`` pairs, deterministic cautious-greedy Alice, optional
tabular Q-learning for Alice/Eve, and public-observation-only learning.

It is intentionally not a hardware simulator. Repeater-backed SeQUeNCe execution
remains in ``protocol.repeater_trial`` / ``game.repeater_game``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..analysis.periods import detect_route_period
from ..claims import CONTROL_GAME
from ..rl.q_learning import QLearningConfig, QTable, epsilon_greedy
from ..routing.features import route_feature_table
from ..routing.route import NoRouteError, Route, enumerate_simple_paths, make_route
from ..topology.ir import TopologyIR

NETWORK_CONTROL_SCOPE = CONTROL_GAME
ATTACK_TYPES = ("no_attack", "node_collision", "edge_collision")
ALICE_POLICIES = ("cautious_greedy", "q_learning")
EVE_POLICIES = ("q_learning", "random", "route_aware_oracle", "fixed")


class NetworkStrategyGameError(ValueError):
    """Invalid graph strategy game configuration or state."""


@dataclass(frozen=True)
class NetworkAttackAction:
    action_id: str
    target_kind: str
    target_id: str
    attack_type: str
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "attack_type": self.attack_type,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class NetworkStrategyConfig:
    alice: str
    bob: str
    num_steps: int = 3000
    seed: int = 1
    alice_policy: str = "cautious_greedy"
    eve_policy: str = "q_learning"
    fixed_eve_action_id: str = "no_attack"
    attack_types: tuple[str, ...] = ("no_attack", "node_collision", "edge_collision")
    avoidance_horizon: int = 2
    attack_cost: float = 0.05
    fallback_penalty: float = 0.2
    alpha: float = 0.2
    gamma: float = 0.9
    alice_epsilon_start: float = 0.3
    eve_epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    final_window_size: int = 20

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise NetworkStrategyGameError("num_steps must be >= 1")
        if self.alice_policy not in ALICE_POLICIES:
            raise NetworkStrategyGameError(f"unsupported Alice policy {self.alice_policy!r}")
        if self.eve_policy not in EVE_POLICIES:
            raise NetworkStrategyGameError(f"unsupported Eve policy {self.eve_policy!r}")
        bad = [attack for attack in self.attack_types if attack not in ATTACK_TYPES]
        if bad:
            raise NetworkStrategyGameError(f"unsupported attack types {bad}")
        if self.attack_cost < 0:
            raise NetworkStrategyGameError("attack_cost must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alice": self.alice,
            "bob": self.bob,
            "num_steps": self.num_steps,
            "seed": self.seed,
            "alice_policy": self.alice_policy,
            "eve_policy": self.eve_policy,
            "fixed_eve_action_id": self.fixed_eve_action_id,
            "attack_types": list(self.attack_types),
            "avoidance_horizon": self.avoidance_horizon,
            "attack_cost": self.attack_cost,
            "fallback_penalty": self.fallback_penalty,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "alice_epsilon_start": self.alice_epsilon_start,
            "eve_epsilon_start": self.eve_epsilon_start,
            "epsilon_end": self.epsilon_end,
            "final_window_size": self.final_window_size,
            "scope": NETWORK_CONTROL_SCOPE,
            "information_gain_reward_enabled": False,
        }


@dataclass(frozen=True)
class NetworkStepRecord:
    step: int
    alice_route_id: str
    alice_path: tuple[str, ...]
    eve_action: NetworkAttackAction
    public_outcome: str
    collision: bool
    alice_reward: float
    eve_reward: float
    fallback_used: bool = False
    route_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        quality_metrics = {"qber": None, "fidelity": None, "delivery_success": None}
        return {
            "step": self.step,
            "mode": "network_strategy_control",
            "scope_label": NETWORK_CONTROL_SCOPE,
            "alice_route_id": self.alice_route_id,
            "alice_path": list(self.alice_path),
            "eve_action": self.eve_action.to_dict(),
            "public_outcome": self.public_outcome,
            "collision": self.collision,
            "collision_or_delivery_failure": self.collision,
            "alice_reward": self.alice_reward,
            "eve_reward": self.eve_reward,
            "fallback_used": self.fallback_used,
            "route_features": dict(self.route_features),
            "quality_metrics": quality_metrics,
            "metrics": {
                "scope_label": NETWORK_CONTROL_SCOPE,
                "information_gain_reward_enabled": False,
                **quality_metrics,
            },
        }


@dataclass
class NetworkStrategyResult:
    config: NetworkStrategyConfig
    routes: list[Route]
    actions: list[NetworkAttackAction]
    steps: list[NetworkStepRecord] = field(default_factory=list)
    alice_q_table: Optional[QTable] = None
    eve_q_table: Optional[QTable] = None

    @property
    def final_window(self) -> list[NetworkStepRecord]:
        return self.steps[-self.config.final_window_size:]

    def summary(self) -> dict[str, Any]:
        route_ids = [step.alice_route_id for step in self.steps if step.alice_route_id]
        collisions = sum(1 for step in self.steps if step.collision)
        switches = sum(1 for a, b in zip(route_ids, route_ids[1:]) if a != b)
        return {
            "scope": NETWORK_CONTROL_SCOPE,
            "config": self.config.to_dict(),
            "routes": [route.to_dict() for route in self.routes],
            "actions": [action.to_dict() for action in self.actions],
            "final_window": [step.to_dict() for step in self.final_window],
            "metrics": {
                "collision_rate": collisions / len(self.steps) if self.steps else 0.0,
                "route_entropy": _route_entropy(route_ids),
                "route_switches": switches,
                "detected_period": detect_route_period(route_ids),
                "information_gain_reward_enabled": False,
            },
            "q_tables": {
                "alice": _q_summary(
                    self.alice_q_table,
                    [{"action_id": route.route_id,
                      "attack_type": "route",
                      "target_id": list(route.path)}
                     for route in self.routes],
                ) if self.alice_q_table is not None else {},
                "eve": _q_summary(
                    self.eve_q_table,
                    [action.to_dict() for action in self.actions],
                ) if self.eve_q_table is not None else {},
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "steps": [step.to_dict() for step in self.steps],
            "final_window": [step.to_dict() for step in self.final_window],
        }


def candidate_repeater_routes(
        topology: TopologyIR,
        alice: str,
        bob: str,
) -> list[Route]:
    """Return every simple cycle-free route in stable order."""

    topology.validate()
    if alice == bob:
        raise NetworkStrategyGameError("Alice and Bob must be distinct")
    paths = enumerate_simple_paths(topology, alice, bob)
    routes = [make_route(topology, path) for path in paths]
    routes.sort(key=lambda r: (r.total_length_m, r.hop_count, r.path))
    return routes


def build_network_attack_actions(
        topology: TopologyIR,
        alice: str,
        bob: str,
        *,
        attack_types: tuple[str, ...] = ("no_attack", "node_collision", "edge_collision"),
        attack_cost: float = 0.05,
) -> list[NetworkAttackAction]:
    """Build Eve's `[target, attack_type]` action set.

    Alice/Bob endpoints are forbidden node targets. Edges are all eligible in
    this control model.
    """

    topology.validate()
    if alice not in topology.nodes or bob not in topology.nodes:
        raise NetworkStrategyGameError("Alice/Bob endpoints must exist in topology")
    actions: list[NetworkAttackAction] = []
    if "no_attack" in attack_types:
        actions.append(NetworkAttackAction("no_attack", "none", "", "no_attack", 0.0))
    if "node_collision" in attack_types:
        for node_id in sorted(nid for nid in topology.nodes if nid not in {alice, bob}):
            actions.append(NetworkAttackAction(
                f"node_collision:{node_id}", "node", node_id,
                "node_collision", attack_cost))
    if "edge_collision" in attack_types:
        for edge in sorted(topology.edges, key=lambda edge: edge.edge_id):
            actions.append(NetworkAttackAction(
                f"edge_collision:{edge.edge_id}", "edge", edge.edge_id,
                "edge_collision", attack_cost))
    if not actions:
        actions.append(NetworkAttackAction("no_attack", "none", "", "no_attack", 0.0))
    return actions


def run_network_strategy_game(topology: TopologyIR,
                              config: NetworkStrategyConfig) -> NetworkStrategyResult:
    """Run the graph-level strategic control game."""

    routes = candidate_repeater_routes(topology, config.alice, config.bob)
    if not routes:
        raise NoRouteError(f"no route from {config.alice!r} to {config.bob!r}")
    actions = build_network_attack_actions(
        topology, config.alice, config.bob,
        attack_types=config.attack_types, attack_cost=config.attack_cost)
    if config.eve_policy == "fixed" and config.fixed_eve_action_id not in {
        action.action_id for action in actions
    }:
        raise NetworkStrategyGameError(
            f"fixed_eve_action_id {config.fixed_eve_action_id!r} is not in Eve action space")
    feature_by_route = {
        row.route_id: row.to_dict()
        for row in route_feature_table(topology, routes, alice=config.alice, bob=config.bob)
    }

    rng = np.random.default_rng(config.seed)
    q_config = QLearningConfig(config.alpha, config.gamma)
    alice_q = QTable(len(routes))
    eve_q = QTable(len(actions))
    result = NetworkStrategyResult(
        config=config,
        routes=routes,
        actions=actions,
        alice_q_table=alice_q if config.alice_policy == "q_learning" else None,
        eve_q_table=eve_q if config.eve_policy == "q_learning" else None,
    )
    avoided_until: dict[str, int] = {}
    previous_route_id = "start"
    previous_action_id = "start"
    previous_outcome = "start"

    for step in range(config.num_steps):
        route, route_index, fallback, alice_state = _select_alice_route(
            routes, alice_q, config, step, rng, avoided_until,
            previous_route_id, previous_outcome)
        action, action_index, eve_state = _select_eve_action(
            actions, eve_q, config, step, rng, route,
            previous_action_id, previous_outcome)
        hit = _collides(topology, route, action)
        public_outcome = "hit" if hit else "miss"
        alice_reward = -1.0 if hit else 1.0
        if fallback:
            alice_reward -= config.fallback_penalty
        eve_reward = (1.0 if hit else 0.0) - action.cost

        if hit and config.alice_policy == "cautious_greedy":
            avoided_until[route.route_id] = step + max(0, config.avoidance_horizon)

        if config.eve_policy == "q_learning":
            eve_q.update(
                eve_state, action_index, eve_reward,
                (action.action_id, public_outcome), q_config)
        if config.alice_policy == "q_learning":
            next_avoid = _avoidance_signature(routes, avoided_until, step)
            alice_q.update(
                alice_state, route_index, alice_reward,
                (route.route_id, public_outcome, next_avoid), q_config)

        result.steps.append(NetworkStepRecord(
            step=step,
            alice_route_id=route.route_id,
            alice_path=route.path,
            eve_action=action,
            public_outcome=public_outcome,
            collision=hit,
            alice_reward=alice_reward,
            eve_reward=eve_reward,
            fallback_used=fallback,
            route_features=feature_by_route[route.route_id],
        ))
        previous_route_id = route.route_id
        previous_action_id = action.action_id
        previous_outcome = public_outcome

    return result


def _select_alice_route(routes: list[Route], alice_q: QTable,
                        config: NetworkStrategyConfig, step: int,
                        rng: np.random.Generator,
                        avoided_until: dict[str, int],
                        previous_route_id: str,
                        previous_outcome: str
                        ) -> tuple[Route, int, bool, tuple[str, ...]]:
    signature = _avoidance_signature(routes, avoided_until, step)
    state = (previous_route_id, previous_outcome, signature)
    if config.alice_policy == "q_learning":
        epsilon = _linear_epsilon(
            config.alice_epsilon_start, config.epsilon_end, step, config.num_steps)
        index = epsilon_greedy(alice_q, state, epsilon, rng)
        return routes[index], index, False, state
    available = [
        route for route in routes
        if step > avoided_until.get(route.route_id, -1)
    ]
    fallback = not available
    route = (available or routes)[0]
    return route, routes.index(route), fallback, state


def _select_eve_action(actions: list[NetworkAttackAction], eve_q: QTable,
                       config: NetworkStrategyConfig, step: int,
                       rng: np.random.Generator, route: Route,
                       previous_action_id: str, previous_outcome: str
                       ) -> tuple[NetworkAttackAction, int, tuple[str, ...]]:
    if config.eve_policy == "route_aware_oracle":
        hits = [action for action in actions if _collides_action(route, action)]
        action = hits[0] if hits else actions[0]
        return action, actions.index(action), ("privileged_route", route.route_id)

    state = (previous_action_id, previous_outcome)
    if config.eve_policy == "random":
        index = int(rng.integers(len(actions)))
        return actions[index], index, state
    if config.eve_policy == "fixed":
        for index, action in enumerate(actions):
            if action.action_id == config.fixed_eve_action_id:
                return action, index, state
        raise NetworkStrategyGameError(
            f"fixed_eve_action_id {config.fixed_eve_action_id!r} is not in Eve action space")

    epsilon = _linear_epsilon(
        config.eve_epsilon_start, config.epsilon_end, step, config.num_steps)
    index = epsilon_greedy(eve_q, state, epsilon, rng)
    return actions[index], index, state


def _collides(topology: TopologyIR, route: Route, action: NetworkAttackAction) -> bool:
    if action.attack_type == "edge_collision" and action.target_kind == "edge":
        return action.target_id in route.edge_ids
    if action.attack_type == "node_collision" and action.target_kind == "node":
        return action.target_id in route.path[1:-1]
    return False


def _collides_action(route: Route, action: NetworkAttackAction) -> bool:
    if action.target_kind == "edge":
        return action.target_id in route.edge_ids
    if action.target_kind == "node":
        return action.target_id in route.path[1:-1]
    return False


def _avoidance_signature(routes: list[Route], avoided_until: dict[str, int],
                         step: int) -> str:
    return "".join(
        "1" if step <= avoided_until.get(route.route_id, -1) else "0"
        for route in routes
    )


def _linear_epsilon(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    t = min(step, total_steps - 1) / (total_steps - 1)
    return float(start + (end - start) * t)


def _route_entropy(route_ids: list[str]) -> float:
    if not route_ids:
        return 0.0
    total = len(route_ids)
    counts = Counter(route_ids)
    return -sum((count / total) * math.log2(count / total)
                for count in counts.values())


def _q_summary(table: Optional[QTable], actions: list[dict[str, Any]],
               limit: int = 8) -> dict[str, Any]:
    if table is None:
        return {}
    scores = [0.0] * len(actions)
    for values in getattr(table, "_q", {}).values():
        for i, value in enumerate(values):
            scores[i] = max(scores[i], float(value))
    ranked = sorted(range(len(actions)), key=lambda i: (-scores[i], actions[i]["action_id"]))
    return {
        "top_actions": [
            {
                **actions[i],
                "q": round(scores[i], 4),
            }
            for i in ranked[:limit]
        ],
        "entries": table.to_dict()["entries"],
    }
