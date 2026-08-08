"""Backend model for the Dash GUI demo.

The binary game here is a toy/control collision model for visualizing routing
and attack-learning dynamics. The repeater function is a thin GUI-facing wrapper
around the current fixed selected-path SeQUeNCe E91 runner; unsupported topologies
return blocked responses rather than simulated physics.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Thread-safe run-progress state read by the GUI ticker callback.
_PROGRESS_LOCK = threading.Lock()
_TRAINING_PROGRESS: dict[str, Any] = {
    "running": False, "phase": "idle", "step": 0, "total": 0, "eve_algo": "",
}


def get_run_progress() -> dict[str, Any]:
    with _PROGRESS_LOCK:
        return dict(_TRAINING_PROGRESS)


def _update_progress(**kwargs: Any) -> None:
    with _PROGRESS_LOCK:
        _TRAINING_PROGRESS.update(kwargs)

import networkx as nx
import numpy as np

from ..claims import BLOCKED, CONTROL_GAME, REPEATER_RUNTIME
from ..eve.repeater_attacks import RepeaterAttackSpec
from ..physical import PhysicalModel
from ..protocol.postprocessing import SiftingConfig
from ..protocol.public_record import PublicStepRecord
from ..protocol.repeater_trial import RepeaterE91RunConfig, run_fixed_repeater_e91_trial
from ..rl.q_learning import QLearningConfig, QTable, epsilon_greedy
from ..routing.route import route_id_for_path
from ..sequence_build.repeater_e91_builder import FixedRepeaterPath
from ..topology import (
    ERTopologyConfig,
    EdgeRecord,
    NodeRecord,
    TopologyIR,
    generate_er_topology,
)

TOY_SCOPE = CONTROL_GAME
REPEATER_SCOPE = REPEATER_RUNTIME
DISABLED_SCOPE = BLOCKED
BLOCKED_ATTACK_TOOLTIP = "blocked: no endpoint transcript, detector model, or information metric implemented."
REPEATER_MOTIF_BLOCK = (
    "repeater-backed mode currently supports one selected simple Alice--repeaters--Bob path; "
    "arbitrary full-graph RouterNetTopo execution is not yet implemented"
)

DISABLED_ATTACKS = [
    {"id": "qnd_resend", "label": "QND/resend", "disabled": True,
     "tooltip": BLOCKED_ATTACK_TOOLTIP},
    {"id": "detector_blinding", "label": "Detector blinding", "disabled": True,
     "tooltip": BLOCKED_ATTACK_TOOLTIP},
    {"id": "time_shift", "label": "Detector time shift", "disabled": True,
     "tooltip": BLOCKED_ATTACK_TOOLTIP},
]

EVE_ALGO_OPTIONS = {
    "q_learning":         {"label": "tabular Q-learning"},
    "deep_q":             {"label": "deep Q-network (DQN)"},
    "random":             {"label": "random"},
    "route_aware_oracle": {"label": "route-aware oracle"},
}

ALICE_ALGO_OPTIONS = {
    "cautious_greedy": {"label": "cautious greedy"},
    "q_learning":      {"label": "tabular Q-learning"},
    "deep_q":          {"label": "deep Q-network (DQN)"},
    "random":          {"label": "random"},
}

_VALID_EVE_ALGOS   = set(EVE_ALGO_OPTIONS)
_VALID_ALICE_ALGOS = set(ALICE_ALGO_OPTIONS)

ACTION_SURFACE_MODELS = {
    "binary_collision": {
        "label": "binary collision model",
        "description": "pure hit/miss on Alice's selected route",
        "attack_kinds": ["node_collision", "edge_collision"],
        "reward_model": "hit_miss_only",
        "information_gain_reward_enabled": False,
    },
    "attack_surface_dos": {
        "label": "attack surface model 1: DoS only",
        "description": "node DoS and edge DoS; Eve reward is denial of service only",
        "attack_kinds": ["node_dos", "edge_dos"],
        "reward_model": "dos_only",
        "information_gain_reward_enabled": False,
    },
    "attack_surface_mixed": {
        "label": "attack surface model 2: mixed reward",
        "description": "node DoS, edge DoS, and information probe; Eve reward mixes denial and information gain",
        "attack_kinds": ["node_dos", "edge_dos", "information_probe"],
        "reward_model": "dos_plus_information_gain",
        "information_gain_reward_enabled": True,
    },
}

REPEATER_ATTACK_OPTIONS = {
    "repeater_memory_measure_X": {
        "label": "pre-swap X memory measurement",
        "description": "selective X-basis measurement on the selected route's first repeater memory",
    },
    "repeater_memory_dephasing_probe_pi2": {
        "label": "pre-swap dephasing probe pi/2",
        "description": "Kraus-sampled reduced Z-dephasing probe before swapping",
    },
    "swap_denial": {
        "label": "swap denial",
        "description": "force swapping failure at the selected repeater",
    },
    "memory_degradation": {
        "label": "memory degradation",
        "description": "lower the selected repeater memory-fidelity template",
    },
    "no_attack": {
        "label": "calibration baseline",
        "description": "honest selected-route request; not a learned Q action",
    },
}

ATTACK_KIND_ALIASES = {
    "none": "none",
    "no_attack": "none",
    "node": "node_collision",
    "node_collision": "node_collision",
    "edge": "edge_collision",
    "edge_collision": "edge_collision",
    "node_dos": "node_dos",
    "edge_dos": "edge_dos",
    "information_probe": "information_probe",
}


class DemoInputError(ValueError):
    """Invalid GUI demo request."""


@dataclass(frozen=True)
class RouteChoice:
    route_id: str
    path: tuple[str, ...]
    edge_ids: tuple[str, ...]
    total_length: float
    hop_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "path": list(self.path),
            "edge_ids": list(self.edge_ids),
            "total_length": self.total_length,
            "hop_count": self.hop_count,
        }


def _edge_key(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v)))


def _edge_id(u: str, v: str) -> str:
    a, b = _edge_key(u, v)
    return f"{a}-{b}"


def _validate_topology(topology: dict[str, Any]) -> None:
    if not topology.get("nodes"):
        raise DemoInputError("topology has no nodes")
    ids = [n["id"] for n in topology["nodes"]]
    if len(set(ids)) != len(ids):
        raise DemoInputError("topology has duplicate node ids")
    known = set(ids)
    seen: set[tuple[str, str]] = set()
    for edge in topology.get("edges", []):
        u, v = edge["u"], edge["v"]
        if u not in known or v not in known:
            raise DemoInputError(f"edge references unknown node: {u}-{v}")
        key = _edge_key(u, v)
        if key in seen:
            raise DemoInputError(f"duplicate edge: {u}-{v}")
        seen.add(key)


def _graph_from_topology(topology: dict[str, Any]) -> nx.Graph:
    _validate_topology(topology)
    graph = nx.Graph()
    for node in sorted(topology["nodes"], key=lambda n: n["id"]):
        graph.add_node(node["id"], x=float(node.get("x", 0.0)), y=float(node.get("y", 0.0)))
    for edge in sorted(topology.get("edges", []), key=lambda e: _edge_id(e["u"], e["v"])):
        graph.add_edge(edge["u"], edge["v"], length=float(edge.get("length", 1.0)))
    return graph


def _topology_ir(topology: dict[str, Any]) -> TopologyIR:
    nodes = {n["id"]: NodeRecord(n["id"]) for n in topology["nodes"]}
    edges = [
        EdgeRecord(_edge_id(e["u"], e["v"]), e["u"], e["v"], float(e.get("length", 1.0)))
        for e in topology.get("edges", [])
    ]
    ir = TopologyIR(nodes=nodes, edges=edges)
    ir.validate()
    return ir


def generate_topology(num_nodes: int, edge_probability: float, seed: int,
                      edge_length_mode: str = "unit") -> dict[str, Any]:
    """Generate a deterministic undirected simple graph for GUI inspection."""

    if num_nodes < 2:
        raise DemoInputError("num_nodes must be >= 2")
    if not 0.0 <= edge_probability <= 1.0:
        raise DemoInputError("edge_probability must be in [0, 1]")
    if edge_length_mode not in {"unit", "random"}:
        raise DemoInputError("edge_length_mode must be 'unit' or 'random'")

    config = ERTopologyConfig(
        n=num_nodes,
        p=edge_probability,
        fixed_edge_length_m=1.0 if edge_length_mode == "unit" else None,
        edge_length_range_m=(1.0, 10.0) if edge_length_mode == "random" else None,
        require_connected=False,
    )
    ir = generate_er_topology(config, seed=seed)
    ids = sorted(ir.nodes)
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    edges = [
        {"u": edge.u, "v": edge.v, "length": round(float(edge.length_m), 3)}
        for edge in ir.edges
    ]
    for edge in edges:
        graph.add_edge(edge["u"], edge["v"], length=edge["length"])

    warnings: list[str] = []
    if not edges:
        warnings.append("generated topology has no edges")
    elif not nx.is_connected(graph):
        warnings.append("generated topology is disconnected; select connected Alice/Bob nodes")

    if edges:
        pos = nx.spring_layout(graph, seed=seed, weight="length")
    else:
        pos = {
            node: (
                math.cos(2 * math.pi * i / num_nodes),
                math.sin(2 * math.pi * i / num_nodes),
            )
            for i, node in enumerate(ids)
        }
    nodes = [
        {
            "id": node,
            "x": round(float(pos[node][0]) * 240 + 320, 3),
            "y": round(float(pos[node][1]) * 220 + 260, 3),
        }
        for node in ids
    ]
    return {
        "nodes": nodes,
        "edges": sorted(edges, key=lambda e: _edge_id(e["u"], e["v"])),
        "warnings": warnings,
        "scope_label": TOY_SCOPE,
        "generator": {
            "kind": "erdos_renyi_gnp",
            "num_nodes": num_nodes,
            "edge_probability": edge_probability,
            "seed": seed,
            "edge_length_mode": edge_length_mode,
        },
    }


def candidate_routes(topology: dict[str, Any], alice: str, bob: str,
                     max_routes: int = 12) -> list[dict[str, Any]]:
    """Stable shortest simple path list, capped for demo readability."""

    if not alice or not bob:
        raise DemoInputError("Alice and Bob must be selected before route computation")
    if alice == bob:
        raise DemoInputError("Alice and Bob must be different nodes")
    graph = _graph_from_topology(topology)
    if alice not in graph or bob not in graph:
        raise DemoInputError("Alice/Bob must exist in the topology")
    if not nx.has_path(graph, alice, bob):
        return []

    paths = []
    for path in nx.shortest_simple_paths(graph, alice, bob, weight="length"):
        path_tuple = tuple(path)
        total = sum(graph[u][v]["length"] for u, v in zip(path, path[1:]))
        edge_ids = tuple(_edge_id(u, v) for u, v in zip(path, path[1:]))
        paths.append(RouteChoice(
            route_id=route_id_for_path(path_tuple),
            path=path_tuple,
            edge_ids=edge_ids,
            total_length=float(total),
            hop_count=len(edge_ids),
        ))
        if len(paths) >= max_routes:
            break
    paths.sort(key=lambda r: (r.total_length, r.hop_count, r.path))
    return [route.to_dict() for route in paths[:max_routes]]


def _attack_actions(
        topology: dict[str, Any],
        alice: str,
        bob: str,
        allowed_kinds: Optional[set[str]] = None,
        attack_node_ids: Optional[set[str]] = None,
        attack_edge_ids: Optional[set[str]] = None,
) -> list[dict[str, str]]:
    allowed = {
        ATTACK_KIND_ALIASES[kind]
        for kind in (allowed_kinds if allowed_kinds is not None
                     else {"none", "node_collision", "edge_collision"})
        if kind in ATTACK_KIND_ALIASES
    }
    nodes = [
        n["id"] for n in topology["nodes"]
        if n["id"] not in {alice, bob}
        and (attack_node_ids is None or n["id"] in attack_node_ids)
    ]
    actions = []
    if "none" in allowed:
        actions.append({
            "id": "no_attack",
            "kind": "none",
            "target": "",
            "attack_type": "no_attack",
        })
    for attack_type in ("node_collision", "node_dos", "information_probe"):
        if attack_type in allowed:
            actions.extend({
                "id": f"{attack_type}:{n}",
                "kind": "node",
                "target": n,
                "attack_type": attack_type,
            } for n in sorted(nodes))
    for attack_type in ("edge_collision", "edge_dos"):
        if attack_type not in allowed:
            continue
        for edge in sorted(topology.get("edges", []), key=lambda e: _edge_id(e["u"], e["v"])):
            target = _edge_id(edge["u"], edge["v"])
            if attack_edge_ids is not None and target not in attack_edge_ids:
                continue
            actions.append({
                "id": f"{attack_type}:{target}",
                "kind": "edge",
                "target": target,
                "attack_type": attack_type,
            })
    if not actions:
        actions.append({
            "id": "no_attack",
            "kind": "none",
            "target": "",
            "attack_type": "no_attack",
        })
    return actions


def _route_edges(path: list[str] | tuple[str, ...]) -> set[str]:
    return {_edge_id(u, v) for u, v in zip(path, path[1:])}


def _route_corridor_attack_scope(
        routes: list[dict[str, Any]],
        alice: str,
        bob: str,
) -> tuple[set[str], set[str]]:
    nodes: set[str] = set()
    edges: set[str] = set()
    for route in routes:
        path = [str(node) for node in route.get("path", [])]
        nodes.update(node for node in path[1:-1] if node not in {alice, bob})
        route_edges = {str(edge_id) for edge_id in route.get("edge_ids", [])}
        route_edges.update(_edge_id(u, v) for u, v in zip(path, path[1:]))
        edges.update(route_edges)
    return nodes, edges


def _collision(route: dict[str, Any], action: dict[str, str]) -> bool:
    if action["kind"] == "none":
        return False
    if action["kind"] == "node":
        return action["target"] in route["path"][1:-1]
    if action["kind"] == "edge":
        return action["target"] in set(route["edge_ids"])
    return False


def _route_entropy(route_ids: list[str]) -> float:
    if not route_ids:
        return 0.0
    total = len(route_ids)
    counts = Counter(route_ids)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _detected_period(route_ids: list[str], max_period: int = 12) -> Optional[int]:
    if len(route_ids) < 8:
        return None
    suffix = route_ids[-min(len(route_ids), 60):]
    for period in range(1, min(max_period, len(suffix) // 2) + 1):
        pattern = suffix[-period:]
        if suffix[-period * 3:] == pattern * 3:
            return period
    return None


def _make_agent(algo: str, num_actions: int):
    """Instantiate a Q-agent; both QTable and DeepQNetwork have identical interface."""
    if algo == "deep_q":
        from ..rl.deep_q_learning import DeepQNetwork
        return DeepQNetwork(num_actions)
    return QTable(num_actions)


def _control_model_config(control_model: str) -> dict[str, Any]:
    if control_model not in ACTION_SURFACE_MODELS:
        raise DemoInputError(f"unsupported control model {control_model!r}")
    return dict(ACTION_SURFACE_MODELS[control_model])


def _is_dos_attack(action: dict[str, str]) -> bool:
    return action.get("attack_type") in {
        "node_collision",
        "edge_collision",
        "node_dos",
        "edge_dos",
    }


def _is_information_attack(action: dict[str, str]) -> bool:
    return action.get("attack_type") == "information_probe"


def _oracle_action(actions: list[dict[str, str]], route: dict[str, Any]) -> dict[str, str]:
    hits = [action for action in actions if action["kind"] != "none" and _collision(route, action)]
    dos_hits = [action for action in hits if _is_dos_attack(action)]
    return (dos_hits or hits or actions)[0]


def _control_outcome(control_model: str,
                     action: dict[str, str],
                     hit: bool,
                     attack_cost: float) -> dict[str, Any]:
    attack_type = action.get("attack_type", "")
    cost = 0.0 if action["kind"] == "none" else float(attack_cost)
    dos_hit = hit and _is_dos_attack(action)
    info_hit = hit and _is_information_attack(action)
    info_enabled = bool(
        ACTION_SURFACE_MODELS[control_model]["information_gain_reward_enabled"])
    information_reward = 1.0 if info_enabled and info_hit else 0.0

    if control_model == "binary_collision":
        public_outcome = "hit" if hit else "miss"
        denial_reward = 1.0 if hit else 0.0
        delivery_success = None
        alice_reward = -1.0 if hit else 1.0
        avoid_route = hit
    else:
        public_outcome = (
            "delivery_failure" if dos_hit else
            "information_exposure" if info_hit else
            "delivered"
        )
        denial_reward = 1.0 if dos_hit else 0.0
        delivery_success = not dos_hit
        alice_reward = -1.0 if dos_hit else -0.5 if info_hit else 1.0
        avoid_route = dos_hit or info_hit

    eve_reward = denial_reward + information_reward - cost
    return {
        "public_outcome": public_outcome,
        "delivery_success": delivery_success,
        "alice_reward": alice_reward,
        "eve_reward": eve_reward,
        "avoid_route": avoid_route,
        "reward_components": {
            "attack_type": attack_type,
            "denial_of_service_reward": denial_reward,
            "information_gain_reward": information_reward,
            "attack_cost": cost,
            "mixed_reward_enabled": info_enabled,
        },
        "information_gain_reward_enabled": info_enabled,
    }


def _q_top_actions(table: QTable, actions: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    scores = [0.0] * len(actions)
    entries = getattr(table, "_q", {})
    for vals in entries.values():
        for i, val in enumerate(vals):
            scores[i] = max(scores[i], float(val))
    ranked = sorted(range(len(actions)), key=lambda i: (-scores[i], actions[i]["id"]))[:limit]
    return [
        {
            "action": actions[i]["id"],
            "kind": actions[i].get("kind", ""),
            "target": actions[i].get("target", ""),
            "attack_type": actions[i].get("attack_type", actions[i].get("kind", "")),
            "q": round(scores[i], 4),
        }
        for i in ranked
    ]


def _standard_step(step: int, mode: str, alice_policy: str, eve_policy: str,
                   route: Optional[dict[str, Any]], action: dict[str, str],
                   public_outcome: str, collision: bool, alice_reward: float,
                   eve_reward: float, fallback_used: bool,
                   metrics: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    merged_metrics = {
        "qber": None,
        "fidelity": None,
        "delivery_success": None,
        **(metrics or {}),
    }
    step_scope = merged_metrics.get("scope_label", TOY_SCOPE)
    quality_metrics = {
        "qber": merged_metrics.get("qber"),
        "fidelity": merged_metrics.get("fidelity"),
        "delivery_success": merged_metrics.get("delivery_success"),
    }
    route_features = _gui_route_features(route, step_scope)
    public_record = PublicStepRecord(
        step=step,
        mode=mode,
        scope_label=step_scope,
        alice_route_id=route["route_id"] if route else "",
        alice_path=tuple(route["path"]) if route else (),
        eve_action={
            "kind": action["kind"],
            "target": action["target"],
            "attack_type": action.get("attack_type", action["kind"]),
        },
        public_outcome=public_outcome,
        collision_or_delivery_failure=bool(
            collision or public_outcome in {"delivery_failure", "qber_abort", "blocked"}
        ),
        alice_reward=float(alice_reward),
        eve_reward=float(eve_reward),
        route_features=route_features,
        quality_metrics=quality_metrics,
    ).to_dict()
    return {
        **public_record,
        "alice_policy": alice_policy,
        "eve_policy": eve_policy,
        "collision": bool(collision),
        "fallback_used": bool(fallback_used),
        "metrics": merged_metrics,
    }


def _gui_route_features(route: Optional[dict[str, Any]], scope_label: str) -> dict[str, Any]:
    if not route:
        return {}
    return {
        "hops": route.get("hop_count"),
        "length": route.get("total_length", route.get("total_length_m")),
        "internal_nodes": list(route.get("path", [])[1:-1]),
        "edges": list(route.get("edge_ids", [])),
        "scope_label": scope_label,
    }


def run_binary_collision(topology: dict[str, Any], alice: str, bob: str, *,
                         eve_algo: str = "q_learning",
                         alice_algo: str = "cautious_greedy",
                         control_model: str = "binary_collision",
                         num_steps: int = 3000,
                         seed: int = 1,
                         avoidance_horizon: int = 2,
                         attack_cost: float = 0.05,
                         final_window_size: int = 20,
                         max_routes: int = 12,
                         route_candidates: Optional[list[dict[str, Any]]] = None,
                         allowed_attack_kinds: Optional[list[str]] = None,
                         step_timing_callback: Optional[Callable[[int, float], None]] = None,
                         include_alice_q_table: bool = False,
                         dt_refit_observer: Any = None,
                         dt_refit_callback: Optional[Callable[[Any, dict[str, Any]], None]] = None) -> dict[str, Any]:
    """Run a scoped toy/control attack-surface game."""

    started = time.perf_counter()
    _update_progress(running=True, phase="Initializing…", step=0, total=num_steps, eve_algo=eve_algo)
    model = _control_model_config(control_model)
    if not alice or not bob:
        _update_progress(running=False, phase="error")
        raise DemoInputError("Alice/Bob must be selected before run")
    if alice_algo not in _VALID_ALICE_ALGOS:
        raise DemoInputError(f"unsupported Alice algorithm {alice_algo!r}")
    if eve_algo not in _VALID_EVE_ALGOS:
        raise DemoInputError(f"unsupported Eve algorithm {eve_algo!r}")
    if eve_algo == "route_aware_oracle" and alice_algo != "cautious_greedy":
        raise DemoInputError(
            "route-aware oracle Eve is only defined against cautious-greedy Alice"
        )
    # Alias so loop body below reads naturally
    alice_policy = alice_algo
    eve_policy   = eve_algo
    if num_steps < 1:
        raise DemoInputError("num_steps must be >= 1")

    routes = (
        [dict(route) for route in route_candidates]
        if route_candidates is not None
        else candidate_routes(topology, alice, bob, max_routes=max_routes)
    )
    if not routes:
        step = _standard_step(
            0, control_model, alice_policy, eve_policy, None,
            {"kind": "none", "target": "", "id": "no_attack"}, "no_route",
            False, -0.5, 0.0, True)
        summary = _summary(
            control_model, seed, alice, bob, [step], [step], final_window_size,
            routes, TOY_SCOPE)
        summary["metrics"].update({
            "training_runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_iterations": 1,
            "training_execution": "batched tabular update; replay visualizes the final window",
            "control_model": control_model,
            "eve_algo": eve_algo,
            "alice_algo": alice_algo,
            "reward_model": model["reward_model"],
            "information_gain_reward_enabled": model["information_gain_reward_enabled"],
        })
        return {"summary": summary, "steps": [step], "final_window": [step],
                "scope_label": TOY_SCOPE}

    rng = np.random.default_rng(seed)
    if allowed_attack_kinds is None:
        allowed = set(model["attack_kinds"])
    else:
        allowed = {str(kind) for kind in allowed_attack_kinds}
    enabled_attack_types = sorted({
        ATTACK_KIND_ALIASES[kind]
        for kind in allowed
        if kind in ATTACK_KIND_ALIASES
    })
    attack_node_ids: Optional[set[str]] = None
    attack_edge_ids: Optional[set[str]] = None
    attack_surface_scope = "full_topology"
    if route_candidates is not None:
        attack_node_ids, attack_edge_ids = _route_corridor_attack_scope(routes, alice, bob)
        attack_surface_scope = "route_corridor"
    actions = _attack_actions(
        topology,
        alice,
        bob,
        allowed,
        attack_node_ids=attack_node_ids,
        attack_edge_ids=attack_edge_ids,
    )
    _update_progress(phase=f"Loading agents: {eve_algo} Eve vs {alice_algo} Alice…")
    eve_q   = _make_agent(eve_algo, len(actions))
    alice_q = _make_agent(alice_algo, len(routes))
    _update_progress(phase=f"Running: {eve_algo} Eve · {len(actions)} actions · {num_steps} steps")
    q_cfg = QLearningConfig(alpha=0.2, gamma=0.9)
    avoided_until: dict[str, int] = {}
    previous_eve_action = "start"
    previous_public_outcome = "start"
    previous_route_id = "start"
    steps: list[dict[str, Any]] = []

    for step_idx in range(num_steps):
        step_started = time.perf_counter()
        if step_idx % 100 == 0:
            is_dqn = eve_algo == "deep_q" or alice_algo == "deep_q"
            pct = int(step_idx / num_steps * 100)
            phase_detail = " · fitting model…" if is_dqn and step_idx > 0 else ""
            _update_progress(
                step=step_idx,
                phase=f"{eve_algo} Eve step {step_idx}/{num_steps} ({pct}%){phase_detail}",
            )
        fallback_used = False
        avoided_signature = "".join(
            "1" if step_idx <= avoided_until.get(route["route_id"], -1) else "0"
            for route in routes
        )
        if alice_algo in {"q_learning", "deep_q"}:
            alice_state = (previous_route_id, previous_public_outcome, avoided_signature)
            epsilon = max(0.05, 0.3 * (1 - step_idx / max(1, num_steps - 1)))
            route_idx = epsilon_greedy(alice_q, alice_state, epsilon, rng)
            route = routes[route_idx]
        elif alice_policy == "random":
            route_idx = int(rng.integers(len(routes)))
            route = routes[route_idx]
            alice_state = None
        else:
            available = [
                route for route in routes
                if step_idx > avoided_until.get(route["route_id"], -1)
            ]
            fallback_used = not available
            route = (available or routes)[0]
            route_idx = routes.index(route)
            alice_state = None

        if eve_policy == "route_aware_oracle":
            action = _oracle_action(actions, route)
            eve_state = ("privileged_route", route["route_id"])
            eve_action_idx = actions.index(action)
        elif eve_policy == "random":
            eve_state = (previous_eve_action, previous_public_outcome)
            eve_action_idx = int(rng.integers(len(actions)))
            action = actions[eve_action_idx]
        else:
            eve_state = (previous_eve_action, previous_public_outcome)
            epsilon = max(0.05, 0.35 * (1 - step_idx / max(1, num_steps - 1)))
            eve_action_idx = epsilon_greedy(eve_q, eve_state, epsilon, rng)
            action = actions[eve_action_idx]

        hit = _collision(route, action)
        outcome = _control_outcome(control_model, action, hit, attack_cost)
        public_outcome = str(outcome["public_outcome"])
        eve_reward = float(outcome["eve_reward"])
        alice_reward = float(outcome["alice_reward"])
        if fallback_used:
            alice_reward -= 0.2
        if outcome["avoid_route"] and alice_policy == "cautious_greedy":
            avoided_until[route["route_id"]] = step_idx + 1

        next_eve_state = (action["id"], public_outcome)
        if eve_algo in {"q_learning", "deep_q"}:
            eve_q.update(eve_state, eve_action_idx, eve_reward, next_eve_state, q_cfg)
        if alice_algo in {"q_learning", "deep_q"}:
            next_avoid = "".join(
                "1" if step_idx <= avoided_until.get(r["route_id"], -1) else "0"
                for r in routes
            )
            next_alice_state = (route["route_id"], public_outcome, next_avoid)
            alice_q.update(alice_state, route_idx, alice_reward, next_alice_state, q_cfg)

        record = _standard_step(
            step_idx, control_model, alice_policy, eve_policy, route, action,
            public_outcome, hit, alice_reward, eve_reward, fallback_used,
            metrics={
                "delivery_success": outcome["delivery_success"],
                "reward_components": outcome["reward_components"],
                "information_gain_reward_enabled": outcome[
                    "information_gain_reward_enabled"],
            })
        record["metrics"]["scope_label"] = TOY_SCOPE
        steps.append(record)
        previous_eve_action = action["id"]
        previous_public_outcome = public_outcome
        previous_route_id = route["route_id"]
        if dt_refit_observer is not None:
            dt_refit_observer.record_transition(
                alice_action=route_idx,
                eve_action=eve_action_idx,
                alice_route=route.get("route_id"),
                eve_target=action.get("target"),
                reward=eve_reward,
            )
            tick = dt_refit_observer.observe(
                step=step_idx + 1,
                alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
            )
            if tick is not None and tick.refit_recommended:
                event = dt_refit_observer.refit_event_from_tick(tick)
                if dt_refit_callback is not None:
                    dt_refit_callback(event, {
                        "step": step_idx + 1,
                        "routes": routes,
                        "actions": actions,
                        "steps": steps,
                        "alice_q": alice_q,
                        "eve_q": eve_q,
                        "alice_algo": alice_algo,
                        "eve_algo": eve_algo,
                        "control_model": control_model,
                    })
                dt_refit_observer.mark_refit(
                    step=step_idx + 1,
                    alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                    eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
                    reason=tick.refit_reason,
                )
        if step_timing_callback is not None:
            step_timing_callback(step_idx, time.perf_counter() - step_started)

    summary = _summary(
        control_model, seed, alice, bob, steps, steps[-final_window_size:],
        final_window_size, routes, TOY_SCOPE)
    summary["q_tables"] = {
        "eve": {
            "top_actions": _q_top_actions(eve_q, actions),
            "raw": eve_q.to_dict(),          # full Q-table for XAI policy explainer
            "action_names": [a["id"] for a in actions],
            "observation_state": ["previous_action", "previous_public_outcome"],
            "private_route_observed": eve_policy == "route_aware_oracle",
            "route_aware_oracle_label": (
                "privileged/control" if eve_policy == "route_aware_oracle" else None
            ),
            "enabled_attack_kinds": enabled_attack_types,
            "attack_surface_scope": attack_surface_scope,
            "attack_surface_node_count": len(attack_node_ids) if attack_node_ids is not None else None,
            "attack_surface_edge_count": len(attack_edge_ids) if attack_edge_ids is not None else None,
            "control_model": control_model,
            "reward_model": model["reward_model"],
            "information_gain_reward_enabled": model["information_gain_reward_enabled"],
        },
        "alice": {
            "top_actions": _q_top_actions(
                alice_q,
                [{"id": route["route_id"], "kind": "route", "target": route["path"]}
                 for route in routes],
            ) if alice_algo in {"q_learning", "deep_q"} else [],
            "raw": (
                alice_q.to_dict()
                if include_alice_q_table and alice_algo in {"q_learning", "deep_q"}
                else None
            ),
            "action_names": [route["route_id"] for route in routes],
            "observation_state": ["previous_route", "previous_public_outcome", "avoided_route_bitset"],
        },
    }
    summary["metrics"].update({
        "training_runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "training_iterations": len(steps),
        "training_execution": "batched tabular update; replay visualizes the final window",
        "control_model": control_model,
        "eve_algo": eve_algo,
        "alice_algo": alice_algo,
        "reward_model": model["reward_model"],
        "enabled_attack_kinds": enabled_attack_types,
        "attack_surface_scope": attack_surface_scope,
        "attack_surface_node_count": len(attack_node_ids) if attack_node_ids is not None else None,
        "attack_surface_edge_count": len(attack_edge_ids) if attack_edge_ids is not None else None,
        "information_gain_reward_enabled": model["information_gain_reward_enabled"],
        "information_exposures": sum(
            1 for step in steps if step.get("public_outcome") == "information_exposure"
        ),
        "delivery_failures": sum(
            1 for step in steps if step.get("public_outcome") == "delivery_failure"
        ),
    })
    summary["disabled_attacks"] = DISABLED_ATTACKS
    final_window = steps[-final_window_size:]
    summary["final_window"] = final_window
    _update_progress(running=False, phase="Done", step=num_steps)
    return {
        "summary": summary,
        "steps": steps,
        "final_window": final_window,
        "scope_label": TOY_SCOPE,
    }


def _summary(mode: str, seed: int, alice: str, bob: str,
             steps: list[dict[str, Any]], final_window: list[dict[str, Any]],
             final_window_size: int, routes: list[dict[str, Any]],
             scope_warning: str) -> dict[str, Any]:
    route_ids = [step["alice_route_id"] for step in steps if step.get("alice_route_id")]
    collisions = sum(1 for step in steps if step.get("collision"))
    switches = sum(1 for a, b in zip(route_ids, route_ids[1:]) if a != b)
    return {
        "run_id": f"{mode}-{seed}-{len(steps)}",
        "seed": seed,
        "mode": mode,
        "num_steps": len(steps),
        "alice": alice,
        "bob": bob,
        "routes": routes,
        "final_window": final_window,
        "metrics": {
            "collision_rate": collisions / len(steps) if steps else 0.0,
            "route_entropy": _route_entropy(route_ids),
            "route_switches": switches,
            "detected_period": _detected_period(route_ids),
            "delivery_failures": None,
            "aborts": None,
            "qber": None,
            "fidelity": None,
            "final_replay_window": final_window_size,
            "information_gain_reward_enabled": False,
        },
        "scope_warning": scope_warning,
    }


def _edge_length(topology: dict[str, Any], u: str, v: str) -> Optional[float]:
    target = _edge_key(u, v)
    for edge in topology.get("edges", []):
        if _edge_key(edge["u"], edge["v"]) == target:
            return float(edge.get("length", 1.0))
    return None


def _toy_memory_model() -> PhysicalModel:
    return PhysicalModel("gui_demo_memory", "memory", "toy", "GUI demo toy memory", {
        "fidelity": 1.0,
        "frequency": 8e7,
        "efficiency": 1.0,
        "coherence_time": -1,
        "wavelength": 500.0,
    })


def _toy_fiber_model() -> PhysicalModel:
    return PhysicalModel("gui_demo_fiber", "fiber_channel", "toy", "GUI demo toy fiber", {
        "attenuation": 0.0,
        "polarization_fidelity": 1.0,
        "light_speed": 2e-4,
        "frequency": 8e7,
    })


def _basis_counts(bases: tuple[str, ...]) -> dict[str, int]:
    return dict(sorted(Counter(bases).items()))


def run_repeater_demo(topology: dict[str, Any], alice: str, bob: str,
                      selected_route: list[str], *,
                      attack: str = "repeater_memory_measure_X",
                      num_trials: int = 20, seed: int = 1,
                      final_window_size: int = 20) -> dict[str, Any]:
    """Run the current supported selected-path SeQUeNCe E91 runner or block."""

    started = time.perf_counter()
    if attack not in REPEATER_ATTACK_OPTIONS:
        return {
            "blocked": True,
            "blocked_reason": BLOCKED_ATTACK_TOOLTIP,
            "scope_label": DISABLED_SCOPE,
            "disabled_attacks": DISABLED_ATTACKS,
        }
    if not alice or not bob or not selected_route:
        return {"blocked": True, "blocked_reason": REPEATER_MOTIF_BLOCK,
                "scope_label": DISABLED_SCOPE, "disabled_attacks": DISABLED_ATTACKS}
    if len(selected_route) < 3 or selected_route[0] != alice or selected_route[-1] != bob:
        return {"blocked": True, "blocked_reason": REPEATER_MOTIF_BLOCK,
                "scope_label": DISABLED_SCOPE, "disabled_attacks": DISABLED_ATTACKS}

    edge_lengths = [
        _edge_length(topology, u, v)
        for u, v in zip(selected_route, selected_route[1:])
    ]
    if any(length is None for length in edge_lengths):
        return {"blocked": True, "blocked_reason": REPEATER_MOTIF_BLOCK,
                "scope_label": DISABLED_SCOPE, "disabled_attacks": DISABLED_ATTACKS}

    repeater = selected_route[1]
    spec = _repeater_attack_spec(attack, repeater)

    run_config = RepeaterE91RunConfig(
        memory_pairs=max(1, int(num_trials)),
        sifting=SiftingConfig(qber_threshold=0.5, min_sifted_samples=1),
        request_fidelity=0.01,
        start_time_ps=1_000_000,
        end_time_ps=80_000_000,
        stop_time_ps=100_000_000,
        swapping_success_prob=1.0,
        swapping_degradation=1.0,
    )
    transcript, built = run_fixed_repeater_e91_trial(
        FixedRepeaterPath(
            nodes=tuple(selected_route),
            edge_lengths_m=tuple(float(length) for length in edge_lengths),
        ),
        memory_model=_toy_memory_model(),
        fiber_model=_toy_fiber_model(),
        run_config=run_config,
        trial_id="gui-repeater",
        seed=seed,
        attack=spec,
    )

    if transcript.accepted:
        outcome = "accepted"
    elif transcript.abort_reason == "qber_above_threshold":
        outcome = "qber_abort"
    else:
        outcome = "delivery_failure"
    route = {
        "route_id": route_id_for_path(tuple(selected_route)),
        "path": selected_route,
        "edge_ids": [_edge_id(u, v) for u, v in zip(selected_route, selected_route[1:])],
        "total_length": sum(float(length) for length in edge_lengths),
        "hop_count": len(selected_route) - 1,
    }
    action = {
        "kind": attack if attack != "no_attack" else "none",
        "target": repeater,
        "attack_type": attack,
    }
    fidelity = transcript.extra.get("mean_actual_fidelity")
    metrics = {
        "qber": transcript.qber_estimate,
        "fidelity": fidelity,
        "delivery_success": transcript.generation_successes > 0,
        "delivered_pairs": transcript.generation_successes,
        "abort_reason": transcript.abort_reason,
        "scope_label": REPEATER_SCOPE,
    }
    built_summary = built.summary() if hasattr(built, "summary") else None
    if built_summary:
        metrics["pre_swap_memory_hook_registered"] = built_summary.get(
            "pre_swap_memory_hook_registered")
        metrics["pre_swap_memory_intervention_public"] = built_summary.get(
            "pre_swap_memory_intervention_public")
    step = _standard_step(
        0, "repeater_availability_quality", "selected_one_repeater_path",
        "selected_repeater_attack", route, action, outcome, False,
        1.0 if transcript.accepted else -1.0,
        1.0 if not transcript.accepted else 0.0,
        False, metrics)
    summary = _summary(
        "repeater_availability_quality", seed, alice, bob, [step], [step],
        final_window_size, [route], REPEATER_SCOPE)
    summary["metrics"].update({
        "delivery_failures": 0 if transcript.generation_successes > 0 else 1,
        "aborts": 0 if transcript.accepted else 1,
        "qber": transcript.qber_estimate,
        "fidelity": fidelity,
        "delivered_pair_count": transcript.generation_successes,
        "amer_style_d_eff_mean": transcript.extra.get("amer_style_d_eff_mean"),
        "endpoint_basis_counts": {
            "alice": _basis_counts(transcript.alice_bases),
            "bob": _basis_counts(transcript.bob_bases),
        },
        "information_gain_reward_enabled": False,
        "runner_runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "runner_execution": "single selected repeater path request",
    })
    summary["disabled_attacks"] = DISABLED_ATTACKS
    summary["limitation"] = REPEATER_MOTIF_BLOCK
    summary["repeater_action_space"] = [
        {
            "id": attack_id,
            **metadata,
        }
        for attack_id, metadata in REPEATER_ATTACK_OPTIONS.items()
    ]
    return {
        "blocked": False,
        "blocked_reason": None,
        "summary": summary,
        "steps": [step],
        "final_window": [step],
        "scope_label": REPEATER_SCOPE,
        "disabled_attacks": DISABLED_ATTACKS,
    }


def _repeater_attack_spec(attack: str, repeater: str) -> RepeaterAttackSpec:
    if attack == "no_attack":
        return RepeaterAttackSpec()
    if attack == "swap_denial":
        return RepeaterAttackSpec(kind="swap_denial", target_node=repeater)
    if attack == "memory_degradation":
        return RepeaterAttackSpec(
            kind="memory_degradation",
            target_node=repeater,
            memory_fidelity_multiplier=0.5,
        )
    if attack == "repeater_memory_measure_X":
        return RepeaterAttackSpec(
            kind="repeater_memory_measurement",
            target_node=repeater,
            basis="X",
            attack_id=f"repeater_memory_measure_X:{repeater}",
        )
    if attack == "repeater_memory_dephasing_probe_pi2":
        return RepeaterAttackSpec(
            kind="repeater_memory_dephasing_probe",
            target_node=repeater,
            theta=math.pi / 2,
            theta_label="pi/2",
            attack_id=f"repeater_memory_dephasing_probe:{repeater}:pi/2",
        )
    raise DemoInputError(f"unsupported repeater attack {attack!r}")
