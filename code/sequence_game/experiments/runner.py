"""Build-and-run plumbing for toy Eve experiments from serializable configs.

A config is a plain JSON-able dict (see ``configs/experiments/`` for examples)
with sections: topology, roles, routing, actions, trial, observation, reward,
training. Everything is toy-scope; ``run_eve_experiment`` stamps scope and
provenance metadata into every output directory.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any

from ..analysis.metrics import aggregate_training_metrics, write_json
from ..environment.env import EveGameEnv, RewardConfig
from ..eve.actions import DenialAttackAction, EveAction, NoAttackAction
from ..eve.observation import ObservationConfig
from ..protocol.postprocessing import SiftingConfig
from ..protocol.toy_trial import ToyTrialConfig
from ..rl.eve_training import EveTrainingConfig, TrainingResult, train_eve
from ..routing.baselines import (
    FixedRoutePolicy,
    SeededRandomSimplePathPolicy,
    ShortestHopPolicy,
    ShortestLengthPolicy,
)
from ..routing.policy import RoutingPolicy
from ..topology.er_generator import ERTopologyConfig, generate_er_topology
from ..topology.ir import TopologyIR
from ..topology.roles import RoleAssignmentConfig, assign_roles


class ExperimentConfigError(ValueError):
    """Invalid experiment configuration."""


_ROUTING_POLICIES = {
    "shortest_hop": lambda spec: ShortestHopPolicy(),
    "shortest_length": lambda spec: ShortestLengthPolicy(),
    "seeded_random_simple_path": lambda spec: SeededRandomSimplePathPolicy(),
    "fixed_route": lambda spec: FixedRoutePolicy(tuple(spec["path"])),
}


def build_topology(config: dict[str, Any]) -> TopologyIR:
    topo_spec = config["topology"]
    er = ERTopologyConfig(
        n=int(topo_spec["n"]),
        p=float(topo_spec["p"]),
        fixed_edge_length_m=topo_spec.get("fixed_edge_length_m"),
        edge_length_range_m=tuple(topo_spec["edge_length_range_m"])
        if topo_spec.get("edge_length_range_m") else None,
        require_connected=bool(topo_spec.get("require_connected", True)),
    )
    ir = generate_er_topology(er, seed=int(topo_spec["seed"]))
    roles_spec = config.get("roles", {})
    role_config = RoleAssignmentConfig(
        alice=roles_spec.get("alice"),
        bob=roles_spec.get("bob"),
        eve_eligible_nodes=_listish(roles_spec.get("eve_eligible_nodes", "none")),
        eve_eligible_edges=_listish(roles_spec.get("eve_eligible_edges", "none")),
        allow_endpoint_eve=bool(roles_spec.get("allow_endpoint_eve", False)),
    )
    return assign_roles(ir, role_config, seed=roles_spec.get("seed"))


def _listish(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def build_routing_policy(config: dict[str, Any]) -> RoutingPolicy:
    spec = config.get("routing", {"policy": "shortest_hop"})
    name = spec["policy"]
    if name not in _ROUTING_POLICIES:
        raise ExperimentConfigError(
            f"unknown routing policy {name!r}; known: {sorted(_ROUTING_POLICIES)}")
    return _ROUTING_POLICIES[name](spec)


def build_actions(config: dict[str, Any]) -> list[EveAction]:
    actions: list[EveAction] = []
    for spec in config["actions"]:
        kind = spec["type"]
        if kind == "no_attack":
            actions.append(NoAttackAction())
        elif kind == "denial":
            actions.append(DenialAttackAction(
                spec["target_type"], spec["target_id"],
                cost=float(spec.get("cost", 1.0))))
        else:
            raise ExperimentConfigError(
                f"unknown action type {kind!r}; known: no_attack, denial")
    return actions


def build_env(config: dict[str, Any]) -> EveGameEnv:
    topology = build_topology(config)
    alices = topology.nodes_with_role("alice")
    bobs = topology.nodes_with_role("bob")
    trial_spec = config["trial"]
    sifting = SiftingConfig(
        qber_threshold=float(trial_spec["qber_threshold"]),
        min_sifted_samples=int(trial_spec["min_sifted_samples"]))
    trial_config = ToyTrialConfig(
        num_pairs=int(trial_spec["num_pairs"]),
        matched_flip_probability=float(trial_spec["matched_flip_probability"]),
        sifting=sifting)
    obs_spec = config.get("observation", {})
    observation = ObservationConfig(
        expose_route_id=bool(obs_spec.get("expose_route_id", False)),
        expose_qber=bool(obs_spec.get("expose_qber", False)),
        expose_latency=bool(obs_spec.get("expose_latency", False)),
        expose_sifted_count=bool(obs_spec.get("expose_sifted_count", False)),
        qber_bucket_edges=tuple(obs_spec.get("qber_bucket_edges", [])),
        latency_bucket_edges_ps=tuple(obs_spec.get("latency_bucket_edges_ps", [])),
    )
    reward_spec = config["reward"]
    reward = RewardConfig(
        accept_reward=float(reward_spec["accept_reward"]),
        abort_reward=float(reward_spec["abort_reward"]),
        cost_weight=float(reward_spec.get("cost_weight", 1.0)))
    return EveGameEnv(
        topology=topology, alice=alices[0], bob=bobs[0],
        routing_policy=build_routing_policy(config),
        actions=build_actions(config),
        trial_config=trial_config, observation_config=observation,
        reward_config=reward, seed=int(config["seed"]))


def build_training_config(config: dict[str, Any]) -> EveTrainingConfig:
    spec = config["training"]
    return EveTrainingConfig(
        episodes=int(spec["episodes"]),
        steps_per_episode=int(spec["steps_per_episode"]),
        alpha=float(spec["alpha"]),
        gamma=float(spec["gamma"]),
        epsilon_start=float(spec["epsilon_start"]),
        epsilon_end=float(spec["epsilon_end"]),
        seed=int(config["seed"]))


def git_commit_or_unknown() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": config.get("seed"),
        "git_commit": git_commit_or_unknown(),
        "scope": "toy",
        "scope_note": ("toy game-mechanics run; not physically valid, "
                       "not a security result"),
    }


def run_eve_experiment(config: dict[str, Any], output_dir: Path) -> TrainingResult:
    """Build env from config, train Eve, write all artifacts to output_dir."""
    output_dir = Path(output_dir)
    env = build_env(config)
    training = build_training_config(config)
    result = train_eve(env, training, output_dir=output_dir)
    write_json(config, output_dir / "experiment_config.json")
    write_json(run_metadata(config), output_dir / "run_metadata.json")
    rewards = [e.total_reward for e in result.episodes]
    write_json(aggregate_training_metrics(rewards),
               output_dir / "training_metrics.json")
    return result


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
