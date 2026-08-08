"""Eve Q-learning training loop over the game environment.

Consumes only the environment's public outputs (observation state keys and
rewards). Outputs (Q-table, config, per-episode metrics, summary) are
serializable and reproducible under the configured seed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..claims import CONTROL_GAME
from ..environment.env import EveGameEnv
from .q_learning import (
    QLearningConfig,
    QTable,
    epsilon_greedy,
    linear_epsilon_schedule,
)


@dataclass(frozen=True)
class EveTrainingConfig:
    episodes: int
    steps_per_episode: int
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "steps_per_episode": self.steps_per_episode,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "seed": self.seed,
        }


@dataclass
class EpisodeMetrics:
    episode: int
    epsilon: float
    total_reward: float
    accepts: int
    aborts: int

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "epsilon": self.epsilon,
            "total_reward": self.total_reward,
            "accepts": self.accepts,
            "aborts": self.aborts,
        }


@dataclass
class TrainingResult:
    q_table: QTable
    episodes: list[EpisodeMetrics] = field(default_factory=list)
    action_ids: list[str] = field(default_factory=list)
    action_histogram: dict[str, int] = field(default_factory=dict)
    final_epsilon: float = 0.0

    def summary(self) -> dict:
        rewards = [e.total_reward for e in self.episodes]
        action_ids = self.action_ids or [str(i) for i in range(self.q_table.num_actions)]
        return {
            "scope_label": CONTROL_GAME,
            "episode_count": len(self.episodes),
            "episodes": len(self.episodes),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "final_reward": rewards[-1] if rewards else 0.0,
            "final_epsilon": self.final_epsilon,
            "reward_trace": rewards,
            "action_histogram": dict(sorted(self.action_histogram.items())),
            "q_top_actions": q_top_actions(self.q_table, action_ids),
            "convergence_claim": None,
            "total_accepts": sum(e.accepts for e in self.episodes),
            "total_aborts": sum(e.aborts for e in self.episodes),
        }


def train_eve(env: EveGameEnv, config: EveTrainingConfig,
              output_dir: Optional[Path] = None) -> TrainingResult:
    rng = np.random.default_rng(config.seed)
    q_config = QLearningConfig(alpha=config.alpha, gamma=config.gamma)
    table = QTable(num_actions=len(env.actions))
    schedule = linear_epsilon_schedule(config.epsilon_start, config.epsilon_end,
                                       config.episodes)
    result = TrainingResult(
        q_table=table,
        action_ids=env.action_ids,
        action_histogram={action_id: 0 for action_id in env.action_ids},
    )

    for episode in range(config.episodes):
        epsilon = schedule(episode)
        result.final_epsilon = epsilon
        obs = env.reset(seed=int(rng.integers(2**31)))
        state = obs.as_state_key()
        total_reward, accepts, aborts = 0.0, 0, 0
        for _ in range(config.steps_per_episode):
            action = epsilon_greedy(table, state, epsilon, rng)
            result.action_histogram[env.action_ids[action]] += 1
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = next_obs.as_state_key()
            table.update(state, action, reward, next_state, q_config)
            total_reward += reward
            if next_obs.outcome == "accept":
                accepts += 1
            elif next_obs.outcome == "abort":
                aborts += 1
            state = next_state
            if terminated or truncated:
                break
        result.episodes.append(EpisodeMetrics(episode, epsilon, total_reward,
                                              accepts, aborts))

    if output_dir is not None:
        save_training_outputs(result, config, env, Path(output_dir))
    return result


def save_training_outputs(result: TrainingResult, config: EveTrainingConfig,
                          env: EveGameEnv, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "q_table.json").write_text(result.q_table.to_json(),
                                             encoding="utf-8")
    run_config = {
        "training": config.to_dict(),
        "action_ids": env.action_ids,
        "reward": env.reward_config.to_dict(),
        "trial": env.trial_config.to_dict(),
        "topology_metadata": env.topology.metadata.to_dict(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    with open(output_dir / "episode_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "epsilon", "total_reward", "accepts", "aborts"])
        writer.writeheader()
        for episode in result.episodes:
            writer.writerow(episode.to_dict())
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True), encoding="utf-8")


def q_top_actions(table: QTable, action_ids: list[str], limit: int = 8) -> list[dict]:
    scores = [0.0] * table.num_actions
    for values in getattr(table, "_q", {}).values():
        for index, value in enumerate(values):
            scores[index] = max(scores[index], float(value))
    ranked = sorted(range(table.num_actions), key=lambda index: (-scores[index], action_ids[index]))
    return [
        {"action_id": action_ids[index], "q": round(scores[index], 6)}
        for index in ranked[:limit]
    ]
