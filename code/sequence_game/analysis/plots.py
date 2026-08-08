"""Basic matplotlib plots over metrics outputs.

Plots visualize toy game metrics only; they are not evidence of physical
performance or security.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import moving_average  # noqa: E402


def plot_episode_rewards(episode_rewards: Sequence[float], out_path: Path, *,
                         window: int = 10, title: Optional[str] = None) -> Path:
    """Per-episode reward plus moving average."""
    fig, ax = plt.subplots(figsize=(6, 4))
    episodes = range(len(episode_rewards))
    ax.plot(episodes, episode_rewards, label="episode reward", alpha=0.5)
    ax.plot(episodes, moving_average(list(episode_rewards), window),
            label=f"moving avg (w={window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward (game units)")
    ax.set_title(title or "Eve training reward (toy scope)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def plot_cumulative_reward(episode_rewards: Sequence[float], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    cumulative = []
    total = 0.0
    for reward in episode_rewards:
        total += reward
        cumulative.append(total)
    ax.plot(range(len(cumulative)), cumulative)
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward (game units)")
    ax.set_title("Cumulative reward (toy scope)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def plot_distribution(distribution: dict[str, float], out_path: Path, *,
                      title: str, xlabel: str = "") -> Path:
    """Bar chart for e.g. route usage or Eve action usage."""
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(distribution)
    ax.bar(range(len(keys)), [distribution[k] for k in keys])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def plot_rates_by_condition(rates: dict[str, dict[str, Optional[float]]],
                            out_path: Path, *,
                            rate_keys: Sequence[str] = ("accept_rate", "abort_rate"),
                            title: str = "Outcome rates by condition (toy scope)") -> Path:
    """Grouped bars: condition name -> {rate_key: value}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    conditions = list(rates)
    width = 0.8 / max(len(rate_keys), 1)
    for j, rate_key in enumerate(rate_keys):
        xs = [i + j * width for i in range(len(conditions))]
        ys = [rates[c].get(rate_key) or 0.0 for c in conditions]
        ax.bar(xs, ys, width=width, label=rate_key)
    ax.set_xticks([i + width * (len(rate_keys) - 1) / 2 for i in range(len(conditions))])
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)
