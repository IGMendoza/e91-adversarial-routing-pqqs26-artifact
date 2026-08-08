"""Metrics aggregation over public trial records and RL episode metrics.

Inputs are plain dicts (public transcript fields, public action results,
rewards), so no private transcript data is required or consumed. Rewards are
game-design quantities; none of these metrics is a security claim, and reward
is never key/information gain.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from ..protocol.toy_trial import DISRUPTED_REASON


class MetricsError(ValueError):
    """Invalid metrics inputs."""


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def aggregate_honest_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accept/abort rates, QBER/latency means, route usage.

    Each record needs ``accepted``; ``qber_estimate``/``latency_ps``/``route_id``
    are optional (None values are skipped).
    """
    if not records:
        raise MetricsError("no records to aggregate")
    decided = [r for r in records if r.get("accepted") is not None]
    accepts = sum(1 for r in decided if r["accepted"])
    qbers = [r["qber_estimate"] for r in records if r.get("qber_estimate") is not None]
    latencies = [r["latency_ps"] for r in records if r.get("latency_ps") is not None]
    routes = Counter(r["route_id"] for r in records if r.get("route_id"))
    return {
        "num_trials": len(records),
        "accept_rate": accepts / len(decided) if decided else None,
        "abort_rate": (len(decided) - accepts) / len(decided) if decided else None,
        "qber_mean": _mean(qbers),
        "latency_ps_mean": _mean(latencies),
        "route_usage": dict(sorted(routes.items())),
    }


def aggregate_game_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Eve-side game metrics: disruption hits, action usage, cost, reward."""
    if not records:
        raise MetricsError("no records to aggregate")
    disruptions = sum(1 for r in records
                      if r.get("abort_reason") == DISRUPTED_REASON)
    attacks = [r for r in records if r.get("action_id") not in (None, "no_attack")]
    actions = Counter(r["action_id"] for r in records if r.get("action_id"))
    costs = [r.get("cost", 0.0) for r in records]
    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    return {
        "num_trials": len(records),
        "disruption_rate": disruptions / len(records),
        "eve_hit_rate": disruptions / len(attacks) if attacks else None,
        "action_usage": dict(sorted(actions.items())),
        "total_attack_cost": sum(costs),
        "cumulative_reward": sum(rewards) if rewards else None,
    }


def moving_average(values: Sequence[float], window: int) -> list[float]:
    if window < 1:
        raise MetricsError("window must be >= 1")
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def aggregate_training_metrics(episode_rewards: Sequence[float],
                               window: int = 10) -> dict[str, Any]:
    if not episode_rewards:
        raise MetricsError("no episode rewards")
    ma = moving_average(episode_rewards, window)
    return {
        "episodes": len(episode_rewards),
        "cumulative_reward": sum(episode_rewards),
        "mean_reward": sum(episode_rewards) / len(episode_rewards),
        "final_moving_average": ma[-1],
        "moving_average_window": window,
    }


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any],
                      keys: Sequence[str]) -> dict[str, Any]:
    """Numeric deltas candidate - baseline for the given keys (None-safe)."""
    out = {}
    for key in keys:
        b, c = baseline.get(key), candidate.get(key)
        out[key] = None if (b is None or c is None) else c - b
    return out


def write_json(data: dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True),
                          encoding="utf-8")


def write_records_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    if not records:
        raise MetricsError("no records to write")
    fieldnames = sorted({k for r in records for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
