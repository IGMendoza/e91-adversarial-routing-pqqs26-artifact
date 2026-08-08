"""Standard route-length profiles and cached clean SeQUeNCe baselines."""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .backend import TurnResult
from .config import (
    BASELINE_PRECOMPUTE_SAMPLES_PER_HOP,
    CORPUS_BASE_EDGE_LENGTH_M,
    CORPUS_EDGE_LENGTH_STEP_M,
    MAX_ROUTE_HOPS,
)

BASELINE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BaselineCacheTask:
    hop_count: int
    sample_index: int
    length_vector_m: tuple[float, ...]
    seed: int

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return route_physics_cache_key_from_vector(self.length_vector_m)

    @property
    def route_id(self) -> str:
        return f"baseline_h{self.hop_count}_s{self.sample_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_count": self.hop_count,
            "sample_index": self.sample_index,
            "length_vector_m": list(self.length_vector_m),
            "seed": self.seed,
            "cache_key": list(self.cache_key),
            "route_id": self.route_id,
        }


def standard_route_length_vector(hop_count: int) -> tuple[float, ...]:
    if hop_count < 1:
        raise ValueError("hop_count must be >= 1")
    return tuple(
        float(CORPUS_BASE_EDGE_LENGTH_M + index * CORPUS_EDGE_LENGTH_STEP_M)
        for index in range(hop_count)
    )


def standard_route_length_profiles(
        max_hops: int = MAX_ROUTE_HOPS,
) -> dict[int, tuple[float, ...]]:
    return {
        hop_count: standard_route_length_vector(hop_count)
        for hop_count in range(1, max_hops + 1)
    }


def route_physics_cache_key(route: dict[str, Any]) -> tuple[Any, ...]:
    edge_lengths = tuple(
        round(float(length), 9)
        for length in route.get("edge_lengths_m", ())
    )
    if not edge_lengths:
        edge_lengths = standard_route_length_vector(int(route["hop_count"]))
    return route_physics_cache_key_from_vector(edge_lengths)


def route_physics_cache_key_from_vector(
        edge_lengths_m: tuple[float, ...],
) -> tuple[Any, ...]:
    edge_lengths = tuple(round(float(length), 9) for length in edge_lengths_m)
    return (
        "clean_route_baseline",
        len(edge_lengths),
        edge_lengths,
    )


def normalize_route_lengths(route: dict[str, Any]) -> dict[str, Any]:
    hop_count = int(route["hop_count"])
    vector = standard_route_length_vector(hop_count)
    total = float(sum(vector))
    return {
        **route,
        "edge_lengths_m": list(vector),
        "total_length": total,
        "total_length_m": total,
        "length_profile_id": f"standard_h{hop_count}",
    }


def normalize_case_payload(payload: dict[str, Any]) -> dict[str, Any]:
    routes = [normalize_route_lengths(route) for route in payload.get("routes", [])]
    features = dict(payload.get("features", {}))
    if routes:
        lengths = [float(route["total_length_m"]) for route in routes]
        hops = [int(route["hop_count"]) for route in routes]
        features.update({
            "mean_route_length_m": sum(lengths) / len(lengths),
            "shortest_hops": float(min(hops)),
            "longest_hops": float(max(hops)),
            "route_hop_ratio": float(max(hops) / max(1, min(hops))),
        })
    topology = dict(payload.get("topology", payload.get("ir_dict", {})))
    metadata = dict(topology.get("metadata", {}))
    params = dict(metadata.get("params", {}))
    params.update({
        "route_length_profile": "standard_by_hop_count",
        "route_length_profiles_m": {
            str(hops): list(vector)
            for hops, vector in standard_route_length_profiles().items()
        },
        "route_edge_lengths_are_canonical_for_sequence_runtime": True,
    })
    metadata["params"] = params
    topology["metadata"] = metadata
    return {
        **payload,
        "routes": routes,
        "features": features,
        "topology": topology,
    }


def initialize_baseline_cache(
        path: Path,
        *,
        metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        with con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS baseline_samples (
                    hop_count INTEGER NOT NULL,
                    sample_index INTEGER NOT NULL,
                    cache_key_json TEXT NOT NULL,
                    length_vector_json TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    public_outcome TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    chsh_s REAL,
                    qber REAL,
                    delivered_count INTEGER NOT NULL,
                    sifted_count INTEGER NOT NULL,
                    wall_seconds REAL NOT NULL,
                    PRIMARY KEY (hop_count, sample_index)
                );

                CREATE INDEX IF NOT EXISTS idx_baseline_cache_key
                ON baseline_samples(cache_key_json);
                """
            )
            payload = {
                "schema_version": BASELINE_CACHE_SCHEMA_VERSION,
                **metadata,
            }
            con.executemany(
                """
                INSERT INTO metadata(key, value_json)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
                """,
                [
                    (str(key), json.dumps(value, sort_keys=True))
                    for key, value in sorted(payload.items())
                ],
            )
    finally:
        con.close()


def insert_baseline_sample(
        path: Path,
        *,
        task: BaselineCacheTask,
        result: TurnResult,
        wall_seconds: float,
) -> None:
    con = sqlite3.connect(path)
    try:
        with con:
            con.execute(
                """
                INSERT INTO baseline_samples(
                    hop_count, sample_index, cache_key_json, length_vector_json,
                    seed, result_json, public_outcome, accepted, chsh_s, qber,
                    delivered_count, sifted_count, wall_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hop_count, sample_index) DO UPDATE SET
                    cache_key_json=excluded.cache_key_json,
                    length_vector_json=excluded.length_vector_json,
                    seed=excluded.seed,
                    result_json=excluded.result_json,
                    public_outcome=excluded.public_outcome,
                    accepted=excluded.accepted,
                    chsh_s=excluded.chsh_s,
                    qber=excluded.qber,
                    delivered_count=excluded.delivered_count,
                    sifted_count=excluded.sifted_count,
                    wall_seconds=excluded.wall_seconds
                """,
                (
                    task.hop_count,
                    task.sample_index,
                    _cache_key_json(task.cache_key),
                    json.dumps(list(task.length_vector_m), sort_keys=True),
                    task.seed,
                    json.dumps(result.to_dict(), sort_keys=True),
                    result.public_outcome,
                    1 if result.accepted else 0,
                    result.chsh_s,
                    result.qber,
                    result.delivered_count,
                    result.sifted_count,
                    float(wall_seconds),
                ),
            )
    finally:
        con.close()


def existing_baseline_sample_keys(path: Path) -> set[tuple[int, int]]:
    path = Path(path)
    if not path.exists():
        return set()
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT hop_count, sample_index FROM baseline_samples"
        ).fetchall()
    finally:
        con.close()
    return {(int(hops), int(sample)) for hops, sample in rows}


def baseline_cache_summary(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT hop_count, COUNT(*), AVG(accepted), AVG(delivered_count),
                   AVG(chsh_s), AVG(wall_seconds)
            FROM baseline_samples
            GROUP BY hop_count
            ORDER BY hop_count
            """
        ).fetchall()
        total = con.execute("SELECT COUNT(*) FROM baseline_samples").fetchone()[0]
    finally:
        con.close()
    return {
        "exists": True,
        "path": str(path),
        "sample_count": int(total),
        "by_hop": {
            str(hops): {
                "samples": int(count),
                "accepted_rate": float(accepted_rate or 0.0),
                "mean_delivered_count": float(mean_delivered or 0.0),
                "mean_chsh_s": None if mean_chsh_s is None else float(mean_chsh_s),
                "mean_wall_seconds": float(mean_wall or 0.0),
            }
            for hops, count, accepted_rate, mean_delivered, mean_chsh_s, mean_wall
            in rows
        },
    }


def baseline_cache_hop_statistics(path: Path) -> list[dict[str, Any]]:
    """Return per-hop plotting statistics from the clean baseline cache."""

    path = Path(path)
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT hop_count, public_outcome, accepted, chsh_s, qber,
                   delivered_count
            FROM baseline_samples
            ORDER BY hop_count, sample_index
            """
        ).fetchall()
    finally:
        con.close()

    grouped: dict[int, list[Any]] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)

    stats = []
    for hop_count, group in sorted(grouped.items()):
        signed_s = [
            float(row[3]) for row in group
            if row[3] is not None
        ]
        abs_s = [abs(value) for value in signed_s]
        qbers = [
            float(row[4]) for row in group
            if row[4] is not None
        ]
        delivered = [float(row[5]) for row in group]
        outcomes: dict[str, int] = {}
        for row in group:
            outcomes[str(row[1])] = outcomes.get(str(row[1]), 0) + 1
        stats.append({
            "hop_count": hop_count,
            "samples": len(group),
            "accepted_rate": (
                sum(1 for row in group if int(row[2])) / len(group)
                if group else 0.0
            ),
            "mean_chsh_s": _stat_mean(signed_s),
            "sd_chsh_s": _stat_stdev(signed_s),
            "mean_abs_chsh_s": _stat_mean(abs_s),
            "sd_abs_chsh_s": _stat_stdev(abs_s),
            "mean_qber": _stat_mean(qbers),
            "sd_qber": _stat_stdev(qbers),
            "mean_delivered_count": _stat_mean(delivered),
            "outcome_counts": dict(sorted(outcomes.items())),
        })
    return stats


def cached_baseline_sample_count(
        path: Path,
        *,
        cache_key: tuple[Any, ...],
) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    con = sqlite3.connect(path)
    try:
        count = con.execute(
            """
            SELECT COUNT(*)
            FROM baseline_samples
            WHERE cache_key_json = ?
            """,
            (_cache_key_json(cache_key),),
        ).fetchone()[0]
    finally:
        con.close()
    return int(count)


def load_cached_baseline_results(
        path: Path,
        *,
        cache_key: tuple[Any, ...],
        sample_count: int,
        seed: int,
) -> list[TurnResult] | None:
    path = Path(path)
    if not path.exists():
        return None
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT result_json
            FROM baseline_samples
            WHERE cache_key_json = ?
            ORDER BY sample_index
            """,
            (_cache_key_json(cache_key),),
        ).fetchall()
    finally:
        con.close()
    if len(rows) < sample_count:
        return None
    rng = random.Random(f"{seed}:{_cache_key_json(cache_key)}:{sample_count}")
    selected = sorted(rng.sample(range(len(rows)), sample_count))
    return [
        turn_result_from_dict(json.loads(rows[index][0]))
        for index in selected
    ]


def turn_result_from_dict(payload: dict[str, Any]) -> TurnResult:
    return TurnResult(
        public_outcome=str(payload["public_outcome"]),
        alice_reward=float(payload["alice_reward"]),
        eve_hit_reward=float(payload["eve_hit_reward"]),
        active_route_attacked=bool(payload["active_route_attacked"]),
        accepted=bool(payload["accepted"]),
        qber=payload.get("qber"),
        chsh_s=payload.get("chsh_s"),
        chsh_adequately_sampled=payload.get("chsh_adequately_sampled"),
        delivered_count=int(payload["delivered_count"]),
        sifted_count=int(payload["sifted_count"]),
        fidelity=payload.get("fidelity"),
        runtime_engine=str(payload["runtime_engine"]),
        runtime_attack_applied=dict(payload.get("runtime_attack_applied") or {}),
        sequence_timing=dict(payload.get("sequence_timing") or {}),
        eve_feedback_reward=(
            None
            if payload.get("eve_feedback_reward") is None
            else float(payload["eve_feedback_reward"])
        ),
        eve_feedback_model=str(
            payload.get("eve_feedback_model")
            or "legacy_target_overlap_diagnostic_only"
        ),
        eve_feedback_observation=dict(
            payload.get("eve_feedback_observation") or {}
        ),
    )


def _stat_mean(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def _stat_stdev(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def build_baseline_tasks(
        *,
        max_hops: int = MAX_ROUTE_HOPS,
        samples_per_hop: int = BASELINE_PRECOMPUTE_SAMPLES_PER_HOP,
        seed: int,
) -> list[BaselineCacheTask]:
    tasks = []
    for hop_count in range(1, max_hops + 1):
        vector = standard_route_length_vector(hop_count)
        for sample_index in range(samples_per_hop):
            tasks.append(BaselineCacheTask(
                hop_count=hop_count,
                sample_index=sample_index,
                length_vector_m=vector,
                seed=seed + hop_count * 10_000 + sample_index,
            ))
    return tasks


def _cache_key_json(cache_key: tuple[Any, ...]) -> str:
    return json.dumps(_jsonable(cache_key), sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
