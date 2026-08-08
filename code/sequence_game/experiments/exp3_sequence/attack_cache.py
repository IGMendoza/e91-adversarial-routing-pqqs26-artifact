"""Cached attacked SeQUeNCe route-profile samples for Exp3 diagnostics."""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .backend import ActionSpec, TurnResult
from .baseline_cache import (
    standard_route_length_vector,
    turn_result_from_dict,
)
from .config import (
    ATTACK_PRECOMPUTE_SAMPLES_PER_HOP,
    ATTACK_PRECOMPUTE_SEED,
    MAX_ROUTE_HOPS,
)

ATTACK_CACHE_SCHEMA_VERSION = 1
ATTACK_KINDS = ("edge_intercept_resend", "memory_degradation")


@dataclass(frozen=True)
class AttackCacheTask:
    attack_kind: str
    hop_count: int
    sample_index: int
    length_vector_m: tuple[float, ...]
    seed: int
    target_kind: str
    target_index: int
    target_count: int
    target_id: str
    target_u: str = ""
    target_v: str = ""

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return attack_route_cache_key_from_vector(
            self.length_vector_m,
            self.attack_kind,
        )

    @property
    def route_id(self) -> str:
        return f"attack_{self.attack_kind}_h{self.hop_count}_s{self.sample_index}"

    @property
    def action(self) -> ActionSpec:
        return ActionSpec(
            f"{self.attack_kind}:{self.target_id}",
            self.target_kind,
            self.target_id,
            self.attack_kind,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_kind": self.attack_kind,
            "hop_count": self.hop_count,
            "sample_index": self.sample_index,
            "length_vector_m": list(self.length_vector_m),
            "seed": self.seed,
            "cache_key": list(self.cache_key),
            "route_id": self.route_id,
            "target_kind": self.target_kind,
            "target_index": self.target_index,
            "target_count": self.target_count,
            "target_id": self.target_id,
            "target_u": self.target_u,
            "target_v": self.target_v,
            "action": self.action.to_dict(),
        }


@dataclass(frozen=True)
class CachedAttackSample:
    attack_kind: str
    hop_count: int
    sample_index: int
    seed: int
    cache_key: tuple[Any, ...]
    length_vector_m: tuple[float, ...]
    target_kind: str
    target_index: int
    target_count: int
    target_id: str
    target_u: str
    target_v: str
    result: TurnResult
    wall_seconds: float
    eve_information_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_kind": self.attack_kind,
            "hop_count": self.hop_count,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "cache_key": list(self.cache_key),
            "length_vector_m": list(self.length_vector_m),
            "target_kind": self.target_kind,
            "target_index": self.target_index,
            "target_count": self.target_count,
            "target_id": self.target_id,
            "target_u": self.target_u,
            "target_v": self.target_v,
            "result": self.result.to_dict(),
            "wall_seconds": self.wall_seconds,
            "eve_information_status": self.eve_information_status,
        }


def attack_route_cache_key(
        route: dict[str, Any],
        attack_kind: str,
) -> tuple[Any, ...]:
    edge_lengths = tuple(
        round(float(length), 9)
        for length in route.get("edge_lengths_m", ())
    )
    if not edge_lengths:
        edge_lengths = standard_route_length_vector(int(route["hop_count"]))
    return attack_route_cache_key_from_vector(edge_lengths, attack_kind)


def attack_route_cache_key_from_vector(
        edge_lengths_m: tuple[float, ...],
        attack_kind: str,
) -> tuple[Any, ...]:
    if attack_kind not in ATTACK_KINDS:
        raise ValueError(f"unsupported attack kind {attack_kind!r}")
    edge_lengths = tuple(round(float(length), 9) for length in edge_lengths_m)
    return (
        "attacked_route_profile",
        attack_kind,
        len(edge_lengths),
        edge_lengths,
    )


def build_attack_tasks(
        *,
        max_hops: int = MAX_ROUTE_HOPS,
        samples_per_hop: int = ATTACK_PRECOMPUTE_SAMPLES_PER_HOP,
        seed: int = ATTACK_PRECOMPUTE_SEED,
        attack_kinds: tuple[str, ...] = ATTACK_KINDS,
) -> list[AttackCacheTask]:
    tasks: list[AttackCacheTask] = []
    for attack_kind in attack_kinds:
        if attack_kind not in ATTACK_KINDS:
            raise ValueError(f"unsupported attack kind {attack_kind!r}")
        for hop_count in range(1, max_hops + 1):
            target_count = _target_count(attack_kind, hop_count)
            if target_count == 0:
                continue
            target_indices = balanced_random_target_indices(
                target_count,
                samples_per_hop,
                seed=seed + _attack_seed_offset(attack_kind) + hop_count * 10_000,
            )
            vector = standard_route_length_vector(hop_count)
            for sample_index, target_index in enumerate(target_indices):
                target = attack_target_metadata(
                    attack_kind,
                    hop_count,
                    target_index,
                )
                tasks.append(AttackCacheTask(
                    attack_kind=attack_kind,
                    hop_count=hop_count,
                    sample_index=sample_index,
                    length_vector_m=vector,
                    seed=(
                        seed
                        + _attack_seed_offset(attack_kind)
                        + hop_count * 10_000
                        + sample_index
                    ),
                    target_kind=target["target_kind"],
                    target_index=target_index,
                    target_count=target_count,
                    target_id=target["target_id"],
                    target_u=target.get("target_u", ""),
                    target_v=target.get("target_v", ""),
                ))
    return tasks


def balanced_random_target_indices(
        target_count: int,
        sample_count: int,
        *,
        seed: int,
) -> tuple[int, ...]:
    if target_count < 1:
        raise ValueError("target_count must be >= 1")
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    indices = [
        index % target_count
        for index in range(sample_count)
    ]
    rng = random.Random(seed)
    rng.shuffle(indices)
    return tuple(indices)


def attack_target_metadata(
        attack_kind: str,
        hop_count: int,
        target_index: int,
) -> dict[str, str]:
    nodes = nodes_for_hops(hop_count)
    if attack_kind == "edge_intercept_resend":
        if not 0 <= target_index < hop_count:
            raise ValueError("edge target_index outside route")
        u, v = nodes[target_index], nodes[target_index + 1]
        return {
            "target_kind": "edge",
            "target_id": f"{u}-{v}",
            "target_u": u,
            "target_v": v,
        }
    if attack_kind == "memory_degradation":
        internal_nodes = nodes[1:-1]
        if not 0 <= target_index < len(internal_nodes):
            raise ValueError("memory target_index outside route")
        return {
            "target_kind": "node",
            "target_id": internal_nodes[target_index],
        }
    raise ValueError(f"unsupported attack kind {attack_kind!r}")


def nodes_for_hops(hops: int) -> list[str]:
    if hops < 1:
        raise ValueError("hops must be >= 1")
    nodes = ["alice"]
    nodes.extend(f"r{index}" for index in range(1, hops))
    nodes.append("bob")
    return nodes


def initialize_attack_cache(
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

                CREATE TABLE IF NOT EXISTS attack_samples (
                    attack_kind TEXT NOT NULL,
                    hop_count INTEGER NOT NULL,
                    sample_index INTEGER NOT NULL,
                    cache_key_json TEXT NOT NULL,
                    length_vector_json TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_index INTEGER NOT NULL,
                    target_count INTEGER NOT NULL,
                    target_id TEXT NOT NULL,
                    target_u TEXT NOT NULL,
                    target_v TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    public_outcome TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    active_route_attacked INTEGER NOT NULL,
                    alice_reward REAL NOT NULL,
                    eve_hit_reward REAL NOT NULL,
                    chsh_s REAL,
                    qber REAL,
                    chsh_adequately_sampled INTEGER,
                    delivered_count INTEGER NOT NULL,
                    sifted_count INTEGER NOT NULL,
                    fidelity REAL,
                    eve_information_status TEXT NOT NULL,
                    wall_seconds REAL NOT NULL,
                    PRIMARY KEY (attack_kind, hop_count, sample_index)
                );

                CREATE INDEX IF NOT EXISTS idx_attack_cache_key
                ON attack_samples(cache_key_json);

                CREATE INDEX IF NOT EXISTS idx_attack_target
                ON attack_samples(attack_kind, hop_count, target_kind, target_index);
                """
            )
            payload = {
                "schema_version": ATTACK_CACHE_SCHEMA_VERSION,
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


def insert_attack_sample(
        path: Path,
        *,
        task: AttackCacheTask,
        result: TurnResult,
        wall_seconds: float,
        eve_information_status: str,
) -> None:
    con = sqlite3.connect(path)
    try:
        with con:
            con.execute(
                """
                INSERT INTO attack_samples(
                    attack_kind, hop_count, sample_index, cache_key_json,
                    length_vector_json, seed, target_kind, target_index,
                    target_count, target_id, target_u, target_v, action_id,
                    result_json, public_outcome, accepted, active_route_attacked,
                    alice_reward, eve_hit_reward, chsh_s, qber,
                    chsh_adequately_sampled, delivered_count, sifted_count,
                    fidelity, eve_information_status, wall_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attack_kind, hop_count, sample_index) DO UPDATE SET
                    cache_key_json=excluded.cache_key_json,
                    length_vector_json=excluded.length_vector_json,
                    seed=excluded.seed,
                    target_kind=excluded.target_kind,
                    target_index=excluded.target_index,
                    target_count=excluded.target_count,
                    target_id=excluded.target_id,
                    target_u=excluded.target_u,
                    target_v=excluded.target_v,
                    action_id=excluded.action_id,
                    result_json=excluded.result_json,
                    public_outcome=excluded.public_outcome,
                    accepted=excluded.accepted,
                    active_route_attacked=excluded.active_route_attacked,
                    alice_reward=excluded.alice_reward,
                    eve_hit_reward=excluded.eve_hit_reward,
                    chsh_s=excluded.chsh_s,
                    qber=excluded.qber,
                    chsh_adequately_sampled=excluded.chsh_adequately_sampled,
                    delivered_count=excluded.delivered_count,
                    sifted_count=excluded.sifted_count,
                    fidelity=excluded.fidelity,
                    eve_information_status=excluded.eve_information_status,
                    wall_seconds=excluded.wall_seconds
                """,
                (
                    task.attack_kind,
                    task.hop_count,
                    task.sample_index,
                    _cache_key_json(task.cache_key),
                    json.dumps(list(task.length_vector_m), sort_keys=True),
                    task.seed,
                    task.target_kind,
                    task.target_index,
                    task.target_count,
                    task.target_id,
                    task.target_u,
                    task.target_v,
                    task.action.action_id,
                    json.dumps(result.to_dict(), sort_keys=True),
                    result.public_outcome,
                    1 if result.accepted else 0,
                    1 if result.active_route_attacked else 0,
                    result.alice_reward,
                    result.eve_hit_reward,
                    result.chsh_s,
                    result.qber,
                    _nullable_bool(result.chsh_adequately_sampled),
                    result.delivered_count,
                    result.sifted_count,
                    result.fidelity,
                    eve_information_status,
                    float(wall_seconds),
                ),
            )
    finally:
        con.close()


def existing_attack_sample_keys(path: Path) -> set[tuple[str, int, int]]:
    path = Path(path)
    if not path.exists():
        return set()
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT attack_kind, hop_count, sample_index
            FROM attack_samples
            """
        ).fetchall()
    finally:
        con.close()
    return {
        (str(kind), int(hops), int(sample_index))
        for kind, hops, sample_index in rows
    }


def attack_cache_summary(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT attack_kind, hop_count, COUNT(*), AVG(accepted),
                   AVG(delivered_count), AVG(chsh_s), AVG(qber),
                   AVG(eve_hit_reward), AVG(wall_seconds)
            FROM attack_samples
            GROUP BY attack_kind, hop_count
            ORDER BY attack_kind, hop_count
            """
        ).fetchall()
        targets = con.execute(
            """
            SELECT attack_kind, hop_count, target_index, COUNT(*)
            FROM attack_samples
            GROUP BY attack_kind, hop_count, target_index
            ORDER BY attack_kind, hop_count, target_index
            """
        ).fetchall()
        total = con.execute("SELECT COUNT(*) FROM attack_samples").fetchone()[0]
    finally:
        con.close()
    by_kind_hop: dict[str, dict[str, Any]] = {}
    for kind, hops, count, accepted, delivered, chsh_s, qber, eve_hit, mean_wall in rows:
        by_kind_hop.setdefault(str(kind), {})[str(hops)] = {
            "samples": int(count),
            "accepted_rate": float(accepted or 0.0),
            "mean_delivered_count": float(delivered or 0.0),
            "mean_chsh_s": None if chsh_s is None else float(chsh_s),
            "mean_qber": None if qber is None else float(qber),
            "mean_eve_hit_reward": float(eve_hit or 0.0),
            "mean_wall_seconds": float(mean_wall or 0.0),
        }
    target_counts: dict[str, dict[str, dict[str, int]]] = {}
    for kind, hops, target_index, count in targets:
        target_counts.setdefault(str(kind), {}).setdefault(str(hops), {})[
            str(target_index)
        ] = int(count)
    return {
        "exists": True,
        "path": str(path),
        "sample_count": int(total),
        "by_attack_kind": by_kind_hop,
        "target_index_counts": target_counts,
    }


def attack_cache_hop_statistics(path: Path) -> list[dict[str, Any]]:
    """Return per-attack/per-hop plotting statistics from the SQLite cache."""

    path = Path(path)
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT attack_kind, hop_count, public_outcome, accepted, chsh_s,
                   qber, delivered_count, target_index
            FROM attack_samples
            ORDER BY attack_kind, hop_count, sample_index
            """
        ).fetchall()
    finally:
        con.close()

    grouped: dict[tuple[str, int], list[Any]] = {}
    for row in rows:
        grouped.setdefault((str(row[0]), int(row[1])), []).append(row)

    stats: list[dict[str, Any]] = []
    for (attack_kind, hop_count), group in sorted(grouped.items()):
        signed_s = [
            float(row[4]) for row in group
            if row[4] is not None
        ]
        abs_s = [abs(value) for value in signed_s]
        qbers = [
            float(row[5]) for row in group
            if row[5] is not None
        ]
        delivered = [float(row[6]) for row in group]
        outcomes: dict[str, int] = {}
        target_counts: dict[int, int] = {}
        for row in group:
            outcomes[str(row[2])] = outcomes.get(str(row[2]), 0) + 1
            target_index = int(row[7])
            target_counts[target_index] = target_counts.get(target_index, 0) + 1
        stats.append({
            "attack_kind": attack_kind,
            "hop_count": hop_count,
            "samples": len(group),
            "accepted_rate": (
                sum(1 for row in group if int(row[3])) / len(group)
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
            "target_index_counts": {
                str(key): value for key, value in sorted(target_counts.items())
            },
        })
    return stats


def cached_attack_sample_count(
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
            FROM attack_samples
            WHERE cache_key_json = ?
            """,
            (_cache_key_json(cache_key),),
        ).fetchone()[0]
    finally:
        con.close()
    return int(count)


def load_cached_attack_samples(
        path: Path,
        *,
        cache_key: tuple[Any, ...],
        sample_count: int,
        seed: int,
) -> list[CachedAttackSample] | None:
    path = Path(path)
    if not path.exists():
        return None
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT attack_kind, hop_count, sample_index, seed, cache_key_json,
                   length_vector_json, target_kind, target_index, target_count,
                   target_id, target_u, target_v, result_json, wall_seconds,
                   eve_information_status
            FROM attack_samples
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
        _cached_attack_sample_from_row(rows[index])
        for index in selected
    ]


def load_cached_attack_results(
        path: Path,
        *,
        cache_key: tuple[Any, ...],
        sample_count: int,
        seed: int,
) -> list[TurnResult] | None:
    samples = load_cached_attack_samples(
        path,
        cache_key=cache_key,
        sample_count=sample_count,
        seed=seed,
    )
    if samples is None:
        return None
    return [sample.result for sample in samples]


def _stat_mean(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def _stat_stdev(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _target_count(attack_kind: str, hop_count: int) -> int:
    if attack_kind == "edge_intercept_resend":
        return max(0, hop_count)
    if attack_kind == "memory_degradation":
        return max(0, hop_count - 1)
    raise ValueError(f"unsupported attack kind {attack_kind!r}")


def _attack_seed_offset(attack_kind: str) -> int:
    if attack_kind == "edge_intercept_resend":
        return 1_000_000
    if attack_kind == "memory_degradation":
        return 2_000_000
    raise ValueError(f"unsupported attack kind {attack_kind!r}")


def _cached_attack_sample_from_row(row: Any) -> CachedAttackSample:
    (
        attack_kind,
        hop_count,
        sample_index,
        seed,
        cache_key_json,
        length_vector_json,
        target_kind,
        target_index,
        target_count,
        target_id,
        target_u,
        target_v,
        result_json,
        wall_seconds,
        eve_information_status,
    ) = row
    return CachedAttackSample(
        attack_kind=str(attack_kind),
        hop_count=int(hop_count),
        sample_index=int(sample_index),
        seed=int(seed),
        cache_key=_tuple_from_json(json.loads(cache_key_json)),
        length_vector_m=tuple(float(value) for value in json.loads(length_vector_json)),
        target_kind=str(target_kind),
        target_index=int(target_index),
        target_count=int(target_count),
        target_id=str(target_id),
        target_u=str(target_u),
        target_v=str(target_v),
        result=turn_result_from_dict(json.loads(result_json)),
        wall_seconds=float(wall_seconds),
        eve_information_status=str(eve_information_status),
    )


def _nullable_bool(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _cache_key_json(cache_key: tuple[Any, ...]) -> str:
    return json.dumps(_jsonable(cache_key), sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _tuple_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_from_json(item) for item in value)
    return value
